"""binary structural causal model on a time grid with threshold noise."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from core.preprocessing import ViewData
from utils import io
from utils.errors import MissingInputError

# jeffreys prior (add 0.5), the non-informative prior of a bernoulli share
SMOOTHING = 0.5
PROB_CLIP = 1e-9


# series helpers


def recording_series(view_data: ViewData, recording_id: str) -> dict[str, np.ndarray]:
    """variable -> float array with nan where the state is unknown."""
    values = view_data.arrays[recording_id].astype(float).copy()
    values[view_data.masks[recording_id]] = np.nan
    return {name: values[:, j] for j, name in enumerate(view_data.var_names)}


def fill_values(view_data: ViewData, recording_ids: Sequence[str]) -> dict[str, float]:
    """most frequent state of every variable over the given recordings (0 if never recorded)."""
    out = {}
    for j, name in enumerate(view_data.var_names):
        observed = np.concatenate(
            [view_data.arrays[r][~view_data.masks[r][:, j], j] for r in recording_ids]
        )
        out[name] = float(np.round(observed.mean())) if observed.size else 0.0
    return out


# mechanisms


# x[t] = 1 if u[t] < p(x[t-1], parents at t, t-1 and extra lags), with u ~ uniform(0, 1)
@dataclass
class Mechanism:
    variable: str
    parents: list[str]
    # previous state -> p(x[t] = 1) for every parent configuration (index = binary code)
    tables: dict[int, np.ndarray] = field(default_factory=dict)
    # previous state -> rows seen for every parent configuration
    counts: dict[int, np.ndarray] = field(default_factory=dict)
    n_rows: dict[int, int] = field(default_factory=dict)
    n_changes: dict[int, int] = field(default_factory=dict)
    # parent -> extra lags (>= 2 steps) used on top of the parent now and one step earlier
    lags: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        """true when there was at least one training row."""
        return sum(self.n_rows.values()) > 0

    @property
    def width(self) -> int:
        return 2 * len(self.parents) + sum(len(self.lags.get(p, ())) for p in self.parents)

    @property
    def max_lag(self) -> int:
        """longest lag the features look back (1 for the first-order model)."""
        return max([1, *(lag for p in self.parents for lag in self.lags.get(p, ()))])

    def prob_one(self, prev: np.ndarray, features: np.ndarray) -> np.ndarray:
        """p(x[t] = 1) for each row, given the previous state and binary parent features."""
        prev = np.asarray(prev, dtype=float)
        codes = (np.asarray(features, dtype=int) @ (1 << np.arange(self.width))) if self.width else np.zeros(len(prev), dtype=int)
        out = np.full(len(prev), np.nan)
        for state in (0, 1):
            rows = prev == state
            if rows.any():
                out[rows] = self.tables[state][codes[rows]]
        return np.clip(out, PROB_CLIP, 1 - PROB_CLIP)


def parent_features(series: dict[str, np.ndarray], parents: Sequence[str], fills: dict[str, float],
                    lags: dict[str, tuple[int, ...]] | None = None) -> np.ndarray:
    """rows t = 1..T-1, columns [p1[t], p1[t-1], p1[t-L] per extra lag, p2[t], ...]."""
    lags = lags or {}
    n = len(next(iter(series.values())))
    steps = np.arange(1, n)
    columns = []
    for p in parents:
        values = np.where(np.isnan(series[p]), fills[p], series[p]) if p in series else np.full(n, fills[p])
        columns.append(values[1:])
        columns.append(values[:-1])
        for lag in lags.get(p, ()):
            columns.append(values[np.maximum(steps - lag, 0)])
    return np.column_stack(columns) if columns else np.zeros((n - 1, 0))


def history_rows(n: int, max_lag: int) -> np.ndarray:
    """rows t = 1..T-1 whose lagged parent values all lie inside the recording (t >= max_lag)."""
    return np.arange(1, n) >= max_lag


def fit_mechanism(
    variable: str,
    parents: Sequence[str],
    training: Iterable[dict[str, np.ndarray]],
    fills: dict[str, float],
    smoothing: float = SMOOTHING,
    lags: dict[str, tuple[int, ...]] | None = None,
) -> Mechanism:
    """smoothed table p(x[t] = 1 | x[t-1], parents now, one step earlier and at extra lags)."""
    mechanism = Mechanism(variable=variable, parents=list(parents), lags=dict(lags or {}))
    width = mechanism.width
    if width > 20:
        raise ValueError(f"{variable}: too many parent features ({width}) for a probability table")
    features, prev, target = [], [], []
    for series in training:
        x = series[variable]
        f = parent_features(series, parents, fills, mechanism.lags)
        ok = ~np.isnan(x[1:]) & ~np.isnan(x[:-1]) & history_rows(len(x), mechanism.max_lag)
        features.append(f[ok])
        prev.append(x[:-1][ok])
        target.append(x[1:][ok])
    features = np.vstack(features) if features else np.zeros((0, width))
    prev = np.concatenate(prev) if prev else np.zeros(0)
    target = np.concatenate(target) if target else np.zeros(0)
    codes = (features.astype(int) @ (1 << np.arange(width))) if width else np.zeros(len(prev), dtype=int)

    size = 1 << width
    for state in (0, 1):
        rows = prev == state
        y = target[rows]
        mechanism.n_rows[state] = int(rows.sum())
        mechanism.n_changes[state] = int((y != state).sum())
        n = np.bincount(codes[rows], minlength=size).astype(float)
        ones = np.bincount(codes[rows], weights=y, minlength=size)
        # back-off for unseen configurations: smoothed share for this previous state
        fallback = (y.sum() + smoothing) / (len(y) + 2 * smoothing) if len(y) else 0.5
        table = np.where(n > 0, (ones + smoothing) / (n + 2 * smoothing), fallback)
        mechanism.tables[state] = table
        mechanism.counts[state] = n
    return mechanism


# the scm


@dataclass
class TemporalSCM:
    graph: nx.DiGraph
    mechanisms: dict[str, Mechanism]
    fills: dict[str, float]
    interval_s: float

    @property
    def order(self) -> list[str]:
        return [v for v in nx.topological_sort(self.graph) if v in self.mechanisms]

    # observational scoring

    def transition_probs(self, series: dict[str, np.ndarray], variable: str) -> np.ndarray:
        """p(x[t] = 1) for t = 1..T-1 given the observed previous state and parents (nan where unknown)."""
        mech = self.mechanisms[variable]
        x = series[variable]
        prev = x[:-1]
        probs = np.full(len(prev), np.nan)
        ok = ~np.isnan(prev)
        if ok.any():
            probs[ok] = mech.prob_one(prev[ok], parent_features(series, mech.parents, self.fills, mech.lags)[ok])
        return probs

    def surprise(self, series: dict[str, np.ndarray], variable: str) -> np.ndarray:
        """-log p(observed x[t] | x[t-1], parents) for t = 1..T-1 (nan where unknown)."""
        x = series[variable][1:]
        p = self.transition_probs(series, variable)
        return np.where(np.isnan(x) | np.isnan(p), np.nan, -np.log(np.where(x == 1, p, 1 - p)))

    # simulation, interventions and counterfactuals

    def simulate(
        self,
        initial: dict[str, float],
        noise: dict[str, np.ndarray],
        interventions: dict[str, tuple[float, int]] | None = None,
        fixed: dict[str, float] | None = None,
        history: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """roll the equations forward for noise of shape (samples, steps). returns variable -> (samples, steps)."""
        # interventions[v] = (value, first step) replaces the equation from that step on.
        # fixed holds context variables at one value. history[v] holds observed states before step 0
        interventions = interventions or {}
        history = history or {}
        fixed = {**{v: self.fills[v] for v, m in self.mechanisms.items() if not m.has_data}, **(fixed or {})}
        n_samples, n_steps = next(iter(noise.values())).shape
        out = {v: np.zeros((n_samples, n_steps)) for v in self.mechanisms}
        for v in self.mechanisms:
            out[v][:, 0] = initial.get(v, self.fills[v])
        order = self.order

        def past(p: str, step: int) -> np.ndarray:
            # state of p at a step before 0: observed history if it reaches back that far, else the initial state
            h = history.get(p)
            if h is not None and len(h) >= -step:
                value = h[len(h) + step]
                return np.full(n_samples, self.fills[p] if np.isnan(value) else value)
            return out[p][:, 0]

        for t in range(1, n_steps):
            for v in order:
                if v in interventions and t >= interventions[v][1]:
                    out[v][:, t] = interventions[v][0]
                    continue
                if v in fixed:
                    out[v][:, t] = fixed[v]
                    continue
                mech = self.mechanisms[v]
                columns = []
                for p in mech.parents:
                    columns.extend([out[p][:, t], out[p][:, t - 1]])
                    columns.extend(out[p][:, t - lag] if t >= lag else past(p, t - lag) for lag in mech.lags.get(p, ()))
                features = np.column_stack(columns) if columns else np.zeros((n_samples, 0))
                prob = mech.prob_one(out[v][:, t - 1], features)
                out[v][:, t] = (noise[v][:, t] < prob).astype(float)
        return out

    def abduct_noise(self, series: dict[str, np.ndarray], n_samples: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """posterior samples of u per variable and step: x = 1 means u < p, x = 0 means u >= p."""
        n_steps = len(next(iter(series.values())))
        noise = {}
        for v in self.mechanisms:
            u = rng.uniform(size=(n_samples, n_steps))
            if v in series:
                x = series[v][1:]
                p = self.transition_probs(series, v)
                known = ~np.isnan(x) & ~np.isnan(p)
                ones = known & (x == 1)
                zeros = known & (x == 0)
                u[:, 1:][:, ones] = u[:, 1:][:, ones] * p[ones]
                u[:, 1:][:, zeros] = p[zeros] + u[:, 1:][:, zeros] * (1 - p[zeros])
            noise[v] = u
        return noise


def fit_scm(
    view_data: ViewData,
    graph: nx.DiGraph,
    train_ids: Sequence[str],
    manipulated: dict[str, set[str]],
    interval_s: float,
    smoothing: float = SMOOTHING,
    lags: dict[tuple[str, str], Sequence[int]] | None = None,
) -> TemporalSCM:
    """one mechanism per graph node. repeated train ids keep their multiplicity, lags adds extra lags."""
    lags = {e: tuple(sorted(int(l) for l in ls)) for e, ls in (lags or {}).items() if len(ls)}
    for (u, v), ls in lags.items():
        if not graph.has_edge(u, v):
            raise ValueError(f"lags given for {u} -> {v}, which is not an edge of the graph")
        if min(ls) < 2:
            raise ValueError(f"{u} -> {v}: extra lags must be >= 2 steps (lags 0 and 1 are always used)")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("the scm needs an acyclic graph (lag-0 parents enter the equations)")
    # an undirected discovered edge must never become a parent by accident (use to_dag first)
    undirected = [(u, v) for u, v, a in graph.edges(data=True) if a.get("oriented", True) is False]
    if undirected:
        raise ValueError(f"graph contains undirected edges {undirected}. convert it with core.graphs.to_dag")
    fills = fill_values(view_data, train_ids)
    # one entry per listed recording: a recording drawn twice in a bootstrap sample counts twice
    cache = {r: recording_series(view_data, r) for r in dict.fromkeys(train_ids)}
    series = [(r, cache[r]) for r in train_ids]
    mechanisms = {}
    for v in graph.nodes:
        if v not in view_data.var_names:
            continue
        parents = sorted(p for p in graph.predecessors(v) if p in view_data.var_names)
        training = [s for r, s in series if v not in manipulated.get(r, set())]
        mech_lags = {p: lags[(p, v)] for p in parents if (p, v) in lags}
        mechanisms[v] = fit_mechanism(v, parents, training, fills, smoothing=smoothing, lags=mech_lags)
    return TemporalSCM(graph=graph.copy(), mechanisms=mechanisms, fills=fills, interval_s=interval_s)


def extra_lags(graph: nx.DiGraph, lag: int | None, max_features: int) -> dict[tuple[str, str], list[int]] | None:
    """the extra lag on every edge whose child keeps at most max_features table features (none: first order)."""
    if lag is None:
        return None
    return {(u, v): [lag] for u, v in graph.edges if 3 * graph.in_degree(v) <= max_features}


def fitted_path(subsystem: str, graph: str, scope: str, directory: Path | None = None) -> Path:
    """<directory>/<subsystem>_<graph>_<scope>.pkl, by default in data/models/scm."""
    return (directory or io.models_dir("scm")) / f"{subsystem}_{graph}_{scope}.pkl"


def save_fitted(model: TemporalSCM, subsystem: str, graph: str, scope: str, directory: Path | None = None) -> None:
    """store a fitted scm (by default in data/models/scm)."""
    path = fitted_path(subsystem, graph, scope, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))


def load_fitted(subsystem: str, graph: str, scope: str, directory: Path | None = None) -> TemporalSCM:
    """a fitted scm written by the scm stage."""
    path = fitted_path(subsystem, graph, scope, directory)
    if not path.exists():
        raise MissingInputError(f"{io.relative(path)} not found: run \"python main.py run scm\" first")
    return pickle.loads(path.read_bytes())


# diagnostics


def held_out_predictions(scm: TemporalSCM, view_data: ViewData, recording_ids: Sequence[str]) -> pd.DataFrame:
    """one row per variable and step with p(x[t] = 1), x[t] and x[t-1]."""
    frames = []
    for r in recording_ids:
        s = recording_series(view_data, r)
        for v in scm.mechanisms:
            if v not in s:
                continue
            p = scm.transition_probs(s, v)
            y, prev = s[v][1:], s[v][:-1]
            ok = ~np.isnan(p) & ~np.isnan(y)
            if ok.any():
                frames.append(pd.DataFrame({"recording_id": r, "variable": v, "step": np.flatnonzero(ok) + 1,
                                            "p": p[ok], "y": y[ok], "prev": prev[ok]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["recording_id", "variable", "step", "p", "y", "prev"])


def quantile_residuals(predictions: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """randomized quantile residuals (dunn and smyth) for binary outcomes. normal(0, 1) if the model is right."""
    from scipy.stats import norm

    p, y = predictions["p"].to_numpy(), predictions["y"].to_numpy()
    lower = np.where(y == 1, 1 - p, 0.0)
    upper = np.where(y == 1, 1.0, 1 - p)
    u = lower + rng.uniform(size=len(p)) * (upper - lower)
    return norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))


def simulated_behaviour(scm: TemporalSCM, view_data: ViewData, recording_ids: Sequence[str],
                        rng: np.random.Generator, n_samples: int = 20) -> pd.DataFrame:
    """share of time in state 1 and changes per minute: observed vs simulated from the same start state."""
    rows = []
    for r in recording_ids:
        s = recording_series(view_data, r)
        n = len(next(iter(s.values())))
        initial = {v: (s[v][0] if v in s and not np.isnan(s[v][0]) else scm.fills[v]) for v in scm.mechanisms}
        fixed = {v: scm.fills[v] for v in scm.mechanisms if v not in s or np.isnan(s[v]).all()}
        noise = {v: rng.uniform(size=(n_samples, n)) for v in scm.mechanisms}
        sim = scm.simulate(initial, noise, fixed=fixed)
        minutes = n * scm.interval_s / 60
        for v in scm.mechanisms:
            if v in fixed:
                continue
            obs = s[v][~np.isnan(s[v])]
            rows.append({
                "recording_id": r, "variable": v,
                "observed_share_on": float(obs.mean()),
                "simulated_share_on": float(sim[v].mean()),
                "observed_changes_per_min": float(np.count_nonzero(np.diff(obs)) / minutes),
                "simulated_changes_per_min": float(np.count_nonzero(np.diff(sim[v], axis=1)) / n_samples / minutes),
            })
    return pd.DataFrame(rows)
