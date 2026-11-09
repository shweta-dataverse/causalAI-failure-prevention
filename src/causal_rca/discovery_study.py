"""stage 12: a focused, reproducible hyperparameter study of the causal discovery step.

what it does and does not do:

* it re-runs the regime-wise pcmci+ of ``causal_discovery`` for a small grid of settings,
  aggregates the regimes exactly as notebooks 03 / 10 do, and scores every graph against the
  expert reference. the selection objective is fixed in ``configs/discovery_study.yaml``
  before any result exists (directed f1, then skeleton f1, then shorter lag, ...).
* selecting by agreement with the expert graph is *reference-guided tuning*: pcmci+ never sees
  expert edges, but the chosen settings do depend on them, so the selected scores are
  optimistic and are not independent test accuracy.
* every pcmci+ run is one regime (normal training recordings or one fault scenario). runs do
  not depend on other regimes, so leave-one-scenario-out graphs reuse them and only drop the
  held-out regime when aggregating; the support threshold and the post-hoc fdr filter also
  only change the aggregation and need no new runs.
* runs are cached on disk by a hash of everything that can change their result (data bytes,
  recordings, variables, settings, tigramite version), executed in separate processes and
  stopped after a timeout. failed or stopped runs are logged and never silently dropped.

scm fitting, rca, interventions and counterfactuals are not touched here.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from causal_rca import causal_discovery as cd
from causal_rca import graph_evaluation as ge
from causal_rca.preprocessing import ViewData, load_view_data

CI_TESTS = ("gsq", "cmisymb")

# the selection order of configs/discovery_study.yaml (checked against the yaml by a test)
SELECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("directed_f1", "max"),
    ("skeleton_f1", "max"),
    ("tau_max_s", "min"),
    ("pc_alpha", "min"),
    ("support_threshold", "max"),
    ("config_id", "min"),
)

# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyConfig:
    """one discovery configuration: pcmci+ settings plus the aggregation settings."""

    ci_test: str = "gsq"
    interval_s: float = 0.5
    tau_max_s: float = 10.0
    pc_alpha: float = 0.01
    max_conds: tuple[int, int, int] = (3, 3, 3)  # max_conds_dim, max_conds_px, max_conds_py
    fdr_method: str = "fdr_bh"
    apply_fdr_to_graph: bool = True
    support_threshold: float = 0.5
    sig_samples: int | None = None  # permutation draws, cmisymb only
    seed: int | None = None  # permutation seed, cmisymb only

    def __post_init__(self) -> None:
        if self.ci_test not in CI_TESTS:
            raise ValueError(f"unknown ci test {self.ci_test!r}")
        steps = self.tau_max_s / self.interval_s
        if self.tau_max_s < self.interval_s or abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"tau_max_s={self.tau_max_s} is not a positive multiple of interval_s={self.interval_s}")
        if not 0.0 < self.pc_alpha < 1.0:
            raise ValueError("pc_alpha must lie in (0, 1)")
        if not 0.0 < self.support_threshold <= 1.0:
            raise ValueError("support_threshold must lie in (0, 1]")
        if len(self.max_conds) != 3 or any(int(c) != c or c < 1 for c in self.max_conds):
            raise ValueError("max_conds must be three positive integers")
        if self.ci_test == "cmisymb" and not self.sig_samples:
            raise ValueError("cmisymb needs sig_samples")
        object.__setattr__(self, "max_conds", tuple(int(c) for c in self.max_conds))

    @property
    def tau_max(self) -> int:
        """maximum lag in grid steps."""
        return int(round(self.tau_max_s / self.interval_s))

    def discovery_config(self) -> cd.DiscoveryConfig:
        """the same settings as the config class of notebooks 03 / 10."""
        dim, px, py = self.max_conds
        return cd.DiscoveryConfig(
            interval_s=self.interval_s, tau_max_s=self.tau_max_s, pc_alpha=self.pc_alpha,
            fdr_method=self.fdr_method, max_conds_dim=dim, max_conds_px=px, max_conds_py=py,
            apply_fdr_to_graph=self.apply_fdr_to_graph, support_threshold=self.support_threshold,
        )

    def run_settings(self) -> dict[str, Any]:
        """settings that change a pcmci+ run (support threshold and fdr filter only aggregate)."""
        settings = {"ci_test": self.ci_test, "interval_s": self.interval_s, "tau_max": self.tau_max,
                    "pc_alpha": self.pc_alpha, "max_conds": list(self.max_conds), "fdr_method": self.fdr_method}
        if self.ci_test == "cmisymb":
            settings.update(sig_samples=self.sig_samples, seed=self.seed)
        return settings

    @property
    def run_id(self) -> str:
        dim, px, py = self.max_conds
        return f"{self.ci_test}_dt{self.interval_s:g}_tau{self.tau_max_s:g}s_a{self.pc_alpha:g}_c{dim}-{px}-{py}_{self.fdr_method}"

    @property
    def config_id(self) -> str:
        fdr = "filter" if self.apply_fdr_to_graph else "nofilter"
        return f"{self.run_id}-{fdr}_s{self.support_threshold:g}"

    def row(self) -> dict[str, Any]:
        """columns describing the configuration in result tables."""
        dim, px, py = self.max_conds
        return {
            "config_id": self.config_id, "ci_test": self.ci_test, "interval_s": self.interval_s,
            "tau_max_s": self.tau_max_s, "tau_max_steps": self.tau_max, "pc_alpha": self.pc_alpha,
            "max_conds_dim": dim, "max_conds_px": px, "max_conds_py": py,
            "conditioning_caps": f"{dim}/{px}/{py}", "fdr_method": self.fdr_method,
            "fdr_filter": self.apply_fdr_to_graph, "support_threshold": self.support_threshold,
        }


def config_from_dict(d: dict[str, Any]) -> StudyConfig:
    d = dict(d)
    if "max_conds" in d:
        d["max_conds"] = tuple(d["max_conds"])
    return StudyConfig(**d)


def grid_configs(grid: dict[str, Any]) -> list[StudyConfig]:
    """the configurations of the yaml grid (alpha x lag x support threshold)."""
    fixed = config_from_dict(grid["fixed"])
    return [replace(fixed, pc_alpha=a, tau_max_s=t, support_threshold=s)
            for a in grid["pc_alpha"] for t in grid["tau_max_s"] for s in grid["support_threshold"]]


def make_cond_ind_test(config: StudyConfig):
    """the tigramite test object for a configuration."""
    if config.ci_test == "gsq":
        return None  # causal_discovery.run_regime builds Gsquared(significance='analytic', mask_type='xyz')
    from tigramite.independence_tests.cmisymb import CMIsymb  # needs numba
    return CMIsymb(significance="shuffle_test", sig_samples=int(config.sig_samples), mask_type="xyz",
                   seed=config.seed)


# ---------------------------------------------------------------------------
# run specifications and the on-disk cache
# ---------------------------------------------------------------------------


def regime_fingerprint(view_data: ViewData, recording_ids: Sequence[str], variables: Sequence[str]) -> str:
    """sha1 of the data and mask bytes a regime run reads (in recording order)."""
    columns = [view_data.var_names.index(v) for v in variables]
    h = hashlib.sha1()
    for r in recording_ids:
        h.update(r.encode())
        h.update(np.ascontiguousarray(view_data.arrays[r][:, columns]).tobytes())
        h.update(np.ascontiguousarray(view_data.masks[r][:, columns]).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class RunSpec:
    """one pcmci+ run of one regime."""

    subsystem: str
    view_name: str
    regime: str
    recording_ids: tuple[str, ...]
    variables: tuple[str, ...]
    config: StudyConfig
    data_sha1: str

    def key(self) -> str:
        payload = {
            "subsystem": self.subsystem, "view": self.view_name, "regime": self.regime,
            "recording_ids": list(self.recording_ids), "variables": list(self.variables),
            "settings": self.config.run_settings(), "data_sha1": self.data_sha1,
            "tigramite": version("tigramite"),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

    def cost(self) -> float:
        """rough relative cost, used only to start expensive runs first."""
        n = len(self.variables)
        factor = self.config.sig_samples or 1
        return float(n * n * self.config.tau_max * factor * max(1, len(self.recording_ids)))


def make_specs(subsystem: str, view_name: str, view_data: ViewData, regimes: dict[str, list[str]],
               config: StudyConfig) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    """run specs for every regime with two usable variables, plus log rows for skipped regimes."""
    specs, skipped = [], []
    for regime, ids in regimes.items():
        variables = cd.regime_variables(view_data, ids)
        if len(variables) < 2:
            skipped.append({"regime": regime, "run_id": config.run_id, "status": "skipped",
                            "reason": f"{len(variables)} usable variable(s)"})
            continue
        specs.append(RunSpec(subsystem, view_name, regime, tuple(ids), tuple(variables), config,
                             regime_fingerprint(view_data, ids, variables)))
    return specs, skipped


class RunCache:
    """one json file per run key: settings, status, runtime and the full link table."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, spec: RunSpec) -> Path:
        return self.directory / spec.subsystem / f"{spec.key()}.json"

    def has(self, spec: RunSpec) -> bool:
        return self.path(spec).exists()

    def load(self, spec: RunSpec) -> dict[str, Any]:
        entry = json.loads(self.path(spec).read_text())
        entry["links"] = pd.DataFrame(entry["links"], columns=cd.LINK_COLUMNS)
        return entry

    def write(self, spec: RunSpec, entry: dict[str, Any]) -> None:
        write_entry(self.path(spec), entry)


def write_entry(path: Path, entry: dict[str, Any]) -> None:
    """atomic json write (a killed process never leaves a half-written cache file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(entry, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray, tuple, set)):
        return list(o)
    raise TypeError(type(o))


def spec_header(spec: RunSpec) -> dict[str, Any]:
    return {"key": spec.key(), "subsystem": spec.subsystem, "regime": spec.regime, "run_id": spec.config.run_id,
            "settings": spec.config.run_settings(), "recording_ids": list(spec.recording_ids),
            "variables": list(spec.variables), "data_sha1": spec.data_sha1, "tigramite": version("tigramite")}


_VIEW_CACHE: dict[tuple[str, str], ViewData] = {}


def execute_spec(spec: RunSpec, processed_dir: str, out_path: str) -> None:
    """run one regime and write its cache entry (also used inside worker processes)."""
    entry = spec_header(spec)
    try:
        key = (processed_dir, spec.view_name)
        if key not in _VIEW_CACHE:
            _VIEW_CACHE[key] = load_view_data(Path(processed_dir), spec.view_name)[0]
        view_data = _VIEW_CACHE[key]
        if regime_fingerprint(view_data, spec.recording_ids, spec.variables) != spec.data_sha1:
            raise RuntimeError("data fingerprint changed since the spec was made")
        result = cd.run_regime(view_data, spec.regime, list(spec.recording_ids), list(spec.variables),
                               spec.config.discovery_config(), cond_ind_test=make_cond_ind_test(spec.config))
        entry.update(status="ok", error="", runtime_s=result.runtime_s, n_samples=result.n_samples,
                     links=cd.links_table(result).to_dict(orient="records"))
    except Exception as exc:  # logged, never hidden
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", runtime_s=np.nan, n_samples=0, links=[],
                     traceback=traceback.format_exc(limit=5))
    write_entry(Path(out_path), entry)


def run_specs(specs: Sequence[RunSpec], cache: RunCache, processed_dir: Path, workers: int = 4,
              timeout_s: float | None = None, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """execute every spec that is not cached, in parallel processes with a per-run timeout.

    returns one event row per executed spec (status ok / failed / timeout).
    """
    pending, seen = [], set()
    for spec in specs:
        if spec.key() not in seen and not cache.has(spec):
            seen.add(spec.key())
            pending.append(spec)
    pending.sort(key=lambda s: -s.cost())
    if not pending:
        return []
    log(f"running {len(pending)} regime runs with {workers} workers")
    ctx = mp.get_context("spawn")
    running: dict[str, tuple[Any, RunSpec, float]] = {}
    events = []
    while pending or running:
        while pending and len(running) < workers:
            spec = pending.pop(0)
            proc = ctx.Process(target=execute_spec, args=(spec, str(processed_dir), str(cache.path(spec))))
            proc.start()
            running[spec.key()] = (proc, spec, time.time())
        time.sleep(0.5)
        for key, (proc, spec, started) in list(running.items()):
            elapsed = time.time() - started
            if proc.is_alive() and timeout_s is not None and elapsed > timeout_s:
                proc.terminate()
                proc.join()
                cache.write(spec, {**spec_header(spec), "status": "timeout", "runtime_s": elapsed, "n_samples": 0,
                                   "links": [], "error": f"stopped after {timeout_s:.0f} s"})
            elif proc.is_alive():
                continue
            proc.join()
            if not cache.has(spec):
                cache.write(spec, {**spec_header(spec), "status": "failed", "runtime_s": elapsed, "n_samples": 0,
                                   "links": [], "error": f"process exited with code {proc.exitcode} without a result"})
            entry = json.loads(cache.path(spec).read_text())
            events.append({"subsystem": spec.subsystem, "regime": spec.regime, "run_id": spec.config.run_id,
                           "key": key, "status": entry["status"], "error": entry.get("error", ""),
                           "wall_s": round(elapsed, 1)})
            log(f"  {spec.subsystem} {spec.config.run_id} {spec.regime}: {entry['status']} "
                f"({elapsed:.0f} s, {len(pending)} queued, {len(running) - 1} running)")
            del running[key]
    return events


# ---------------------------------------------------------------------------
# aggregation and scoring
# ---------------------------------------------------------------------------


@dataclass
class RegimeRun:
    regime: str
    variables: list[str]
    links: pd.DataFrame
    status: str
    runtime_s: float
    error: str = ""


def load_runs(cache: RunCache, specs: Sequence[RunSpec]) -> dict[str, RegimeRun]:
    runs = {}
    for spec in specs:
        if not cache.has(spec):
            runs[spec.regime] = RegimeRun(spec.regime, list(spec.variables), pd.DataFrame(columns=cd.LINK_COLUMNS),
                                          "missing", np.nan, "not run")
            continue
        e = cache.load(spec)
        runs[spec.regime] = RegimeRun(spec.regime, list(spec.variables), e["links"], e["status"],
                                      float(e["runtime_s"]) if e["runtime_s"] is not None else np.nan, e.get("error", ""))
    return runs


def regime_relations(runs: dict[str, RegimeRun], apply_fdr_to_graph: bool) -> pd.DataFrame:
    """static relations per regime, with the same link filter as ``causal_discovery.discover``."""
    frames = [r.links for r in runs.values() if r.status == "ok" and len(r.links)]
    if not frames:
        return pd.DataFrame(columns=["regime", "a", "b", "relation", "basis", "lags"])
    links = pd.concat(frames, ignore_index=True)
    links["significant"] = links["significant"].astype(bool)
    used = links[links["significant"]] if apply_fdr_to_graph else links
    return cd.regime_relations(used)


def aggregate(runs: dict[str, RegimeRun], config: StudyConfig, nodes: Sequence[str],
              exclude: Sequence[str] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pair table and static edge table of one configuration (regimes in ``exclude`` are dropped)."""
    ok = {r: run for r, run in runs.items() if run.status == "ok"}
    relations = regime_relations(ok, config.apply_fdr_to_graph)
    variables_by_regime = {r: run.variables for r, run in ok.items()}
    pairs = cd.aggregate_relations(relations, variables_by_regime, nodes, threshold=config.support_threshold,
                                   exclude=exclude)
    return pairs, cd.static_edges(pairs)


def graph_metrics(pairs: pd.DataFrame, edges: pd.DataFrame, reference: nx.DiGraph,
                  nodes: Sequence[str]) -> dict[str, Any]:
    """agreement with the reference plus edge counts and testable-pair coverage."""
    graph = cd.edges_to_graph(edges, nodes)
    m = ge.compare_graphs(graph, reference)
    testable = {frozenset((a, b)) for a, b, d in zip(pairs["a"], pairs["b"], pairs["decision"]) if d != "untestable"}
    ref_pairs = {frozenset(e) for e in ge.simple_digraph(reference).edges() if e[0] != e[1]}
    n_both = int((pairs["decision"] == "both directions").sum())
    return {
        **{k: m[k] for k in ("skeleton_precision", "skeleton_recall", "skeleton_f1", "directed_precision",
                             "directed_recall", "directed_f1", "shd", "orientation_accuracy")},
        "n_adjacencies": m["n_pred_edges"],
        "n_directed_edges": m["n_pred_directed"],
        "n_undirected_edges": m["n_undirected_pred"],
        "n_both_directions": n_both,
        "unresolved_orientations": m["n_undirected_pred"] + n_both,
        "n_reference_edges": m["n_true_edges"],
        "n_pairs": len(pairs),
        "n_testable_pairs": len(testable),
        "testable_pair_share": len(testable) / len(pairs) if len(pairs) else np.nan,
        "reference_edges_testable": len(ref_pairs & testable),
        "reference_edge_coverage": len(ref_pairs & testable) / len(ref_pairs) if ref_pairs else np.nan,
        "orientation_conflicts": int(pairs["orientation_conflict"].sum()) if len(pairs) else 0,
    }


def evaluate_config(runs: dict[str, RegimeRun], config: StudyConfig, nodes: Sequence[str], reference: nx.DiGraph,
                    exclude: Sequence[str] = ()) -> tuple[dict[str, Any], pd.DataFrame]:
    """one result row and the edge table; incomplete when a regime in scope did not finish."""
    in_scope = {r: run for r, run in runs.items() if r not in set(exclude)}
    bad = {r: run.status for r, run in in_scope.items() if run.status != "ok"}
    row: dict[str, Any] = {
        **config.row(),
        "regimes_in_scope": ",".join(sorted(in_scope)),
        "runtime_s": float(np.nansum([run.runtime_s for run in in_scope.values()])),
        "run_status": "ok" if not bad else "incomplete (" + ", ".join(f"{r}: {s}" for r, s in sorted(bad.items())) + ")",
    }
    if bad:
        return row, pd.DataFrame(columns=["source", "target", "oriented", "support", "orientation_conflict"])
    pairs, edges = aggregate(runs, config, nodes, exclude=exclude)
    row.update(graph_metrics(pairs, edges, reference, nodes))
    return row, edges


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def rank_configs(table: pd.DataFrame, order: Sequence[tuple[str, str]] = SELECTION_ORDER) -> pd.DataFrame:
    """eligible rows (run_status ok) sorted by the lexicographic selection order, with a rank column."""
    eligible = table[table["run_status"] == "ok"].copy()
    keys, ascending = [], []
    for column, direction in order:
        if direction not in ("max", "min"):
            raise ValueError(f"direction must be max or min, got {direction!r}")
        if eligible[column].dtype.kind == "f":
            eligible[f"_{column}"] = eligible[column].round(12).fillna(-np.inf if direction == "max" else np.inf)
            keys.append(f"_{column}")
        else:
            keys.append(column)
        ascending.append(direction == "min")
    ranked = eligible.sort_values(keys, ascending=ascending, kind="mergesort")
    ranked = ranked.drop(columns=[k for k in keys if k.startswith("_")])
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def select(table: pd.DataFrame, order: Sequence[tuple[str, str]] = SELECTION_ORDER) -> tuple[pd.Series, dict[str, int]]:
    """best eligible row and how many rows tie with it on the objective and on objective + first tie-break."""
    ranked = rank_configs(table, order)
    if ranked.empty:
        raise ValueError("no eligible configuration")
    best = ranked.iloc[0]
    (c1, _), (c2, _) = order[0], order[1]
    same1 = np.isclose(ranked[c1].astype(float), float(best[c1]), rtol=0, atol=1e-12)
    same2 = same1 & np.isclose(ranked[c2].astype(float), float(best[c2]), rtol=0, atol=1e-12)
    return best, {"n_eligible": len(ranked), "n_tied_objective": int(same1.sum()),
                  "n_tied_objective_and_first_tiebreak": int(same2.sum())}


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def sample_availability(view_data: ViewData, recording_ids: Sequence[str], variables: Sequence[str],
                        tau_max: int) -> dict[str, Any]:
    """rows and state changes that pcmci+ tests can use at a maximum lag.

    tigramite 5.2 tests (``CondIndTest.run_test``) use ``cut_off='2xtau_max'``: the first
    2 * tau_max steps of every recording are never the time t of a test sample. a change that
    happens there can still appear as a lagged value, but never as the tested present.
    """
    columns = [view_data.var_names.index(v) for v in variables]
    rows = rows_after = changes = changes_after = contributing = 0
    for r in recording_ids:
        x, m = view_data.arrays[r][:, columns], view_data.masks[r][:, columns]
        c = np.zeros(x.shape, dtype=bool)
        c[1:] = (np.diff(x, axis=0) != 0) & ~m[1:] & ~m[:-1]
        cut = 2 * tau_max
        rows += len(x)
        rows_after += max(0, len(x) - cut)
        contributing += int(len(x) > cut)
        changes += int(c.sum())
        changes_after += int(c[cut:].sum())
    return {"recordings": len(recording_ids), "recordings_contributing": contributing, "rows": rows,
            "rows_after_cut": rows_after, "row_share_after_cut": rows_after / rows if rows else np.nan,
            "state_changes": changes, "state_changes_after_cut": changes_after,
            "change_share_after_cut": changes_after / changes if changes else np.nan}


def co_change_profile(view_data: ViewData, recording_ids: Sequence[str], variables: Sequence[str],
                      max_lag: int, interval_s: float, bins: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """expert-free timing evidence: how often a change of b follows a change of a after k steps.

    lift = co-changes at lag k / (changes of a * changes of b / rows); lift 1 = no excess.
    """
    columns = {v: view_data.var_names.index(v) for v in variables}
    ch = {}
    for r in recording_ids:
        x, m = view_data.arrays[r], view_data.masks[r]
        c = np.zeros(x.shape, dtype=bool)
        c[1:] = (np.diff(x, axis=0) != 0) & ~m[1:] & ~m[:-1]
        ch[r] = c
    rows_total = sum(len(view_data.arrays[r]) for r in recording_ids)
    out = []
    for a in variables:
        for b in variables:
            if a == b:
                continue
            ja, jb = columns[a], columns[b]
            counts = np.zeros(max_lag + 1)
            na = nb = 0
            for r in recording_ids:
                ca, cb = ch[r][:, ja], ch[r][:, jb]
                na += int(ca.sum())
                nb += int(cb.sum())
                ta = np.flatnonzero(ca)
                for k in range(max_lag + 1):
                    t = ta + k
                    counts[k] += cb[t[t < len(cb)]].sum()
            expected = na * nb / rows_total if rows_total else 0.0
            lift = counts / expected if expected > 0 else np.zeros_like(counts)
            row = {"a": a, "b": b, "changes_a": na, "changes_b": nb}
            for name, (lo, hi) in bins.items():
                row[f"cochanges_{name}"] = int(counts[lo:hi + 1].sum())
                row[f"max_lift_{name}"] = float(lift[lo:hi + 1].max()) if hi >= lo else np.nan
            ok = np.flatnonzero(counts >= 3)
            row["peak_lag_s"] = float(ok[np.argmax(lift[ok])] * interval_s) if len(ok) else np.nan
            out.append(row)
    return pd.DataFrame(out)


def fdr_diagnostics(runs: dict[str, RegimeRun], config: StudyConfig) -> pd.DataFrame:
    """what the post-hoc filter removes per regime, and which lag-0 orientations it could undermine.

    pcmci+ orients lag-0 links (collider and meek rules) on the graph of uncorrected tests; a
    lagged link X(t-tau) -> Y(t) can take part in orienting a lag-0 link that touches Y. such a
    lag-0 orientation is counted as *possibly dependent* when the filter removes a lagged link
    (autodependencies included) into one of its endpoints. this is an upper bound, not a proof
    that the orientation changes.
    """
    rows = []
    for regime, run in runs.items():
        if run.status != "ok":
            continue
        links = run.links
        lagged = links[links["lag"] > 0]
        removed = lagged[~lagged["significant"].astype(bool)]
        oriented0 = links[(links["kind"] == "contemporaneous") & (links["link"] == "-->")]
        unoriented0 = links[(links["kind"] == "contemporaneous") & (links["link"] != "-->")]
        hit_targets = set(removed["target"])
        possibly = oriented0[oriented0["source"].isin(hit_targets) | oriented0["target"].isin(hit_targets)]
        rows.append({
            "config_run": config.run_id, "regime": regime,
            "lagged_links": int((lagged["kind"] == "lagged").sum()),
            "lagged_removed": int((removed["kind"] == "lagged").sum()),
            "auto_links": int((lagged["kind"] == "auto").sum()),
            "auto_removed": int((removed["kind"] == "auto").sum()),
            "lag0_oriented": len(oriented0), "lag0_unoriented": len(unoriented0),
            "lag0_oriented_possibly_dependent_on_removed": len(possibly),
            "removed_corrected_p_range": (f"{removed['p_value'].min():.3g}-{removed['p_value'].max():.3g}"
                                          if len(removed) else ""),
        })
    return pd.DataFrame(rows)


def bootstrap_regimes(regimes: dict[str, list[str]], n_boot: int, seed: int) -> list[dict[str, list[str]]]:
    """recordings resampled with replacement within every regime (the draws of ``cd.bootstrap_edges``)."""
    rng = np.random.default_rng(seed)
    return [{r: [str(x) for x in rng.choice(ids, size=len(ids), replace=True)] for r, ids in regimes.items()}
            for _ in range(n_boot)]


def environment() -> dict[str, str]:
    import platform
    import sys

    pkgs = {}
    for name in ("tigramite", "numpy", "scipy", "pandas", "networkx", "numba"):
        try:
            pkgs[name] = version(name)
        except Exception:
            pkgs[name] = "not installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(), **pkgs}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_dict(config: StudyConfig) -> dict[str, Any]:
    return asdict(config)
