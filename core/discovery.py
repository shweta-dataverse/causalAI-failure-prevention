"""regime-wise pcmci+ discovery, aggregation by regime support, run cache and settings selection."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from tigramite import data_processing as tg_pp
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.pcmci import PCMCI

from core.data import protected_ids
from core.graphs import compare_graphs, simple_digraph
from core.preprocessing import ViewData, read_states, states_file

LINK_COLUMNS = ["regime", "source", "target", "lag", "link", "kind", "p_value", "p_corrected", "g_stat", "significant"]


@dataclass(frozen=True)
class DiscoveryConfig:
    interval_s: float = 0.5
    tau_max_s: float = 10.0
    pc_alpha: float = 0.01
    fdr_method: str = "fdr_bh"
    # caps on conditioning sets keep the discrete contingency tables estimable
    max_conds_dim: int | None = 3
    max_conds_px: int | None = 3
    max_conds_py: int | None = 3
    # pcmci+ decides links with uncorrected tests and bh-corrects only lagged p-values.
    # the filter removes lagged links whose corrected p-value exceeds pc_alpha afterwards
    apply_fdr_to_graph: bool = True
    support_threshold: float = 0.5

    @property
    def tau_max(self) -> int:
        """maximum lag in grid steps."""
        return max(1, int(round(self.tau_max_s / self.interval_s)))


@dataclass
class RegimeResult:
    regime: str
    recording_ids: list[str]
    variables: list[str]
    graph: np.ndarray  # (N, N, tau_max + 1) tigramite link strings
    val_matrix: np.ndarray
    p_matrix: np.ndarray
    n_samples: int
    runtime_s: float
    config: DiscoveryConfig = field(repr=False)


# regimes and variables


def define_regimes(splits: pd.DataFrame, exclude: Sequence[str] = (), exclude_related: bool = True) -> dict[str, list[str]]:
    """normal training recordings and every fault scenario as regimes, without the excluded scenarios."""
    regimes = {"normal": splits.loc[splits["split"] == "train", "recording_id"].tolist()}
    fault = splits[splits["split"] == "fault"]
    for fold, group in fault.groupby("fold", sort=True):
        regimes[str(fold)] = group["recording_id"].tolist()
    if exclude:
        drop = protected_ids(fault.loc[fault["fold"].isin(exclude), "recording_id"].tolist()) if exclude_related else set()
        regimes = {r: [i for i in ids if i not in drop] for r, ids in regimes.items() if r not in set(exclude)}
        regimes = {r: ids for r, ids in regimes.items() if ids}
    return regimes


def regime_variables(view_data: ViewData, recording_ids: Sequence[str], min_changes: int = 1) -> list[str]:
    """variables recorded in every recording of the regime that change at least ``min_changes`` times."""
    selected = []
    for j, name in enumerate(view_data.var_names):
        recorded = all(not view_data.masks[r][:, j].all() for r in recording_ids)
        if not recorded:
            continue
        changes = 0
        for r in recording_ids:
            values = view_data.arrays[r][~view_data.masks[r][:, j], j]
            changes += int(np.count_nonzero(np.diff(values)))
        if changes >= min_changes:
            selected.append(name)
    return selected


# running pcmci+


def run_pcmciplus(view_data: ViewData, recording_ids: Sequence[str], variables: Sequence[str],
                  config: DiscoveryConfig) -> dict[str, Any]:
    """pcmci+ with the g-squared test and each recording a separate dataset."""
    columns = [view_data.var_names.index(v) for v in variables]
    frame = tg_pp.DataFrame({k: view_data.arrays[r][:, columns].astype(float) for k, r in enumerate(recording_ids)},
                            mask={k: view_data.masks[r][:, columns] for k, r in enumerate(recording_ids)},
                            analysis_mode="multiple", var_names=list(variables))
    pcmci = PCMCI(dataframe=frame, cond_ind_test=Gsquared(significance="analytic", mask_type="xyz"), verbosity=0)
    return pcmci.run_pcmciplus(tau_min=0, tau_max=config.tau_max, pc_alpha=config.pc_alpha,
                               fdr_method=config.fdr_method, max_conds_dim=config.max_conds_dim,
                               max_conds_px=config.max_conds_px, max_conds_py=config.max_conds_py)


def run_regime(
    view_data: ViewData,
    regime: str,
    recording_ids: Sequence[str],
    variables: Sequence[str],
    config: DiscoveryConfig,
) -> RegimeResult:
    """pcmci+ on one regime."""
    if len(variables) < 2:
        raise ValueError(f"regime {regime!r} has fewer than two usable variables")
    start = time.perf_counter()
    results = run_pcmciplus(view_data, recording_ids, variables, config)
    runtime = time.perf_counter() - start
    return RegimeResult(
        regime=regime,
        recording_ids=list(recording_ids),
        variables=list(variables),
        graph=results["graph"],
        val_matrix=results["val_matrix"],
        p_matrix=results["p_matrix"],
        n_samples=int(sum(len(view_data.arrays[r]) for r in recording_ids)),
        runtime_s=runtime,
        config=config,
    )


# link tables


def links_table(result: RegimeResult) -> pd.DataFrame:
    """one row per detected link (auto, lagged or contemporaneous, the last listed once per pair)."""
    rows = []
    names = result.variables
    n, _, n_lags = result.graph.shape
    for i in range(n):
        for j in range(n):
            for lag in range(n_lags):
                link = result.graph[i, j, lag]
                if link == "" or (lag == 0 and i >= j):
                    continue
                source, target = names[i], names[j]
                if lag == 0 and link == "<--":
                    source, target, link = target, source, "-->"
                kind = "auto" if i == j else ("lagged" if lag > 0 else "contemporaneous")
                rows.append(
                    {
                        "regime": result.regime,
                        "source": source,
                        "target": target,
                        "lag": lag,
                        "link": link,
                        "kind": kind,
                        "p_value": float(result.p_matrix[i, j, lag]),
                        "p_corrected": bool(lag > 0 and result.config.fdr_method not in (None, "none")),
                        "g_stat": float(result.val_matrix[i, j, lag]),
                        "significant": bool(result.p_matrix[i, j, lag] <= result.config.pc_alpha),
                    }
                )
    return pd.DataFrame(rows, columns=LINK_COLUMNS)


def regime_relations(links: pd.DataFrame) -> pd.DataFrame:
    """static relations per regime: directed a -> b or undirected (only unoriented lag-0 evidence)."""
    cross = links[links["kind"] != "auto"]
    rows = []
    for regime, group in cross.groupby("regime"):
        directed = group[(group["kind"] == "lagged") | (group["link"] == "-->")]
        directed_pairs = set(zip(directed["source"], directed["target"]))
        for (a, b), g in directed.groupby(["source", "target"]):
            rows.append({"regime": regime, "a": a, "b": b, "relation": "directed",
                         "basis": "+".join(sorted(set(g["kind"]))), "lags": sorted(set(g["lag"]))})
        unoriented = group[(group["kind"] == "contemporaneous") & (group["link"] != "-->")]
        for row in unoriented.itertuples(index=False):
            if (row.source, row.target) in directed_pairs or (row.target, row.source) in directed_pairs:
                continue
            a, b = sorted([row.source, row.target])
            rows.append({"regime": regime, "a": a, "b": b, "relation": "undirected",
                         "basis": f"contemporaneous ({row.link})", "lags": [0]})
    return pd.DataFrame(rows, columns=["regime", "a", "b", "relation", "basis", "lags"])


# aggregation over regimes and static projection


def aggregate_relations(
    relations: pd.DataFrame,
    variables_by_regime: dict[str, Sequence[str]],
    nodes: Sequence[str],
    threshold: float = 0.5,
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """one row per unordered pair with its support over the testable regimes and the decision."""
    regimes = [r for r in variables_by_regime if r not in set(exclude)]
    relations = relations[relations["regime"].isin(regimes)]
    rows = []
    nodes = sorted(nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            testable = [r for r in regimes if a in variables_by_regime[r] and b in variables_by_regime[r]]
            pair = relations[((relations["a"] == a) & (relations["b"] == b)) | ((relations["a"] == b) & (relations["b"] == a))]
            ab = pair[(pair["relation"] == "directed") & (pair["a"] == a)]["regime"].nunique()
            ba = pair[(pair["relation"] == "directed") & (pair["a"] == b)]["regime"].nunique()
            adjacent = pair["regime"].nunique()
            n = len(testable)
            row = {
                "a": a,
                "b": b,
                "n_testable": n,
                "n_adjacent": adjacent,
                "n_a_to_b": ab,
                "n_b_to_a": ba,
                "support": adjacent / n if n else np.nan,
                "support_a_to_b": ab / n if n else np.nan,
                "support_b_to_a": ba / n if n else np.nan,
                "orientation_conflict": bool(ab > 0 and ba > 0),
                "mixed_marks": bool(0 < (ab + ba) and pair[pair["relation"] == "undirected"]["regime"].nunique() > 0),
                "lags": sorted({lag for lags in pair["lags"] for lag in lags}),
                "regimes": sorted(pair["regime"].unique()),
            }
            if n == 0:
                row["decision"] = "untestable"
            elif row["support"] < threshold:
                row["decision"] = "absent"
            elif row["support_a_to_b"] >= threshold and row["support_b_to_a"] >= threshold:
                row["decision"] = "both directions"
            elif row["support_a_to_b"] >= threshold:
                row["decision"] = "a -> b"
            elif row["support_b_to_a"] >= threshold:
                row["decision"] = "b -> a"
            else:
                row["decision"] = "undirected"
            rows.append(row)
    return pd.DataFrame(rows)


def static_edges(pairs: pd.DataFrame) -> pd.DataFrame:
    """edge list of the summary graph: source, target, oriented, support, orientation_conflict."""
    rows = []
    for row in pairs.itertuples(index=False):
        conflict = bool(getattr(row, "orientation_conflict", False))
        if row.decision in ("a -> b", "both directions"):
            rows.append({"source": row.a, "target": row.b, "oriented": True, "support": row.support_a_to_b,
                         "orientation_conflict": conflict})
        if row.decision in ("b -> a", "both directions"):
            rows.append({"source": row.b, "target": row.a, "oriented": True, "support": row.support_b_to_a,
                         "orientation_conflict": conflict})
        if row.decision == "undirected":
            rows.append({"source": row.a, "target": row.b, "oriented": False, "support": row.support,
                         "orientation_conflict": conflict})
    return pd.DataFrame(rows, columns=["source", "target", "oriented", "support", "orientation_conflict"])


def edges_to_graph(edges: pd.DataFrame, nodes: Sequence[str]) -> nx.DiGraph:
    """directed graph. undirected edges are stored once with oriented=False."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, oriented=bool(row.oriented), support=float(row.support))
    return graph


# edge identities


def edge_key(edges: pd.DataFrame) -> set[tuple[str, str, bool]]:
    """comparable edge identities. undirected edges use sorted endpoints."""
    keys = set()
    for row in edges.itertuples(index=False):
        a, b = (row.source, row.target) if row.oriented else tuple(sorted((row.source, row.target)))
        keys.add((a, b, bool(row.oriented)))
    return keys


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


# study configurations

# the selection order of discovery.selection.order in config.yaml (checked by a test)
SELECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("directed_f1", "max"),
    ("skeleton_f1", "max"),
    ("tau_max_s", "min"),
    ("pc_alpha", "min"),
    ("support_threshold", "max"),
    ("config_id", "min"),
)


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

    def __post_init__(self) -> None:
        if self.ci_test != "gsq":
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
        object.__setattr__(self, "max_conds", tuple(int(c) for c in self.max_conds))

    @property
    def tau_max(self) -> int:
        """maximum lag in grid steps."""
        return int(round(self.tau_max_s / self.interval_s))

    def discovery_config(self) -> DiscoveryConfig:
        """the pcmci+ and aggregation settings of this configuration."""
        dim, px, py = self.max_conds
        return DiscoveryConfig(
            interval_s=self.interval_s, tau_max_s=self.tau_max_s, pc_alpha=self.pc_alpha,
            fdr_method=self.fdr_method, max_conds_dim=dim, max_conds_px=px, max_conds_py=py,
            apply_fdr_to_graph=self.apply_fdr_to_graph, support_threshold=self.support_threshold,
        )

    def run_settings(self) -> dict[str, Any]:
        """settings that change a pcmci+ run (support threshold and fdr filter only aggregate)."""
        return {"ci_test": self.ci_test, "interval_s": self.interval_s, "tau_max": self.tau_max,
                "pc_alpha": self.pc_alpha, "max_conds": list(self.max_conds), "fdr_method": self.fdr_method}

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


# run specifications and the on-disk cache


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
        return float(n * n * self.config.tau_max * max(1, len(self.recording_ids)))


def make_specs(subsystem: str, view_name: str, view_data: ViewData, regimes: dict[str, list[str]],
               config: StudyConfig) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    """run specs for every regime with two usable variables, plus log rows for skipped regimes."""
    specs, skipped = [], []
    for regime, ids in regimes.items():
        variables = regime_variables(view_data, ids)
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
        entry["links"] = pd.DataFrame(entry["links"], columns=LINK_COLUMNS)
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
            _VIEW_CACHE[key] = read_states(states_file(Path(processed_dir), spec.subsystem))
        view_data = _VIEW_CACHE[key]
        if regime_fingerprint(view_data, spec.recording_ids, spec.variables) != spec.data_sha1:
            raise RuntimeError("data fingerprint changed since the spec was made")
        result = run_regime(view_data, spec.regime, list(spec.recording_ids), list(spec.variables),
                            spec.config.discovery_config())
        entry.update(status="ok", error="", runtime_s=result.runtime_s, n_samples=result.n_samples,
                     links=links_table(result).to_dict(orient="records"))
    except Exception as exc:  # logged, never hidden
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", runtime_s=np.nan, n_samples=0, links=[],
                     traceback=traceback.format_exc(limit=5))
    write_entry(Path(out_path), entry)


def run_specs(specs: Sequence[RunSpec], cache: RunCache, processed_dir: Path, workers: int = 4,
              timeout_s: float | None = None, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """run every uncached spec in parallel processes with a timeout, one event row per executed spec."""
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


# aggregation and scoring


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
            runs[spec.regime] = RegimeRun(spec.regime, list(spec.variables), pd.DataFrame(columns=LINK_COLUMNS),
                                          "missing", np.nan, "not run")
            continue
        e = cache.load(spec)
        runs[spec.regime] = RegimeRun(spec.regime, list(spec.variables), e["links"], e["status"],
                                      float(e["runtime_s"]) if e["runtime_s"] is not None else np.nan, e.get("error", ""))
    return runs


def run_relations(runs: dict[str, RegimeRun], apply_fdr_to_graph: bool) -> pd.DataFrame:
    """static relations per regime of cached runs. with the filter, only significant links are used."""
    frames = [r.links for r in runs.values() if r.status == "ok" and len(r.links)]
    if not frames:
        return pd.DataFrame(columns=["regime", "a", "b", "relation", "basis", "lags"])
    links = pd.concat(frames, ignore_index=True)
    links["significant"] = links["significant"].astype(bool)
    used = links[links["significant"]] if apply_fdr_to_graph else links
    return regime_relations(used)


def aggregate(runs: dict[str, RegimeRun], config: StudyConfig, nodes: Sequence[str],
              exclude: Sequence[str] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pair table and static edge table of one configuration (regimes in ``exclude`` are dropped)."""
    ok = {r: run for r, run in runs.items() if run.status == "ok"}
    relations = run_relations(ok, config.apply_fdr_to_graph)
    variables_by_regime = {r: run.variables for r, run in ok.items()}
    pairs = aggregate_relations(relations, variables_by_regime, nodes, threshold=config.support_threshold,
                                   exclude=exclude)
    return pairs, static_edges(pairs)


def graph_metrics(pairs: pd.DataFrame, edges: pd.DataFrame, reference: nx.DiGraph,
                  nodes: Sequence[str]) -> dict[str, Any]:
    """agreement with the reference plus edge counts and testable-pair coverage."""
    graph = edges_to_graph(edges, nodes)
    m = compare_graphs(graph, reference)
    testable = {frozenset((a, b)) for a, b, d in zip(pairs["a"], pairs["b"], pairs["decision"]) if d != "untestable"}
    ref_pairs = {frozenset(e) for e in simple_digraph(reference).edges() if e[0] != e[1]}
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
    """one result row and the edge table. incomplete when a regime in scope did not finish."""
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


# selection


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


# bootstrap and reruns


def bootstrap_regimes(regimes: dict[str, list[str]], n_boot: int, seed: int) -> list[dict[str, list[str]]]:
    """recordings resampled with replacement within every regime ."""
    rng = np.random.default_rng(seed)
    return [{r: [str(x) for x in rng.choice(ids, size=len(ids), replace=True)] for r, ids in regimes.items()}
            for _ in range(n_boot)]


class RefreshCache(RunCache):
    """run cache that treats entries written before ``since`` (a unix time) as missing, so they are rerun."""

    def __init__(self, directory: Path, since: float):
        super().__init__(directory)
        self.since = since

    def has(self, spec: RunSpec) -> bool:
        path = self.path(spec)
        return path.exists() and path.stat().st_mtime >= self.since


# why an expert edge is missing


def state_changes(view_data: ViewData, recording_id: str, variable: str) -> list[tuple[int, int]]:
    """(step, new value) of every observed change of one variable."""
    j = view_data.var_names.index(variable)
    x, m = view_data.arrays[recording_id][:, j], view_data.masks[recording_id][:, j]
    steps = np.flatnonzero((np.diff(x) != 0) & ~m[1:] & ~m[:-1]) + 1
    return [(int(i), int(x[i])) for i in steps]


def fault_intervals(view_data: ViewData, recording_id: str, variable: str) -> list[tuple[int, int]]:
    """[first step at 0, recovery step) of every period in which a health state is 0."""
    intervals, start = [], None
    for step, value in state_changes(view_data, recording_id, variable):
        if value == 0:
            start = step
        elif start is not None:
            intervals.append((start, step))
            start = None
    if start is not None:
        intervals.append((start, len(view_data.arrays[recording_id])))
    return intervals


def pair_separation(view_data: ViewData, recording_ids: Sequence[str], variables: Sequence[str],
                    config: StudyConfig, a: str, b: str) -> tuple[float, list[dict[str, Any]]]:
    """smallest p-value of the pair in one pcmci+ run and the conditioning sets that separated it."""
    results = run_pcmciplus(view_data, recording_ids, variables, config.discovery_config())
    i, j = list(variables).index(a), list(variables).index(b)
    p_min = float(np.nanmin(np.r_[results["p_matrix"][i, j, :], results["p_matrix"][j, i, :]]))
    separations = []
    for ((x, lag), y), conditions in (results.get("sepsets") or {}).items():
        if {x, y} == {i, j} and conditions:
            separations.append({"x": variables[x], "lag": int(lag), "y": variables[y],
                                "conditions": [(variables[c], int(t)) for c, t in conditions]})
    return p_min, separations


def pair_links(runs: dict[str, RegimeRun], a: str, b: str) -> pd.DataFrame:
    """links between two variables in cached regime runs (kept or removed by the fdr filter)."""
    frames = [run.links.assign(regime=regime) for regime, run in runs.items() if run.status == "ok" and len(run.links)]
    if not frames:
        return pd.DataFrame(columns=LINK_COLUMNS)
    links = pd.concat(frames, ignore_index=True)
    return links[((links["source"] == a) & (links["target"] == b)) | ((links["source"] == b) & (links["target"] == a))]
