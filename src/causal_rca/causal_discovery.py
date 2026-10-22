"""pcmci+ causal discovery per regime, link tables, aggregation and static projection.

design (see notebook 03 for the reasoning):

* one *regime* = one set of recordings with the same recorded variables: the
  normal training recordings, and each fault scenario separately. fault runs only
  record the variables of their scenario, and tigramite either fails on tests
  without common samples or returns p = 1 ("independent") for constant data, so
  pooling everything would turn missing information into false independence.
* inside a regime only variables that are recorded in every recording and change
  at least once are used. a pair of variables is *testable* in a regime when both
  are used there; pairs never testable are reported as untestable, not absent.
* the conditional independence test is the g-squared test for discrete data
  (no parcorr on binary states). recordings enter tigramite as separate datasets,
  so lags never cross recording boundaries.
* regime results are combined into one static graph by support: the share of
  testable regimes in which the relation was found.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from tigramite import data_processing as tg_pp
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.pcmci import PCMCI

from causal_rca.preprocessing import ViewData

LINK_COLUMNS = ["regime", "source", "target", "lag", "link", "kind", "p_value", "p_corrected", "g_stat", "significant"]


@dataclass(frozen=True)
class DiscoveryConfig:
    interval_s: float = 0.5
    tau_max_s: float = 10.0
    pc_alpha: float = 0.01
    fdr_method: str = "fdr_bh"
    # limits on conditioning sets keep the discrete contingency tables estimable:
    # max_conds_dim for the condition-selection steps, max_conds_px/py for the number
    # of strongest lagged parents of x and y used in each mci test
    max_conds_dim: int | None = 3
    max_conds_px: int | None = 3
    max_conds_py: int | None = 3
    # tigramite 5.2 pcmci+ decides every link with uncorrected tests at pc_alpha and then
    # bh-corrects only the reported p-values of *lagged* links (contemporaneous p-values
    # stay uncorrected). with this flag lagged links whose corrected p-value exceeds
    # pc_alpha are removed afterwards; lag-0 links are unaffected (their max p <= pc_alpha).
    apply_fdr_to_graph: bool = True
    support_threshold: float = 0.5

    @property
    def tau_max(self) -> int:
        """maximum lag in grid steps."""
        return max(1, int(round(self.tau_max_s / self.interval_s)))

    @property
    def label(self) -> str:
        fdr = "" if self.apply_fdr_to_graph else "_nofdrfilter"
        return f"dt{self.interval_s:g}_tau{self.tau_max_s:g}s_alpha{self.pc_alpha:g}{fdr}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "tau_max": self.tau_max, "label": self.label}


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


# ---------------------------------------------------------------------------
# regimes and variables
# ---------------------------------------------------------------------------


def define_regimes(splits: pd.DataFrame) -> dict[str, list[str]]:
    """normal training recordings as one regime, every fault scenario as its own regime."""
    regimes = {"normal": splits.loc[splits["split"] == "train", "recording_id"].tolist()}
    fault = splits[splits["split"] == "fault"]
    for fold, group in fault.groupby("fold", sort=True):
        regimes[str(fold)] = group["recording_id"].tolist()
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


# ---------------------------------------------------------------------------
# running pcmci+
# ---------------------------------------------------------------------------


def run_regime(
    view_data: ViewData,
    regime: str,
    recording_ids: Sequence[str],
    variables: Sequence[str],
    config: DiscoveryConfig,
) -> RegimeResult:
    """pcmci+ with the g-squared test on one regime; each recording is its own dataset."""
    if len(variables) < 2:
        raise ValueError(f"regime {regime!r} has fewer than two usable variables")
    columns = [view_data.var_names.index(v) for v in variables]
    data = {k: view_data.arrays[r][:, columns].astype(float) for k, r in enumerate(recording_ids)}
    mask = {k: view_data.masks[r][:, columns] for k, r in enumerate(recording_ids)}
    frame = tg_pp.DataFrame(data, mask=mask, analysis_mode="multiple", var_names=list(variables))

    pcmci = PCMCI(dataframe=frame, cond_ind_test=Gsquared(significance="analytic", mask_type="xyz"), verbosity=0)
    start = time.perf_counter()
    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=config.tau_max,
        pc_alpha=config.pc_alpha,
        fdr_method=config.fdr_method,
        max_conds_dim=config.max_conds_dim,
        max_conds_px=config.max_conds_px,
        max_conds_py=config.max_conds_py,
    )
    runtime = time.perf_counter() - start
    n_samples = int(sum(len(a) for a in data.values()))
    return RegimeResult(
        regime=regime,
        recording_ids=list(recording_ids),
        variables=list(variables),
        graph=results["graph"],
        val_matrix=results["val_matrix"],
        p_matrix=results["p_matrix"],
        n_samples=n_samples,
        runtime_s=runtime,
        config=config,
    )


def run_all_regimes(
    view_data: ViewData, regimes: dict[str, list[str]], config: DiscoveryConfig, min_changes: int = 1
) -> dict[str, RegimeResult]:
    """run every regime that has at least two usable variables."""
    results = {}
    for regime, ids in regimes.items():
        variables = regime_variables(view_data, ids, min_changes=min_changes)
        if len(variables) >= 2:
            results[regime] = run_regime(view_data, regime, ids, variables, config)
    return results


# ---------------------------------------------------------------------------
# link tables
# ---------------------------------------------------------------------------


def links_table(result: RegimeResult) -> pd.DataFrame:
    """one row per detected link.

    kind: 'auto' (a variable's own past), 'lagged' (source at t - lag -> target at t,
    oriented by time order) or 'contemporaneous' (lag 0; link '-->' oriented by
    pcmci+ rules, 'o-o' unoriented, 'x-x' conflicting). contemporaneous links are
    listed once per pair. p_value is the largest p-value of all tests of the link;
    p_corrected tells whether tigramite bh-corrected it (lagged links only).
    significant = p_value <= pc_alpha.
    """
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
    """static relations per regime: 'directed' a -> b (lagged link or oriented lag-0 link)
    or 'undirected' {a, b} (only unoriented or conflicting lag-0 evidence)."""
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


# ---------------------------------------------------------------------------
# aggregation over regimes and static projection
# ---------------------------------------------------------------------------


def aggregate_relations(
    relations: pd.DataFrame,
    variables_by_regime: dict[str, Sequence[str]],
    nodes: Sequence[str],
    threshold: float = 0.5,
    exclude: Sequence[str] = (),
) -> pd.DataFrame:
    """one row per unordered variable pair with its support over the regimes.

    a pair is testable in a regime when both variables were used there. support =
    regimes with the relation / testable regimes. the pair is kept as 'a -> b' when
    the directed support reaches the threshold, as undirected when only the
    adjacency support does, and is 'untestable' when no regime could test it.
    regimes in ``exclude`` are ignored (used for leave-one-scenario-out graphs).

    conflicting evidence is kept, not hidden: ``orientation_conflict`` marks pairs found
    in both directions in different regimes, ``mixed_marks`` pairs found directed in
    some regimes and undirected in others.
    """
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
    """edge list of the static graph: source, target, oriented (False = undirected edge),
    support, and whether some regimes found the opposite direction (orientation_conflict)."""
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
    """directed graph; undirected edges are stored once with oriented=False."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, oriented=bool(row.oriented), support=float(row.support))
    return graph


# ---------------------------------------------------------------------------
# full experiment helpers
# ---------------------------------------------------------------------------


@dataclass
class Discovery:
    """everything one discovery configuration produced."""

    config: DiscoveryConfig
    results: dict[str, RegimeResult]
    links: pd.DataFrame
    relations: pd.DataFrame

    @property
    def variables_by_regime(self) -> dict[str, list[str]]:
        return {r: res.variables for r, res in self.results.items()}

    def pairs(self, nodes: Sequence[str], exclude: Sequence[str] = ()) -> pd.DataFrame:
        return aggregate_relations(self.relations, self.variables_by_regime, nodes,
                                   threshold=self.config.support_threshold, exclude=exclude)

    def edges(self, nodes: Sequence[str], exclude: Sequence[str] = ()) -> pd.DataFrame:
        return static_edges(self.pairs(nodes, exclude=exclude))


def discover(view_data: ViewData, regimes: dict[str, list[str]], config: DiscoveryConfig) -> Discovery:
    """run pcmci+ on every regime and collect links and static relations."""
    results = run_all_regimes(view_data, regimes, config)
    links = pd.concat([links_table(r) for r in results.values()], ignore_index=True)
    used = links[links["significant"]] if config.apply_fdr_to_graph else links
    return Discovery(config=config, results=results, links=links, relations=regime_relations(used))


def bootstrap_edges(
    view_data: ViewData, regimes: dict[str, list[str]], config: DiscoveryConfig, nodes: Sequence[str],
    n_boot: int = 30, seed: int = 0,
) -> pd.DataFrame:
    """edge frequency when the recordings of every regime are resampled with replacement.

    returns one row per (bootstrap draw, edge) of the aggregated static graph.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        resampled = {r: list(rng.choice(ids, size=len(ids), replace=True)) for r, ids in regimes.items()}
        edges = discover(view_data, resampled, config).edges(nodes)
        rows.append(edges.assign(draw=b))
    return pd.concat(rows, ignore_index=True)


def edge_key(edges: pd.DataFrame) -> set[tuple[str, str, bool]]:
    """comparable edge identities; undirected edges use sorted endpoints."""
    keys = set()
    for row in edges.itertuples(index=False):
        a, b = (row.source, row.target) if row.oriented else tuple(sorted((row.source, row.target)))
        keys.add((a, b, bool(row.oriented)))
    return keys


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0
