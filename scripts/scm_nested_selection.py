"""nested selection of SCM settings with outer leave-one-scenario-out evaluation.

grid: smoothing alpha in {0.01, 0.1, 0.5, 1.0} x extra lag in {none, 2, 4} (lag only for graphs with edges).
for every outer scenario k and graph type (expert, discovered, empty):
  * inner: for every j != k, fit each setting on normal train + fault runs except k and j, with the
    graph that has seen neither k nor j (discovered: regime-wise aggregation excluding k and j),
    and score the runs of j;
  * select one setting per objective: (a) lowest pooled one-step log loss, (b) highest mean AP@3;
    ties go to the thesis baseline (alpha 0.5, no lag), then to fewer lags and alpha closer to 0.5;
  * outer: fit the baseline and both selected settings on normal train + fault runs except k with
    the graph that has not seen k, and score the runs of k.
the normal validation/test recordings are not used. the research code is imported unchanged.

usage: python scripts/scm_nested_selection.py <research_repo> <out_dir> [check]
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
CHECK_ONLY = len(sys.argv) > 3 and sys.argv[3] == "check"
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from causal_rca import causal_discovery as cd  # noqa: E402
from causal_rca import data  # noqa: E402
from causal_rca import graph_evaluation as ge  # noqa: E402
from causal_rca import preprocessing as pp  # noqa: E402
from causal_rca import root_cause_analysis as rca  # noqa: E402
from causal_rca import scm as sc  # noqa: E402
from scm_variants_validation import fit_scores, predictions  # noqa: E402

INTERVAL = 0.5
TABLES = {"coolant": ("02_coolant_splits", "03_coolant_links", "03_coolant_edges"),
          "probe": ("10_probe_splits", "10_probe_links", "10_probe_edges")}
ALPHAS = (0.01, 0.1, 0.5, 1.0)
LAGS = (None, 2, 4)
BASELINE = (0.5, None)


def preference(setting):
    """tie-break order: baseline first, then no lag before lag, then alpha closest to 0.5."""
    alpha, lag = setting
    return (setting != BASELINE, lag is not None, lag or 0, abs(np.log(alpha / 0.5)))


class Graphs:
    """expert, empty, and discovered graphs rebuilt from the saved per-regime links."""

    def __init__(self, subsystem, view_data, splits, links, nodes):
        self.nodes = nodes
        self.expert = ge.simple_digraph(data.load_expert_graph(subsystem=subsystem))
        self.empty = nx.DiGraph()
        self.empty.add_nodes_from(nodes)
        regimes = cd.define_regimes(splits)
        self.variables = {r: v for r, ids in regimes.items()
                          if len(v := cd.regime_variables(view_data, ids)) >= 2}
        config = cd.DiscoveryConfig(interval_s=INTERVAL, tau_max_s=10.0, pc_alpha=0.01)
        used = links[links["significant"]] if config.apply_fdr_to_graph else links
        self.relations = cd.regime_relations(used)
        self.threshold = config.support_threshold
        self.cache = {}

    def discovered_edges(self, exclude):
        pairs = cd.aggregate_relations(self.relations, self.variables, self.nodes,
                                       threshold=self.threshold, exclude=list(exclude))
        return cd.static_edges(pairs)

    def get(self, kind, exclude):
        if kind == "expert":
            return self.expert
        if kind == "empty":
            return self.empty
        key = tuple(sorted(exclude))
        if key not in self.cache:
            self.cache[key] = ge.to_dag(cd.edges_to_graph(self.discovered_edges(key), self.nodes))[0]
        return self.cache[key]


def fit(view_data, graph, train_ids, manipulated, setting):
    alpha, lag = setting
    lags = None
    if lag is not None:
        lags = {(u, v): [lag] for u, v in graph.edges if 3 * graph.in_degree(v) <= 20}
    return sc.fit_scm(view_data, graph, train_ids, manipulated, INTERVAL, smoothing=alpha, lags=lags)


def score_runs(model, view_data, ids, manipulated, alarms, nodes):
    """per-run rca metrics and pooled prediction rows for the runs of one scenario."""
    metrics = []
    for rid in ids:
        series = sc.recording_series(view_data, rid)
        window = rca.analysis_window(series, alarms, INTERVAL)
        if window is None:
            continue
        ranking = rca.score_candidates(model, series, rca.candidates(series, nodes, alarms), window)
        metrics.append({"recording_id": rid, **rca.ranking_metrics(ranking["variable"], set(manipulated[rid]))})
    pred = predictions(model, view_data, ids, manipulated)
    lik = np.where(pred.y == 1, pred.p, 1 - pred.p)
    return pd.DataFrame(metrics), float(-np.log(lik).sum()), len(pred)


def check_reproduction(graphs, saved_edges, folds):
    """the rebuilt leave-one-out graphs must equal the saved ones."""
    for k in [None, *folds]:
        gid = "discovered" if k is None else f"discovered_without_{k}"
        saved = saved_edges[saved_edges.graph_id == gid]
        rebuilt = graphs.discovered_edges([] if k is None else [k])
        if cd.edge_key(saved) != cd.edge_key(rebuilt):
            raise SystemExit(f"rebuilt graph {gid} differs from the saved one")
    print("rebuilt discovered graphs match the saved ones for", len(folds) + 1, "graphs", flush=True)


def main():
    runs = data.load_fault_runs()
    manipulated = data.manipulated_by_recording(runs, data.load_experiments())
    inner_rows, selection_rows, outer_rows = [], [], []
    for subsystem, (split_t, links_t, edges_t) in TABLES.items():
        tables = data.RESULTS_DIR / "tables"
        view_data, _ = pp.load_view_data(data.PROCESSED_DIR, f"{subsystem}_dt{INTERVAL:g}")
        splits = pd.read_csv(tables / f"{split_t}.csv")
        folds = sorted(splits.loc[splits.split == "fault", "fold"].unique())
        normal_train = splits.loc[splits.split == "train", "recording_id"].tolist()
        expert = ge.simple_digraph(data.load_expert_graph(subsystem=subsystem))
        nodes = sorted(expert.nodes)
        alarms = sorted(n for n in nodes if expert.nodes[n]["kind"] == "Alarm")
        graphs = Graphs(subsystem, view_data, splits, pd.read_csv(tables / f"{links_t}.csv"), nodes)
        check_reproduction(graphs, pd.read_csv(tables / f"{edges_t}.csv"), folds)
        if CHECK_ONLY:
            continue

        def train_without(*held):
            return normal_train + splits.loc[(splits.split == "fault") & (~splits.fold.isin(held)), "recording_id"].tolist()

        def runs_of(fold):
            return splits.loc[splits.fold == fold, "recording_id"].tolist()

        for kind in ("expert", "discovered", "empty"):
            settings = [(a, l) for a in ALPHAS for l in LAGS if kind != "empty" or l is None]
            for k in folds:
                inner = {s: {"ap": [], "nll": 0.0, "rows": 0} for s in settings}
                for j in folds:
                    if j == k:
                        continue
                    graph = graphs.get(kind, [k, j])
                    for s in settings:
                        m, nll, n = score_runs(fit(view_data, graph, train_without(k, j), manipulated, s),
                                               view_data, runs_of(j), manipulated, alarms, nodes)
                        inner[s]["ap"].extend(m["ap@3"].tolist())
                        inner[s]["nll"] += nll
                        inner[s]["rows"] += n
                        inner_rows.append({"subsystem": subsystem, "graph": kind, "outer_fold": k, "inner_fold": j,
                                           "alpha": s[0], "lag": s[1], "ap@3": m["ap@3"].mean(), "nll": nll, "rows": n})
                summary = {s: (np.mean(v["ap"]), v["nll"] / v["rows"]) for s, v in inner.items()}
                best_ap = max(summary.values(), key=lambda x: x[0])[0]
                best_ll = min(summary.values(), key=lambda x: x[1])[1]
                chosen = {
                    "baseline": BASELINE,
                    "selected by AP@3": min((s for s in settings if np.isclose(summary[s][0], best_ap)), key=preference),
                    "selected by log loss": min((s for s in settings if np.isclose(summary[s][1], best_ll)), key=preference),
                }
                for rule, s in chosen.items():
                    selection_rows.append({"subsystem": subsystem, "graph": kind, "outer_fold": k, "rule": rule,
                                           "alpha": s[0], "lag": s[1], "inner_ap@3": summary[s][0], "inner_log_loss": summary[s][1]})
                graph = graphs.get(kind, [k])
                for rule, s in chosen.items():
                    model = fit(view_data, graph, train_without(k), manipulated, s)
                    m, nll, n = score_runs(model, view_data, runs_of(k), manipulated, alarms, nodes)
                    pred = predictions(model, view_data, runs_of(k), manipulated)
                    for row in m.to_dict("records"):
                        outer_rows.append({"subsystem": subsystem, "graph": kind, "outer_fold": k, "rule": rule,
                                           "alpha": s[0], "lag": s[1], **row})
                    outer_rows.append({"subsystem": subsystem, "graph": kind, "outer_fold": k, "rule": rule,
                                       "alpha": s[0], "lag": s[1], "recording_id": "__fit__", **fit_scores(pred)})
                print(subsystem, kind, k, {r: s for r, s in chosen.items()}, flush=True)
    if not CHECK_ONLY:
        pd.DataFrame(inner_rows).to_csv(out / "13_scm_nested_inner.csv", index=False)
        pd.DataFrame(selection_rows).to_csv(out / "13_scm_nested_selection.csv", index=False)
        pd.DataFrame(outer_rows).to_csv(out / "13_scm_nested_outer.csv", index=False)


if __name__ == "__main__":
    main()
