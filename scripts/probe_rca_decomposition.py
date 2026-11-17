"""stage 15d: which change explains the probe rca of the selected pipeline, the graph or the scm settings?

four combinations per graph, all with the leave-one-scenario-out fold models: original or selected
discovered graph (expert graph: always the same) x default or selected scm settings. nothing is
selected here; the table only decomposes the difference between stage 14 and stage 15.
also counts how many different rankings each model produces over the 34 runs.
output: results/tables/15_probe_rca_decomposition.csv
usage: PYTHONPATH=src .venv/bin/python scripts/probe_rca_decomposition.py
"""

import pandas as pd

from causal_rca import causal_discovery as cd
from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import graph_evaluation as ge
from causal_rca import preprocessing as pp
from causal_rca import root_cause_analysis as rca
from causal_rca import scm as sc

T = data.RESULTS_DIR / "tables"
INTERVAL = 0.5


def main():
    view, _ = pp.load_view_data(data.PROCESSED_DIR, "probe_dt0.5")
    splits = pd.read_csv(T / "10_probe_splits.csv")
    manipulated = data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments())
    expert = ge.simple_digraph(data.load_expert_graph(subsystem="probe"))
    nodes = sorted(expert.nodes)
    alarms = sorted(n for n in nodes if expert.nodes[n]["kind"] == "Alarm")
    graphs = {}
    # both discovered graphs come from the same grouped scopes (related partners of held-out runs removed)
    for label, file in (("original", "12_probe_original_edges.csv"), ("selected", "15_probe_discovery_edges.csv")):
        e = pd.read_csv(T / file)
        graphs[label] = {gid: ge.to_dag(cd.edges_to_graph(g, nodes))[0] for gid, g in e.groupby("graph_id")}
    sel = pd.read_csv(T / "15_scm_selection.csv")
    sel = sel[(sel.subsystem == "probe") & (sel.track == "F")]
    rows = []
    for graph_name, graph_version in (("expert", "-"), ("discovered", "original"), ("discovered", "selected")):
        for scm_version in ("default", "selected"):
            def graph_for(fold):
                return expert if graph_name == "expert" else graphs[graph_version][f"discovered_without_{fold}"]

            def fit(fold, train):
                g = graph_for(fold)
                if scm_version == "default":
                    return sc.fit_scm(view, g, train, manipulated, INTERVAL)
                # the settings selected for this graph type and fold in stage 15a
                r = sel[(sel.graph == graph_name) & (sel.fold == fold)].iloc[0]
                lags = None if pd.isna(r.lag) else {(u, v): [int(r.lag)] for u, v in g.edges if 3 * g.in_degree(v) <= 20}
                return sc.fit_scm(view, g, train, manipulated, INTERVAL, smoothing=float(r.alpha), lags=lags)

            metrics, rankings, _ = rca.evaluate_rca(view, splits, manipulated, graph_for, alarms, nodes, INTERVAL,
                                                    seed=0, fit_model=fit)
            m = metrics[metrics.scored]
            orders = rankings.sort_values(["recording_id", "rank"]).groupby("recording_id").variable.apply(tuple)
            per = m.groupby("fold")["ap@3"].mean()
            rows.append({"graph": graph_name, "discovered_graph": graph_version, "scm_settings": scm_version,
                         "top1_run_mean": m.top1.mean(), "map3_run_mean": m["ap@3"].mean(),
                         "ap3_scenario_macro": per.mean(), "distinct_rankings": orders.nunique(),
                         **{f"ap3 {k}": v for k, v in per.items()}})
            print(rows[-1], flush=True)
    ev.save_table(pd.DataFrame(rows), "15_probe_rca_decomposition")


if __name__ == "__main__":
    main()
