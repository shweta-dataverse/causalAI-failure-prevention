"""stage 15e: diagnostics of the coolant normal-prediction model with the selected settings.

randomized quantile residuals on the normal validation and test recordings, and simulated
against observed behaviour (share of time in state 1, changes per minute) on the normal test
recordings, for the expert graph with the smoothing and extra lag selected for track n.
output: results/tables/15_coolant_normal_diagnostics.csv
usage: PYTHONPATH=src .venv/bin/python scripts/normal_diagnostics.py
"""

import numpy as np
import pandas as pd

from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import graph_evaluation as ge
from causal_rca import preprocessing as pp
from causal_rca import scm as sc

T = data.RESULTS_DIR / "tables"


def main():
    view, _ = pp.load_view_data(data.PROCESSED_DIR, "coolant_dt0.5")
    splits = pd.read_csv(T / "02_coolant_splits.csv")
    manipulated = data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments())
    graph = ge.simple_digraph(data.load_expert_graph(subsystem="coolant"))
    sel = pd.read_csv(T / "15_scm_selection.csv")
    r = sel[(sel.subsystem == "coolant") & (sel.graph == "expert") & (sel.track == "N")].iloc[0]
    lags = None if pd.isna(r.lag) else {(u, v): [int(r.lag)] for u, v in graph.edges if 3 * graph.in_degree(v) <= 20}
    model = sc.fit_scm(view, graph, data.fold_training_ids(splits, None), manipulated, 0.5,
                       smoothing=float(r.alpha), lags=lags)
    held = splits.loc[splits.split.isin(["validation", "test"]), "recording_id"].tolist()
    pred = sc.held_out_predictions(model, view, held)
    res = sc.quantile_residuals(pred, np.random.default_rng(0))
    rows = [{"quantity": "quantile residual sd (validation + test)", "value": float(res.std())},
            {"quantity": "share of |residual| > 3", "value": float((np.abs(res) > 3).mean())}]
    test = splits.loc[splits.split == "test", "recording_id"].tolist()
    beh = sc.simulated_behaviour(model, view, test, np.random.default_rng(0), n_samples=20)
    for v, d in beh.groupby("variable"):
        if d.observed_changes_per_min.sum() == 0:
            continue
        for k in ("observed_share_on", "simulated_share_on", "observed_changes_per_min", "simulated_changes_per_min"):
            rows.append({"quantity": f"{v}: {k}", "value": float(d[k].mean())})
    out = pd.DataFrame(rows)
    ev.save_table(out, "15_coolant_normal_diagnostics")
    print(out.to_string())


if __name__ == "__main__":
    main()
