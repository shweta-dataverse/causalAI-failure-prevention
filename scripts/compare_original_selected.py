"""stage 15b: headline results of the original settings (14_) next to the selected settings (15_).

reads the stored tables only. output: results/tables/15_original_vs_selected.csv.
usage: PYTHONPATH=src .venv/bin/python scripts/compare_original_selected.py
"""

import pandas as pd

from causal_rca import data
from causal_rca import evaluation as ev

T = data.RESULTS_DIR / "tables"


def load(stage, name):
    return pd.read_csv(T / f"{stage}_{name}.csv")


def rows_for(stage):
    out = []
    g = load(stage, "graph_agreement")
    g = g[(g.graph_id == "discovered") & (g.pair_scope == "all pairs")]
    for r in g.itertuples():
        st = "raw" if "raw" in r.graph_stage else "scm dag"
        for m in ("n_pred_edges", "skeleton_precision", "skeleton_recall", "skeleton_f1", "directed_precision",
                  "directed_recall", "directed_f1", "shd"):
            out.append((r.subsystem, "graph", f"discovered {st}: {m}", "", getattr(r, m)))
    p = load(stage, "scm_prediction")
    for r in p.itertuples():
        if r.track == "N" and r.split in ("validation", "test"):
            for m in ("log_loss", "change_log_loss"):
                out.append((r.subsystem, "scm track N", f"{r.split} {m}", r.graph, getattr(r, m)))
        if r.track == "F" and r.split.startswith("all held-out") and r.variables.startswith("non-manipulated"):
            for m in ("log_loss", "change_log_loss"):
                out.append((r.subsystem, "scm track F", m, r.graph, getattr(r, m)))
    rc = load(stage, "rca_summary")
    for r in rc.itertuples():
        for m in ("top1_run_mean", "map3_run_mean", "top1_scenario_macro", "ap3_scenario_macro"):
            out.append((r.subsystem, "rca", m, r.graph, getattr(r, m)))
    ps = load(stage, "rca_per_scenario")
    for r in ps.itertuples():
        out.append((r.subsystem, "rca per scenario", f"ap3 {r.fold}", r.graph, r.ap3))
    rp = load(stage, "coolant_replay_summary")
    for r in rp[rp.scope == "all scenarios"].itertuples():
        for m in ("supported", "activation_probability_supported_mean", "onset_abs_error_median_s", "onset_abs_error_mean_s"):
            out.append(("coolant", "intervention analysis", m, r.graph, getattr(r, m)))
    rep = load(stage, "coolant_replay")
    obs = rep[rep.observed_fired & rep.alarm_supported & (rep.p_fires > 0)]
    for gname, d in obs.groupby("graph"):
        out.append(("coolant", "intervention analysis", "onsets within one grid step", gname,
                    int((d.abs_onset_error_s <= 0.5 + 1e-9).sum())))
    cf = load(stage, "coolant_cf_summary")
    for r in cf.to_dict("records"):
        for m in ("valid_share_independent (2000 fresh worlds)", "validity_independent_mean", "alarms_covered_pooled",
                  "actions_mean", "improvement_mean (fresh worlds)", "action_precision_mean", "action_recall_mean",
                  "refit_validity_mean", "refit_validity_min", "runs_valid_in_all_refits"):
            out.append(("coolant", "counterfactuals", m, r["graph"], r[m]))
    return pd.DataFrame(out, columns=["subsystem", "component", "metric", "graph", stage])


def main():
    a, b = rows_for("14"), rows_for("15")
    key = ["subsystem", "component", "metric", "graph"]
    m = a.merge(b, on=key, how="outer").rename(columns={"14": "original settings (14)", "15": "selected settings (15)"})
    ev.save_table(m, "15_original_vs_selected")
    pd.set_option("display.width", 220, "display.max_rows", 400, "display.max_colwidth", 60)
    print(m.to_string())


if __name__ == "__main__":
    main()
