"""stage 17b: scenario-only protocol (commit 00c0767) next to the protocol with related-recording exclusion.

the scenario-only results are read from git at the last commit before the correction, the
corrected results from the working tree. metric definitions and aggregation are unchanged.
also verifies that every coolant result is identical (coolant has no related recordings).
outputs: results/tables/17_protocol_comparison.csv, 17_protocol_affected_runs.csv,
17_protocol_coolant_identity.csv
usage: PYTHONPATH=src .venv/bin/python scripts/protocol_comparison.py
"""

import io
import subprocess

import numpy as np
import pandas as pd

from causal_rca import data
from causal_rca import evaluation as ev

BEFORE = "00c0767"
T = data.RESULTS_DIR / "tables"


def before(name):
    text = subprocess.run(["git", "show", f"{BEFORE}:results/tables/{name}.csv"], capture_output=True, text=True,
                          cwd=data.PROJECT_ROOT, check=True).stdout
    return pd.read_csv(io.StringIO(text))


def after(name):
    return pd.read_csv(T / f"{name}.csv")


def rows(get):
    out = []
    r = get("15_rca_summary")
    for x in r[r.subsystem == "probe"].itertuples():
        for m, lab in (("top1_run_mean", "Top-1 (run mean)"), ("map3_run_mean", "MAP@3 (run mean)"),
                       ("ap3_scenario_macro", "AP@3 (scenario mean)")):
            out.append(("C. root-cause ranking", f"causal RCA, {x.graph}", lab, getattr(x, m)))
    ps = get("15_rca_per_scenario")
    for x in ps[ps.subsystem == "probe"].itertuples():
        out.append(("C. root-cause ranking", f"causal RCA, {x.graph}", f"AP@3 {x.fold}", x.ap3))
    lstm = get("16_lstm_shap_summary")
    for x in lstm[lstm.subsystem == "probe"].itertuples():
        for m, lab in (("top1_run_mean", "Top-1 (run mean)"), ("map3_run_mean", "MAP@3 (run mean)"),
                       ("ap3_scenario_macro", "AP@3 (scenario mean)"), ("map3_seed_min", "MAP@3 lowest seed"),
                       ("map3_seed_max", "MAP@3 highest seed")):
            out.append(("C. root-cause ranking", "LSTM + SHAP", lab, getattr(x, m)))
    lps = get("16_lstm_shap_per_scenario")
    for _, x in lps[lps.subsystem == "probe"].iterrows():
        out.append(("C. root-cause ranking", "LSTM + SHAP", f"AP@3 {x.fold}", x["ap@3"]))
    p = get("15_scm_prediction")
    p = p[(p.subsystem == "probe") & (p.track == "F") & (p.split == "all held-out scenarios (scenario macro)")
          & (p.variables == "non-manipulated (mechanism prediction)")]
    for x in p.itertuples():
        out.append(("B. state prediction", f"SCM, {x.graph}", "log loss (held-out scenarios)", x.log_loss))
        out.append(("B. state prediction", f"SCM, {x.graph}", "change-row log loss (held-out scenarios)", x.change_log_loss))
        out.append(("B. state prediction", f"SCM, {x.graph}", "Brier score (held-out scenarios)", x.brier))
    g = get("15_graph_agreement")
    g = g[(g.subsystem == "probe") & (g.pair_scope == "all pairs") & (g.graph_id != "discovered")]
    for stage, d in g.groupby("graph_stage"):
        lab = "raw" if "raw" in stage else "SCM DAG"
        out.append(("A. graph agreement", f"fold graphs, {lab}", "mean directed F1 over the 6 folds", d.directed_f1.mean()))
        out.append(("A. graph agreement", f"fold graphs, {lab}", "mean skeleton F1 over the 6 folds", d.skeleton_f1.mean()))
        for x in d.itertuples():
            out.append(("A. graph agreement", f"fold graphs, {lab}", f"edges {x.graph_id.removeprefix('discovered_without_')}", x.n_pred_edges))
    sel = get("15_probe_discovery_selection")
    for x in sel[sel.graph_id.str.fullmatch(r"discovered_without_[a-z0-9_]+") & ~sel.graph_id.str.contains("_and_")].itertuples():
        out.append(("settings", "discovery", f"setting {x.graph_id.removeprefix('discovered_without_')}",
                    f"tau {x.tau_max_s:g} s, alpha {x.pc_alpha:g}, kappa {x.support_threshold:g}"))
    s = get("15_scm_selection")
    for x in s[(s.subsystem == "probe") & (s.track == "F")].itertuples():
        out.append(("settings", f"SCM, {x.graph}", f"setting {x.fold}",
                    f"a {x.alpha:g}, lag {'none' if pd.isna(x.lag) else int(x.lag)}"))
    return pd.DataFrame(out, columns=["part", "method", "metric", "value"])


def main():
    b, a = rows(before), rows(after)
    table = b.merge(a, on=["part", "method", "metric"], how="outer", suffixes=("_scenario_only", "_with_exclusion"))
    num = pd.to_numeric(table.value_scenario_only, errors="coerce"), pd.to_numeric(table.value_with_exclusion, errors="coerce")
    table["difference"] = num[1] - num[0]
    ev.save_table(table, "17_protocol_comparison")
    print(table.to_string(index=False))

    # the four runs whose partner was in training, and every other probe run
    related = set(data.related_partners())
    out = []
    for label, get in (("scenario only", before), ("with exclusion", after)):
        r = get("15_rca_runs")
        r = r[(r.subsystem == "probe") & r.scored]
        l = get("16_lstm_shap_runs")
        l = l[(l.subsystem == "probe") & (l.steps == "top20") & (l.aggregation == "abs")]
        l = l.groupby("recording_id")["ap@3"].mean()
        for _, x in r.iterrows():
            out.append({"protocol": label, "recording_id": x.recording_id, "fold": x.fold, "method": f"causal {x.graph}",
                        "related": x.recording_id in related, "ap3": x["ap@3"]})
        for rid, v in l.items():
            out.append({"protocol": label, "recording_id": rid, "fold": rid.rsplit("_run", 1)[0].removeprefix("probe_"),
                        "method": "LSTM + SHAP", "related": rid in related, "ap3": v})
    runs = pd.DataFrame(out)
    ev.save_table(runs, "17_protocol_affected_runs")
    piv = runs.pivot_table(index=["method", "related"], columns="protocol", values="ap3", aggfunc="mean")
    print(piv.round(3))
    print(runs[runs.related].pivot_table(index=["recording_id", "method"], columns="protocol", values="ap3").round(3))

    # coolant must be unchanged
    ident = []
    for name in ("15_rca_runs", "15_scm_prediction", "15_graph_agreement", "15_coolant_cf_runs", "15_coolant_cf_summary",
                 "15_coolant_replay_summary", "15_coolant_cf_refits", "15_scm_selection", "16_lstm_shap_runs"):
        x, y = before(name), after(name)
        if "subsystem" in x.columns:
            x, y = x[x.subsystem == "coolant"], y[y.subsystem == "coolant"]
        x = x.drop(columns=[c for c in ("scored",) if c in x.columns and c not in y.columns])
        y = y[[c for c in x.columns]]
        same = len(x) == len(y) and all(
            np.allclose(pd.to_numeric(x[c], errors="coerce").fillna(-9), pd.to_numeric(y[c], errors="coerce").fillna(-9), rtol=0, atol=1e-12)
            if pd.api.types.is_numeric_dtype(x[c]) else (x[c].fillna("").astype(str).to_numpy() == y[c].fillna("").astype(str).to_numpy()).all()
            for c in x.columns)
        ident.append({"table": name, "coolant_rows": len(x), "identical": bool(same)})
    ident = pd.DataFrame(ident)
    ev.save_table(ident, "17_protocol_coolant_identity")
    print(ident.to_string(index=False))


if __name__ == "__main__":
    main()
