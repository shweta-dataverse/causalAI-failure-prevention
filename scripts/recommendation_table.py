"""stage 17e: the coolant counterfactual results as maintenance recommendations (no new result).

reads the stored stage-15 counterfactual table and adds the recorded onset of every activated alarm
(first 0 -> 1 rise, as in the rca window). one row per run and graph: recommended action and its
start, the modelled alarms it addresses with their recorded onsets, the alarms it cannot address,
the validity with its wilson interval, and whether the recommendation is accepted.
outputs: results/tables/17_coolant_recommendations_runs.csv, 17_coolant_recommendations_scenarios.csv
usage: PYTHONPATH=src .venv/bin/python scripts/recommendation_table.py
"""

import pandas as pd

from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import preprocessing as pp
from causal_rca import root_cause_analysis as rca
from causal_rca import scm as sc

T = data.RESULTS_DIR / "tables"
INTERVAL = 0.5


def split(text):
    return [a for a in str(text).split(", ") if a and a != "nan"]


def main():
    view, _ = pp.load_view_data(data.PROCESSED_DIR, "coolant_dt0.5")
    cf = pd.read_csv(T / "15_coolant_cf_runs.csv")
    rows = []
    for r in cf.itertuples(index=False):
        series = sc.recording_series(view, r.recording_id)
        onsets = rca.alarm_onsets(series, split(r.activated_alarms))
        rows.append({
            "graph": r.graph, "scenario": r.fold, "recording_id": r.recording_id,
            "injected_causes": r.true_causes, "action": r.actions, "action_start": r.action_start_s,
            "addressed_alarms": "; ".join(f"{a} at {onsets[a] * INTERVAL:g} s" for a in split(r.covered_alarms) if a in onsets),
            "unmodelled_alarms": "; ".join(f"{a} at {onsets[a] * INTERVAL:g} s" for a in split(r.uncovered_alarms) if a in onsets),
            "validity": r.validity_independent, "wilson_low": r.wilson_low, "wilson_high": r.wilson_high,
            "status": "accepted recommendation" if bool(r.valid_independent) else "fallback proposal, validity threshold not met",
        })
    runs = pd.DataFrame(rows)
    ev.save_table(runs, "17_coolant_recommendations_runs")
    scen = (runs.groupby(["graph", "scenario"])
            .agg(runs=("recording_id", "size"), action=("action", lambda x: "; ".join(sorted(set(x)))),
                 accepted=("status", lambda x: int((x == "accepted recommendation").sum())),
                 validity_min=("validity", "min"), validity_max=("validity", "max"))
            .reset_index())
    ev.save_table(scen, "17_coolant_recommendations_scenarios")
    print(scen.to_string(index=False))
    print(runs[runs.graph == "expert"][["recording_id", "action_start", "addressed_alarms", "unmodelled_alarms", "validity", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
