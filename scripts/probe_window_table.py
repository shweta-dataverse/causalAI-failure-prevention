"""stage 17d: how the diagnosis window of every probe fault run is built (no new result).

per run: annotated fault start and end, the first observed change of each injected sensor after the
annotated start (0.5 s grid), the first 0 -> 1 rise of every recorded alarm (the onsets that the rca
uses), the resulting analysis window and whether a fallback was needed. the window is computed by
the same function as the evaluation (root_cause_analysis.analysis_window).
output: results/tables/17_probe_window_runs.csv
usage: PYTHONPATH=src .venv/bin/python scripts/probe_window_table.py
"""

import numpy as np
import pandas as pd

from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import graph_evaluation as ge
from causal_rca import preprocessing as pp
from causal_rca import root_cause_analysis as rca
from causal_rca import scm as sc

INTERVAL = 0.5


def main():
    view, _ = pp.load_view_data(data.PROCESSED_DIR, "probe_dt0.5")
    splits = pd.read_csv(data.RESULTS_DIR / "tables" / "10_probe_splits.csv")
    runs = data.load_fault_runs().set_index("recording_id")
    manipulated = data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments())
    expert = ge.simple_digraph(data.load_expert_graph(subsystem="probe"))
    alarms = sorted(n for n in expert.nodes if expert.nodes[n]["kind"] == "Alarm")
    rows = []
    for rid, fold in splits.loc[splits.split == "fault", ["recording_id", "fold"]].itertuples(index=False):
        series = sc.recording_series(view, rid)
        start, end = runs.loc[rid, "cause_start_at"], runs.loc[rid, "cause_end_at"]
        first_change = {}
        for v in sorted(manipulated[rid]):
            x = series[v]
            ok = ~np.isnan(x)
            ch = np.flatnonzero(ok[1:] & ok[:-1] & (x[1:] != x[:-1])) + 1
            ch = ch[ch * INTERVAL >= start]
            first_change[v] = ch[0] * INTERVAL if len(ch) else np.nan
        onsets = rca.alarm_onsets(series, alarms)
        window = rca.analysis_window(series, alarms, INTERVAL)
        first = np.nanmin(list(first_change.values())) if first_change else np.nan
        rows.append({
            "recording_id": rid, "scenario": fold, "injected": ",".join(sorted(manipulated[rid])),
            "annotated_start_s": start, "annotated_end_s": end,
            "first_target_change_s": first,
            "first_target_change_is_release": bool(0 <= first - end <= 1.5) if np.isfinite(first) else False,
            "alarm_onsets_used": "; ".join(f"{a} {t * INTERVAL:g} s" for a, t in sorted(onsets.items(), key=lambda x: x[1])),
            "first_onset_s": min(onsets.values()) * INTERVAL if onsets else np.nan,
            "onset_after_annotated_start": bool(onsets) and min(onsets.values()) * INTERVAL >= start,
            "onset_before_annotated_end": bool(onsets) and max(onsets.values()) * INTERVAL <= end,
            "onset_after_first_target_change": bool(onsets) and np.isfinite(first) and min(onsets.values()) * INTERVAL >= first,
            "window_start_s": window[0] * INTERVAL if window else np.nan,
            "window_end_s": window[1] * INTERVAL if window else np.nan,
            "window_contains_annotated_start": bool(window) and window[0] * INTERVAL <= start,
            "fallback_used": "none" if window else "not scored",
        })
    table = pd.DataFrame(rows)
    ev.save_table(table, "17_probe_window_runs")
    print(table.groupby("scenario").agg(runs=("recording_id", "size"),
                                        onset_after_first_change=("onset_after_first_target_change", "sum"),
                                        release_first=("first_target_change_is_release", "sum"),
                                        onset_after_start=("onset_after_annotated_start", "sum"),
                                        onset_before_end=("onset_before_annotated_end", "sum"),
                                        window_has_start=("window_contains_annotated_start", "sum")).to_string())
    print("runs with one alarm onset:", int((table.alarm_onsets_used.str.count(";") == 0).sum()), "of", len(table))
    print(table[table.scenario == "exp_6"].T.to_string())


if __name__ == "__main__":
    main()
