"""summary statistics of the nested scm selection (reads 13_scm_nested_outer.csv, fits nothing).

writes
  13_scm_nested_per_scenario.csv  mean rca metrics per subsystem, graph, rule and held-out scenario
  13_scm_nested_comparisons.csv   paired differences in ap@3, at the level of scenarios (primary)
                                  and of runs (exploratory: runs of one scenario are correlated)

usage: python scripts/scm_nested_summary.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_rca.root_cause_analysis import bootstrap_ci  # noqa: E402

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"
METRICS = ["top1", "hit@3", "ap@3", "rr"]
COMPARISONS = [  # (graph a, rule a, graph b, rule b): difference a - b
    ("expert", "selected by AP@3", "expert", "baseline"),
    ("expert", "selected by log loss", "expert", "baseline"),
    ("expert", "selected by AP@3", "empty", "selected by AP@3"),
    ("expert", "selected by AP@3", "discovered", "selected by AP@3"),
    ("expert", "baseline", "empty", "baseline"),
]


def paired(table, a, b, unit):
    left = table[(table.graph == a[0]) & (table.rule == a[1])].set_index(unit)["ap@3"]
    right = table[(table.graph == b[0]) & (table.rule == b[1])].set_index(unit)["ap@3"]
    return (left - right.reindex(left.index)).dropna()


def describe(diff):
    nonzero = diff[~np.isclose(diff, 0)]
    p = wilcoxon(nonzero).pvalue if len(nonzero) else np.nan
    low, high = bootstrap_ci(diff.values)
    return {"n": len(diff), "better": int((diff > 1e-9).sum()), "worse": int((diff < -1e-9).sum()),
            "unchanged": int(np.isclose(diff, 0).sum()), "mean_difference": diff.mean(),
            "ci_low": low, "ci_high": high, "wilcoxon_p": p}


def main():
    outer = pd.read_csv(TABLES / "13_scm_nested_outer.csv")
    runs = outer[outer.recording_id != "__fit__"]
    per_scenario = (runs.groupby(["subsystem", "graph", "rule", "outer_fold"], as_index=False)
                    .agg(runs=("recording_id", "size"), alpha=("alpha", "first"), lag=("lag", "first"),
                         **{m: (m, "mean") for m in METRICS}))
    per_scenario.to_csv(TABLES / "13_scm_nested_per_scenario.csv", index=False)

    rows = []
    for subsystem in sorted(runs.subsystem.unique()):
        for a_graph, a_rule, b_graph, b_rule in COMPARISONS:
            for level, table, unit in (("scenario", per_scenario, "outer_fold"), ("run", runs, "recording_id")):
                sub = table[table.subsystem == subsystem]
                diff = paired(sub, (a_graph, a_rule), (b_graph, b_rule), unit)
                rows.append({"subsystem": subsystem, "a": f"{a_graph} {a_rule}", "b": f"{b_graph} {b_rule}",
                             "level": level, **describe(diff)})
    pd.DataFrame(rows).to_csv(TABLES / "13_scm_nested_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
