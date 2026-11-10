"""stage 15c: where the coolant discovery missed expert adjacencies, and why the extra edges appear.

for every expert adjacency missing from the selected all-scenario coolant graph:
  * testability: the regimes in which both variables are recorded and change;
  * co-change: whether the partner changes at all while the cause variable is in its fault state;
  * stage of exclusion in the selected configuration: dropped by a conditional-independence
    test (with the separating set from tigramite), removed by the bh filter, below the support
    threshold, or removed in the dag conversion;
  * evaluated changes: every stage-12 full-data run (grid, caps, 60 s) in which the pair appears,
    with and without the bh filter, plus the trade-off of the best such configuration;
  * bootstrap frequency of the pair (stage 12d).
for every extra adjacency between two alarms: whether the two alarms reset at the same step.

the expert graph is used only to name the pairs; nothing here changes the pipeline.
outputs: results/tables/15_coolant_missing_edges.csv, 15_coolant_extra_alarm_edges.csv.

usage: PYTHONPATH=src .venv/bin/python scripts/edge_diagnostics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tigramite import data_processing as tg_pp
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.pcmci import PCMCI

sys.path.insert(0, str(Path(__file__).parent))
from causal_rca import causal_discovery as cd  # noqa: E402
from causal_rca import data  # noqa: E402
from causal_rca import evaluation as ev  # noqa: E402
import run_discovery_study as rds  # noqa: E402

TABLES = data.RESULTS_DIR / "tables"
CACHE_DIR = data.PROJECT_ROOT / "data" / "cache" / "discovery_study" / "coolant"
INTERVAL = 0.5


def changes(view, rid, v):
    j = view.var_names.index(v)
    x, m = view.arrays[rid][:, j], view.masks[rid][:, j]
    idx = np.flatnonzero((np.diff(x) != 0) & ~m[1:] & ~m[:-1]) + 1
    return [(i, int(x[i])) for i in idx]


def fault_intervals(view, rid, v):
    """steps during which v is 0 (unhealthy), from its first drop to its recovery."""
    ch = changes(view, rid, v)
    out, start = [], None
    for i, val in ch:
        if val == 0:
            start = i
        elif start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(view.arrays[rid])))
    return out


def separating_set(s, regime, config, a, b):
    """rerun pcmci+ on one regime and return the links of the pair and its separating sets."""
    ids = s.regimes[regime]
    var = cd.regime_variables(s.view, ids)
    cols = [s.view.var_names.index(v) for v in var]
    frame = tg_pp.DataFrame({k: s.view.arrays[r][:, cols].astype(float) for k, r in enumerate(ids)},
                            mask={k: s.view.masks[r][:, cols] for k, r in enumerate(ids)},
                            analysis_mode="multiple", var_names=var)
    pcmci = PCMCI(dataframe=frame, cond_ind_test=Gsquared(significance="analytic", mask_type="xyz"), verbosity=0)
    res = pcmci.run_pcmciplus(tau_min=0, tau_max=int(round(config.tau_max_s / INTERVAL)), pc_alpha=config.pc_alpha,
                              fdr_method="fdr_bh", max_conds_dim=3, max_conds_px=3, max_conds_py=3)
    i, j = var.index(a), var.index(b)
    p_min = float(np.nanmin(np.r_[res["p_matrix"][i, j, :], res["p_matrix"][j, i, :]]))
    seps = []
    for key, cond in (res.get("sepsets") or {}).items():
        (x, lag), y = key
        if {x, y} == {i, j} and cond:
            seps.append(f"{var[x]}(t{lag:+d}) - {var[y]}(t) | " + ", ".join(f"{var[c]}(t{t:+d})" for c, t in cond))
    return p_min, sorted(set(seps))


def cached_full_runs(s, regime, pair):
    """stage-12 full-data runs of a regime (not bootstrap draws) in which the pair has a link."""
    full = sorted(s.regimes[regime])
    hits = []
    for f in CACHE_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        if d["regime"] != regime or d["status"] != "ok" or sorted(d["recording_ids"]) != full:
            continue
        for link in d["links"]:
            if {link["source"], link["target"]} == set(pair):
                hits.append(f"{d['run_id']}: lag {link['lag']}, {'kept' if link['significant'] else 'removed by bh filter'}")
    return sorted(set(hits))


def main():
    study = rds.load_study()
    s = rds.Setting("coolant")
    sel = pd.read_csv(TABLES / "15_coolant_discovery_selection.csv").set_index("graph_id")
    config = {c.config_id: c for c in rds.ds.grid_configs(study["grid"])}[sel.loc["discovered", "config_id"]]
    edges = pd.read_csv(TABLES / "15_coolant_discovery_edges.csv")
    graph = edges[edges.graph_id == "discovered"]
    found = {frozenset((a, b)) for a, b in zip(graph.source, graph.target)}
    boot = pd.read_csv(TABLES / "12_coolant_bootstrap_edges.csv")
    boot = boot[boot.config_id == config.config_id]
    fdr_off = pd.read_csv(TABLES / "12_coolant_fdr_off.csv")
    fdr_off = fdr_off[fdr_off.scope == "all scenarios"]

    rows = []
    for u, v in sorted(s.expert.edges):
        if frozenset((u, v)) in found:
            continue
        testable = [r for r, ids in s.regimes.items() if {u, v} <= set(cd.regime_variables(s.view, ids))]
        # does the partner change while the cause is in its fault state?
        co, partner_values = 0, set()
        n_runs = 0
        for r in testable:
            for rid in s.regimes[r]:
                for a0, a1 in fault_intervals(s.view, rid, u):
                    n_runs += 1
                    j = s.view.var_names.index(v)
                    seg = s.view.arrays[rid][a0:a1, j]
                    partner_values |= set(np.unique(seg[~np.isnan(seg)]).astype(int).tolist())
                    co += any(a0 - 1 <= i <= a1 for i, _ in changes(s.view, rid, v))
        seps, p_min = [], np.nan
        for r in testable:
            p, sp = separating_set(s, r, config, u, v)
            p_min = p if np.isnan(p_min) else min(p_min, p)
            seps += [f"{r}: {x}" for x in sp]
        hits = [f"{r}: {h}" for r in testable for h in cached_full_runs(s, r, (u, v))]
        bf = boot[boot.edge.str.contains(u) & boot.edge.str.contains(v)].bootstrap_frequency
        rows.append({
            "expert_edge": f"{u} -> {v}",
            "testable_regimes": ", ".join(testable),
            "fault_intervals_of_cause": n_runs,
            "intervals_with_partner_change": co,
            "partner_values_during_fault": ", ".join(map(str, sorted(partner_values))),
            "stage_of_exclusion (selected config)": "conditional-independence test in PCMCI+"
                                                    + (" (unconditional test already independent)" if not seps else ""),
            "separating_sets": "; ".join(seps) if seps else "none (dropped without conditioning)",
            "smallest_p_value_in_regime": p_min,
            "full_data_runs_with_a_link": "; ".join(hits) if hits else "none of the evaluated settings",
            "bootstrap_frequency (selected config)": float(bf.iloc[0]) if len(bf) else 0.0,
        })
    ev.save_table(pd.DataFrame(rows), "15_coolant_missing_edges")
    print(pd.DataFrame(rows).T.to_string())

    # trade-off of the evaluated setting that recovers a missed adjacency (filter off, alpha 0.05)
    tr = fdr_off[(fdr_off.pc_alpha == 0.05) & (fdr_off.support_threshold == 0.5)]
    print("\nfilter off, alpha 0.05, support 0.5:")
    print(tr[["config_id", "skeleton_recall", "skeleton_precision", "directed_f1", "n_adjacencies"]].to_string())

    extra = []
    for a, b in zip(graph.source, graph.target):
        if s.expert.has_edge(a, b) or s.expert.has_edge(b, a):
            continue
        both_alarms = all(s.expert.nodes[x]["kind"] == "Alarm" for x in (a, b))
        regimes = [r for r, ids in s.regimes.items() if {a, b} <= set(cd.regime_variables(s.view, ids))]
        same_reset, same_onset, n = 0, 0, 0
        for r in regimes:
            for rid in s.regimes[r]:
                ca, cb = changes(s.view, rid, a), changes(s.view, rid, b)
                n += 1
                same_reset += any(i == k for i, x in ca if x == 0 for k, y in cb if y == 0)
                same_onset += any(abs(i - k) <= 1 for i, x in ca if x == 1 for k, y in cb if y == 1)
        extra.append({"extra_edge": f"{a} - {b}", "both_alarms": both_alarms, "regimes": ", ".join(regimes), "runs": n,
                      "runs_with_identical_reset_step": same_reset, "runs_with_onsets_within_one_step": same_onset})
    ev.save_table(pd.DataFrame(extra), "15_coolant_extra_alarm_edges")
    print(pd.DataFrame(extra).to_string())


if __name__ == "__main__":
    main()
