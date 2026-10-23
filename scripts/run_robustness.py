"""robustness extensions of the replay and counterfactual evaluation (stage 11).

x1  time-limited replay: every replay of notebooks 07, 09 and 10 is repeated with the
    manipulation released at its observed reversion step (b) next to the existing sustained
    replay (a), with identical scms, start states, history and noise. the reversion steps are
    read retrospectively from the recordings; the framework does not predict them.
x2  independent monte carlo validity: the action sets selected in notebook 09 (100 samples,
    first set with validity >= 0.9) are not reselected; each is re-evaluated with 2,000 fresh
    abducted worlds from a separate random stream, with wilson intervals.

existing result files are never written; all outputs start with 11_. every run first checks
that the sustained replay and the recomputed rca rankings reproduce the saved baselines.

usage: PYTHONPATH=src python scripts/run_robustness.py [x1|x2|all]
"""

from __future__ import annotations

import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from causal_rca import causal_discovery as cd
from causal_rca import counterfactuals as cfx
from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import graph_evaluation as ge
from causal_rca import interventions as itv
from causal_rca import preprocessing as pp
from causal_rca import root_cause_analysis as rca
from causal_rca import scm as sc

INTERVAL = 0.5
SEED = 0
KINDS = ["delete", "insert", "reverse"]
FRACTIONS = [0.1, 0.25, 0.5]
SEEDS = range(5)
X2_ENTROPY = 20261023  # separate from every seed used during selection
TABLES = data.RESULTS_DIR / "tables"


def step(t: float) -> int:
    """grid step of a time, the rule used by preprocessing and the notebooks."""
    return int(np.ceil(round(t / INTERVAL, 9)))


def setting(sub: str) -> dict:
    """data, graphs and observed onsets and reversions of one subsystem."""
    prefix = "02" if sub == "coolant" else "10"
    view, _ = pp.load_view_data(data.PROCESSED_DIR, f"{sub}_dt{INTERVAL:g}")
    splits = pd.read_csv(TABLES / f"{prefix}_{sub}_splits.csv")
    runs, experiments = data.load_fault_runs(), data.load_experiments()
    manipulated = data.manipulated_by_recording(runs, experiments)
    expert = ge.simple_digraph(data.load_expert_graph(subsystem=sub))
    nodes = sorted(expert.nodes)
    alarms = sorted(n for n in nodes if expert.nodes[n]["kind"] == "Alarm")
    events = pp.mark_changes(pp.parse_values(pp.load_all_events(
        [r for r in pp.discover_recordings(data.RAW_DIR) if r.subsystem == sub], data.RAW_DIR)))
    events = events[events.source == "dig_twin"]
    changes = data.fault_variable_changes(events, runs[runs.subsystem == sub], experiments)
    onsets, reversions = {}, {}
    for r, v, t in zip(changes.recording_id, changes.variable, changes.first_change_after_start_s):
        if np.isnan(t):
            continue
        onsets[(r, v)] = step(t)
        later = events[(events.recording_id == r) & (events.node == v) & events.is_change & (events.time_s > t)]
        if len(later):
            reversions[(r, v)] = step(float(later.time_s.min()))
    edges = pd.read_csv(TABLES / ("03_coolant_edges.csv" if sub == "coolant" else "10_probe_edges.csv"))
    discovered = {gid: ge.to_dag(cd.edges_to_graph(g, nodes))[0] for gid, g in edges.groupby("graph_id")}
    empty = nx.DiGraph()
    empty.add_nodes_from(nodes)
    variants = {"expert": ("expert", 0.0, -1, lambda fold: expert, expert),
                "discovered": ("discovered", np.nan, -1, lambda fold: discovered[f"discovered_without_{fold}"],
                               discovered["discovered"]),
                "empty": ("empty", 1.0, -1, lambda fold: empty, empty)}
    for kind in KINDS:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                g = ge.perturb_graph(expert, kind, fraction, seed)
                variants[f"{kind} {fraction:g} s{seed}"] = (kind, fraction, seed, (lambda graph: (lambda fold: graph))(g), g)
    folds = sorted(splits.loc[splits.split == "fault", "fold"].unique())
    return dict(sub=sub, view=view, splits=splits, manipulated=manipulated, expert=expert, nodes=nodes, alarms=alarms,
                onsets=onsets, reversions=reversions, variants=variants, folds=folds, discovered=discovered)


def fold_models(s: dict, getter, lags_for=None) -> dict:
    """one scm per held-out scenario, fitted exactly as in the notebooks."""
    return {fold: sc.fit_scm(s["view"], getter(fold), data.fold_training_ids(s["splits"], fold), s["manipulated"],
                             INTERVAL, lags=None if lags_for is None else lags_for(fold)) for fold in s["folds"]}


def paired_replays(s: dict, models: dict, n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame, list[bool]]:
    alarm_frames, target_frames, identical = [], [], []
    for fold, model in models.items():
        for rid in s["splits"].loc[s["splits"].fold == fold, "recording_id"]:
            series = sc.recording_series(s["view"], rid)
            on = {v: s["onsets"][(rid, v)] for v in s["manipulated"][rid] if (rid, v) in s["onsets"]}
            rev = {v: s["reversions"][(rid, v)] for v in on if (rid, v) in s["reversions"]}
            a, t, same = itv.replay_paired(model, series, on, rev, s["alarms"], n_samples, SEED)
            alarm_frames.append(a.assign(fold=fold, recording_id=rid))
            target_frames.append(t.assign(fold=fold, recording_id=rid))
            identical.append(same)
    return pd.concat(alarm_frames, ignore_index=True), pd.concat(target_frames, ignore_index=True), identical


def replay_summary(alarms: pd.DataFrame) -> dict:
    """the replay measures of notebooks 07 and 09 for both modes, on observed activations."""
    fired = alarms[alarms.observed_fired]
    sup = fired[fired.alarm_supported]
    out = {"pairs_fired": len(fired), "pairs_fired_supported": len(sup)}
    for mode in ("sustained", "limited"):
        err = (sup[f"predicted_median_delay_steps_{mode}"] - sup.observed_delay_steps).abs() * INTERVAL
        out[f"p_fires_all_{mode}"] = fired[f"p_fires_{mode}"].mean()
        out[f"p_fires_supported_{mode}"] = sup[f"p_fires_{mode}"].mean()
        out[f"median_abs_delay_error_s_{mode}"] = float(np.nanmedian(err)) if len(err) else np.nan
        out[f"never_predicted_{mode}"] = int(sup[f"predicted_median_delay_steps_{mode}"].isna().sum())
    diff = (fired.p_fires_limited - fired.p_fires_sustained).abs()
    out["pairs_with_different_p_fires"] = int((diff > 0).sum())
    out["max_abs_p_fires_difference"] = float(diff.max()) if len(diff) else np.nan
    return out


def spearman_rows(table: pd.DataFrame, quality: str, targets: list[str], label: str) -> list[dict]:
    """spearman over the 45 perturbed graphs and over the nine level means (replicates share their f1)."""
    perturbed = table[table.kind.isin(KINDS)]
    levels = perturbed.groupby(["kind", "fraction"])[[quality, *targets]].mean()
    rows = []
    for unit, frame in (("45 graphs", perturbed), ("9 level means", levels)):
        for target in targets:
            x, y = frame[quality].to_numpy(float), frame[target].to_numpy(float)
            ok = ~np.isnan(x) & ~np.isnan(y)
            res = stats.spearmanr(x[ok], y[ok]) if ok.sum() > 2 and np.std(y[ok]) > 0 else None
            rows.append({"analysis": label, "unit": unit, "quality": quality, "downstream": target,
                         "spearman": res.statistic if res else np.nan, "p_exploratory": res.pvalue if res else np.nan,
                         "n": int(ok.sum())})
    return rows


# ---------------------------------------------------------------------------
# x1
# ---------------------------------------------------------------------------


def run_x1() -> None:
    start = time.perf_counter()
    corr_rows, check_rows = [], []
    for sub in ("coolant", "probe"):
        s = setting(sub)
        # (1) the notebook-07 setting for coolant: three graphs and the lag-aware model, 200 samples
        if sub == "coolant":
            lag_table = pd.read_csv(TABLES / "05_coolant_lag_selection.csv")
            chosen = lag_table[lag_table.decision == "selected"]
            lag_sets = {gid: {(u, v): tuple(sorted(g.lag)) for (u, v), g in chosen[chosen.graph_id == gid].groupby(["source", "target"])}
                        for gid in s["discovered"]}
            configs = {name: (s["variants"][name][3], None) for name in ("expert", "discovered", "empty")}
            configs["discovered + lags"] = (s["variants"]["discovered"][3], lambda fold: lag_sets[f"discovered_without_{fold}"])
            ref_frames, target_frames, summaries = [], [], []
            for name, (getter, lags_for) in configs.items():
                alarms, targets, identical = paired_replays(s, fold_models(s, getter, lags_for), 200)
                ref_frames.append(alarms.assign(graph=name))
                target_frames.append(targets.assign(graph=name))
                summaries.append({"graph": name, "identical_before_release": all(identical), **replay_summary(alarms)})
            ref = pd.concat(ref_frames, ignore_index=True)
            # reproduction check against the saved notebook-07 replay (sustained mode)
            saved = pd.concat([pd.read_csv(TABLES / "07_coolant_replay.csv"), pd.read_csv(TABLES / "07_coolant_replay_lagged.csv")])
            merged = ref.merge(saved, on=["graph", "recording_id", "alarm"], suffixes=("", "_saved"))
            check_rows.append({"check": "coolant notebook-07 replay reproduced", "pairs": len(merged), "expected_pairs": len(saved),
                               "max_abs_difference": float((merged.p_fires_sustained - merged.p_fires).abs().max())})
            ev.save_table(ref, "11_coolant_replay_timelimited")
            ev.save_table(pd.concat(target_frames, ignore_index=True), "11_coolant_replay_targets")
            ev.save_table(pd.DataFrame(summaries), "11_coolant_replay_timelimited_summary")
        # (2) the perturbation-study setting (notebooks 09 and 10): all 48 graphs, 100 samples
        rows, target_frames, probe_ref = [], [], []
        for name, (kind, fraction, seed, getter, graph) in s["variants"].items():
            alarms, targets, identical = paired_replays(s, fold_models(s, getter), 100)
            quality = ge.compare_graphs(graph, s["expert"])
            rows.append({"graph": name, "kind": kind, "fraction": fraction, "seed": seed,
                         "directed_f1": quality["directed_f1"], "skeleton_f1": quality["skeleton_f1"],
                         "identical_before_release": all(identical), **replay_summary(alarms)})
            target_frames.append(targets.assign(graph=name))
            if name in ("expert", "discovered", "empty"):
                probe_ref.append(alarms.assign(graph=name))
        table = pd.DataFrame(rows)
        saved_name = "09_coolant_summary" if sub == "coolant" else "10_probe_perturbation_summary"
        saved = pd.read_csv(TABLES / f"{saved_name}.csv").set_index("graph")
        diff = (table.set_index("graph").p_fires_all_sustained - saved.loc[table.graph, "replay_p_fires_all"].to_numpy()).abs()
        check_rows.append({"check": f"{sub} perturbation-study replay reproduced", "pairs": len(table), "expected_pairs": len(saved),
                           "max_abs_difference": float(diff.max())})
        ev.save_table(table, f"11_{sub}_perturbation_replay_timelimited")
        ev.save_table(pd.concat(target_frames, ignore_index=True), f"11_{sub}_perturbation_replay_targets")
        if sub == "probe":
            ev.save_table(pd.concat(probe_ref, ignore_index=True), "11_probe_replay_timelimited")
        for mode in ("sustained", "limited"):
            corr_rows += [dict(r, subsystem=sub) for r in
                          spearman_rows(table, "directed_f1", [f"p_fires_all_{mode}", f"p_fires_supported_{mode}"], f"replay {mode}")]
    ev.save_table(pd.DataFrame(corr_rows), "11_replay_timelimited_correlations")
    ev.save_table(pd.DataFrame(check_rows), "11_replay_reproduction_checks")
    print(pd.DataFrame(check_rows).to_string(index=False))
    print(f"x1 finished in {time.perf_counter() - start:.0f} s")


# ---------------------------------------------------------------------------
# x2
# ---------------------------------------------------------------------------


def parse_actions(text: str, ranking: pd.DataFrame) -> list[itv.Action]:
    detection = dict(zip(ranking.variable, ranking.detection_step))
    names = [a.split("=")[0].removeprefix("do(") for a in text.split(" + ")]
    return [itv.Action(v, itv.allowed_value(v), max(1, int(detection[v]))) for v in names]


def run_x2() -> None:
    start = time.perf_counter()
    s = setting("coolant")
    per_run = pd.read_csv(TABLES / "09_coolant_per_run.csv")
    summary09 = pd.read_csv(TABLES / "09_coolant_summary.csv").set_index("graph")
    root = np.random.SeedSequence(X2_ENTROPY)
    streams = dict(zip([(g, r) for g in s["variants"] for r in sorted(per_run.recording_id.unique())],
                       root.spawn(len(s["variants"]) * per_run.recording_id.nunique())))
    rows, max_ap_diff = [], 0.0
    for name, (kind, fraction, seed, getter, graph) in s["variants"].items():
        models = fold_models(s, getter)
        saved = per_run[per_run.graph == name].set_index("recording_id")
        for fold, model in models.items():
            for rid in s["splits"].loc[s["splits"].fold == fold, "recording_id"]:
                series = sc.recording_series(s["view"], rid)
                window = rca.analysis_window(series, s["alarms"], INTERVAL)
                ranking = rca.score_candidates(model, series, rca.candidates(series, s["nodes"], s["alarms"]), window)
                ranking["is_cause"] = ranking.variable.isin(s["manipulated"][rid])
                ap = rca.ranking_metrics(ranking.variable, set(s["manipulated"][rid]))["ap@3"]
                max_ap_diff = max(max_ap_diff, abs(ap - saved.loc[rid, "ap@3"]))
                text = saved.loc[rid, "actions"]
                base = {"graph": name, "kind": kind, "fraction": fraction, "seed": seed, "fold": fold, "recording_id": rid,
                        "status_selection": saved.loc[rid, "status"], "actions": text,
                        "validity_selection": saved.loc[rid, "validity_probability"]}
                if not isinstance(text, str):
                    rows.append({**base, "n_worlds": 0})
                    continue
                recorded = {v for v in series if not np.isnan(series[v]).all()}
                outcome = [a for a in s["alarms"] if a in recorded and model.mechanisms[a].has_data]
                acts = parse_actions(text, ranking)
                feasible, _ = itv.check_feasibility(acts, recorded=recorded)
                res = cfx.reevaluate_validity(model, series, acts, outcome, window[0], recorded, streams[(name, rid)])
                rows.append({**base, **res, "feasible": feasible})
    out = pd.DataFrame(rows)
    out["valid_selection"] = out.status_selection == "valid"
    out["valid_independent"] = out.feasible.fillna(False).astype(bool) & (out.validity_independent >= 0.9)
    ev.save_table(out, "11_coolant_cf_independent_validity")
    # per graph: selection vs independent estimate
    g = out.groupby("graph", sort=False)
    summary = pd.DataFrame({
        "kind": g.kind.first(), "fraction": g.fraction.first(), "seed": g.seed.first(),
        "directed_f1": summary09.loc[list(g.groups), "directed_f1"].to_numpy(),
        "valid_share_selection": g.valid_selection.mean(), "valid_share_independent": g.valid_independent.mean(),
        "validity_mean_selection": g.validity_selection.mean(), "validity_mean_independent": g.validity_independent.mean(),
        "selected_valid": g.valid_selection.sum(),
        "selected_valid_still_above_090": g.apply(lambda d: int((d.valid_selection & d.valid_independent).sum())),
        "selected_valid_now_below_090": g.apply(lambda d: int((d.valid_selection & ~d.valid_independent).sum())),
        "selected_valid_interval_below_090": g.apply(lambda d: int((d.valid_selection & (d.wilson_high < 0.9)).sum())),
    }).reset_index()
    ev.save_table(summary, "11_coolant_cf_independent_summary")
    corr = []
    for label, cols in (("selection", ["valid_share_selection", "validity_mean_selection"]),
                        ("independent", ["valid_share_independent", "validity_mean_independent"])):
        corr += spearman_rows(summary, "directed_f1", cols, f"counterfactual {label}")
    ev.save_table(pd.DataFrame(corr), "11_coolant_cf_independent_correlations")
    checks = pd.DataFrame([{"check": "rca rankings reproduced (max |ap@3 difference|)", "value": max_ap_diff},
                           {"check": "graph configurations", "value": len(s["variants"])},
                           {"check": "of which perturbed", "value": sum(v[0] in KINDS for v in s["variants"].values())}])
    ev.save_table(checks, "11_coolant_cf_reproduction_checks")
    print(checks.to_string(index=False))
    print(f"x2 finished in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("x1", "all"):
        run_x1()
    if which in ("x2", "all"):
        run_x2()
