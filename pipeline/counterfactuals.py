from decimal import ROUND_HALF_UP, Decimal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from core import counterfactuals as cfx
from core import data, scm
from core import interventions as itv
from core.plots import COLORS, INK, PAGE_WIDTH, set_print_style
from core.rca import alarm_onsets, analysis_window
from pipeline.interventions import Setting, load_setting
from utils.config import setting
from utils.io import read_input, relative, save_figure, save_table, write_output
from utils.logger import log


# ---------------------------------------------------------------------------
# formatting shared with the evaluation
# ---------------------------------------------------------------------------


def printed(value: float, digits: int = 2) -> str:
    """value rounded half up as printed (0.945 -> 0.95), not to the nearest binary float."""
    return str(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def printed_range(values: pd.Series, digits: int = 2) -> str:
    low, high = printed(values.min(), digits), printed(values.max(), digits)
    return low if low == high else f"{low}–{high}"


def split_names(text) -> list[str]:
    """names of a comma separated cell (empty or missing -> none)."""
    return [x for x in str(text).split(", ") if x and x != "nan"] if isinstance(text, str) else []


def action_names(text: str) -> list[str]:
    """'do(A=1) + do(B=1)' -> ['A', 'B']."""
    return [a.split("=")[0].removeprefix("do(") for a in text.split(" + ")]


def parse_actions(text: str, ranking: pd.DataFrame) -> list[itv.Action]:
    """actions of a recommendation, each from the detection step of its variable."""
    detection = dict(zip(ranking["variable"], ranking["detection_step"]))
    return [itv.Action(v, itv.allowed_value(v), max(1, int(detection[v]))) for v in action_names(text)]


def run_ranking(rankings: pd.DataFrame, graph: str, rid: str) -> pd.DataFrame:
    rows = rankings[(rankings["graph"] == graph) & (rankings["recording_id"] == rid)]
    return rows.sort_values("rank")


def independent_streams(cfg: dict, s: Setting) -> dict[tuple[str, str], np.random.SeedSequence]:
    """one fresh seed sequence per graph and fault run, never used during the search."""
    run_ids = sorted(s.splits.loc[s.splits["split"] == "fault", "recording_id"])
    graphs = setting(cfg, "graphs")
    root = np.random.SeedSequence(setting(cfg, "counterfactuals", "independent_entropy"))
    return dict(zip([(g, r) for g in graphs for r in run_ids], root.spawn(len(graphs) * len(run_ids))))


def fold_settings(scm_settings: pd.DataFrame, s: Setting, graph: str, fold: str) -> tuple[float, int | None]:
    row = scm_settings[(scm_settings["subsystem"] == s.subsystem) & (scm_settings["graph"] == graph)
                       & (scm_settings["scope"] == data.scope_name(fold))].iloc[0]
    return float(row["alpha"]), None if pd.isna(row["lag"]) else int(row["lag"])


def refit(cfg: dict, s: Setting, graph: nx.DiGraph, train_ids: list[str], alpha: float, lag: int | None):
    """the fold scm refitted on other training data with the same graph and settings."""
    lags = scm.extra_lags(graph, lag, setting(cfg, "scm", "max_table_features"))
    return scm.fit_scm(s.view, graph, train_ids, s.manipulated, s.interval_s, smoothing=alpha, lags=lags)


# ---------------------------------------------------------------------------
# search, independent assessment and bootstrap refits
# ---------------------------------------------------------------------------


def explain_graph(cfg: dict, s: Setting, graph: str, rankings: pd.DataFrame, scm_settings: pd.DataFrame,
                  streams: dict) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    c = setting(cfg, "counterfactuals")
    models = {fold: s.fold_model(graph, fold) for fold in s.folds}
    explained, tried = cfx.explain_runs(models, s.view, s.splits, rankings[rankings["graph"] == graph], s.manipulated,
                                        s.alarms, s.interval_s, n_samples=c["selection_worlds"],
                                        seed=c["selection_seed"], top_k=c["top_k"],
                                        validity_threshold=c["validity_threshold"])
    refits = {}
    run_rows, refit_rows = [], []
    for _, r in explained.iterrows():
        fold, rid = r["fold"], r["recording_id"]
        model = models[fold]
        series = scm.recording_series(s.view, rid)
        ranking = run_ranking(rankings, graph, rid)
        window = analysis_window(series, s.alarms, s.interval_s)
        recorded = {v for v in series if not np.isnan(series[v]).all()}
        activated = list(alarm_onsets(series, s.alarms))
        covered = [a for a in activated if model.mechanisms[a].has_data]
        base = {"graph": graph, "fold": fold, "recording_id": rid,
                "model_id": f"{s.subsystem}-{graph}-{data.scope_name(fold)}",
                "status_selection": r["status"], "actions": r.get("actions"), "n_actions": r.get("n_actions", 0),
                "selection_worlds": c["selection_worlds"], "validity_selection": r.get("validity_probability", np.nan),
                "improvement_selection": r.get("improvement", np.nan),
                "factual_alarm_share": r.get("factual_alarm_share", np.nan),
                "action_precision": r["action_precision"], "action_recall": r["action_recall"],
                "true_causes": r["true_causes"],
                "activated_alarms": ", ".join(activated), "covered_alarms": ", ".join(covered),
                "uncovered_alarms": ", ".join(a for a in activated if a not in covered),
                "alarm_coverage": len(covered) / len(activated) if activated else np.nan,
                "rca_top3": " > ".join(ranking["variable"].head(c["top_k"])),
                "window_start_s": window[0] * s.interval_s if window else np.nan}
        if not isinstance(base["actions"], str):
            run_rows.append({**base, "independent_worlds": 0})
            continue
        actions = parse_actions(base["actions"], ranking)
        outcome = [a for a in s.alarms if a in recorded and model.mechanisms[a].has_data]
        feasible, _ = itv.check_feasibility(actions, recorded=recorded, max_actions=c["max_actions"])
        stream = streams[(graph, rid)]
        result = cfx.reevaluate_validity(model, series, actions, outcome, window[0], recorded, stream,
                                         n_batches=c["independent_batches"], batch_size=c["independent_batch_size"])
        # reduction of alarm-active time on one further fresh batch (spawned after the validity batches)
        fresh = cfx.counterfactual(model, series, actions, outcome, window[0], c["independent_batch_size"],
                                   np.random.default_rng(stream.spawn(1)[0]), recorded)
        threshold = c["validity_threshold"]
        run_rows.append({**base, "independent_worlds": result["n_worlds"],
                         "validity_independent": result["validity_independent"],
                         "wilson_low": result["wilson_low"], "wilson_high": result["wilson_high"],
                         "feasible": feasible,
                         "valid_independent": bool(feasible and result["validity_independent"] >= threshold),
                         "borderline (interval contains 0.90)":
                             bool(result["wilson_low"] < threshold <= result["wilson_high"]),
                         "improvement_fresh": fresh.improvement,
                         "action_start_s": "; ".join(f"{a.variable} at {a.start_step * s.interval_s:g} s"
                                                     for a in actions)})
        # robustness to the mechanism estimates: bootstrap refits of the fold training data
        train = data.fold_training_ids(s.splits, fold)
        alpha, lag = fold_settings(scm_settings, s, graph, fold)
        for b in range(c["refits_per_fold"]):
            sample = list(np.random.default_rng(c["refit_sample_seed"] + b).choice(train, size=len(train), replace=True))
            if (fold, b) not in refits:
                refits[(fold, b)] = refit(cfg, s, model.graph, sample, alpha, lag)
            refitted = refits[(fold, b)]
            refit_outcome = [a for a in s.alarms if a in recorded and refitted.mechanisms[a].has_data]
            check = cfx.counterfactual(refitted, series, actions, refit_outcome, window[0], c["selection_worlds"],
                                       np.random.default_rng(c["refit_world_seed"] + b), recorded)
            refit_rows.append({"graph": graph, "fold": fold, "recording_id": rid, "refit": b,
                               "unique_recordings": len(set(sample)), "draws": len(sample),
                               "worlds": c["selection_worlds"], "validity": check.validity_probability})
    return run_rows, tried.assign(graph=graph), refit_rows


# ---------------------------------------------------------------------------
# recommendation tables
# ---------------------------------------------------------------------------


def seconds(value: float) -> str:
    return "0" if value == 0 else f"{value:.1f}"


def alarm_lead_times(s: Setting, runs: pd.DataFrame, rankings: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """(scenario, addressed alarm) -> time from the start of the action reaching it to its recorded onset."""
    leads: dict[tuple[str, str], list[float]] = {}
    for r in runs.itertuples(index=False):
        series = scm.recording_series(s.view, r.recording_id)
        model_graph = s.fold_model(r.graph, r.fold).graph
        actions = parse_actions(r.actions, run_ranking(rankings, r.graph, r.recording_id))
        onsets = alarm_onsets(series, split_names(r.covered_alarms))
        for alarm, onset in onsets.items():
            start = max(a.start_step for a in actions if nx.has_path(model_graph, a.variable, alarm))
            leads.setdefault((r.fold, alarm), []).append((onset - start) * s.interval_s)
    return leads


def action_text(names: list[str], component_names: dict[str, str], below_threshold: bool) -> str:
    if len(names) == 1:
        text = f"keep the {component_names[names[0]]} healthy ({names[0]})"
    else:
        parts = [f"the {component_names[v]} ({v})" for v in names]
        text = f"keep {', '.join(parts[:-1])} and {parts[-1]} healthy"
    if below_threshold:
        return f"Proposal below the validity threshold: {text}"
    return text[0].upper() + text[1:]


def recommendation_table(cfg: dict, s: Setting, runs: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """recommended action per scenario with the addressed alarms, validity and valid runs."""
    first, second = setting(cfg, "graphs")
    names = setting(cfg, "counterfactuals", "component_names")
    main = runs[runs["graph"] == first]
    leads = alarm_lead_times(s, main, rankings)
    rows = []
    for fold, d in main.groupby("fold"):
        other = runs[(runs["graph"] == second) & (runs["fold"] == fold)]
        n_valid = int(d["valid_independent"].astype(bool).sum())
        actions = list(dict.fromkeys(v for text in d["actions"] for v in action_names(text)))
        addressed = list(dict.fromkeys(a for text in d["covered_alarms"] for a in split_names(text)))
        uncovered = list(dict.fromkeys(a for text in d["uncovered_alarms"] for a in split_names(text)))
        timing = []
        for alarm in addressed:
            lead = leads[(fold, alarm)]
            low, high = seconds(min(lead)), seconds(max(lead))
            timing.append(f"{alarm} ({low if low == high else f'{low}–{high}'} s)")
        rows.append({
            "Scenario": fold,
            "Recommended action": action_text(actions, names, below_threshold=n_valid == 0),
            "Alarm addressed (time before observed alarm)": ", ".join(timing),
            "Alarm without training data": ", ".join(uncovered) or "–",
            "Validity, expert (discovered)": f"{printed_range(d['validity_independent'])} "
                                             f"({printed_range(other['validity_independent'])})",
            "Valid": f"{n_valid} / {len(d)}",
            "_order": (n_valid == 0, fold),
        })
    table = pd.DataFrame(rows).sort_values("_order")
    return table.drop(columns="_order").reset_index(drop=True)


def short_name(variable: str) -> str:
    """HP_Pump_Ok -> HP, HP_A_700304 -> HP_A."""
    if variable[-1].isdigit():
        return variable.rsplit("_", 1)[0]
    return variable.split("_")[0]


def per_run_table(cfg: dict, s: Setting, runs: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """actions, evaluated alarm onsets and validity of every run (first graph)."""
    graph = setting(cfg, "graphs")[0]
    rows = []
    for r in runs[runs["graph"] == graph].itertuples(index=False):
        series = scm.recording_series(s.view, r.recording_id)
        actions = parse_actions(r.actions, run_ranking(rankings, graph, r.recording_id))
        onsets = alarm_onsets(series, split_names(r.covered_alarms))
        rows.append({
            "Scenario": r.fold,
            "Run": int(r.recording_id.rsplit("_", 1)[1]),
            "Action from (s)": ", ".join(f"{short_name(a.variable)} {a.start_step * s.interval_s:g}" for a in actions),
            "Evaluated alarm (s)": ", ".join(f"{short_name(a)} {t * s.interval_s:g}" for a, t in onsets.items()),
            "Validity": f"{printed(r.validity_independent)} [{printed(r.wilson_low)}, {printed(r.wilson_high)}]",
            "Valid": "yes" if r.valid_independent else "no",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# counterfactual example figure
# ---------------------------------------------------------------------------


def example_figure(cfg: dict, s: Setting, runs: pd.DataFrame, rankings: pd.DataFrame) -> plt.Figure:
    """observed states and counterfactual probability of state 1 for the example run."""
    graph = setting(cfg, "graphs")[0]
    rid = setting(cfg, "figure_examples", "coolant_run")
    fold = setting(cfg, "figure_examples", "coolant_fold")
    c = setting(cfg, "counterfactuals")
    model = s.fold_model(graph, fold)
    row = runs[(runs["graph"] == graph) & (runs["recording_id"] == rid)].iloc[0]
    actions = parse_actions(row["actions"], run_ranking(rankings, graph, rid))
    series = scm.recording_series(s.view, rid)
    stream = independent_streams(cfg, s)[(graph, rid)]
    world = cfx.independent_worlds(model, series, actions, stream, c["independent_batches"],
                                   c["independent_batch_size"])
    recorded = {v for v in series if not np.isnan(series[v]).all()}
    window = analysis_window(series, s.alarms, s.interval_s)
    outcome = [a for a in s.alarms if a in recorded and model.mechanisms[a].has_data]
    any_alarm = np.max(np.stack([world[a][:, window[0]:] for a in outcome]), axis=0)
    observed = np.column_stack([np.nan_to_num(series[a][window[0]:]) for a in outcome]).max(axis=1)
    log.debug(f"example {rid}: validity {(any_alarm.max(axis=1) == 0).mean():.4f} "
              f"(table {row['validity_independent']:.4f}), reduction of alarm-active time "
              f"{observed.mean() - any_alarm.mean():.4f}")

    set_print_style()
    targets = [a.variable for a in actions]
    alarms = [a for a in s.alarms if a in recorded]
    shown = targets + alarms
    roles = {v: "action target" for v in targets}
    roles.update({a: "alarm with training data" if model.mechanisms[a].has_data else "alarm without training data"
                  for a in alarms})
    t = np.arange(len(series[shown[0]])) * s.interval_s
    fig, axes = plt.subplots(len(shown), 1, figsize=(PAGE_WIDTH, 1.35 * len(shown) + 1.0), sharex=True,
                             layout="constrained")
    for ax, v in zip(axes, shown):
        ax.step(t, series[v], where="post", color=INK["muted"], linewidth=2.2)
        if model.mechanisms[v].has_data:
            ax.plot(t, world[v].mean(axis=0), color=COLORS[0], linewidth=1.8)
        for a in actions:
            ax.axvline(a.start_step * s.interval_s, color=COLORS[1], linewidth=1.4)
        ax.set_ylim(-0.1, 1.15)
        ax.set_yticks([0, 1])
        ax.set_title(f"{v} ({roles[v]})", loc="left", fontsize=9.5)
    axes[-1].set_xlabel("Time since recording start (s)")
    act = ", ".join(f"do({a.variable} = {int(a.value)}) at {a.start_step * s.interval_s:g} s" for a in actions)
    handles = [Line2D([], [], color=INK["muted"], lw=2.2, label="Observed state"),
               Line2D([], [], color=COLORS[0], lw=1.8, label="Counterfactual probability of state 1"),
               Line2D([], [], color=COLORS[1], lw=1.4, label=f"Action: {act}")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    return fig


def run(cfg: dict, force: bool = False) -> None:
    s = load_setting(cfg, setting(cfg, "counterfactuals", "subsystem"))
    rankings = read_input("rca_rankings")
    rankings = rankings[rankings["subsystem"] == s.subsystem]
    scm_settings = read_input("scm_settings")
    streams = independent_streams(cfg, s)
    run_rows, tried, refit_rows = [], [], []
    for graph in setting(cfg, "graphs"):
        rows, tried_sets, refits = explain_graph(cfg, s, graph, rankings, scm_settings, streams)
        run_rows += rows
        tried.append(tried_sets)
        refit_rows += refits
    runs = pd.DataFrame(run_rows)
    file = write_output(runs, "recommendations")
    write_output(pd.concat(tried, ignore_index=True), "tried_action_sets")
    write_output(pd.DataFrame(refit_rows), "recommendation_refits")
    n_valid = {g: int(d["valid_independent"].astype(bool).sum()) for g, d in runs.groupby("graph", sort=False)}
    log.info(f"counterfactual explanations done. {len(runs)} recommendations "
             f"({', '.join(f'{g} {n} valid' for g, n in n_valid.items())} of {runs['recording_id'].nunique()} runs) "
             f"saved to {relative(file)}")
    save_table(recommendation_table(cfg, s, runs, rankings), "6.7", "Recommended coolant maintenance actions",
               "pipeline")
    save_table(per_run_table(cfg, s, runs, rankings), "A.7", "Coolant counterfactual results by run", "pipeline")
    save_figure(example_figure(cfg, s, runs, rankings), "6.9", "Counterfactual proposal below the validity threshold",
                "pipeline")
