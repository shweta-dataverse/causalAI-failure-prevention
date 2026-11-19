import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core import counterfactuals as cfx
from core import data, scm
from core import interventions as itv
from core.plots import COLORS, GRAPH_COLORS, GRAPH_LABELS, INK, PAGE_WIDTH, set_print_style
from core.rca import analysis_window
from pipeline.counterfactuals import (
    fold_settings,
    independent_streams,
    printed,
    printed_range,
    refit,
    run_ranking,
    split_names,
)
from pipeline.interventions import Setting, load_setting
from utils.config import setting
from utils.io import read_input, relative, save_figure, save_table, tables_dir
from utils.logger import log


def write_detail(table: pd.DataFrame, name: str) -> None:
    """unnumbered detail table of the evaluation (outputs/evaluation/tables/<name>.csv)."""
    file = tables_dir("evaluation") / f"{name}.csv"
    file.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(file, index=False)
    log.debug(f"wrote {relative(file)} ({len(table)} rows)")


def count(part: int, total: int) -> str:
    return f"{part} / {total}"


def valid_mask(runs: pd.DataFrame) -> pd.Series:
    return runs["valid_independent"].fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# intervention analysis
# ---------------------------------------------------------------------------


def replay_summary(replay: pd.DataFrame, folds: list[str], graphs: list[str]) -> pd.DataFrame:
    """observed activations, simulated activation probability and onset error per graph and scope."""
    rows = []
    scopes = [(g, "all scenarios") for g in graphs] + [(g, f) for g in graphs for f in folds]
    for graph, scope in scopes:
        in_scope = (replay["graph"] == graph) & (True if scope == "all scenarios" else replay["fold"] == scope)
        d = replay[in_scope & replay["observed_fired"]]
        supported = d[d["alarm_supported"]]
        scored = supported[supported["p_fires"] > 0]
        rows.append({
            "graph": graph, "scope": scope, "observed_activations": len(d), "supported": len(supported),
            "unsupported (no mechanism data)": len(d) - len(supported),
            "activation_probability_supported_mean": supported["p_fires"].mean() if len(supported) else np.nan,
            "miss_probability_supported_mean": (1 - supported["p_fires"]).mean() if len(supported) else np.nan,
            "supported_never_activated": int((supported["p_fires"] == 0).sum()),
            "onset_error_scored_pairs": int(scored["predicted_delay_s"].notna().sum()),
            "onset_abs_error_median_s": float(np.nanmedian(scored["abs_onset_error_s"])) if len(scored) else np.nan,
            "onset_abs_error_mean_s": float(np.nanmean(scored["abs_onset_error_s"])) if len(scored) else np.nan,
            "negatives (not activated in recording)": int((in_scope & ~replay["observed_fired"]).sum()),
        })
    return pd.DataFrame(rows)


def intervention_figure(replay: pd.DataFrame, graphs: list[str], interval_s: float) -> plt.Figure:
    """observed activations, simulated activation probability and activation-time error per scenario."""
    set_print_style()
    observed = replay[replay["observed_fired"]]
    folds = sorted(observed["fold"].unique())
    x = np.arange(len(folds))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 1, figsize=(PAGE_WIDTH, 7.6), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 0.75, 1.1]})
    first = observed[observed["graph"] == graphs[0]]
    modelled = first[first["alarm_supported"]].groupby("fold").size().reindex(folds, fill_value=0)
    total = first.groupby("fold").size().reindex(folds)

    ax = axes[0]
    ax.bar(x, modelled, width=0.5, color=COLORS[0], label="Alarm with training data")
    ax.bar(x, total - modelled, bottom=modelled, width=0.5, color="white", edgecolor=INK["muted"], hatch="///",
           label="Alarm without training data")
    for i, f in enumerate(folds):
        ax.text(i, total[f] + 0.4, f"{modelled[f]} of {total[f]}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, total.max() * 1.25)
    ax.set_xticks(x, folds)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Activations")
    ax.set_title("(a) Observed alarm activations", loc="left")

    ticks = [f"{f}\nn = {int(modelled[f])}" for f in folds]
    ax = axes[1]
    for k, g in enumerate(graphs):
        d = observed[(observed["graph"] == g) & observed["alarm_supported"]]
        for i, f in enumerate(folds):
            v = d.loc[d["fold"] == f, "p_fires"]
            ax.scatter(np.full(len(v), i + (k - 0.5) * 0.3) + rng.uniform(-0.06, 0.06, len(v)), v, s=16,
                       color=GRAPH_COLORS[g], zorder=3)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks(x, ticks)
    ax.set_ylabel("Probability")
    ax.set_title("(b) Predicted activation probability", loc="left")

    ax = axes[2]
    for k, g in enumerate(graphs):
        d = observed[(observed["graph"] == g) & observed["alarm_supported"] & observed["abs_onset_error_s"].notna()]
        for i, f in enumerate(folds):
            v = d.loc[d["fold"] == f, "abs_onset_error_s"]
            offset = i + (k - 0.5) * 0.3
            ax.scatter(np.full(len(v), offset) + rng.uniform(-0.06, 0.06, len(v)), v, s=16, color=GRAPH_COLORS[g],
                       alpha=0.8, zorder=3)
            ax.plot([offset - 0.12, offset + 0.12], [v.median()] * 2, color=INK["primary"], linewidth=2, zorder=4)
    ax.axhline(interval_s, color=INK["secondary"], linestyle="--", linewidth=1)
    ax.set_xticks(x, ticks)
    ax.set_ylim(-0.15, 3.4)
    ax.set_ylabel("Activation-time error (s)")
    ax.set_xlabel("Test scenario")
    ax.set_title("(c) Alarm activation-time error", loc="left")

    handles = [Patch(color=COLORS[0], label="Alarm with training data"),
               Patch(facecolor="white", edgecolor=INK["muted"], hatch="///", label="Alarm without training data"),
               *[Line2D([], [], marker="o", linestyle="", color=GRAPH_COLORS[g], label=GRAPH_LABELS[g]) for g in graphs],
               Line2D([], [], color=INK["primary"], lw=2, label="Scenario median"),
               Line2D([], [], color=INK["secondary"], lw=1, linestyle="--", label=f"One grid step ({interval_s:g} s)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    return fig


def block_line(label: str, values: list[str]) -> str:
    """one metric line with one column per graph."""
    return (f"  {label:<40}" + "".join(f"{v:<14}" for v in values)).rstrip()


def replay_stats(replay: pd.DataFrame, interval_s: float) -> dict:
    observed = replay[replay["observed_fired"]]
    supported = observed[observed["alarm_supported"]]
    errors = supported["abs_onset_error_s"]
    return {"runs": replay["recording_id"].nunique(), "observed": len(observed), "supported": len(supported),
            "unsupported": len(observed) - len(supported),
            "unsupported_alarms": observed.loc[~observed["alarm_supported"], "alarm"].nunique(),
            "reproduced": int((supported["p_fires"] > 0.5).sum()), "lowest_p": supported["p_fires"].min(),
            "within": int((errors <= interval_s).sum()), "median": errors.median(), "max": errors.max(),
            "negatives": int((~replay["observed_fired"]).sum())}


def report_interventions(replay: pd.DataFrame, graphs: list[str], subsystem: str, interval_s: float) -> None:
    stats = {g: replay_stats(replay[replay["graph"] == g], interval_s) for g in graphs}
    first = stats[graphs[0]]
    worlds = int(replay["worlds"].iloc[0])
    p_by_graph = replay.pivot_table(index=["recording_id", "alarm"], columns="graph", values="p_fires")
    p_gap = (p_by_graph.max(axis=1) - p_by_graph.min(axis=1)).max()
    log.info("\n".join([
        f"\nintervention analysis ({subsystem}, {first['runs']} held-out runs, {worlds} simulations per run)",
        block_line("", graphs),
        block_line("observed alarm activations", [str(stats[g]["observed"]) for g in graphs]),
        block_line("with training data in the fold", [str(stats[g]["supported"]) for g in graphs]),
        block_line("without training data (not simulated)",
                   [f"{stats[g]['unsupported']} ({stats[g]['unsupported_alarms']} alarms)" for g in graphs]),
        block_line("activations reproduced (p > 0.5)",
                   [f"{stats[g]['reproduced']} of {stats[g]['supported']}" for g in graphs]),
        block_line("lowest activation probability", [f"{stats[g]['lowest_p']:.3f}" for g in graphs]),
        block_line(f"onset error within one step ({interval_s:g} s)",
                   [f"{stats[g]['within']} of {stats[g]['supported']}" for g in graphs]),
        block_line("onset error median / max (s)", [f"{stats[g]['median']:.1f} / {stats[g]['max']:.1f}" for g in graphs]),
        block_line("recorded alarms that never activate", [str(stats[g]["negatives"]) for g in graphs]),
        block_line("largest probability gap between graphs", [f"{p_gap:.3f}"]),
    ]))
    least = int(round(min(stats[g]["lowest_p"] for g in graphs) * worlds))
    sentence = (f"the fold models reproduce {first['reproduced']} of the {first['supported']} alarm activations that "
                f"have training data (each in at least {least} of {worlds} simulations), {first['within']} of them "
                f"within one grid step, so the simulated fault responses agree with the recordings; the "
                f"{first['unsupported']} activations without training data are not covered")
    if first["negatives"] == 0:
        sentence += ", every recorded alarm activates, so false activations cannot be assessed"
    log.info(sentence + ", and the check is retrospective on recorded fault injections, not a test of repairs.")


# ---------------------------------------------------------------------------
# counterfactual explanations
# ---------------------------------------------------------------------------


def coverage_count(runs: pd.DataFrame) -> str:
    covered = sum(len(split_names(x)) for x in runs["covered_alarms"])
    activated = sum(len(split_names(x)) for x in runs["activated_alarms"])
    return f"{covered}/{activated}"


def results_table(runs: pd.DataFrame, refits: pd.DataFrame, graphs: list[str], threshold: float) -> pd.DataFrame:
    """counterfactual results per graph for all proposals and for the valid recommendations."""
    rows = {}
    for g in graphs:
        d = runs[runs["graph"] == g]
        valid = d[valid_mask(d)]
        lowest_refit = refits[refits["graph"] == g].groupby("recording_id")["validity"].min()
        whole = {
            "Reaching 0.90 in the search (200 simulations)": count(int((d["status_selection"] == "valid").sum()), len(d)),
            "Valid recommendations (2,000 simulations)": count(len(valid), len(d)),
            "Valid for all five resampled SCMs": count(int((lowest_refit >= threshold).sum()), len(d)),
        }
        for name, part in (("All", d), ("Valid", valid)):
            rows[(g, name)] = {
                **{k: (v if name == "All" else "") for k, v in whole.items()},
                "Mean validity": printed(part["validity_independent"].mean(), 3),
                "Lowest lower limit of the 95% Wilson interval": printed(part["wilson_low"].min()),
                "Alarm coverage, count of activations": coverage_count(part),
                "Alarm coverage, mean per run": printed(part["alarm_coverage"].mean()),
                "Mean number of actions": printed(part["n_actions"].mean()),
                "Mean reduction in time with an active alarm": printed(part["improvement_fresh"].mean()),
                "Action precision": printed(part["action_precision"].mean()),
                "Action recall": printed(part["action_recall"].mean()),
            }
    order = ["Reaching 0.90 in the search (200 simulations)", "Valid recommendations (2,000 simulations)",
             "Mean validity", "Lowest lower limit of the 95% Wilson interval", "Alarm coverage, count of activations",
             "Alarm coverage, mean per run", "Mean number of actions", "Mean reduction in time with an active alarm",
             "Action precision", "Action recall", "Valid for all five resampled SCMs"]
    table = pd.DataFrame({f"{GRAPH_LABELS[g]}, {name.lower()}": [rows[(g, name)][m] for m in order]
                          for g, name in rows})
    table.insert(0, "Metric", order)
    return table


def summary(runs: pd.DataFrame, refits: pd.DataFrame, graphs: list[str], threshold: float) -> pd.DataFrame:
    """detail summary per graph (validity, coverage, actions, refits)."""
    rows = []
    for g in graphs:
        d = runs[runs["graph"] == g]
        rf = refits[refits["graph"] == g]
        rows.append({
            "graph": g, "runs": len(d), "runs_with_actions": int(d["actions"].notna().sum()),
            "valid_share_selection (200 worlds)": (d["status_selection"] == "valid").mean(),
            "valid_share_independent (2000 fresh worlds)": valid_mask(d).mean(),
            "validity_independent_mean": d["validity_independent"].mean(),
            "validity_independent_scenario_macro": d.groupby("fold")["validity_independent"].mean().mean(),
            "borderline_runs": int(d["borderline (interval contains 0.90)"].fillna(False).astype(bool).sum()),
            "alarm_coverage_run_mean": d["alarm_coverage"].mean(),
            "alarm_coverage_scenario_macro": d.groupby("fold")["alarm_coverage"].mean().mean(),
            "alarms_covered_pooled": coverage_count(d),
            "runs_all_alarms_covered": int((d["alarm_coverage"] == 1).sum()),
            "actions_mean": d["n_actions"].mean(), "improvement_mean (fresh worlds)": d["improvement_fresh"].mean(),
            "action_precision_mean": d["action_precision"].mean(), "action_recall_mean": d["action_recall"].mean(),
            "refit_validity_mean": rf["validity"].mean(), "refit_validity_min": rf["validity"].min(),
            "runs_valid_in_all_refits": int((rf.groupby("recording_id")["validity"].min() >= threshold).sum()),
        })
    return pd.DataFrame(rows)


def settings_check(cfg: dict, s: Setting, rankings: pd.DataFrame) -> pd.DataFrame:
    """validity of one fixed action in one fold under several scm settings, with the independent streams."""
    check = setting(cfg, "counterfactuals", "settings_check")
    c = setting(cfg, "counterfactuals")
    fold, alarm, target = check["fold"], check["alarm"], check["action"]
    train = data.fold_training_ids(s.splits, fold)
    rows = []
    for graph in setting(cfg, "graphs"):
        fold_graph = s.fold_model(graph, fold).graph
        for alpha, lag, label in check["settings"]:
            # fresh streams per setting: spawning advances a seed sequence
            streams = independent_streams(cfg, s)
            model = refit(cfg, s, fold_graph, train, alpha, lag)
            stays = cfx.persistence(model, alarm)
            for rid in s.runs_of(fold):
                series = scm.recording_series(s.view, rid)
                ranking = run_ranking(rankings, graph, rid)
                detection = dict(zip(ranking["variable"], ranking["detection_step"]))
                start = max(1, int(detection[target]))
                actions = [itv.Action(target, itv.allowed_value(target), start)]
                window = analysis_window(series, s.alarms, s.interval_s)
                recorded = {v for v in series if not np.isnan(series[v]).all()}
                outcome = [a for a in s.alarms if a in recorded and model.mechanisms[a].has_data]
                result = cfx.reevaluate_validity(model, series, actions, outcome, window[0], recorded, streams[(graph, rid)],
                                                 n_batches=c["independent_batches"],
                                                 batch_size=c["independent_batch_size"])
                batch = cfx.counterfactual(model, series, actions, outcome, window[0], check["worlds"],
                                           np.random.default_rng(streams[(graph, rid)].spawn(1)[0]), recorded)
                recovery = next(i for i in range(start + 1, len(series[target])) if series[target][i] == 1)
                # where do the simulations that fail activate the alarm: before or after the recorded recovery
                noise = model.abduct_noise(series, check["worlds"], np.random.default_rng(check["persistence_seed"]))
                world = model.simulate(cfx.initial_state(model, series), noise,
                                       interventions={target: (actions[0].value, start)},
                                       fixed=cfx.unrecorded_context(model, series))
                active = world[alarm][:, window[0]:]
                fired = active.max(axis=1) > 0
                first = np.argmax(active > 0, axis=1)[fired] + window[0]
                rows.append({"graph": graph, "setting": label, "pseudo_count": alpha, "extra_lag": lag or 0,
                             "recording_id": rid, "p_alarm_stays_with_healthy_pump": stays,
                             "validity_2000": result["validity_independent"],
                             "valid": result["validity_independent"] >= c["validity_threshold"],
                             "pump_recovery_s": recovery * s.interval_s,
                             "share_failures_after_recovery": float((first >= recovery).mean()) if fired.any() else np.nan,
                             "validity_check_batch": batch.validity_probability})
    return pd.DataFrame(rows)


def persistence_text(value: float) -> str:
    return printed(value) if value >= 0.01 else f"{value:.1g}"


def settings_table(cfg: dict, s: Setting, check: pd.DataFrame, scm_settings: pd.DataFrame) -> pd.DataFrame:
    """validity of the fixed action per scm setting (first graph)."""
    graph = setting(cfg, "graphs")[0]
    fold = setting(cfg, "counterfactuals", "settings_check", "fold")
    baseline = setting(cfg, "scm", "baseline")
    named = {(baseline["alpha"], baseline["lag"]): "original", fold_settings(scm_settings, s, graph, fold): "selected"}
    rows = []
    for alpha, lag, label in setting(cfg, "counterfactuals", "settings_check", "settings"):
        d = check[(check["graph"] == graph) & (check["setting"] == label)]
        name = f"a = {alpha:g}, " + ("no extra lag" if lag is None else f"extra lag {lag} steps")
        if (alpha, lag) in named:
            name += f" ({named[(alpha, lag)]})"
        rows.append({f"SCM setting of the {fold} fold": name,
                     "Persistence": persistence_text(d["p_alarm_stays_with_healthy_pump"].iloc[0]),
                     "Validity (expert)": printed_range(d["validity_2000"]),
                     "Valid runs": count(int(d["valid"].sum()), len(d))})
    return pd.DataFrame(rows)


def pick_examples(runs: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """fixed rule: the valid run with the highest coverage, the lowest coverage and the lowest validity."""
    picks = []
    valid = runs[valid_mask(runs)].sort_values(["alarm_coverage", "recording_id"], ascending=[False, True])
    if len(valid):
        picks.append(("success", valid.iloc[0]))
    picks.append(("lowest_coverage", runs.sort_values(["alarm_coverage", "recording_id"]).iloc[0]))
    picks.append(("lowest_validity", runs.sort_values(["validity_independent", "recording_id"]).iloc[0]))
    return picks


def join_words(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def example_rows(cfg: dict, number: int, rule: str, r: pd.Series) -> list[dict]:
    c = setting(cfg, "counterfactuals")
    names = c["alarm_names"]
    valid = bool(r["valid_independent"])
    starts = [part.split(" at ") for part in r["action_start_s"].split("; ")]
    actions = [f"Keep {starts[0][0]} healthy from {starts[0][1]}", *[f"{v} from {t}" for v, t in starts[1:]]]
    covered, uncovered = split_names(r["covered_alarms"]), split_names(r["uncovered_alarms"])
    verb = "remains" if len(covered) == 1 else "remain"
    if valid:
        outcome = f"{join_words(covered)} {verb} inactive"
    else:
        labels = [f"the {names[a]} {a}" if a in names else a for a in covered]
        share = printed(100 * r["validity_independent"], 0)
        outcome = f"{join_words(labels)} {verb} inactive in {share}% of the simulations"
        outcome = outcome[0].upper() + outcome[1:]
    n_activated = len(covered) + len(uncovered)
    if uncovered:
        has = "has" if len(uncovered) == 1 else "have"
        coverage = f"{len(covered)} of {n_activated}: {join_words(uncovered)} {has} no training data in this fold"
    else:
        coverage = f"{len(covered)} of {n_activated} activated alarms have training data"
    validity = f"{printed(r['validity_independent'])}, interval [{printed(r['wilson_low'])}, {printed(r['wilson_high'])}]"
    run = int(r["recording_id"].rsplit("_", 1)[1])
    case = f"Case {number}"
    return [
        {"Case": case, "Item": "Run", "Text": f"{r['fold']}, run {run} — {c['examples'][rule]['title']}"},
        {"Case": case, "Item": "Counterfactual actions" if valid else "Proposal below threshold",
         "Text": " and ".join(actions)},
        {"Case": case, "Item": "Model-predicted outcome", "Text": outcome},
        {"Case": case, "Item": "Validity", "Text": validity if valid else f"{validity}; not valid"},
        {"Case": case, "Item": "Coverage", "Text": coverage},
        {"Case": case, "Item": "Limitation", "Text": c["examples"][rule]["limitation"]},
    ]


def examples_detail(picks: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    columns = ["fold", "recording_id", "activated_alarms", "rca_top3", "true_causes", "actions", "action_start_s",
               "validity_independent", "wilson_low", "wilson_high", "covered_alarms", "uncovered_alarms",
               "alarm_coverage", "improvement_fresh"]
    return pd.DataFrame([{"example": rule, **r[columns].to_dict()} for rule, r in picks])


def counterfactual_figure(runs: pd.DataFrame, graphs: list[str], threshold: float) -> plt.Figure:
    """validity of the selected action set per run and alarm coverage per scenario."""
    set_print_style()
    folds = sorted(runs["fold"].unique())
    x = np.arange(len(folds))
    rng = np.random.default_rng(0)
    counts = runs[runs["graph"] == graphs[0]].groupby("fold").size()
    ticks = [f"{f}\nn = {int(counts[f])}" for f in folds]
    fig, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH, 5.6), layout="constrained")
    ax = axes[0]
    for k, g in enumerate(graphs):
        d = runs[runs["graph"] == g]
        for i, f in enumerate(folds):
            v = d.loc[d["fold"] == f, "validity_independent"].dropna()
            offset = i + (k - 0.5) * 0.3
            ax.scatter(np.full(len(v), offset) + rng.uniform(-0.06, 0.06, len(v)), v, s=18, color=GRAPH_COLORS[g],
                       zorder=3)
            ax.plot([offset - 0.12, offset + 0.12], [v.mean()] * 2, color=INK["primary"], linewidth=2, zorder=4)
    ax.axhline(threshold, color=INK["secondary"], linestyle="--", linewidth=1)
    ax.set_ylim(0.78, 1.01)
    ax.set_xticks(x, ticks)
    ax.set_ylabel("Counterfactual validity")
    ax.set_title("(a) Counterfactual validity", loc="left")
    ax = axes[1]
    for k, g in enumerate(graphs):
        coverage = runs[runs["graph"] == g].groupby("fold")["alarm_coverage"].mean().reindex(folds)
        ax.bar(x + (k - 0.5) * 0.36, coverage, width=0.36, color=GRAPH_COLORS[g])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, ticks)
    ax.set_ylabel("Alarm coverage")
    ax.set_xlabel("Test scenario")
    ax.set_title("(b) Alarm coverage", loc="left")
    handles = [Line2D([], [], marker="o", linestyle="", color=GRAPH_COLORS[g], label=GRAPH_LABELS[g]) for g in graphs]
    handles += [Line2D([], [], color=INK["primary"], lw=2, label="Scenario mean"),
                Line2D([], [], color=INK["secondary"], lw=1, linestyle="--",
                       label=f"Validity threshold: {threshold:.2f}")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    return fig


def report_counterfactuals(cfg: dict, runs: pd.DataFrame, refits: pd.DataFrame, check: pd.DataFrame,
                           subsystem: str) -> None:
    graphs = setting(cfg, "graphs")
    c = setting(cfg, "counterfactuals")
    threshold = c["validity_threshold"]
    worlds = c["independent_batches"] * c["independent_batch_size"]

    stats = {}
    for g in graphs:
        d = runs[runs["graph"] == g]
        valid = valid_mask(d)
        rf = refits[refits["graph"] == g]
        lowest = rf.groupby("recording_id")["validity"].min()
        stats[g] = {
            "runs": len(d), "valid": int(valid.sum()), "search": int((d["status_selection"] == "valid").sum()),
            "range_valid": printed_range(d.loc[valid, "validity_independent"]),
            "range_below": printed_range(d.loc[~valid, "validity_independent"]) if (~valid).any() else "-",
            "crossing": int(d["borderline (interval contains 0.90)"].fillna(False).astype(bool).sum()),
            "coverage": coverage_count(d),
            "stay_valid": int((lowest[d.loc[valid, "recording_id"]] >= threshold).sum()),
            "refit_share": (rf["validity"] >= threshold).mean(), "refit_n": len(rf),
            "below_folds": sorted(d.loc[~valid, "fold"].unique()),
        }
    first = graphs[0]
    labels = [label for _, _, label in c["settings_check"]["settings"]]

    def settings_valid(g: str) -> str:
        d = check[check["graph"] == g]
        valid = [int(d.loc[d["setting"] == label, "valid"].sum()) for label in (labels[0], labels[-1])]
        return f"{valid[0]} / {valid[1]} of {int((d['setting'] == labels[0]).sum())}"

    log.info("\n".join([
        f"\ncounterfactual explanations ({subsystem}, {stats[first]['runs']} held-out runs, "
        f"{worlds:,} independent simulations per proposal)",
        block_line("", graphs),
        block_line("valid recommendations", [count(stats[g]["valid"], stats[g]["runs"]) for g in graphs]),
        block_line("reaching 0.90 in the search", [count(stats[g]["search"], stats[g]["runs"]) for g in graphs]),
        block_line("validity range, valid", [stats[g]["range_valid"] for g in graphs]),
        block_line("validity range, below threshold", [stats[g]["range_below"] for g in graphs]),
        block_line("wilson intervals containing 0.90", [str(stats[g]["crossing"]) for g in graphs]),
        block_line("alarm coverage (activations)", [stats[g]["coverage"] for g in graphs]),
        block_line("valid runs valid in all refits", [count(stats[g]["stay_valid"], stats[g]["valid"]) for g in graphs]),
        block_line("refit evaluations reaching 0.90",
                   [f"{stats[g]['refit_share']:.0%} of {stats[g]['refit_n']}" for g in graphs]),
        block_line(f"{c['settings_check']['fold']} valid, {labels[0]} / {labels[-1]} scm",
                   [settings_valid(g) for g in graphs]),
    ]))
    same = all(stats[g]["valid"] == stats[first]["valid"] for g in graphs)
    valid_ranges = runs.loc[valid_mask(runs), "validity_independent"]
    lead = (f"both graphs give valid recommendations for {stats[first]['valid']} of {stats[first]['runs']} runs"
            if same else "the graphs give " + " and ".join(f"{stats[g]['valid']} ({g})" for g in graphs)
            + f" valid recommendations of {stats[first]['runs']} runs")
    stable = all(stats[g]["stay_valid"] == stats[g]["valid"] for g in graphs)
    below = sorted({f for g in graphs for f in stats[g]["below_folds"]})
    log.info(
        f"{lead} (validity {printed_range(valid_ranges)} in {worlds:,} independent simulations)"
        + (", and every valid recommendation stays valid in all five bootstrap-refitted scms" if stable else
           ", but not every valid recommendation stays valid in all five bootstrap-refitted scms")
        + (f"; the proposals of {', '.join(below)} stay below {threshold:.2f}" if below else "")
        + f". validity is model-based (prevention predicted by the scm, not observed), the "
        f"{coverage_uncovered(runs, first)} alarm activations without training data are not covered, and the "
        "action timing is retrospective, not an advance warning."
    )


def coverage_uncovered(runs: pd.DataFrame, graph: str) -> int:
    return sum(len(split_names(x)) for x in runs.loc[runs["graph"] == graph, "uncovered_alarms"])


def run(cfg: dict, force: bool = False) -> None:
    graphs = setting(cfg, "graphs")
    threshold = setting(cfg, "counterfactuals", "validity_threshold")
    s = load_setting(cfg, setting(cfg, "counterfactuals", "subsystem"))
    replay = read_input("replay")
    runs = read_input("recommendations")
    refits = read_input("recommendation_refits")
    rankings = read_input("rca_rankings")
    rankings = rankings[rankings["subsystem"] == s.subsystem]
    scm_settings = read_input("scm_settings")

    write_detail(replay_summary(replay, s.folds, graphs), "replay_summary")
    save_figure(intervention_figure(replay, graphs, s.interval_s), "6.7", "Alarm responses to fault interventions",
                "evaluation")
    report_interventions(replay, graphs, s.subsystem, s.interval_s)

    write_detail(summary(runs, refits, graphs, threshold), "counterfactual_summary")
    save_table(results_table(runs, refits, graphs, threshold), "6.8", "Counterfactual results for coolant", "evaluation")
    check = settings_check(cfg, s, rankings)
    write_detail(check, "settings_check")
    save_table(settings_table(cfg, s, check, scm_settings), "6.9", "Validity under different SCM settings",
               "evaluation")
    picks = pick_examples(runs[runs["graph"] == graphs[0]])
    write_detail(examples_detail(picks), "counterfactual_examples")
    rows = [row for i, (rule, r) in enumerate(picks, start=1) for row in example_rows(cfg, i, rule, r)]
    save_table(pd.DataFrame(rows), "6.10", "Examples of counterfactual explanations", "evaluation")
    save_figure(counterfactual_figure(runs, graphs, threshold), "6.8", "Counterfactual validity and alarm coverage",
                "evaluation")
    report_counterfactuals(cfg, runs, refits, check, s.subsystem)
