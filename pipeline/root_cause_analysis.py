import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core import data, rca, scm
from core.plots import COLORS, INK, PAGE_WIDTH, set_print_style
from pipeline.scm_fitting import Subsystem, load_subsystem
from utils import io
from utils.config import setting
from utils.logger import log


def analysis_window(cfg: dict, sub: Subsystem, series: dict) -> tuple[int, int] | None:
    rc = setting(cfg, "rca")
    return rca.analysis_window(series, sub.alarms, sub.interval_s, rc["window_before_s"], rc["window_after_s"])


def ranking_of(cfg: dict, sub: Subsystem, model: scm.TemporalSCM, series: dict) -> pd.DataFrame | None:
    """candidates of one run ranked by mechanism break (None when no alarm switches on)."""
    window = analysis_window(cfg, sub, series)
    if window is None:
        return None
    return rca.score_candidates(model, series, rca.candidates(series, sub.nodes, sub.alarms), window)


def rank_run(cfg: dict, sub: Subsystem, model: scm.TemporalSCM, rid: str) -> tuple[dict, pd.DataFrame | None]:
    """metrics of one held-out run and its candidate ranking (None when no alarm switches on)."""
    rc = setting(cfg, "rca")
    series = scm.recording_series(sub.view, rid)
    truth = set(sub.manipulated[rid])
    base = {"recording_id": rid, "n_causes": len(truth)}
    window = analysis_window(cfg, sub, series)
    if window is None:
        return {**base, "scored": False}, None
    cands = rca.candidates(series, sub.nodes, sub.alarms)
    ranking = rca.score_candidates(model, series, cands, window)
    ranking["is_cause"] = ranking["variable"].isin(truth)
    chance = rca.chance_metrics(cands, truth, rc["top_k"], rc["chance_draws"], rc["seed"])
    metrics = {**base, "scored": True, "n_candidates": len(cands),
               "window_start_s": window[0] * sub.interval_s, "window_end_s": window[1] * sub.interval_s,
               **rca.ranking_metrics(ranking["variable"], truth, rc["top_k"]),
               **{f"chance_{k}": v for k, v in chance.items()}}
    return metrics, ranking


def rank_subsystem(cfg: dict, sub: Subsystem, graph: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """every held-out run ranked with the scm of its fold."""
    run_rows, rankings = [], []
    for fold in sub.folds:
        scope = data.scope_name(fold)
        model = scm.load_fitted(sub.name, graph, scope)
        for rid in sub.runs_of(fold):
            metrics, ranking = rank_run(cfg, sub, model, rid)
            run_rows.append({"fold": fold, **metrics, "subsystem": sub.name, "graph": graph,
                             "model_id": f"{sub.name}-{graph}-{scope}"})
            if ranking is not None:
                rankings.append(ranking.assign(fold=fold, recording_id=rid, subsystem=sub.name, graph=graph))
    return pd.DataFrame(run_rows), pd.concat(rankings, ignore_index=True)


# ---------------------------------------------------------------------------
# probe analysis windows
# ---------------------------------------------------------------------------


def first_change(series: dict, variables: set[str], start_s: float, interval_s: float) -> float:
    """first observed change of an injected variable at or after the annotated fault start."""
    times = []
    for v in sorted(variables):
        x = series[v]
        known = ~np.isnan(x)
        steps = np.flatnonzero(known[1:] & known[:-1] & (x[1:] != x[:-1])) + 1
        steps = steps[steps * interval_s >= start_s]
        if len(steps):
            times.append(steps[0] * interval_s)
    return min(times) if times else np.nan


def window_table(cfg: dict, sub: Subsystem) -> pd.DataFrame:
    """annotated fault interval, first injected change, alarm onsets and analysis window of every run."""
    annotations = data.load_fault_runs().set_index("recording_id")
    rows = []
    for fold in sub.folds:
        for rid in sub.runs_of(fold):
            series = scm.recording_series(sub.view, rid)
            start, end = annotations.loc[rid, ["cause_start_at", "cause_end_at"]]
            change = first_change(series, sub.manipulated[rid], start, sub.interval_s)
            # a first change right after the annotated end is the release of the fault
            release = bool(0 <= change - end <= 3 * sub.interval_s)
            onsets = sorted(rca.alarm_onsets(series, sub.alarms).items(), key=lambda item: item[1])
            window = analysis_window(cfg, sub, series)
            rows.append({
                "Scenario": fold, "Run": int(rid.rsplit("_", 1)[1]), "Start": f"{start:.1f}", "End": f"{end:.1f}",
                "First change": f"{change:.1f}" + ("^r" if release else ""),
                "Alarm onset": "; ".join(f"{a} {t * sub.interval_s:.1f}" for a, t in onsets),
                "Window": "not scored" if window is None else
                f"{window[0] * sub.interval_s:.1f}–{window[1] * sub.interval_s:.1f}",
            })
    return pd.DataFrame(rows).sort_values(["Scenario", "Run"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# surprise timeline of one run
# ---------------------------------------------------------------------------


def surprise_figure(cfg: dict, sub: Subsystem) -> plt.Figure:
    """surprise of every candidate over one coolant run, with the analysis window and the alarm onsets."""
    examples = setting(cfg, "figure_examples")
    rid = examples["coolant_run"]
    model = scm.load_fitted(sub.name, "expert", data.scope_name(examples["coolant_fold"]))
    series = scm.recording_series(sub.view, rid)
    window = analysis_window(cfg, sub, series)
    cands = rca.candidates(series, sub.nodes, sub.alarms)
    causes = sub.manipulated[rid]
    t = np.arange(1, len(series[cands[0]])) * sub.interval_s
    set_print_style()
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.9), layout="constrained")
    ax.axvspan(window[0] * sub.interval_s, window[1] * sub.interval_s, color=INK["grid"], zorder=0)
    for k, v in enumerate(cands):
        cause = v in causes
        ax.plot(t, model.surprise(series, v), linestyle="-" if cause else ":", linewidth=2.0 if cause else 1.5,
                color=COLORS[k % len(COLORS)], label=v + (" (injected cause)" if cause else ""))
    for step in rca.alarm_onsets(series, sub.alarms).values():
        ax.axvline(step * sub.interval_s, color=INK["secondary"], linewidth=0.9, linestyle="--")
    ax.set_xlabel("Time since recording start (s)")
    ax.set_ylabel("Surprise score (nats)")
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(color=INK["grid"], label="Analysis window"),
                Line2D([], [], color=INK["secondary"], linestyle="--", lw=0.9, label="First alarm activation")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


def run(cfg: dict, force: bool = False) -> None:
    run_tables, ranking_tables, counts = [], [], {}
    subsystems = {name: load_subsystem(cfg, name) for name in setting(cfg, "subsystems")}
    for name, sub in subsystems.items():
        for graph in setting(cfg, "graphs"):
            runs, rankings = rank_subsystem(cfg, sub, graph)
            run_tables.append(runs)
            ranking_tables.append(rankings)
            counts[name] = int(runs.scored.sum())
    io.write_output(pd.concat(run_tables, ignore_index=True), "rca_runs")
    io.write_output(pd.concat(ranking_tables, ignore_index=True), "rca_rankings")
    io.save_table(window_table(cfg, subsystems["probe"]), "A.6", "Probe analysis windows", "pipeline")
    io.save_figure(surprise_figure(cfg, subsystems["coolant"]), "6.4", "Root-cause scores during a coolant fault",
                   "pipeline")
    log.info(f"root cause analysis done. {counts['coolant']} coolant and {counts['probe']} probe runs ranked with "
             f"both graphs, saved to {io.relative(io.contract_path('rca_runs'))}")
