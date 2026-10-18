"""scm-based root cause analysis: which variable's own mechanism breaks?

an intervention on a variable replaces its structural equation, so a root cause
shows up as values its fitted mechanism cannot explain *given its parents*, while
downstream effects are explained by their (changed) parents. the score of a candidate
is its largest surprise -log p(x[t] | x[t-1], parents) inside the analysis window.

leakage rules:
* the analysis window is anchored on observed alarm onsets (visible at run time),
  never on causes.json.
* ground-truth causes are used only to evaluate rankings and, for *training*
  recordings, to skip manipulated rows when fitting mechanisms.
* candidates are the non-alarm variables recorded in the run. which variables a run
  records already hints at the scenario, so every result is compared with the chance
  level under the same candidate set.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from causal_rca.data import fold_training_ids
from causal_rca.preprocessing import ViewData
from causal_rca.scm import TemporalSCM, fit_scm, recording_series


def alarm_onsets(series: dict[str, np.ndarray], alarms: Sequence[str]) -> dict[str, int]:
    """first step at which each recorded alarm switches from 0 to 1."""
    onsets = {}
    for a in alarms:
        if a not in series:
            continue
        x = series[a]
        rises = np.flatnonzero((x[1:] == 1) & (x[:-1] == 0)) + 1
        if len(rises):
            onsets[a] = int(rises[0])
    return onsets


def analysis_window(
    series: dict[str, np.ndarray], alarms: Sequence[str], interval_s: float,
    before_s: float = 60.0, after_s: float = 5.0,
) -> tuple[int, int] | None:
    """steps [first alarm onset - before_s, last alarm onset + after_s]; None without any alarm."""
    onsets = alarm_onsets(series, alarms)
    if not onsets:
        return None
    n = len(next(iter(series.values())))
    start = max(1, min(onsets.values()) - int(round(before_s / interval_s)))
    end = min(n - 1, max(onsets.values()) + int(round(after_s / interval_s)))
    return start, end


def candidates(series: dict[str, np.ndarray], variables: Sequence[str], alarms: Sequence[str]) -> list[str]:
    """non-alarm variables with at least one known state in this run."""
    alarm_set = set(alarms)
    return [v for v in variables if v not in alarm_set and v in series and not np.isnan(series[v]).all()]


def score_candidates(
    scm: TemporalSCM, series: dict[str, np.ndarray], candidate_vars: Sequence[str], window: tuple[int, int]
) -> pd.DataFrame:
    """rank candidates by maximum surprise in the window (ties: summed surprise, then name)."""
    start, end = window
    rows = []
    for v in candidate_vars:
        s = scm.surprise(series, v)  # s[k] belongs to step k + 1
        window_values = s[start - 1:end]
        known = ~np.isnan(window_values)
        part = window_values[known]
        rows.append({
            "variable": v,
            "score": float(part.max()) if part.size else np.nan,
            "score_sum": float(part.sum()) if part.size else np.nan,
            # step of the largest surprise: when the mechanism break was detected
            "detection_step": int(start + np.nanargmax(np.where(known, window_values, -np.inf))) if part.size else start,
            "n_parents": len(scm.mechanisms[v].parents) if v in scm.mechanisms else 0,
        })
    table = pd.DataFrame(rows)
    table = table.sort_values(["score", "score_sum", "variable"], ascending=[False, False, True], na_position="last")
    table["rank"] = np.arange(1, len(table) + 1)
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------


def ranking_metrics(ranked: Sequence[str], truth: set[str], k: int = 3) -> dict[str, float]:
    """top-1, hit@k, recall@k, average precision@k and reciprocal rank for one ranking.

    multiple true causes: hit@k needs any of them in the top k, recall@k counts the
    share found, ap@k = (1 / min(|truth|, k)) * sum of precision@i at hits i <= k.
    """
    ranked = list(ranked)
    hits = [v in truth for v in ranked]
    top = hits[:k]
    precisions = [sum(hits[:i + 1]) / (i + 1) for i in range(len(top)) if top[i]]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    return {
        "top1": float(bool(hits[:1]) and hits[0]),
        f"hit@{k}": float(any(top)),
        f"recall@{k}": sum(top) / len(truth) if truth else np.nan,
        f"ap@{k}": sum(precisions) / min(len(truth), k) if truth else np.nan,
        "rr": 1.0 / first if first else 0.0,
    }


def chance_metrics(candidate_vars: Sequence[str], truth: set[str], k: int = 3, n_draws: int = 2000,
                   seed: int = 0) -> dict[str, float]:
    """expected metrics of a uniformly random ranking of the same candidates."""
    rng = np.random.default_rng(seed)
    draws = [ranking_metrics(list(rng.permutation(list(candidate_vars))), truth, k) for _ in range(n_draws)]
    return pd.DataFrame(draws).mean().to_dict()


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """percentile bootstrap interval of the mean over runs."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


# ---------------------------------------------------------------------------
# leave-one-scenario-out evaluation
# ---------------------------------------------------------------------------


def evaluate_rca(
    view_data: ViewData,
    splits: pd.DataFrame,
    manipulated: dict[str, set[str]],
    graph_for: Callable[[str], nx.DiGraph],
    alarms: Sequence[str],
    nodes: Sequence[str],
    interval_s: float,
    folds: Sequence[str] | None = None,
    n_chance: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, TemporalSCM]]:
    """fit the scm without each fault scenario and rank the candidates of its runs.

    ``graph_for(fold)`` returns the (acyclic) graph to use for that fold. returns per-run
    metrics (with the chance level of the same candidate set), per-candidate rankings,
    and the fitted scm of every fold.
    """
    folds = list(folds) if folds is not None else sorted(splits.loc[splits["split"] == "fault", "fold"].unique())
    metric_rows, ranking_frames, scms = [], [], {}
    for fold in folds:
        model = fit_scm(view_data, graph_for(fold), fold_training_ids(splits, fold), manipulated, interval_s)
        scms[fold] = model
        for rid in splits.loc[splits["fold"] == fold, "recording_id"]:
            series = recording_series(view_data, rid)
            truth = set(manipulated[rid])
            window = analysis_window(series, alarms, interval_s)
            base = {"fold": fold, "recording_id": rid, "n_causes": len(truth)}
            if window is None:
                metric_rows.append({**base, "scored": False})
                continue
            cands = candidates(series, nodes, alarms)
            ranking = score_candidates(model, series, cands, window)
            ranking["is_cause"] = ranking["variable"].isin(truth)
            ranking_frames.append(ranking.assign(fold=fold, recording_id=rid))
            chance = chance_metrics(cands, truth, n_draws=n_chance, seed=seed)
            metric_rows.append({
                **base, "scored": True, "n_candidates": len(cands),
                "window_start_s": window[0] * interval_s, "window_end_s": window[1] * interval_s,
                **ranking_metrics(ranking["variable"], truth),
                **{f"chance_{k}": v for k, v in chance.items()},
            })
    rankings = pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame()
    return pd.DataFrame(metric_rows), rankings, scms


def summarize_rca(metrics: pd.DataFrame, columns: Sequence[str] = ("top1", "hit@3", "recall@3", "ap@3", "rr"),
                  n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """mean of every metric over scored runs with a bootstrap interval, and the chance level."""
    scored = metrics[metrics["scored"]]
    rows = []
    for c in columns:
        low, high = bootstrap_ci(scored[c], n_boot=n_boot, seed=seed)
        rows.append({"metric": c, "mean": scored[c].mean(), "ci_low": low, "ci_high": high,
                     "chance": scored[f"chance_{c}"].mean(), "n_runs": len(scored)})
    return pd.DataFrame(rows)
