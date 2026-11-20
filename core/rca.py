"""root cause analysis by mechanism breaks: candidates ranked by their largest surprise near the alarms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.scm import TemporalSCM


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
    """steps [first alarm onset - before_s, last alarm onset + after_s], or None without any alarm."""
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


# ranking metrics


def ranking_metrics(ranked: Sequence[str], truth: set[str], k: int = 3) -> dict[str, float]:
    """top-1, hit@k, recall@k, ap@k (normalised by min(|truth|, k)) and reciprocal rank of one ranking."""
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


def metric_names(k: int) -> list[str]:
    return ["top1", f"hit@{k}", f"recall@{k}", f"ap@{k}", "rr"]
