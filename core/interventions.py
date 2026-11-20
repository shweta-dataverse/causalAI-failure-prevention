"""interventions on the scm with actionability and feasibility rules (effects are model-predicted)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.scm import TemporalSCM

# actionable coolant variables from the node comments of the expert graph: (kind, value an action may set, repair).
# a maintenance action only restores a component, a control command may be set to 0 or 1
ACTIONABILITY: dict[str, tuple[str, float | None, str]] = {
    "CLF_Filter_Ok": ("maintenance", 1.0, "clean or replace the fine-particle coolant filter"),
    "F_Filter_Ok": ("maintenance", 1.0, "replace the empty fleece filter"),
    "HP_Pump_Ok": ("maintenance", 1.0, "repair the high-pressure pump / reset its motor protection"),
    "LP_Pump_Ok": ("maintenance", 1.0, "repair the low-pressure pump / reset its motor protection"),
    "LT_Level_Ok": ("maintenance", 1.0, "clear the lifting-tank overflow"),
    "CLT_Level_lt_Min": ("maintenance", 0.0, "refill the coolant tank above the minimum level"),
    "LP_Pump_On": ("control", None, "command the low-pressure pump on or off (plc output)"),
}


@dataclass(frozen=True)
class Action:
    variable: str
    value: float
    start_step: int


def allowed_value(variable: str) -> float | None:
    """the value an action may set (none = both values, for control commands)."""
    return ACTIONABILITY[variable][1] if variable in ACTIONABILITY else None


def check_feasibility(actions: Sequence[Action], recorded: set[str] | None = None,
                      max_actions: int = 3) -> tuple[bool, list[str]]:
    """feasibility rules: actionable variables only, allowed values, recorded in the run, few actions."""
    problems = []
    if len(actions) > max_actions:
        problems.append(f"more than {max_actions} actions")
    for a in actions:
        if a.variable not in ACTIONABILITY:
            problems.append(f"{a.variable} is not actionable")
            continue
        value = allowed_value(a.variable)
        if value is not None and a.value != value:
            problems.append(f"{a.variable} may only be set to {value:g}")
        if a.value not in (0.0, 1.0):
            problems.append(f"{a.variable} is binary")
        if recorded is not None and a.variable not in recorded:
            problems.append(f"{a.variable} is not recorded in this run")
    return not problems, problems


def first_activation(trajectory: np.ndarray, start: int = 1) -> np.ndarray:
    """per sample: first step >= start with a 0 -> 1 transition (nan if never)."""
    if start < 1:
        raise ValueError("start must be >= 1 (a transition needs the previous step)")
    rises = (trajectory[:, start:] == 1) & (trajectory[:, start - 1:-1] == 0)
    return np.where(rises.any(axis=1), rises.argmax(axis=1) + start, np.nan)


def replay_manipulation(
    scm: TemporalSCM, series: dict[str, np.ndarray], onsets: dict[str, int], alarms: Sequence[str],
    n_samples: int, rng: np.random.Generator,
) -> pd.DataFrame:
    """replay the observed manipulations of a fault run as interventions from their onsets and compare the alarms."""
    t0 = min(onsets.values()) - 1
    n_steps = len(next(iter(series.values()))) - t0
    initial = {v: (series[v][t0] if v in series and not np.isnan(series[v][t0]) else scm.fills[v]) for v in scm.mechanisms}
    fixed = {v: scm.fills[v] for v in scm.mechanisms if v not in series or np.isnan(series[v]).all()}
    interventions = {v: (float(series[v][step]), step - t0) for v, step in onsets.items()}
    noise = {v: rng.uniform(size=(n_samples, n_steps)) for v in scm.mechanisms}
    history = {v: series[v][:t0] for v in scm.mechanisms if v in series and v not in fixed}
    sim = scm.simulate(initial, noise, interventions=interventions, fixed=fixed, history=history)
    rows = []
    for a in alarms:
        if a not in series or np.isnan(series[a]).all() or a not in sim:
            continue
        x = series[a][t0:]
        rises = np.flatnonzero((x[1:] == 1) & (x[:-1] == 0)) + 1
        predicted = first_activation(sim[a], start=1)
        fired = ~np.isnan(predicted)
        rows.append({
            "alarm": a,
            "observed_fired": bool(len(rises)),
            "observed_delay_steps": float(rises[0] - 1) if len(rises) else np.nan,
            "p_fires": float(fired.mean()),
            "predicted_median_delay_steps": float(np.median(predicted[fired] - 1)) if fired.any() else np.nan,
            "alarm_supported": scm.mechanisms[a].has_data,
        })
    return pd.DataFrame(rows)


def onset_steps(changes: pd.DataFrame, interval_s: float) -> dict[tuple[str, str], int]:
    """(run, manipulated variable) -> grid step of its first change after the fault start."""
    steps = {}
    for rid, variable, t in zip(changes["recording_id"], changes["variable"], changes["first_change_after_start_s"]):
        if not np.isnan(t):
            steps[(rid, variable)] = int(np.ceil(round(t / interval_s, 9)))
    return steps
