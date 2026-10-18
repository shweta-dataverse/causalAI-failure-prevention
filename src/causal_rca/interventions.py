"""population-level interventions on the scm, with actionability and feasibility rules.

an intervention do(X = x) replaces X's structural equation by the constant x from a
given step on. effects are *model-predicted*: they hold only if the graph and the
fitted mechanisms are right, and they say nothing about real failure prevention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_rca.scm import TemporalSCM

# domain semantics of the coolant variables, from the node comments of the expert graph.
# kind 'maintenance': a component condition a technician can restore (only to `value`);
# kind 'control': a plc command that may be set to 0 or 1. anything not listed here is
# treated as non-actionable.
ACTIONABILITY: dict[str, tuple[str, float | None, str]] = {
    "CLF_Filter_Ok": ("maintenance", 1.0, "clean or replace the fine-particle coolant filter"),
    "F_Filter_Ok": ("maintenance", 1.0, "replace the empty fleece filter"),
    "HP_Pump_Ok": ("maintenance", 1.0, "repair the high-pressure pump / reset its motor protection"),
    "LP_Pump_Ok": ("maintenance", 1.0, "repair the low-pressure pump / reset its motor protection"),
    "LT_Level_Ok": ("maintenance", 1.0, "clear the lifting-tank overflow"),
    "CLT_Level_lt_Min": ("maintenance", 0.0, "refill the coolant tank above the minimum level"),
    "LP_Pump_On": ("control", None, "command the low-pressure pump on or off (plc output)"),
}

# why the other coolant variables are not actionable
NOT_ACTIONABLE_REASONS: dict[str, str] = {
    "HP_Pump_isOff": "feedback state of the high-pressure pump, set by the plc logic (derived)",
    "LT_Pump_Ok": "supervision feedback of the lifting-tank pump (derived)",
}


@dataclass(frozen=True)
class Action:
    variable: str
    value: float
    start_step: int


def actionability_table(nodes: Sequence[str], types: dict[str, str]) -> pd.DataFrame:
    """actionability, allowed values and the reason for every variable."""
    rows = []
    for v in sorted(nodes):
        if types.get(v) == "Alarm":
            rows.append({"variable": v, "actionable": False, "kind": "alarm", "allowed_values": "",
                         "reason": "alarm: observable outcome, never an action"})
        elif v in ACTIONABILITY:
            kind, value, reason = ACTIONABILITY[v]
            rows.append({"variable": v, "actionable": True, "kind": kind,
                         "allowed_values": "0, 1" if value is None else f"{value:g}", "reason": reason})
        else:
            rows.append({"variable": v, "actionable": False, "kind": "not actionable", "allowed_values": "",
                         "reason": NOT_ACTIONABLE_REASONS.get(v, "unknown actionability: treated as not actionable")})
    return pd.DataFrame(rows)


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


# ---------------------------------------------------------------------------
# population-level effects
# ---------------------------------------------------------------------------


def simulate_from_state(
    scm: TemporalSCM, initial: dict[str, float], n_steps: int, n_samples: int, rng: np.random.Generator,
    interventions: dict[str, tuple[float, int]] | None = None, fixed: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """monte-carlo trajectories from one initial state with fresh noise."""
    noise = {v: rng.uniform(size=(n_samples, n_steps)) for v in scm.mechanisms}
    return scm.simulate(initial, noise, interventions=interventions, fixed=fixed)


def first_activation(trajectory: np.ndarray, start: int = 1) -> np.ndarray:
    """per sample: first step >= start at which the variable is 1 after being 0 at start - 1 (nan if never)."""
    on = trajectory[:, start:] == 1
    first = np.where(on.any(axis=1), on.argmax(axis=1) + start, np.nan)
    return first


def alarm_response(
    scm: TemporalSCM, initial: dict[str, float], alarms: Sequence[str], interventions: dict[str, tuple[float, int]],
    horizon_steps: int, n_samples: int, rng: np.random.Generator, fixed: dict[str, float] | None = None,
) -> pd.DataFrame:
    """predicted probability that each alarm fires within the horizon and its onset delay (steps)."""
    sim = simulate_from_state(scm, initial, horizon_steps, n_samples, rng, interventions, fixed)
    start = min((s for _, s in interventions.values()), default=1)
    rows = []
    for a in alarms:
        if a not in sim:
            continue
        onset = first_activation(sim[a], start=max(start, 1))
        fired = ~np.isnan(onset)
        rows.append({
            "alarm": a,
            "p_fires": float(fired.mean()),
            "median_delay_steps": float(np.median(onset[fired] - start)) if fired.any() else np.nan,
        })
    return pd.DataFrame(rows)


def effect_matrix(
    scm: TemporalSCM, variables: Sequence[str], outcomes: Sequence[str], horizon_steps: int,
    n_samples: int, rng: np.random.Generator, start_step: int = 1,
) -> pd.DataFrame:
    """predicted average effect of each variable on each outcome, from the usual (healthy) state.

    maintenance variables: do(faulty) vs do(healthy); control variables: do(1) vs do(0).
    effect = p(outcome away from its usual state at the horizon | first) - p(... | second).
    """
    initial = {v: scm.fills[v] for v in scm.mechanisms}
    rows = []
    for v in variables:
        healthy = allowed_value(v)
        first, second = ((1.0 - healthy, healthy) if healthy is not None else (1.0, 0.0))
        sims = {}
        for label, value in (("first", first), ("second", second)):
            noise = {u: rng.uniform(size=(n_samples, horizon_steps)) for u in scm.mechanisms}
            sims[label] = scm.simulate(initial, noise, interventions={v: (value, start_step)})
        for o in outcomes:
            if o == v or o not in scm.mechanisms:
                continue
            usual = scm.fills[o]
            p_first = float((sims["first"][o][:, -1] != usual).mean())
            p_second = float((sims["second"][o][:, -1] != usual).mean())
            rows.append({"variable": v, "outcome": o, "do_first": first, "do_second": second,
                         "p_first": p_first, "p_second": p_second, "effect": p_first - p_second,
                         "outcome_supported": scm.mechanisms[o].has_data})
    return pd.DataFrame(rows)


def replay_manipulation(
    scm: TemporalSCM, series: dict[str, np.ndarray], onsets: dict[str, int], alarms: Sequence[str],
    n_samples: int, rng: np.random.Generator,
) -> pd.DataFrame:
    """replay the observed manipulations of one fault run as interventions and compare alarms.

    every manipulated variable is set to its observed post-onset value from its observed
    onset step; the simulation starts one step before the first onset from the observed
    state, with unrecorded variables held at their usual state. returns per alarm the
    observed onset delay and the predicted firing probability and median delay (steps,
    relative to the first manipulation onset).
    """
    t0 = min(onsets.values()) - 1
    n_steps = len(next(iter(series.values()))) - t0
    initial = {v: (series[v][t0] if v in series and not np.isnan(series[v][t0]) else scm.fills[v]) for v in scm.mechanisms}
    fixed = {v: scm.fills[v] for v in scm.mechanisms if v not in series or np.isnan(series[v]).all()}
    interventions = {v: (float(series[v][step]), step - t0) for v, step in onsets.items()}
    noise = {v: rng.uniform(size=(n_samples, n_steps)) for v in scm.mechanisms}
    sim = scm.simulate(initial, noise, interventions=interventions, fixed=fixed)
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
