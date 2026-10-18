"""individual counterfactual explanations by abduction, action and prediction.

1. abduction: sample the noise u of every variable and step that is consistent with
   the observed run (uniform-threshold noise, see scm.py).
2. action: replace the equations of the chosen actionable variables by their allowed
   value from the step where their mechanism break was detected.
3. prediction: roll the equations forward with the abducted noise.

the result answers "had these variables been kept healthy, would the alarms have
fired?" *under the model*. it is not identified from data alone (it relies on the
monotone threshold noise), and it is not an observed outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from causal_rca.interventions import Action, allowed_value, check_feasibility
from causal_rca.scm import TemporalSCM


@dataclass
class CounterfactualResult:
    actions: list[Action]
    feasible: bool
    problems: list[str]
    factual_alarm_share: float
    counterfactual_alarm_share: float
    validity_probability: float
    proximity_s: float
    outcome_alarms: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        return self.factual_alarm_share - self.counterfactual_alarm_share

    @property
    def sparsity(self) -> int:
        return len(self.actions)

    def valid(self, threshold: float = 0.9) -> bool:
        return self.feasible and self.validity_probability >= threshold

    def to_dict(self) -> dict:
        return {
            "actions": " + ".join(f"do({a.variable}={a.value:g})" for a in self.actions),
            "n_actions": self.sparsity,
            "feasible": self.feasible,
            "problems": "; ".join(self.problems),
            "factual_alarm_share": self.factual_alarm_share,
            "counterfactual_alarm_share": self.counterfactual_alarm_share,
            "improvement": self.improvement,
            "validity_probability": self.validity_probability,
            "valid": self.valid(),
            "proximity_s": self.proximity_s,
            "outcome_alarms": ", ".join(self.outcome_alarms),
        }


def initial_state(scm: TemporalSCM, series: dict[str, np.ndarray]) -> dict[str, float]:
    """first known value of every variable (or its usual state)."""
    out = {}
    for v in scm.mechanisms:
        x = series.get(v)
        out[v] = float(x[0]) if x is not None and not np.isnan(x[0]) else scm.fills[v]
    return out


def unrecorded_context(scm: TemporalSCM, series: dict[str, np.ndarray]) -> dict[str, float]:
    """variables without any known state in the run, held at their usual state."""
    return {v: scm.fills[v] for v in scm.mechanisms if v not in series or np.isnan(series[v]).all()}


def counterfactual(
    scm: TemporalSCM,
    series: dict[str, np.ndarray],
    actions: Sequence[Action],
    outcome_alarms: Sequence[str],
    outcome_start: int,
    n_samples: int,
    rng: np.random.Generator,
    recorded: set[str] | None = None,
) -> CounterfactualResult:
    """abduction, action and prediction for one run and one set of actions.

    outcomes are measured from ``outcome_start`` (the start of the label-free analysis
    window) to the end of the run, the same for every action set. an action set is
    valid when no outcome alarm is active in that period in the counterfactual world.
    """
    feasible, problems = check_feasibility(actions, recorded=recorded)
    noise = scm.abduct_noise(series, n_samples, rng)
    fixed = {v: x for v, x in unrecorded_context(scm, series).items() if v not in {a.variable for a in actions}}
    interventions = {a.variable: (a.value, a.start_step) for a in actions}
    world = scm.simulate(initial_state(scm, series), noise, interventions=interventions, fixed=fixed)

    if outcome_alarms:
        observed = np.column_stack([np.nan_to_num(series[a][outcome_start:]) for a in outcome_alarms])
        factual_share = float(observed.max(axis=1).mean())
        cf_any = np.max(np.stack([world[a][:, outcome_start:] for a in outcome_alarms]), axis=0)
        cf_share = float(cf_any.mean())
        validity = float((cf_any.max(axis=1) == 0).mean())
    else:
        factual_share = cf_share = validity = np.nan

    proximity = 0.0
    for a in actions:
        x = series.get(a.variable)
        if x is not None:
            changed = ~np.isnan(x[a.start_step:]) & (x[a.start_step:] != a.value)
            proximity += float(changed.sum()) * scm.interval_s
    return CounterfactualResult(list(actions), feasible, problems, factual_share, cf_share, validity, proximity,
                                list(outcome_alarms))


def recommend(
    scm: TemporalSCM,
    series: dict[str, np.ndarray],
    ranking: pd.DataFrame,
    detection_step: dict[str, int],
    outcome_alarms: Sequence[str],
    outcome_start: int,
    n_samples: int,
    rng: np.random.Generator,
    recorded: set[str],
    top_k: int = 3,
    max_actions: int = 3,
    validity_threshold: float = 0.9,
) -> tuple[CounterfactualResult | None, list[CounterfactualResult]]:
    """smallest valid set of maintenance actions among the top-ranked actionable candidates.

    candidates are the rca ranking restricted to actionable variables (top_k of them),
    each set to its allowed healthy value from its detection step. sets are tried by
    size, then by rank. returns the first valid result (or None) and all tried results.
    """
    actionable = [v for v in ranking["variable"] if allowed_value(v) is not None][:top_k]
    tried = []
    for size in range(1, min(max_actions, len(actionable)) + 1):
        for subset in combinations(actionable, size):
            actions = [Action(v, allowed_value(v), max(1, detection_step[v])) for v in subset]
            result = counterfactual(scm, series, actions, outcome_alarms, outcome_start, n_samples, rng,
                                    recorded=recorded)
            tried.append(result)
            if result.valid(validity_threshold):
                return result, tried
    return None, tried


def abduction_consistency(scm: TemporalSCM, series: dict[str, np.ndarray], n_samples: int,
                          rng: np.random.Generator) -> float:
    """share of recorded cells reproduced when the abducted noise is replayed without any action.

    variables without a mechanism estimate (never recorded in training) are held at their
    usual state by design and are left out of the check.
    """
    noise = scm.abduct_noise(series, n_samples, rng)
    fixed = unrecorded_context(scm, series)
    world = scm.simulate(initial_state(scm, series), noise, fixed=fixed)
    matches, total = 0, 0
    for v in scm.mechanisms:
        if v in fixed or v not in series or not scm.mechanisms[v].has_data:
            continue
        known = ~np.isnan(series[v])
        matches += int((world[v][:, known] == series[v][known]).sum())
        total += int(known.sum()) * n_samples
    return matches / total if total else np.nan


def explain_runs(
    scms: dict[str, TemporalSCM],
    view_data,
    splits: pd.DataFrame,
    rankings: pd.DataFrame,
    manipulated: dict[str, set[str]],
    alarms: Sequence[str],
    interval_s: float,
    n_samples: int = 200,
    seed: int = 0,
    top_k: int = 3,
    validity_threshold: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """counterfactual recommendation for every held-out run (``scms[fold]`` never saw the fold).

    ``rankings`` are the rca rankings of the same folds (with detection steps). the
    ground truth is used only afterwards, to compare the recommendation with the true causes.
    returns one row per run and one row per tried action set.
    """
    from causal_rca.root_cause_analysis import analysis_window
    from causal_rca.scm import recording_series

    rng = np.random.default_rng(seed)
    run_rows, tried_rows = [], []
    for fold, model in scms.items():
        for rid in splits.loc[splits["fold"] == fold, "recording_id"]:
            series = recording_series(view_data, rid)
            ranking = rankings[rankings["recording_id"] == rid].sort_values("rank")
            window = analysis_window(series, alarms, interval_s)
            if window is None or ranking.empty:
                continue
            recorded = {v for v in series if not np.isnan(series[v]).all()}
            outcome = [a for a in alarms if a in recorded and model.mechanisms[a].has_data]
            unsupported = [a for a in alarms if a in recorded and not model.mechanisms[a].has_data]
            detection = dict(zip(ranking["variable"], ranking["detection_step"]))
            best, tried = recommend(model, series, ranking, detection, outcome, window[0], n_samples, rng,
                                    recorded, top_k=top_k, validity_threshold=validity_threshold)
            for result in tried:
                tried_rows.append({"fold": fold, "recording_id": rid, **result.to_dict()})
            chosen = best if best is not None else max(tried, key=lambda r: r.improvement, default=None)
            truth = set(manipulated[rid])
            actions = {a.variable for a in chosen.actions} if chosen else set()
            run_rows.append({
                "fold": fold, "recording_id": rid,
                "status": "valid" if best is not None else ("partial" if chosen else "none"),
                **(chosen.to_dict() if chosen else {}),
                "outcome_unsupported": ", ".join(unsupported),
                "true_causes": ", ".join(sorted(truth)),
                "action_precision": len(actions & truth) / len(actions) if actions else np.nan,
                "action_recall": len(actions & truth) / len(truth) if truth else np.nan,
                "abduction_consistency": abduction_consistency(model, series, 20, rng),
            })
    return pd.DataFrame(run_rows), pd.DataFrame(tried_rows)
