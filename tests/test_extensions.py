"""tests for the robustness extensions: time-limited replay (x1) and independent validity (x2)."""

from __future__ import annotations

import numpy as np
import pytest

from causal_rca import counterfactuals as cfx
from causal_rca import interventions as itv
from causal_rca import scm as sc
from test_core import faulty_run, fitted_scm


def _noise(model, n_samples=200, n_steps=60, seed=0):
    rng = np.random.default_rng(seed)
    return {v: rng.uniform(size=(n_samples, n_steps)) for v in model.mechanisms}


INITIAL = {"r": 1.0, "p": 0.0, "c": 0.0, "a": 0.0}


def test_without_releases_the_replay_is_sustained_as_before():
    _, model = fitted_scm()
    noise = _noise(model)
    default = model.simulate(INITIAL, noise, interventions={"p": (1.0, 10)})
    none = model.simulate(INITIAL, noise, interventions={"p": (1.0, 10)}, releases=None)
    # a release after the last step never takes effect, so it equals the sustained run
    late = model.simulate(INITIAL, noise, interventions={"p": (1.0, 10)}, releases={"p": 1000})
    for v in model.mechanisms:
        assert np.array_equal(default[v], none[v]) and np.array_equal(default[v], late[v])
    assert (default["p"][:, 10:] == 1).all()


def test_release_keeps_the_trajectory_before_release_and_then_follows_the_mechanism():
    _, model = fitted_scm()
    noise = _noise(model, n_samples=2000)
    sustained = model.simulate(INITIAL, noise, interventions={"p": (1.0, 10)})
    limited = model.simulate(INITIAL, noise, interventions={"p": (1.0, 10)}, releases={"p": 20})
    for v in model.mechanisms:
        assert np.array_equal(sustained[v][:, :20], limited[v][:, :20])
    # after release p is drawn from its fitted mechanism with previous state 1 (switches about 5% of steps)
    switch_off = 1 - model.mechanisms["p"].prob_one(np.array([1.0]), np.zeros((1, 0)))[0]
    observed = (limited["p"][:, 20] == 0).mean()
    assert abs(observed - switch_off) < 0.03
    assert 0 < (limited["p"][:, 20:] == 0).any(axis=1).mean() < 1
    # c still follows p after the release
    assert np.array_equal(limited["c"][:, 25:] == 1, limited["p"][:, 25:] == 1) or (limited["c"][:, 25:] == limited["p"][:, 25:]).mean() > 0.9


def test_after_release_an_unseen_faulty_state_follows_the_fallback_not_a_forced_repair():
    # r never takes the faulty value 0 in training, so its table has no rows for previous state 0
    # and falls back to 0.5; after release about half of the samples leave 0 in the first step.
    # a forced repair would move all samples back at once.
    _, model = fitted_scm()
    mech = model.mechanisms["r"]
    assert mech.n_rows[0] == 0
    p_back = mech.prob_one(np.array([0.0]), np.zeros((1, 0)))[0]
    assert p_back == pytest.approx(0.5)
    noise = _noise(model, n_samples=4000)
    limited = model.simulate(INITIAL, noise, interventions={"r": (0.0, 10)}, releases={"r": 20})
    assert abs((limited["r"][:, 20] == 1).mean() - p_back) < 0.03
    assert (limited["r"][:, 19] == 0).all()


def test_release_arguments_are_validated():
    _, model = fitted_scm()
    noise = _noise(model)
    with pytest.raises(ValueError):
        model.simulate(INITIAL, noise, interventions={"p": (1.0, 10)}, releases={"p": 10})
    with pytest.raises(ValueError):
        model.simulate(INITIAL, noise, interventions={"p": (1.0, 10)}, releases={"c": 20})


def test_paired_replay_agrees_before_release_and_reports_the_return_share():
    vd, model = fitted_scm(extra_runs={"test_0": faulty_run()})
    s = sc.recording_series(vd, "test_0")
    alarms, targets, identical = itv.replay_paired(model, s, {"r": 30}, {"r": 45}, ["a"], 200, seed=0)
    assert identical
    assert set(alarms.columns) >= {"p_fires_sustained", "p_fires_limited", "observed_fired"}
    row = targets.set_index("variable").loc["r"]
    # the share that returns comes from the fitted (fallback) mechanism, it is not forced to one
    assert row.released and 0.9 < row.p_returns_after_release <= 1.0
    # the default replay is unchanged by the new arguments
    plain = itv.replay_manipulation(model, s, {"r": 30}, ["a"], 200, np.random.default_rng(0))
    assert np.isclose(plain.p_fires.iloc[0], alarms.p_fires_sustained.iloc[0])
    # without a reversion the manipulation stays sustained and is reported as not released
    _, targets_none, identical_none = itv.replay_paired(model, s, {"r": 30}, {}, ["a"], 50, seed=0)
    assert identical_none and not targets_none.released.iloc[0]


def test_wilson_interval_known_values():
    low, high = cfx.wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12) and 0.25 < high < 0.35
    low, high = cfx.wilson_interval(1900, 2000)
    assert low < 0.95 < high and high - low < 0.025


def test_independent_validity_is_reproducible_and_separate_from_selection():
    vd, model = fitted_scm(extra_runs={"test_0": faulty_run()})
    s = sc.recording_series(vd, "test_0")
    itv.ACTIONABILITY["r"] = ("maintenance", 1.0, "synthetic root")
    try:
        actions = [itv.Action("r", 1.0, 30)]
        first = cfx.reevaluate_validity(model, s, actions, ["a"], 20, set(s), np.random.SeedSequence(7), 4, 50)
        again = cfx.reevaluate_validity(model, s, actions, ["a"], 20, set(s), np.random.SeedSequence(7), 4, 50)
        assert first == again and first["n_worlds"] == 200
        assert first["wilson_low"] <= first["validity_independent"] <= first["wilson_high"]
        assert first["validity_independent"] > 0.9
    finally:
        del itv.ACTIONABILITY["r"]
