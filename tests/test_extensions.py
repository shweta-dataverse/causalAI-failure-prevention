"""tests of the wilson interval and the independent validity check."""

from __future__ import annotations

import numpy as np
import pytest

from core import counterfactuals as cfx
from core import interventions as itv
from core import scm as sc
from test_core import faulty_run, fitted_scm


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
