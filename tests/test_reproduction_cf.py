"""the intervention and counterfactual stages reproduce the stored results and the printed thesis tables."""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import counterfactuals as cfx
from core import interventions as itv
from core import scm
from core.rca import analysis_window
from evaluation import counterfactuals as ev
from pipeline import counterfactuals as cf
from pipeline.interventions import replay_runs
from pipeline.scm_fitting import load_subsystem
from utils import io
from utils.config import load_config
from utils.errors import MissingInputError

EXPECTED = Path(__file__).parent / "expected"
THESIS = EXPECTED / "thesis"


def expected(name: str) -> pd.DataFrame:
    return pd.read_csv(EXPECTED / f"{name}.csv")


def thesis_table(stem: str) -> pd.DataFrame:
    return pd.read_csv(THESIS / f"{stem}.csv", dtype=str, keep_default_na=False)


def as_printed(table: pd.DataFrame) -> pd.DataFrame:
    """the table as it is written to csv and read back as text."""
    return pd.read_csv(StringIO(table.to_csv(index=False)), dtype=str, keep_default_na=False)


def assert_same(actual: pd.DataFrame, stored: pd.DataFrame, skip: tuple[str, ...] = ()) -> None:
    """same columns and rows; floats within 1e-9, everything else exactly."""
    assert list(actual.columns) == list(stored.columns)
    assert len(actual) == len(stored)
    for column in stored.columns:
        if column in skip:
            continue
        a, e = actual[column].reset_index(drop=True), stored[column].reset_index(drop=True)
        if pd.api.types.is_float_dtype(e):
            np.testing.assert_allclose(a.astype(float), e.astype(float), rtol=0, atol=1e-9, equal_nan=True,
                                       err_msg=column)
        else:
            assert a.fillna("").astype(str).tolist() == e.fillna("").astype(str).tolist(), column


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def coolant(cfg):
    return load_subsystem(cfg, cfg["counterfactuals"]["subsystem"])


@pytest.fixture(scope="module")
def rankings(coolant):
    table = io.read_input("rca_rankings")
    return table[table["subsystem"] == coolant.name]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_printed_numbers_round_half_up_like_the_tables():
    assert io.printed(0.945) == "0.95"
    assert io.printed(0.8355) == "0.84"
    assert io.printed(0.93346, 3) == "0.933"
    assert io.printed(85.05, 0) == "85"
    assert cf.printed_range(pd.Series([0.9965, 0.9995])) == "1.00"
    assert cf.printed_range(pd.Series([0.939, 0.9555])) == "0.94–0.96"


def test_onsets_are_rounded_up_to_the_next_grid_step():
    changes = pd.DataFrame({"recording_id": ["r", "r", "r"], "variable": ["a", "b", "c"],
                            "first_change_after_start_s": [10.0, 10.2, np.nan]})
    assert itv.onset_steps(changes, 0.5) == {("r", "a"): 20, ("r", "b"): 21}


def test_actions_parse_with_their_detection_steps():
    ranking = pd.DataFrame({"variable": ["CLF_Filter_Ok", "LP_Pump_Ok"], "detection_step": [195, 0]})
    actions = cf.parse_actions("do(CLF_Filter_Ok=1) + do(LP_Pump_Ok=1)", ranking)
    assert actions == [itv.Action("CLF_Filter_Ok", 1.0, 195), itv.Action("LP_Pump_Ok", 1.0, 1)]


def test_missing_recommendations_name_the_stage_to_run(monkeypatch, tmp_path):
    monkeypatch.setattr(io, "tables_dir", lambda part="pipeline": tmp_path)
    with pytest.raises(MissingInputError, match=r'run "python main.py run counterfactuals" first'):
        io.read_input("recommendation_refits")


# ---------------------------------------------------------------------------
# reproduction of the stored results
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_replay_reproduces_the_stored_replay(cfg, coolant):
    assert_same(replay_runs(cfg, coolant), expected("coolant_replay"))


def test_independent_worlds_give_the_stored_validity_of_the_example_run(cfg, coolant, rankings):
    graph, rid = "expert", cfg["figure_examples"]["coolant_run"]
    model = coolant.fold_model(graph, cfg["figure_examples"]["coolant_fold"])
    stored = expected("coolant_cf_runs").set_index(["graph", "recording_id"]).loc[(graph, rid)]
    actions = cf.parse_actions(stored["actions"], cf.run_ranking(rankings, graph, rid))
    series = scm.recording_series(coolant.view, rid)
    recorded = {v for v in series if not np.isnan(series[v]).all()}
    start = analysis_window(series, coolant.alarms, coolant.interval_s)[0]
    outcome = [a for a in coolant.alarms if a in recorded and model.mechanisms[a].has_data]
    result = cfx.reevaluate_validity(model, series, actions, outcome, start, recorded,
                                     cf.independent_streams(cfg, coolant)[(graph, rid)])
    world = cfx.independent_worlds(model, series, actions, cf.independent_streams(cfg, coolant)[(graph, rid)])
    alarm_free = np.max(np.stack([world[a][:, start:] for a in outcome]), axis=0).max(axis=1) == 0
    assert result["validity_independent"] == pytest.approx(stored["validity_independent"], abs=1e-12)
    assert alarm_free.mean() == pytest.approx(stored["validity_independent"], abs=1e-12)
    assert result["wilson_low"] == pytest.approx(stored["wilson_low"], abs=1e-12)


def test_recommendation_files_match_the_stored_results():
    assert_same(io.read_input("recommendations"), expected("coolant_cf_runs"))
    assert_same(io.read_input("recommendation_refits"), expected("coolant_cf_refits"))
    tried = io.read_input("tried_action_sets")
    for graph in ("expert", "discovered"):
        assert_same(tried[tried["graph"] == graph], expected(f"coolant_cf_tried_{graph}"))


def test_summaries_from_the_stored_results(cfg):
    graphs, threshold = cfg["graphs"], cfg["counterfactuals"]["validity_threshold"]
    replay = expected("coolant_replay")
    folds = sorted(replay["fold"].unique())
    assert_same(ev.replay_summary(replay, folds, graphs), expected("coolant_replay_summary"))
    runs, refits = expected("coolant_cf_runs"), expected("coolant_cf_refits")
    assert_same(ev.summary(runs, refits, graphs, threshold), expected("coolant_cf_summary"))
    picks = ev.pick_examples(runs[runs["graph"] == graphs[0]])
    assert_same(ev.examples_detail(picks), expected("coolant_cf_examples"), skip=("example",))


DETAIL_FILES = {"replay_summary": "coolant_replay_summary", "counterfactual_summary": "coolant_cf_summary",
                "settings_check": "coolant_exp22_settings", "counterfactual_examples": "coolant_cf_examples"}


@pytest.mark.parametrize("name", list(DETAIL_FILES))
def test_evaluation_detail_files_match_the_stored_results(name):
    file = io.tables_dir("evaluation") / f"{name}.csv"
    if not file.exists():
        pytest.skip('run "python main.py evaluate counterfactuals" first')
    assert_same(pd.read_csv(file), expected(DETAIL_FILES[name]), skip=("example",))


# ---------------------------------------------------------------------------
# thesis tables
# ---------------------------------------------------------------------------


def test_thesis_tables_from_the_stored_results(cfg, coolant, rankings):
    runs, refits = expected("coolant_cf_runs"), expected("coolant_cf_refits")
    threshold = cfg["counterfactuals"]["validity_threshold"]
    pd.testing.assert_frame_equal(as_printed(cf.recommendation_table(cfg, coolant, runs, rankings)),
                                  thesis_table("table_6_7_recommended_coolant_maintenance_actions"))
    pd.testing.assert_frame_equal(as_printed(cf.per_run_table(cfg, coolant, runs, rankings)),
                                  thesis_table("table_a_7_coolant_counterfactual_results_by_run"))
    pd.testing.assert_frame_equal(as_printed(ev.results_table(runs, refits, cfg["graphs"], threshold)),
                                  thesis_table("table_6_8_counterfactual_results_for_coolant"))
    pd.testing.assert_frame_equal(
        as_printed(ev.settings_table(cfg, coolant, expected("coolant_exp22_settings"), io.read_input("scm_settings"))),
        thesis_table("table_6_9_validity_under_different_scm_settings"))
    picks = ev.pick_examples(runs[runs["graph"] == cfg["graphs"][0]])
    rows = [row for i, (rule, r) in enumerate(picks, start=1) for row in ev.example_rows(cfg, i, rule, r)]
    pd.testing.assert_frame_equal(as_printed(pd.DataFrame(rows)),
                                  thesis_table("table_6_10_examples_of_counterfactual_explanations"))


SAVED_TABLES = {
    "table_6_7_recommended_coolant_maintenance_actions": "pipeline",
    "table_a_7_coolant_counterfactual_results_by_run": "pipeline",
    "table_6_8_counterfactual_results_for_coolant": "evaluation",
    "table_6_9_validity_under_different_scm_settings": "evaluation",
    "table_6_10_examples_of_counterfactual_explanations": "evaluation",
}


@pytest.mark.parametrize("stem", list(SAVED_TABLES))
def test_saved_thesis_tables_match_the_printed_tables(stem):
    file = io.tables_dir(SAVED_TABLES[stem]) / f"{stem}.csv"
    if not file.exists():
        pytest.skip("the command that writes this table has not run yet")
    pd.testing.assert_frame_equal(pd.read_csv(file, dtype=str, keep_default_na=False), thesis_table(stem))
