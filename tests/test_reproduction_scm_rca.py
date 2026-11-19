"""the scm and rca stages and their evaluation reproduce the stored results and the printed tables."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import rca as rca_evaluation
from evaluation import scm as scm_evaluation
from pipeline import root_cause_analysis, scm_fitting
from utils import io
from utils.config import load_config

EXPECTED = Path(__file__).parent / "expected"
THESIS = EXPECTED / "thesis"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def written(part: str, name: str, **read_csv_args) -> pd.DataFrame:
    """a table written by a command; skipped when the command has not been run."""
    file = io.tables_dir(part) / f"{name}.csv"
    if not file.exists():
        pytest.skip(f"{io.relative(file)} not written yet")
    return pd.read_csv(file, **read_csv_args)


def numbered(part: str, prefix: str) -> pd.DataFrame:
    files = sorted(io.tables_dir(part).glob(f"{prefix}_*.csv"))
    if not files:
        pytest.skip(f"{prefix} not written yet")
    return pd.read_csv(files[0], dtype=str, keep_default_na=False)


def assert_same(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """same columns and rows; integers and strings exact, floats within 1e-9."""
    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected)
    for column in expected.columns:
        a, e = actual[column], expected[column]
        if e.dtype.kind == "f" or a.dtype.kind == "f":
            a, e = a.astype(float).to_numpy(), e.astype(float).to_numpy()
            assert (np.isnan(a) == np.isnan(e)).all(), column
            assert np.nanmax(np.abs(a - e), initial=0.0) < 1e-9, column
        else:
            assert a.astype(str).tolist() == e.astype(str).tolist(), column


def thesis_rows(name: str) -> list[list[str]]:
    with open(THESIS / f"{name}.csv", newline="", encoding="utf-8") as f:
        return [row for row in csv.reader(f)]


def fill_down(rows: list[list[str]]) -> list[list[str]]:
    """repeat the first cell where the printed table leaves it empty."""
    out, last = [], ""
    for row in rows:
        last = row[0] or last
        out.append([last, *row[1:]])
    return out


def cells(table: pd.DataFrame) -> list[list[str]]:
    return [[c for c in row if c != ""] for row in table.astype(str).values.tolist()]


def printed(value: str, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


# ---------------------------------------------------------------------------
# selection rules (no data needed)
# ---------------------------------------------------------------------------


def test_ties_go_to_the_baseline_then_no_lag_then_the_closest_smoothing():
    base = (0.5, None)
    tied = {(0.01, 2): 1.0, (0.5, None): 1.0, (1.0, None): 1.0}
    assert scm_fitting.best_setting(tied, base) == (0.5, None)
    tied = {(0.01, 2): 1.0, (0.1, None): 1.0, (1.0, 2): 1.0}
    assert scm_fitting.best_setting(tied, base) == (0.1, None)
    assert scm_fitting.best_setting({(0.1, None): 1.0, (1.0, None): 1.0}, base) == (1.0, None)
    assert scm_fitting.best_setting({(0.01, 4): 0.9, (0.5, None): 1.0}, base) == (0.01, 4)


def test_values_shared_by_both_subsystems_are_printed_once():
    assert scm_fitting.per_subsystem({"coolant": "0.01", "probe": "0.01"}) == "0.01"
    assert scm_fitting.per_subsystem({"coolant": "5", "probe": "20"}) == "5 / 20"


def test_aggregation_weights_recordings_and_scenarios_equally():
    pred = pd.DataFrame({"recording_id": ["a", "a", "a", "b"], "variable": ["x", "x", "y", "x"],
                         "log_loss": [1.0, 3.0, 4.0, 10.0], "brier": [0.0] * 4, "change": [True, False, False, True],
                         "configuration_seen": [True] * 4, "mechanism_has_data": [True] * 4})
    # recording a: (mean(1, 3) + 4) / 2 = 3, recording b: 10
    assert scm_evaluation.macro(pred)["log_loss"] == pytest.approx(6.5)
    # scenario s1 = {a}, s2 = {b}
    assert scm_evaluation.macro(pred, {"a": "s1", "b": "s2"})["log_loss"] == pytest.approx(6.5)
    assert scm_evaluation.macro(pred, {"a": "s1", "b": "s1"})["change_log_loss"] == pytest.approx(5.5)


def test_first_change_ignores_changes_before_the_annotated_start():
    series = {"v": np.array([0, 1, 0, 0, 1, 1], dtype=float)}
    assert root_cause_analysis.first_change(series, {"v"}, 1.2, 0.5) == 2.0
    assert np.isnan(root_cause_analysis.first_change(series, {"v"}, 9.0, 0.5))


def test_a_graph_that_ranks_the_same_with_both_settings_is_printed_once():
    parts = pd.DataFrame([
        {"graph": "discovered", "discovered_graph": "original", "scm_settings": s, "top1_run_mean": 0.24,
         "map3_run_mean": 0.56, "ap3_scenario_macro": 0.61, "distinct_rankings": 4} for s in ("default", "selected")])
    table = rca_evaluation.decomposition_table(load_config(), parts)
    assert table["SCM settings"].tolist() == ["Default"]


# ---------------------------------------------------------------------------
# scm stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["scm_graphs", "dag_removals", "scm_settings", "scm_selection_inner"])
def test_scm_stage_reproduces_the_contract_files(name):
    assert_same(written("pipeline", name), pd.read_csv(EXPECTED / f"{name}.csv"))


def test_table_5_7_matches_the_thesis():
    expected = thesis_rows("table_5_7")
    actual = cells(numbered("pipeline", "table_5_7"))
    assert len(actual) == len(expected)
    for mine, thesis in zip(actual, expected):
        # symbols are typeset in latex, so only the text columns are compared
        assert (mine[1:] if len(thesis) == 5 else mine) == (thesis[1:] if len(thesis) == 5 else thesis)


def test_table_a_2_matches_the_thesis():
    assert cells(numbered("pipeline", "table_a_2")) == fill_down(thesis_rows("table_a_2"))


def test_every_selected_scm_is_saved():
    from core import scm
    settings = written("pipeline", "scm_settings")
    for row in settings.itertuples():
        assert scm.fitted_path(row.subsystem, row.graph, row.scope).exists()


# ---------------------------------------------------------------------------
# rca stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rca_runs", "rca_rankings"])
def test_rca_stage_reproduces_the_contract_files(name):
    assert_same(written("pipeline", name), pd.read_csv(EXPECTED / f"{name}.csv"))


def test_table_a_6_matches_the_thesis():
    assert cells(numbered("pipeline", "table_a_6")) == thesis_rows("table_a_6")


# ---------------------------------------------------------------------------
# scm evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["scm_prediction", "scm_prediction_per_variable", "scm_refit_stability",
                                  "scm_normal_diagnostics", "scm_objective_inner", "scm_objective_selection",
                                  "scm_objective_outer"])
def test_scm_evaluation_reproduces_the_stored_results(name):
    assert_same(written("evaluation", name), pd.read_csv(EXPECTED / f"{name}.csv"))


def test_table_6_3_matches_the_thesis():
    actual = [[s, d, *(printed(v, 2 if i % 3 == 2 else 3) for i, v in enumerate(values))]
              for s, d, *values in cells(numbered("evaluation", "table_6_3"))]
    assert actual == fill_down(thesis_rows("table_6_3"))


def test_table_a_8_matches_the_thesis():
    expected, section = [], ""
    for row in thesis_rows("table_a_8"):
        if len(row) == 1:
            section = row[0]
        else:
            expected.append([section, *row])
    assert cells(numbered("evaluation", "table_a_8")) == expected


def test_table_a_9_matches_the_thesis():
    actual = numbered("evaluation", "table_a_9").astype(str).values.tolist()
    actual = [[scenario, runs, *(printed(v, 2) for v in values)] for scenario, runs, *values in actual]
    assert actual == thesis_rows("table_a_9")


# ---------------------------------------------------------------------------
# rca evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rca_per_scenario", "rca_summary", "rca_decomposition"])
def test_rca_evaluation_reproduces_the_stored_results(name):
    assert_same(written("evaluation", name), pd.read_csv(EXPECTED / f"{name}.csv"))


@pytest.mark.parametrize("name", ["lstm_shap_tuning", "lstm_shap_summary", "lstm_shap_per_scenario",
                                  "lstm_shap_sensitivity", "lstm_shap_runs", "lstm_shap_importance"])
def test_lstm_shap_baseline_reproduces_the_stored_results(name):
    assert_same(written("evaluation", name), pd.read_csv(EXPECTED / f"{name}.csv"))


def test_table_5_8_matches_the_thesis():
    actual = cells(numbered("evaluation", "table_5_8"))
    assert actual == thesis_rows("table_5_8")


def test_table_6_4_matches_the_thesis():
    actual = [[s, m, *(printed(v, 2) for v in values)] for s, m, *values in cells(numbered("evaluation", "table_6_4"))]
    assert actual == fill_down(thesis_rows("table_6_4"))


def test_table_6_5_matches_the_thesis():
    actual = [[g, s, printed(t, 2), printed(m, 2), printed(a, 2), o]
              for g, s, t, m, a, o in cells(numbered("evaluation", "table_6_5"))]
    assert actual == fill_down(thesis_rows("table_6_5"))


def test_table_6_6_matches_the_thesis():
    actual = [[steps, *(printed(v, 2) for v in values)] for steps, *values in cells(numbered("evaluation", "table_6_6"))]
    assert actual == thesis_rows("table_6_6")
