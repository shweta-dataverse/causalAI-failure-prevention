"""the appendix comparisons reproduce the stored results of the original configuration and of the probe pipeline
without the near-duplicate exclusion, and the printed tables a.3 and a.5."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import comparisons
from evaluation.rca import lstm_choice
from utils import io
from utils.config import load_config

EXPECTED = Path(__file__).parent / "expected"
STORED = EXPECTED / "comparisons"
THESIS = EXPECTED / "thesis"
TABLE_A_3 = "table_a_3_results_with_original_and_selected_settings"
TABLE_A_5 = "table_a_5_results_before_and_after_near_duplicate_exclusion"


def stored(name: str, **read_csv_args) -> pd.DataFrame:
    return pd.read_csv(STORED / f"{name}.csv", **read_csv_args)


def variant_table(variant: str, name: str) -> pd.DataFrame:
    """a table of an alternative run; skipped when "python main.py evaluate comparisons" has not run."""
    file = comparisons.variant_dir(variant) / f"{name}.csv"
    if not file.exists():
        pytest.skip(f"{io.relative(file)} not written yet")
    return pd.read_csv(file)


def thesis_table(stem: str) -> pd.DataFrame:
    return pd.read_csv(THESIS / f"{stem}.csv", dtype=str, keep_default_na=False)


def as_printed(table: pd.DataFrame) -> pd.DataFrame:
    return comparisons.printed_columns(table).astype(str).reset_index(drop=True)


def assert_same(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """the expected columns and rows; integers and strings exact, floats within 1e-9."""
    assert len(actual) == len(expected)
    for column in expected.columns:
        a, e = actual[column], expected[column]
        if e.dtype.kind == "f" or a.dtype.kind == "f":
            a, e = a.astype(float).to_numpy(), e.astype(float).to_numpy()
            assert (np.isnan(a) == np.isnan(e)).all(), column
            assert np.nanmax(np.abs(a - e), initial=0.0) < 1e-9, column
        else:
            assert a.astype(str).tolist() == e.astype(str).tolist(), column


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ---------------------------------------------------------------------------
# tables from the stored results (no pipeline run needed)
# ---------------------------------------------------------------------------


def selected_tables(cfg: dict) -> dict:
    tables = {"discovery_settings": pd.read_csv(EXPECTED / "discovery_settings.csv", keep_default_na=False),
              "scm_graphs": pd.read_csv(EXPECTED / "scm_graphs.csv"),
              "scm_prediction": pd.read_csv(EXPECTED / "scm_prediction.csv"),
              "rca_runs": pd.read_csv(EXPECTED / "rca_runs.csv"),
              "replay": pd.read_csv(EXPECTED / "coolant_replay.csv"),
              "recommendations": pd.read_csv(EXPECTED / "coolant_cf_runs.csv"),
              "recommendation_refits": pd.read_csv(EXPECTED / "coolant_cf_refits.csv")}
    return comparisons.with_rca_summary(cfg, tables)


def original_tables(cfg: dict) -> dict:
    tables = {name: stored(f"original_{name}") for name in
              ("scm_prediction", "rca_runs", "replay", "recommendations", "recommendation_refits")}
    # the discovered graph of all scenarios: directed f1 of the raw graph and the edges of its scm dag
    agreement = stored("original_graph_agreement")
    raw = agreement[agreement.stage == "raw"]
    tables["discovery_settings"] = raw.assign(scope="all")[["subsystem", "scope", "directed_f1"]]
    dag = agreement[agreement.stage == "scm dag"]
    tables["scm_graphs"] = pd.DataFrame([{"subsystem": r.subsystem, "graph": "discovered", "scope": "all"}
                                         for r in dag.itertuples() for _ in range(r.n_pred_edges)])
    return comparisons.with_rca_summary(cfg, tables)


def lstm(runs: pd.DataFrame) -> dict:
    return {"runs": runs, "chosen": lstm_choice(pd.read_csv(EXPECTED / "lstm_shap_tuning.csv"))}


def test_table_a_3_from_the_stored_results(cfg):
    table = comparisons.original_table(cfg, original_tables(cfg), selected_tables(cfg))
    pd.testing.assert_frame_equal(as_printed(table), thesis_table(TABLE_A_3))


def test_table_a_5_from_the_stored_results(cfg):
    without = comparisons.with_rca_summary(cfg, {"rca_runs": stored("without_exclusion_rca_runs"),
                                                 "scm_prediction": stored("without_exclusion_scm_prediction")})
    main = selected_tables(cfg)
    before = comparisons.protocol_values(cfg, without, lstm(stored("without_exclusion_lstm_shap_runs")), "probe")
    after = comparisons.protocol_values(cfg, main, lstm(pd.read_csv(EXPECTED / "lstm_shap_runs.csv")), "probe")
    settings = stored("without_exclusion_discovery_settings")
    changed = comparisons.changed_settings(settings, main["discovery_settings"], "probe")
    table = comparisons.exclusion_table(cfg, before, after, changed)
    pd.testing.assert_frame_equal(as_printed(table), thesis_table(TABLE_A_5))


def test_directions_follow_the_numbers():
    assert comparisons.direction("3.04/3.03", "2.98/2.88") == "lower"
    assert comparisons.direction("0.48/0.56", "0.67/0.63") == "raise"
    assert comparisons.direction("1.00", "1.00") == "keep"
    assert comparisons.direction("0.44/0.24", "0.40/0.53") == "change"


# ---------------------------------------------------------------------------
# alternative runs of "python main.py evaluate comparisons"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rca_runs", "scm_prediction", "replay", "recommendations", "recommendation_refits"])
def test_original_configuration_reproduces_the_stored_results(name):
    assert_same(variant_table("original", name), stored(f"original_{name}"))


def test_original_discovered_graphs_have_the_stored_agreement():
    settings = variant_table("original", "discovery_settings")
    edges = variant_table("original", "scm_graphs")
    for r in stored("original_graph_agreement").itertuples():
        if r.stage == "raw":
            row = settings[(settings.subsystem == r.subsystem) & (settings.scope == "all")]
            assert abs(row.directed_f1.iloc[0] - r.directed_f1) < 1e-9
        else:
            dag = edges[(edges.subsystem == r.subsystem) & (edges.graph == "discovered") & (edges.scope == "all")]
            assert len(dag) == r.n_pred_edges


@pytest.mark.parametrize("name", ["rca_runs", "scm_prediction"])
def test_probe_without_exclusion_reproduces_the_stored_results(name):
    assert_same(variant_table("without_exclusion", name), stored(f"without_exclusion_{name}"))


@pytest.mark.parametrize("name", ["discovery_settings", "scm_settings"])
def test_probe_without_exclusion_selects_the_stored_settings(name):
    actual = variant_table("without_exclusion", name)
    expected = stored(f"without_exclusion_{name}")
    keys = ["subsystem", "graph", "scope"] if name == "scm_settings" else ["subsystem", "scope"]
    actual = actual.set_index(keys).loc[expected.set_index(keys).index].reset_index()
    assert_same(actual, expected)


@pytest.mark.parametrize("stem", [TABLE_A_3, TABLE_A_5])
def test_saved_tables_match_the_printed_tables(stem):
    file = io.tables_dir("evaluation") / f"{stem}.csv"
    if not file.exists():
        pytest.skip("python main.py evaluate comparisons has not run yet")
    pd.testing.assert_frame_equal(pd.read_csv(file, dtype=str, keep_default_na=False), thesis_table(stem))
