"""Tests for causal_rca.preprocessing on small synthetic event logs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_rca import preprocessing as pp

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "causRCA"


def make_info(recording_id: str = "rec_a", source: str = "dig_twin") -> pp.RecordingInfo:
    return pp.RecordingInfo(
        recording_id=recording_id,
        source=source,
        subsystem=None if source == "real_op" else "coolant",
        experiment_id=None if source == "real_op" else "exp_1",
        run_id=None if source == "real_op" else "run_1",
        source_file=f"{recording_id}.csv",
    )


def raw_frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame([[str(x) for x in row] for row in rows], columns=list(pp.EVENT_COLUMNS))


def parsed(rows: list[tuple], recording_id: str = "rec_a", source: str = "dig_twin") -> pd.DataFrame:
    return pp.parse_values(pp.prepare_events(raw_frame(rows), make_info(recording_id, source)))


def processed(*recordings: tuple[str, list[tuple]], source: str = "dig_twin") -> pd.DataFrame:
    events = pd.concat([parsed(rows, rid, source) for rid, rows in recordings], ignore_index=True)
    events, _ = pp.resolve_duplicates(events)
    return pp.mark_changes(events)


def column(states: pd.DataFrame, recording_id: str, node: str) -> list:
    values = states.loc[states["recording_id"] == recording_id, node]
    return [None if pd.isna(v) else v for v in values]


# --- parsing ---------------------------------------------------------------


def test_binary_strings_are_parsed_explicitly():
    events = parsed(
        [
            (0.0, "A", "True", "Binary"),
            (1.0, "A", "false", "Binary"),
            (2.0, "A", " TRUE ", "Binary"),
            (3.0, "A", "0", "Binary"),
            (4.0, "B", "1.0", "Alarm"),
        ]
    )
    assert events["value_num"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert events["value_cat"].isna().all()
    assert events["raw_value"].tolist()[0] == "True"


def test_invalid_binary_value_raises_unless_lenient():
    rows = [(0.0, "A", "True", "Binary"), (1.0, "A", "yes", "Binary")]
    with pytest.raises(pp.ValueParseError):
        parsed(rows)
    lenient = pp.parse_values(pp.prepare_events(raw_frame(rows), make_info()), strict_binary=False)
    assert np.isnan(lenient["value_num"].iloc[1])


def test_numeric_looking_categorical_stays_string():
    events = parsed(
        [(0.0, "Prog_LineNo", "1080", "Categorical"), (1.0, "Prog_LineNo", "007", "Categorical")]
    )
    assert events["value_cat"].tolist() == ["1080", "007"]
    assert events["value_num"].isna().all()
    states = pp.reconstruct_states(pp.mark_changes(events), 1.0)
    assert str(states["Prog_LineNo"].dtype) == "string"
    assert column(states, "rec_a", "Prog_LineNo") == ["1080", "007"]


def test_unparsable_continuous_value_raises():
    with pytest.raises(pp.ValueParseError):
        parsed([(0.0, "P", "12.5", "Continuous"), (1.0, "P", "abc", "Continuous")])


@pytest.mark.parametrize(
    "rows",
    [
        [(0.0, "A", "True", "Unknown")],
        [(-1.0, "A", "True", "Binary")],
        [("x", "A", "True", "Binary")],
    ],
)
def test_schema_violations_raise(rows):
    with pytest.raises(pp.SchemaError):
        pp.prepare_events(raw_frame(rows), make_info())


def test_wrong_columns_and_conflicting_types_raise():
    with pytest.raises(pp.SchemaError):
        pp.prepare_events(pd.DataFrame({"t": ["0"], "node": ["A"]}), make_info())
    with pytest.raises(pp.SchemaError):
        parsed([(0.0, "A", "True", "Binary"), (1.0, "A", "2.0", "Continuous")])


# --- duplicates and changes -------------------------------------------------


DUPLICATE_ROWS = [
    (0.0, "A", "False", "Binary"),
    (1.0, "A", "True", "Binary"),
    (1.0, "A", "False", "Binary"),
]


@pytest.mark.parametrize("policy, expected", [("last", 0.0), ("first", 1.0)])
def test_duplicate_policy_is_deterministic(policy, expected):
    events, log = pp.resolve_duplicates(parsed(DUPLICATE_ROWS), policy)
    assert len(events) == 2
    assert events.loc[events["time_s"] == 1.0, "value_num"].item() == expected
    assert len(log) == 2 and log["kept"].sum() == 1


def test_duplicate_policy_error_and_reconstruction_guard():
    with pytest.raises(pp.DuplicateEventError):
        pp.resolve_duplicates(parsed(DUPLICATE_ROWS), "error")
    with pytest.raises(pp.DuplicateEventError):
        pp.reconstruct_states(parsed(DUPLICATE_ROWS), 1.0)


def test_mark_changes_separates_initial_change_and_repeat():
    events = processed(
        ("rec_a", [(0.0, "A", "False", "Binary"), (1.0, "A", "False", "Binary"), (2.0, "A", "True", "Binary")])
    )
    assert events["is_initial"].tolist() == [True, False, False]
    assert events["is_repeat"].tolist() == [False, True, False]
    assert events["is_change"].tolist() == [False, False, True]


# --- state reconstruction ----------------------------------------------------


def test_forward_fill_between_events():
    events = processed(("rec_a", [(0.0, "A", "True", "Binary"), (2.5, "A", "False", "Binary")]))
    states = pp.reconstruct_states(events, 1.0)
    assert states["t_s"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert column(states, "rec_a", "A") == [1, 1, 1, 0]
    assert str(states["A"].dtype) == "Int8"


def test_unknown_initial_state_is_not_backfilled():
    events = processed(
        ("rec_a", [(0.0, "A", "True", "Binary"), (1.5, "P", "3.5", "Continuous"), (3.0, "A", "False", "Binary")])
    )
    states = pp.reconstruct_states(events, 1.0)
    assert column(states, "rec_a", "P") == [None, None, 3.5, 3.5]


def test_recording_boundaries_are_preserved():
    events = processed(
        ("rec_a", [(0.0, "X", "True", "Binary"), (1.0, "Y", "5.0", "Continuous")]),
        ("rec_b", [(0.0, "Y", "1.0", "Continuous"), (2.0, "X", "False", "Binary")]),
    )
    states = pp.reconstruct_states(events, 1.0)
    # rec_b must not inherit rec_a's X = True before its own first X event.
    assert column(states, "rec_b", "X") == [None, None, 0]
    assert column(states, "rec_a", "Y") == [None, 5.0]
    assert states.groupby("recording_id")["step"].min().tolist() == [0, 0]
    assert set(states.loc[states["recording_id"] == "rec_b", "experiment_id"]) == {"exp_1"}


def test_events_on_grid_points_are_included_at_that_time():
    events = processed(
        ("rec_a", [(0.0, "A", "False", "Binary"), (0.3, "A", "True", "Binary"), (0.31, "B", "True", "Binary")])
    )
    states = pp.reconstruct_states(events, 0.1)
    assert column(states, "rec_a", "A") == [0, 0, 0, 1, 1]
    # B's first event at 0.31 lands on the next grid point, 0.4.
    assert column(states, "rec_a", "B") == [None, None, None, None, 1]


def test_constant_variables_are_kept_in_archival_table():
    events = processed(("rec_a", [(0.0, "C", "True", "Binary"), (5.0, "A", "True", "Binary")]))
    states = pp.reconstruct_states(events, 1.0)
    assert column(states, "rec_a", "C") == [1] * 6


def test_short_pulse_is_hidden_by_coarse_grid_but_counted():
    events = processed(
        ("rec_a", [(0.0, "A", "False", "Binary"), (1.2, "A", "True", "Binary"), (1.6, "A", "False", "Binary")])
    )
    coarse = pp.reconstruct_states(events, 1.0)
    assert 1 not in column(coarse, "rec_a", "A")
    bins = pp.transitions_per_bin(events, 1.0)
    assert bins[["step", "node", "n_changes"]].values.tolist() == [[2, "A", 2]]

    sensitivity = pp.sampling_sensitivity(events, [0.1, 1.0]).set_index(["interval_s", "variable_type"])
    assert sensitivity.loc[(1.0, "Binary"), "n_true_changes"] == 2
    assert sensitivity.loc[(1.0, "Binary"), "n_visible_transitions"] == 0
    assert sensitivity.loc[(1.0, "Binary"), "n_bins_with_masked_change"] == 1
    assert sensitivity.loc[(0.1, "Binary"), "transition_retention"] == 1.0


def test_multi_change_fraction_flags_simultaneous_changes():
    events = processed(
        (
            "rec_a",
            [
                (0.0, "A", "False", "Binary"),
                (0.0, "B", "False", "Binary"),
                (1.0, "A", "True", "Binary"),
                (1.0, "B", "True", "Binary"),
                (3.0, "A", "False", "Binary"),
            ],
        )
    )
    row = pp.sampling_sensitivity(events, [1.0]).query("variable_type == 'ALL'").iloc[0]
    assert row["multi_change_step_fraction"] == pytest.approx(0.5)


# --- verification and diagnostics -------------------------------------------


def test_verification_passes_and_detects_corruption():
    events = processed(
        ("rec_a", [(0.0, "A", "True", "Binary"), (1.5, "L", "12", "Categorical"), (2.2, "A", "False", "Binary")]),
        ("rec_b", [(0.0, "P", "1.5", "Continuous"), (0.7, "P", "2.5", "Continuous")]),
    )
    states = pp.reconstruct_states(events, 0.5)
    result = pp.verify_states(events, states, 0.5)
    assert result.passed and result.n_cells_checked == len(states) * 3

    corrupted = states.copy()
    corrupted.loc[corrupted.index[1], "A"] = 0
    corrupted.loc[corrupted.index[0], "L"] = "12"  # value before L's first event
    bad = pp.verify_states(events, corrupted, 0.5)
    assert not bad.passed and bad.n_mismatches == 2

    truncated = states.iloc[:-1]
    assert pp.verify_states(events, truncated, 0.5).n_grid_errors == 1


def test_event_statistics():
    events = processed(
        (
            "rec_a",
            [
                (0.0, "A", "True", "Binary"),
                (1.0, "A", "True", "Binary"),
                (4.0, "A", "False", "Binary"),
                (2.0, "K", "10", "Counter"),
                (3.0, "K", "2", "Counter"),
            ],
        )
    )
    stats = pp.event_statistics(events).set_index("node")
    assert stats.loc["A", ["n_events", "n_changes", "n_repeats"]].tolist() == [3, 1, 1]
    assert stats.loc["A", "min_gap_s"] == 1.0 and stats.loc["A", "max_gap_s"] == 3.0
    assert bool(stats.loc["A", "has_t0_snapshot"]) and not bool(stats.loc["K", "has_t0_snapshot"])
    assert stats.loc["K", "n_counter_resets"] == 1


# --- model views -------------------------------------------------------------


def view_fixture() -> tuple[pd.DataFrame, dict[str, str]]:
    def rows(level: str, pressure: tuple[str, str], counter: tuple[str, str], const: str):
        return [
            (0.0, "Line", level, "Categorical"),
            (0.0, "P", pressure[0], "Continuous"),
            (2.0, "P", pressure[1], "Continuous"),
            (0.0, "K", counter[0], "Counter"),
            (1.0, "K", counter[1], "Counter"),
            (0.0, "Const", const, "Binary"),
            (1.0, "Alarm", "True", "Alarm"),
            (3.0, "Alarm", "False", "Alarm"),
        ]

    events = processed(
        ("train_1", rows("10", ("1.0", "3.0"), ("100", "101"), "True")),
        ("train_2", rows("20", ("2.0", "6.0"), ("500", "502"), "True")),
        ("test_1", rows("99", ("100.0", "100.0"), ("7", "8"), "False")),
    )
    return pp.reconstruct_states(events, 1.0), pp.node_types(events)


def test_discrete_view_uses_training_vocabulary_and_excludes_constants():
    states, types = view_fixture()
    view = pp.ModelView.fit(states, types, ["train_1", "train_2"], "discrete")
    assert view.columns == ["Alarm", "Line"]
    assert view.vocabularies == {"Line": ["10", "20"]}
    reasons = {e["column"]: e["reason"] for e in view.exclusions}
    assert reasons == {
        "Const": "constant_in_training",
        "K": "numeric_type_not_in_discrete_view",
        "P": "numeric_type_not_in_discrete_view",
    }
    data = view.transform(states, ["test_1"])
    line = data.arrays["test_1"][:, data.var_names.index("Line")]
    assert set(line) == {2.0}  # unseen category gets the reserved code len(vocabulary)
    assert data.data_type.tolist() == [1, 1]


def test_continuous_view_standardizes_with_training_statistics_only():
    states, types = view_fixture()
    view = pp.ModelView.fit(states, types, ["train_1", "train_2"], "continuous")
    assert view.columns == ["P"]
    assert {e["column"]: e["reason"] for e in view.exclusions}["K"] == "cumulative_counter_nonstationary"
    train_p = states.loc[states["recording_id"].isin(["train_1", "train_2"]), "P"].astype(float)
    assert view.means["P"] == pytest.approx(train_p.mean())
    assert view.stds["P"] == pytest.approx(train_p.std(ddof=0))
    test = view.transform(states, ["test_1"]).arrays["test_1"][:, 0]
    assert test == pytest.approx((100.0 - train_p.mean()) / train_p.std(ddof=0))


def test_counter_differences_stay_within_recordings_and_mask_unknown_states():
    states, types = view_fixture()
    view = pp.ModelView.fit(states, types, ["train_1", "train_2"], "mixed", counter_first_differences=True)
    assert "K__diff" in view.columns
    data = view.transform(states, ["train_2"])
    k = view.columns.index("K__diff")
    alarm = view.columns.index("Alarm")
    # first row has no previous value inside the recording -> masked, not
    # differenced against train_1's last value.
    assert data.masks["train_2"][:, k].tolist() == [True, False, False, False]
    assert data.masks["train_2"][:, alarm].tolist() == [True, False, False, False]
    assert data.data_type[view.columns.index("P")] == 0


def test_view_rejects_unknown_kind_and_missing_training_ids():
    states, types = view_fixture()
    with pytest.raises(ValueError):
        pp.ModelView.fit(states, types, ["train_1"], "parcorr")
    with pytest.raises(KeyError):
        pp.ModelView.fit(states, types, ["nope"], "discrete")


# --- discovery, config and pipeline -----------------------------------------


def write_raw_tree(root: Path) -> Path:
    raw = root / "raw"
    (raw / "real_op").mkdir(parents=True)
    raw_frame([(0.0, "A", "True", "Binary"), (1.5, "A", "False", "Binary"), (0.0, "L", "5", "Categorical")]).to_csv(
        raw / "real_op" / "dataset_real_op_42.csv", index=False
    )
    run = raw / "dig_twin" / "exp_probe" / "exp_2_3" / "run_1"
    run.mkdir(parents=True)
    raw_frame([(0.0, "A", "False", "Binary"), (2.0, "A", "True", "Binary")]).to_csv(
        run / "faultDataset_probe_exp2_run_1.csv", index=False
    )
    (run / "causes.json").write_text("{}")
    return raw


def test_discover_recordings_uses_folder_ids(tmp_path):
    infos = pp.discover_recordings(write_raw_tree(tmp_path))
    assert [i.recording_id for i in infos] == ["real_op_42", "probe_exp_2_3_run_1"]
    fault = infos[1]
    assert (fault.subsystem, fault.experiment_id, fault.run_id) == ("probe", "exp_2_3", "run_1")
    assert "exp2" in fault.naming_note
    assert infos[0].naming_note is None


def test_discover_recordings_rejects_unexpected_names(tmp_path):
    raw = write_raw_tree(tmp_path)
    (raw / "real_op" / "notes.csv").write_text("x")
    with pytest.raises(pp.SchemaError):
        pp.discover_recordings(raw)


def test_config_loads_repository_yaml_and_rejects_unknown_keys(tmp_path):
    config = pp.PreprocessingConfig.from_yaml(REPO_ROOT / "configs" / "preprocessing.yaml")
    assert config.duplicate_policy == "last" and config.sampling_interval_s == 1.0
    bad = tmp_path / "bad.yaml"
    bad.write_text("sampling_interval: 1.0\n")
    with pytest.raises(ValueError):
        pp.PreprocessingConfig.from_yaml(bad)
    with pytest.raises(ValueError):
        pp.PreprocessingConfig(sampling_interval_s=0)


def test_run_preprocessing_writes_outputs(tmp_path):
    config = pp.PreprocessingConfig(
        raw_dir=write_raw_tree(tmp_path),
        output_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
        sensitivity_intervals_s=(0.5, 1.0),
    )
    report = pp.run_preprocessing(config)
    assert report["verification"]["passed"]
    states = pd.read_parquet(config.output_dir / "states_dt1.parquet")
    assert set(states["recording_id"]) == {"real_op_42", "probe_exp_2_3_run_1"}
    assert column(states, "probe_exp_2_3_run_1", "L") == [None, None, None]
    for name in ["data_quality.json", "data_quality.md", "sampling_sensitivity.csv", "view_exclusions_real_op.csv"]:
        assert (config.report_dir / name).exists()
    saved = json.loads((config.report_dir / "data_quality.json").read_text())
    assert saved["numeric_looking_categoricals"] == ["L"]


# --- real data ---------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not RAW_DIR.is_dir(), reason="raw causRCA data not available")
def test_real_recordings_reconstruct_and_verify():
    infos = pp.discover_recordings(RAW_DIR)
    assert sum(i.source == "real_op" for i in infos) == 170
    assert sum(i.source == "dig_twin" for i in infos) == 100
    sample = [next(i for i in infos if i.source == "real_op"), next(i for i in infos if i.subsystem == "coolant")]
    events = pp.parse_values(pp.load_all_events(sample, RAW_DIR))
    events, _ = pp.resolve_duplicates(events)
    events = pp.mark_changes(events)
    states = pp.reconstruct_states(events, 1.0)
    assert pp.verify_states(events, states, 1.0).passed
    assert str(states["Prog_LineNo"].dtype) == "string"
