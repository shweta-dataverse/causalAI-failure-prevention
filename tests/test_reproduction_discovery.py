"""preprocessing, causal discovery and its evaluation reproduce the stored contract files, the state
views and the numbers of the thesis tables (expected values copied from the LaTeX tables)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import discovery
from core import preprocessing as pp
from utils import io
from utils.config import load_config

EXPECTED = Path(__file__).parent / "expected"
THESIS = EXPECTED / "thesis"
PIPELINE_TABLES = [
    "table_5_1_analysed_subsystems_of_the_causrca_dataset",
    "table_5_2_fault_scenarios_and_injected_causes",
    "table_5_3_coolant_variables_and_their_roles",
    "table_5_4_raw_coolant_event_log",
    "table_5_5_reconstructed_coolant_states",
    "table_5_6_coolant_discovery_regimes",
    "table_a_4_near_duplicate_probe_recordings",
]
EVALUATION_TABLES = [
    "table_6_1_agreement_with_the_expert_graphs",
    "table_a_1_original_and_selected_discovery_settings",
]
GRAPH_FIGURES = [
    "figure_6_1_expert_and_discovered_coolant_graphs",
    "figure_6_2_expert_and_discovered_probe_graphs",
]
needs_raw = pytest.mark.skipif(not io.raw_dir().is_dir(), reason="needs the raw causRCA dataset in data/raw/causRCA")
needs_views = pytest.mark.skipif(not (io.processed_dir() / "coolant_dt0.5.npz").exists(),
                                 reason="needs the state views (python main.py run preprocessing)")
needs_cache = pytest.mark.skipif(not io.cache_dir("discovery_study").is_dir(),
                                 reason="needs the pcmci+ run cache in data/cache/discovery_study")


def read_text_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def assert_same_contract(name: str) -> None:
    """strings and integers exactly, floats to 1e-9 (the stored files print floats to 16 digits)."""
    produced = pd.read_csv(io.contract_path(name), keep_default_na=False, float_precision="round_trip")
    expected = pd.read_csv(EXPECTED / f"{name}.csv", keep_default_na=False, float_precision="round_trip")
    assert list(produced.columns) == list(expected.columns)
    assert len(produced) == len(expected)
    for column in expected.columns:
        left, right = produced[column], expected[column]
        if "f" in (left.dtype.kind, right.dtype.kind):
            left, right = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
            assert (left.isna() == right.isna()).all(), column
            assert np.allclose(left.fillna(0), right.fillna(0), rtol=0, atol=1e-9), column
        else:
            assert (left.astype(str) == right.astype(str)).all(), column


def numbers(text: str) -> list[float]:
    """the numbers written in a sentence (identifiers such as LT_A_700317 are not numbers)."""
    found = []
    for token in re.split(r"[\s,;:()]+", text):
        token = token.rstrip(".")
        if re.fullmatch(r"-?\d+(\.\d+)?", token):
            found.append(float(token))
    return found


def view_fingerprint(directory: Path, name: str) -> str:
    view, spec = pp.load_view_data(directory, name)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    digest.update(json.dumps(view.var_names).encode())
    digest.update(np.asarray(view.data_type, dtype=np.int64).tobytes())
    for recording_id in view.arrays:
        digest.update(recording_id.encode())
        digest.update(np.ascontiguousarray(view.arrays[recording_id], dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(view.masks[recording_id], dtype=bool).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# contract files and state views
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["splits", "discovered_graphs", "discovery_settings"])
def test_contract_file_equals_the_stored_results(name):
    assert_same_contract(name)


def test_every_scope_of_both_subsystems_has_a_selected_graph():
    settings = io.read_input("discovery_settings", keep_default_na=False)
    edges = io.read_input("discovered_graphs")
    assert settings.groupby("subsystem").size().to_dict() == {"coolant": 17, "probe": 37}
    assert set(zip(edges.subsystem, edges.scope)) == set(zip(settings.subsystem, settings.scope))


@needs_views
@pytest.mark.parametrize("name", ["coolant_dt0.5", "probe_dt0.5", "coolant_dt1"])
def test_stored_state_view_equals_the_view_of_the_thesis(name):
    expected = json.loads((EXPECTED / "view_fingerprints.json").read_text())
    assert view_fingerprint(io.processed_dir(), name) == expected[name]


@pytest.mark.integration
@needs_raw
def test_splits_rebuilt_from_the_raw_data_equal_the_stored_splits():
    from pipeline.preprocessing import build_splits

    recordings = pd.DataFrame([asdict(i) for i in pp.discover_recordings(io.raw_dir())])
    rebuilt = build_splits(load_config(), recordings)
    assert rebuilt.to_csv(index=False) == (EXPECTED / "splits.csv").read_text()


@pytest.mark.integration
@needs_raw
def test_coolant_views_rebuilt_from_the_raw_events_are_identical(tmp_path):
    from core import data
    from pipeline.preprocessing import build_splits, build_view, subsystem_events

    infos = pp.discover_recordings(io.raw_dir())
    recordings = pd.DataFrame([asdict(i) for i in infos])
    runs = data.load_fault_runs()
    splits = build_splits(load_config(), recordings)
    events = subsystem_events(infos, runs, "coolant")
    ids = [i.recording_id for i in infos if i.source == "real_op"] + runs.loc[runs.subsystem == "coolant", "recording_id"].tolist()
    expected = json.loads((EXPECTED / "view_fingerprints.json").read_text())
    for spec in load_config()["state_views"]:
        if spec["subsystem"] != "coolant":
            continue
        build_view(spec, events, ids, splits[splits.subsystem == "coolant"], tmp_path)
        name = f"coolant_dt{spec['interval_s']:g}"
        assert view_fingerprint(tmp_path, name) == expected[name]


@needs_views
@needs_cache
@pytest.mark.parametrize("subsystem, scope", [("coolant", "all"), ("coolant", "without_exp_27_exp_35"),
                                              ("probe", "without_exp_2_3")])
def test_selection_from_the_cache_reproduces_the_stored_settings_and_graph(subsystem, scope):
    from core import data
    from pipeline.causal_discovery import DiscoveryContext, select_scope, selection_order

    cfg = load_config()
    context = DiscoveryContext(cfg, subsystem, io.read_input("splits"))
    row, graph = select_scope(context, data.scope_held_out(scope), discovery.grid_configs(cfg["discovery"]["grid"]),
                              selection_order(cfg["discovery"]))
    settings = pd.read_csv(EXPECTED / "discovery_settings.csv", keep_default_na=False)
    stored = settings[(settings.subsystem == subsystem) & (settings.scope == scope)].iloc[0]
    assert row["config_id"] == stored["config_id"]
    assert row["related_removed"] == stored["related_removed"]
    assert row["n_tied_objective"] == stored["n_tied_objective"]
    edges = pd.read_csv(EXPECTED / "discovered_graphs.csv")
    stored_edges = edges[(edges.subsystem == subsystem) & (edges.scope == scope)]
    assert discovery.edge_key(graph) == discovery.edge_key(stored_edges)


# ---------------------------------------------------------------------------
# thesis tables and figures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("part, name", [("pipeline", n) for n in PIPELINE_TABLES] +
                                       [("evaluation", n) for n in EVALUATION_TABLES])
def test_thesis_table_equals_the_latex_table(part, name):
    produced = read_text_table(io.tables_dir(part) / f"{name}.csv")
    expected = read_text_table(THESIS / f"{name}.csv")
    pd.testing.assert_frame_equal(produced, expected)


def test_missed_edge_table_has_the_numbers_of_the_latex_table():
    produced = read_text_table(io.tables_dir("evaluation") / "table_6_2_analysis_of_the_missed_coolant_edges.csv")
    expected = read_text_table(THESIS / "table_6_2_analysis_of_the_missed_coolant_edges.csv")
    assert list(produced.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(produced[["Missing edge", "Testable", "Lost in"]],
                                  expected[["Missing edge", "Testable", "Lost in"]])
    for column in ("Evidence", "Change and trade-off"):
        assert [numbers(t) for t in produced[column]] == [numbers(t) for t in expected[column]], column


@pytest.mark.parametrize("name", GRAPH_FIGURES)
def test_graph_figure_tikz_equals_the_thesis_input(name):
    """identical apart from the first comment line, which names the generating command."""
    produced = (io.figures_dir("evaluation") / f"{name}.tex").read_text().splitlines()
    expected = (THESIS / f"{name}.tex").read_text().splitlines()
    assert produced[0].startswith("%")
    assert produced[1:] == expected[1:]
    assert (io.figures_dir("evaluation") / f"{name}.pdf").exists()


def test_fault_timeline_figure_is_written():
    assert (io.figures_dir("pipeline") / "figure_5_1_machine_states_during_a_coolant_fault.pdf").exists()


def test_number_parser_ignores_identifiers():
    assert numbers("LT_A_700317 drops in 5 of 6 runs; smallest p-value 0.14") == [5.0, 6.0, 0.14]


# ---------------------------------------------------------------------------
# reruns: stored files are only replaced on purpose
# ---------------------------------------------------------------------------


def small_view(directory: Path, value: float) -> None:
    view = pp.ModelView.from_schema({"a": "Binary", "b": "Alarm"}, ["a", "b"], "discrete")
    arrays = {"r1": np.array([[0.0, value], [1.0, 0.0]])}
    masks = {"r1": np.zeros((2, 2), dtype=bool)}
    pp.save_view_data(view, pp.ViewData(["a", "b"], arrays, masks, view.data_type), directory, "toy_dt0.5", 0.5)


def test_a_view_that_differs_from_the_stored_one_is_replaced_only_with_force(tmp_path):
    from pipeline.preprocessing import place_view
    from utils.errors import DataError

    stored, new = tmp_path / "stored", tmp_path / "new"
    small_view(stored, 0.0)
    small_view(new, 0.0)
    assert place_view(new, stored, "toy_dt0.5", force=False).startswith("identical")
    small_view(new, 1.0)
    with pytest.raises(DataError):
        place_view(new, stored, "toy_dt0.5", force=False)
    assert place_view(new, stored, "toy_dt0.5", force=True).startswith("replaced")
    assert pp.load_view_data(stored, "toy_dt0.5")[0].arrays["r1"][0, 1] == 1.0


def test_refresh_cache_treats_entries_written_before_the_start_as_missing(tmp_path):
    import os
    import time

    spec = discovery.RunSpec("toy", "toy_dt0.5", "normal", ("r1",), ("a", "b"), discovery.StudyConfig(), "sha")
    discovery.RunCache(tmp_path).write(spec, {"status": "ok"})
    old = time.time() - 60
    os.utime(discovery.RunCache(tmp_path).path(spec), (old, old))
    cache = discovery.RefreshCache(tmp_path, since=time.time())
    assert discovery.RunCache(tmp_path).has(spec)
    assert not cache.has(spec)
    cache.write(spec, {"status": "ok"})
    assert cache.has(spec)
