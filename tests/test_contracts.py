"""the contract files between pipeline stages load and keep the agreed columns and scope names."""

from pathlib import Path

import pandas as pd
import pytest

from core import data, discovery, graphs
from utils import io
from utils.errors import MissingInputError

EXPECTED = Path(__file__).parent / "expected"

SCOPED = ["subsystem", "scope"]
CONTRACTS = {
    "splits": ["recording_id", "source", "subsystem", "experiment_id", "split", "fold"],
    "discovered_graphs": [*SCOPED, "source", "target", "oriented", "support", "orientation_conflict"],
    "discovery_settings": [
        *SCOPED, "held_out", "related_removed", "config_id", "ci_test", "interval_s", "tau_max_s", "tau_max_steps",
        "pc_alpha", "max_conds_dim", "max_conds_px", "max_conds_py", "conditioning_caps", "fdr_method", "fdr_filter",
        "support_threshold", "regimes_in_scope", "skeleton_precision", "skeleton_recall", "skeleton_f1",
        "directed_precision", "directed_recall", "directed_f1", "shd", "orientation_accuracy", "n_adjacencies",
        "n_directed_edges", "n_undirected_edges", "n_both_directions", "unresolved_orientations", "n_reference_edges",
        "n_pairs", "n_testable_pairs", "testable_pair_share", "reference_edges_testable", "reference_edge_coverage",
        "orientation_conflicts", "n_eligible", "n_tied_objective", "n_tied_objective_and_first_tiebreak",
    ],
    "scm_graphs": ["subsystem", "graph", "scope", "source", "target"],
    "dag_removals": [*SCOPED, "source", "target", "action"],
    "scm_settings": ["subsystem", "graph", "scope", "alpha", "lag", "selection_log_loss", "baseline_log_loss",
                     "selection_data"],
}
SUBSYSTEMS = ("coolant", "probe")


def scenarios(splits: pd.DataFrame, sub: str) -> list[str]:
    part = splits[(splits.subsystem == sub) & (splits.split == "fault")]
    return sorted(part.fold.unique())


def all_scopes(folds: list[str]) -> set[str]:
    """all scenarios, each outer fold, and each ordered (outer, inner) pair."""
    outer = {data.scope_name(k) for k in folds}
    inner = {data.scope_name((k, j)) for k in folds for j in folds if j != k}
    return {"all"} | outer | inner


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_contract_file_loads_with_the_agreed_columns(name):
    table = io.read_input(name)
    assert list(table.columns) == CONTRACTS[name]
    assert len(table)
    assert list(pd.read_csv(EXPECTED / f"{name}.csv").columns) == CONTRACTS[name]


def test_splits_hold_both_subsystems_with_one_split_per_recording():
    splits = io.read_input("splits")
    assert set(splits.subsystem) == set(SUBSYSTEMS)
    assert not splits.duplicated(["subsystem", "recording_id"]).any()
    for sub in SUBSYSTEMS:
        normal = splits[(splits.subsystem == sub) & (splits.source == "real_op")]
        assert normal.split.value_counts().to_dict() == {"train": 119, "validation": 26, "test": 25}
    assert scenarios(splits, "coolant") == ["exp_22", "exp_27", "exp_28", "exp_35"]
    assert scenarios(splits, "probe") == ["exp_1", "exp_2_3", "exp_4", "exp_5", "exp_6", "exp_7"]


def test_scope_names_round_trip():
    for held in [(), ("exp_22",), ("exp_22", "exp_27"), ("exp_2_3", "exp_1"), ("exp_1", "exp_2_3")]:
        assert data.scope_held_out(data.scope_name(held)) == held
    assert data.scope_name(None) == "all"
    assert data.scope_name("exp_4") == "without_exp_4"


def test_discovery_files_cover_every_scope_of_both_subsystems():
    splits = io.read_input("splits")
    settings = io.read_input("discovery_settings", keep_default_na=False)
    edges = io.read_input("discovered_graphs")
    removals = io.read_input("dag_removals")
    scm_graphs = io.read_input("scm_graphs")
    for sub in SUBSYSTEMS:
        scopes = all_scopes(scenarios(splits, sub))
        assert set(settings.loc[settings.subsystem == sub, "scope"]) == scopes
        assert set(edges.loc[edges.subsystem == sub, "scope"]) == scopes
        assert set(removals.loc[removals.subsystem == sub, "scope"]) <= scopes
        for graph in ("expert", "discovered"):
            part = scm_graphs[(scm_graphs.subsystem == sub) & (scm_graphs.graph == graph)]
            assert set(part.scope) == scopes
    held = [tuple(h.split(",")) if h else () for h in settings.held_out]
    assert [data.scope_held_out(s) for s in settings.scope] == held


def test_scm_settings_cover_all_and_outer_scopes():
    splits = io.read_input("splits")
    table = io.read_input("scm_settings")
    for sub in SUBSYSTEMS:
        expected = {"all"} | {data.scope_name(k) for k in scenarios(splits, sub)}
        for graph in ("expert", "discovered"):
            part = table[(table.subsystem == sub) & (table.graph == graph)]
            assert set(part.scope) == expected and len(part) == len(expected)
    assert set(table.alpha) <= {0.01, 0.1, 0.5, 1.0}
    assert set(table.lag.dropna()) <= {2, 4}


def test_scm_graphs_are_the_dag_conversion_of_the_discovered_graphs():
    edges = io.read_input("discovered_graphs")
    scm_graphs = io.read_input("scm_graphs")
    for sub in SUBSYSTEMS:
        expert = graphs.simple_digraph(data.load_expert_graph(subsystem=sub))
        nodes = sorted(expert.nodes)
        for scope, part in edges[edges.subsystem == sub].groupby("scope"):
            dag, _ = graphs.to_dag(discovery.edges_to_graph(part, nodes))
            rows = scm_graphs[(scm_graphs.subsystem == sub) & (scm_graphs.scope == scope)]
            discovered = rows[rows.graph == "discovered"]
            assert set(zip(discovered.source, discovered.target)) == set(dag.edges) and list(dag.nodes) == nodes
            assert set(zip(rows.loc[rows.graph == "expert", "source"], rows.loc[rows.graph == "expert", "target"])) \
                == set(expert.edges)


def test_missing_input_names_the_stage_to_run(monkeypatch, tmp_path):
    monkeypatch.setattr(io, "tables_dir", lambda part="pipeline": tmp_path)
    with pytest.raises(MissingInputError, match=r'run "python main.py run discovery" first'):
        io.read_input("discovered_graphs")


def test_thesis_names_are_snake_case_with_the_number():
    assert io.thesis_name("table", "6.4", "Root-cause ranking performance") == \
        "table_6_4_root_cause_ranking_performance"
    assert io.thesis_name("figure", "A.1", "Settings of the LSTM + SHAP baseline") == \
        "figure_a_1_settings_of_the_lstm_shap_baseline"
