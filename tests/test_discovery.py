"""tests of the discovery settings study: lag conversion, aggregation, selection, caching,
held-out exclusion and the tigramite sample cut it relies on."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from core import discovery
from core.preprocessing import ViewData
from utils.config import load_config


# ---------------------------------------------------------------------------
# configurations and lag conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau_s, steps", [(5.0, 10), (10.0, 20), (20.0, 40), (60.0, 120)])
def test_lag_seconds_convert_to_grid_steps(tau_s, steps):
    config = discovery.StudyConfig(tau_max_s=tau_s)
    assert config.tau_max == steps
    # the same number of steps as the pcmci+ config
    assert config.discovery_config().tau_max == steps


def test_lags_that_are_not_whole_grid_steps_are_rejected():
    with pytest.raises(ValueError):
        discovery.StudyConfig(tau_max_s=10.25)
    with pytest.raises(ValueError):
        discovery.StudyConfig(tau_max_s=0.0)


def test_original_config_matches_the_original_discovery_configuration():
    study = load_config()["discovery"]
    original = discovery.config_from_dict(study["original"]).discovery_config()
    assert original == discovery.DiscoveryConfig(interval_s=0.5, tau_max_s=10.0, pc_alpha=0.01)


def test_grid_has_27_configurations_but_only_9_pcmci_runs():
    study = load_config()["discovery"]
    grid = discovery.grid_configs(study["grid"])
    assert len(grid) == 27
    assert len({c.config_id for c in grid}) == 27
    assert len({c.run_id for c in grid}) == 9


def test_yaml_selection_order_is_the_implemented_order():
    study = load_config()["discovery"]
    order = tuple(tuple(next(iter(d.items()))) for d in study["selection"]["order"])
    assert order == discovery.SELECTION_ORDER


def test_aggregation_settings_do_not_change_the_run_key():
    base = discovery.StudyConfig()
    view = toy_view()
    spec = lambda c: discovery.RunSpec("toy", "toy", "r", ("a",), ("X", "Y"), c, "sha")
    assert spec(replace(base, support_threshold=0.75)).key() == spec(base).key()
    assert spec(replace(base, apply_fdr_to_graph=False)).key() == spec(base).key()
    assert spec(replace(base, pc_alpha=0.05)).key() != spec(base).key()
    assert spec(replace(base, max_conds=(2, 2, 2))).key() != spec(base).key()
    assert spec(replace(base, tau_max_s=20.0)).key() != spec(base).key()
    assert view.var_names == ["X", "Y"]


# ---------------------------------------------------------------------------
# aggregation, fdr filter and held-out exclusion
# ---------------------------------------------------------------------------


def link(regime, source, target, lag, significant=True, link="-->"):
    kind = "auto" if source == target else ("lagged" if lag > 0 else "contemporaneous")
    return {"regime": regime, "source": source, "target": target, "lag": lag, "link": link, "kind": kind,
            "p_value": 0.001 if significant else 0.2, "p_corrected": lag > 0, "g_stat": 10.0,
            "significant": significant}


def runs_from(links_by_regime, variables_by_regime, status=None):
    status = status or {}
    return {r: discovery.RegimeRun(r, variables_by_regime[r], pd.DataFrame(links_by_regime.get(r, []), columns=discovery.LINK_COLUMNS),
                            status.get(r, "ok"), 1.0) for r in variables_by_regime}


def test_support_threshold_counts_only_testable_regimes():
    variables = {"r1": ["A", "B"], "r2": ["A", "B"], "r3": ["A", "C"]}
    runs = runs_from({"r1": [link("r1", "A", "B", 2)]}, variables)
    nodes = ["A", "B", "C"]
    keep, _ = discovery.aggregate(runs, discovery.StudyConfig(support_threshold=0.5), nodes)
    drop, _ = discovery.aggregate(runs, discovery.StudyConfig(support_threshold=0.75), nodes)
    row = lambda pairs: pairs[(pairs.a == "A") & (pairs.b == "B")].iloc[0]
    assert row(keep).n_testable == 2 and row(keep).decision == "a -> b"
    assert row(drop).decision == "absent"


def test_fdr_filter_removes_only_non_significant_links():
    variables = {"r1": ["A", "B"]}
    runs = runs_from({"r1": [link("r1", "A", "B", 3, significant=False)]}, variables)
    _, filtered = discovery.aggregate(runs, discovery.StudyConfig(apply_fdr_to_graph=True), ["A", "B"])
    _, unfiltered = discovery.aggregate(runs, discovery.StudyConfig(apply_fdr_to_graph=False), ["A", "B"])
    assert filtered.empty
    assert list(zip(unfiltered.source, unfiltered.target)) == [("A", "B")]


def test_held_out_regime_never_affects_its_fold():
    nodes = ["A", "B", "C"]
    variables = {"normal": ["A", "B"], "s1": ["A", "B", "C"], "s2": ["A", "B", "C"]}
    base = {"normal": [link("normal", "A", "B", 1)], "s1": [link("s1", "A", "B", 1)]}
    config = discovery.StudyConfig()
    reference = nx.DiGraph([("A", "B"), ("B", "C")])
    row_a, edges_a = discovery.evaluate_config(runs_from(base, variables), config, nodes, reference, exclude=["s2"])
    # whatever the held-out regime found, and even if its run failed, the fold is unchanged
    changed = {**base, "s2": [link("s2", "B", "C", 1), link("s2", "A", "C", 2), link("s2", "B", "A", 1)]}
    row_b, edges_b = discovery.evaluate_config(runs_from(changed, variables), config, nodes, reference, exclude=["s2"])
    row_c, edges_c = discovery.evaluate_config(runs_from(base, variables, status={"s2": "timeout"}), config, nodes,
                                        reference, exclude=["s2"])
    assert discovery.edge_key(edges_a) == discovery.edge_key(edges_b) == discovery.edge_key(edges_c)
    assert row_a["directed_f1"] == row_b["directed_f1"] == row_c["directed_f1"]
    assert row_c["run_status"] == "ok" and "s2" not in row_c["regimes_in_scope"].split(",")
    # but the all-scenario scope is incomplete when s2 failed
    row_all, _ = discovery.evaluate_config(runs_from(base, variables, status={"s2": "timeout"}), config, nodes, reference)
    assert row_all["run_status"].startswith("incomplete")


def test_graph_metrics_count_unresolved_orientations_and_coverage():
    variables = {"r1": ["A", "B", "C"]}
    runs = runs_from({"r1": [link("r1", "A", "B", 1), link("r1", "B", "C", 0, link="o-o")]}, variables)
    reference = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "D")])
    row, _ = discovery.evaluate_config(runs, discovery.StudyConfig(), ["A", "B", "C", "D"], reference)
    assert row["n_adjacencies"] == 2 and row["n_directed_edges"] == 1 and row["unresolved_orientations"] == 1
    assert row["directed_precision"] == 1.0 and row["directed_recall"] == pytest.approx(1 / 3)
    assert row["reference_edges_testable"] == 2 and row["reference_edge_coverage"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# selection rule
# ---------------------------------------------------------------------------


def table(rows):
    base = {"skeleton_f1": 0.5, "tau_max_s": 10.0, "pc_alpha": 0.01, "support_threshold": 0.5, "run_status": "ok"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_selection_uses_directed_f1_first():
    t = table([{"config_id": "a", "directed_f1": 0.6, "skeleton_f1": 0.9},
               {"config_id": "b", "directed_f1": 0.7, "skeleton_f1": 0.1}])
    best, ties = discovery.select(t)
    assert best.config_id == "b" and ties["n_tied_objective"] == 1


def test_selection_tie_breaks_follow_the_declared_order():
    rows = [{"config_id": "a", "directed_f1": 0.7, "skeleton_f1": 0.8, "tau_max_s": 20.0},
            {"config_id": "b", "directed_f1": 0.7, "skeleton_f1": 0.8, "tau_max_s": 10.0, "pc_alpha": 0.05},
            {"config_id": "c", "directed_f1": 0.7, "skeleton_f1": 0.8, "tau_max_s": 10.0, "pc_alpha": 0.01,
             "support_threshold": 0.25},
            {"config_id": "d", "directed_f1": 0.7, "skeleton_f1": 0.8, "tau_max_s": 10.0, "pc_alpha": 0.01,
             "support_threshold": 0.75},
            {"config_id": "e", "directed_f1": 0.7, "skeleton_f1": 0.7}]
    best, ties = discovery.select(table(rows))
    assert best.config_id == "d"
    assert ties == {"n_eligible": 5, "n_tied_objective": 5, "n_tied_objective_and_first_tiebreak": 4}
    ranked = discovery.rank_configs(table(rows))
    assert list(ranked.config_id) == ["d", "c", "b", "a", "e"]


def test_float_noise_does_not_break_ties_and_config_id_is_the_last_rule():
    rows = [{"config_id": "z", "directed_f1": 0.75}, {"config_id": "y", "directed_f1": 0.75 + 1e-15}]
    best, ties = discovery.select(table(rows))
    assert best.config_id == "y" and ties["n_tied_objective"] == 2


def test_incomplete_configurations_are_never_selected():
    t = table([{"config_id": "a", "directed_f1": 0.9, "run_status": "incomplete (normal: timeout)"},
               {"config_id": "b", "directed_f1": 0.5}])
    best, ties = discovery.select(t)
    assert best.config_id == "b" and ties["n_eligible"] == 1
    with pytest.raises(ValueError):
        discovery.select(t[t.config_id == "a"])


# ---------------------------------------------------------------------------
# tigramite sample cut and availability
# ---------------------------------------------------------------------------


def test_tigramite_tests_cut_two_tau_max_steps_per_recording():
    from tigramite import data_processing as tg_pp
    from tigramite.independence_tests.independence_tests_base import CondIndTest

    assert inspect.signature(CondIndTest.run_test).parameters["cut_off"].default == "2xtau_max"
    rng = np.random.default_rng(0)
    data = {0: rng.integers(0, 2, (50, 2)).astype(float), 1: rng.integers(0, 2, (30, 2)).astype(float)}
    frame = tg_pp.DataFrame(data, analysis_mode="multiple")
    array, *_ = frame.construct_array(X=[(0, -1)], Y=[(1, 0)], Z=[], tau_max=4, cut_off="2xtau_max")
    assert array.shape[1] == (50 - 8) + (30 - 8)


def test_sample_availability_counts_rows_and_changes_after_the_cut():
    view = toy_view()
    info = discovery.sample_availability(view, ["a", "b"], ["X", "Y"], tau_max=3)
    # a: 20 rows, b: 5 rows (shorter than 2 * 3 + 1 contributes nothing)
    assert info["rows"] == 25 and info["rows_after_cut"] == 14 and info["recordings_contributing"] == 1
    # x changes at t = 4 and t = 10 in recording a (only t >= 6 survives), y never changes
    assert info["state_changes"] == 2 + 1 and info["state_changes_after_cut"] == 1


def toy_view() -> ViewData:
    xa = np.zeros((20, 2))
    xa[4:10, 0] = 1
    xb = np.zeros((5, 2))
    xb[3:, 0] = 1
    return ViewData(var_names=["X", "Y"], arrays={"a": xa, "b": xb},
                    masks={"a": np.zeros((20, 2), bool), "b": np.zeros((5, 2), bool)}, data_type=np.ones(2))


def test_bootstrap_draws_resample_recordings_within_every_regime():
    regimes = {"normal": [f"n{i}" for i in range(7)], "s1": ["a", "b", "c"]}
    draws = discovery.bootstrap_regimes(regimes, n_boot=3, seed=0)
    rng = np.random.default_rng(0)
    for draw in draws:
        expected = {r: list(rng.choice(ids, size=len(ids), replace=True)) for r, ids in regimes.items()}
        assert draw == expected
    assert discovery.bootstrap_regimes(regimes, 3, 0) == draws


# ---------------------------------------------------------------------------
# cache and process execution
# ---------------------------------------------------------------------------


def write_view(directory: Path, name: str, view: ViewData) -> None:
    ids = list(view.arrays)
    arrays = {f"data__{r}": view.arrays[r] for r in ids} | {f"mask__{r}": view.masks[r] for r in ids}
    np.savez_compressed(directory / f"{name}.npz", var_names=np.array(view.var_names), data_type=view.data_type,
                        recording_ids=np.array(ids), **arrays)
    (directory / f"{name}.json").write_text(json.dumps({"name": name}))


def chain_view() -> ViewData:
    rng = np.random.default_rng(1)
    arrays, masks = {}, {}
    for k in range(3):
        x = (rng.random(400) < 0.5).astype(float)
        x = np.repeat(x[:100], 4)  # persistent states
        y = np.roll(x, 2)
        y[:2] = 0
        arrays[f"r{k}"] = np.column_stack([x, y])
        masks[f"r{k}"] = np.zeros((400, 2), bool)
    return ViewData(var_names=["X", "Y"], arrays=arrays, masks=masks, data_type=np.ones(2))


def test_runs_are_cached_reproducible_and_logged(tmp_path):
    view = chain_view()
    write_view(tmp_path, "chain", view)
    cache = discovery.RunCache(tmp_path / "cache")
    config = discovery.StudyConfig(tau_max_s=2.0)
    specs, skipped = discovery.make_specs("toy", "chain", view, {"r": ["r0", "r1", "r2"]}, config)
    assert not skipped and len(specs) == 1
    events = discovery.run_specs(specs, cache, tmp_path, workers=1, timeout_s=300)
    assert [e["status"] for e in events] == ["ok"]
    # a second call finds the cache and runs nothing
    assert discovery.run_specs(specs, cache, tmp_path, workers=1, timeout_s=300) == []
    cached = cache.load(specs[0])
    direct = discovery.links_table(discovery.run_regime(view, "r", ["r0", "r1", "r2"], ["X", "Y"], config.discovery_config()))
    pd.testing.assert_frame_equal(cached["links"].reset_index(drop=True), direct.reset_index(drop=True),
                                  check_dtype=False)
    assert ((cached["links"].source == "X") & (cached["links"].target == "Y") & (cached["links"].lag == 2)).any()


def test_timeouts_and_changed_data_are_logged_not_hidden(tmp_path):
    view = chain_view()
    write_view(tmp_path, "chain", view)
    cache = discovery.RunCache(tmp_path / "cache")
    specs, _ = discovery.make_specs("toy", "chain", view, {"r": ["r0", "r1"]}, discovery.StudyConfig(tau_max_s=2.0))
    events = discovery.run_specs(specs, cache, tmp_path, workers=1, timeout_s=0.0)
    assert events[0]["status"] == "timeout" and cache.load(specs[0])["status"] == "timeout"
    stale = replace(specs[0], regime="r_stale", data_sha1="0" * 40)
    discovery.execute_spec(stale, str(tmp_path), str(cache.path(stale)))
    entry = cache.load(stale)
    assert entry["status"] == "failed" and "fingerprint" in entry["error"]
    runs = discovery.load_runs(cache, [specs[0]])
    row, _ = discovery.evaluate_config(runs, discovery.StudyConfig(tau_max_s=2.0), ["X", "Y"], nx.DiGraph([("X", "Y")]))
    assert row["run_status"].startswith("incomplete")


# ---------------------------------------------------------------------------
# what tigramite 5.2's fdr option does inside pcmci+ (the basis of the post-hoc filter)
# ---------------------------------------------------------------------------


def test_pcmciplus_fdr_corrects_lagged_p_values_only_and_leaves_the_graph_unchanged():
    from tigramite.pcmci_base import PCMCIbase

    assert inspect.signature(PCMCIbase.get_corrected_pvalues).parameters["exclude_contemporaneous"].default is True
    view = chain_view()
    plain = discovery.run_regime(view, "r", list(view.arrays), ["X", "Y"], replace(discovery.StudyConfig(tau_max_s=2.0),
                                                                            fdr_method="none").discovery_config())
    bh = discovery.run_regime(view, "r", list(view.arrays), ["X", "Y"], discovery.StudyConfig(tau_max_s=2.0).discovery_config())
    # the graph is decided with uncorrected tests, so it is identical
    assert (plain.graph == bh.graph).all()
    # lag-0 p-values are not corrected, lagged ones are never smaller after correction
    assert np.allclose(plain.p_matrix[:, :, 0], bh.p_matrix[:, :, 0])
    assert (bh.p_matrix[:, :, 1:] >= plain.p_matrix[:, :, 1:] - 1e-15).all()
    assert (bh.p_matrix[:, :, 1:] > plain.p_matrix[:, :, 1:] + 1e-15).any()
