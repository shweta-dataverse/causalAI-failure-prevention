"""tests for the stage-1 helpers in data.py and the view file format."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_rca import data
from causal_rca import preprocessing as pp


def make_events(recordings: dict[str, list[tuple]], source: str = "dig_twin") -> pd.DataFrame:
    """parsed, de-duplicated and change-marked events from (time, node, value, type) rows."""
    frames = []
    for recording_id, rows in recordings.items():
        fault = source == "dig_twin"
        info = pp.RecordingInfo(
            recording_id=recording_id,
            source=source,
            subsystem="coolant" if fault else None,
            experiment_id="exp_1" if fault else None,
            run_id="run_1" if fault else None,
            source_file=f"{recording_id}.csv",
        )
        raw = pd.DataFrame([[str(x) for x in row] for row in rows], columns=list(pp.EVENT_COLUMNS))
        frames.append(pp.parse_values(pp.prepare_events(raw, info)))
    events, _ = pp.resolve_duplicates(pd.concat(frames, ignore_index=True))
    return pp.mark_changes(events)


def test_variable_coverage_and_co_observation():
    events = make_events(
        {
            "run_a": [(0.0, "A", "True", "Binary"), (0.0, "B", "False", "Binary")],
            "run_b": [(0.0, "A", "True", "Binary"), (1.0, "C", "True", "Alarm")],
        }
    )
    coverage = data.variable_coverage(events, ["A", "B", "C"])
    assert coverage.loc["run_a"].tolist() == [True, True, False]
    assert coverage.loc["run_b"].tolist() == [True, False, True]

    co = data.co_observation(coverage)
    assert co.loc["A", "A"] == 2
    assert co.loc["A", "B"] == 1 and co.loc["B", "A"] == 1
    # never recorded together -> untestable pair
    assert co.loc["B", "C"] == 0


def splits_fixture() -> pd.DataFrame:
    numbers = [30, 10, 20, 90, 50, 70, 40, 100, 60, 80]
    normal = [
        {"recording_id": f"real_op_{n}", "source": "real_op", "subsystem": None, "experiment_id": None}
        for n in numbers
    ]
    fault = [
        {"recording_id": "coolant_exp_1_run_1", "source": "dig_twin", "subsystem": "coolant", "experiment_id": "exp_1"},
        {"recording_id": "coolant_exp_1_run_2", "source": "dig_twin", "subsystem": "coolant", "experiment_id": "exp_1"},
        {"recording_id": "coolant_exp_2_run_1", "source": "dig_twin", "subsystem": "coolant", "experiment_id": "exp_2"},
        {"recording_id": "probe_exp_7_run_1", "source": "dig_twin", "subsystem": "probe", "experiment_id": "exp_7"},
    ]
    return pd.DataFrame(normal + fault)


def test_make_splits_orders_normal_recordings_and_groups_fault_runs():
    splits = data.make_splits(splits_fixture(), "coolant").set_index("recording_id")
    normal = splits[splits.source == "real_op"]
    number = np.array([int(rid.rsplit("_", 1)[-1]) for rid in normal.index])
    split = normal.split.to_numpy()

    # 10 recordings -> 7 train, 2 validation, 1 test, in recording-number order
    assert normal.split.value_counts().to_dict() == {"train": 7, "validation": 2, "test": 1}
    assert number[split == "train"].max() < number[split == "validation"].min()
    assert number[split == "test"].tolist() == [100]

    fault = splits[splits.source == "dig_twin"]
    assert set(fault.index) == {"coolant_exp_1_run_1", "coolant_exp_1_run_2", "coolant_exp_2_run_1"}
    assert (fault.fold == fault.experiment_id).all()
    # every scenario sits in exactly one fold
    assert fault.groupby("experiment_id").fold.nunique().eq(1).all()
    assert splits.index.is_unique


def test_make_splits_is_deterministic_and_validates_input():
    recordings = splits_fixture()
    first = data.make_splits(recordings, "coolant")
    second = data.make_splits(recordings.sample(frac=1, random_state=3), "coolant")
    pd.testing.assert_frame_equal(
        first.sort_values("recording_id").reset_index(drop=True),
        second.sort_values("recording_id").reset_index(drop=True),
    )
    with pytest.raises(ValueError):
        data.make_splits(recordings, "coolant", fractions=(0.8, 0.3, 0.1))
    with pytest.raises(ValueError):
        data.make_splits(recordings, "hydraulics")


def test_view_files_round_trip(tmp_path):
    events = make_events(
        {
            "run_a": [(0.0, "A", "False", "Binary"), (1.0, "A", "True", "Binary"), (2.0, "B", "True", "Alarm")],
            "run_b": [(0.0, "A", "True", "Binary"), (0.0, "B", "False", "Alarm"), (3.0, "B", "True", "Alarm")],
        }
    )
    states = pp.reconstruct_states(events, 0.5)
    view = pp.ModelView.fit(states, pp.node_types(events), ["run_a", "run_b"], "discrete")
    view_data = view.transform(states, ["run_a", "run_b"])

    pp.save_view_data(view, view_data, tmp_path, "toy_dt0.5", interval_s=0.5)
    loaded, spec = pp.load_view_data(tmp_path, "toy_dt0.5")

    assert loaded.var_names == view_data.var_names
    assert spec["interval_s"] == 0.5 and spec["recording_ids"] == ["run_a", "run_b"]
    np.testing.assert_array_equal(loaded.data_type, view_data.data_type)
    for rid in ["run_a", "run_b"]:
        np.testing.assert_array_equal(loaded.arrays[rid], view_data.arrays[rid])
        np.testing.assert_array_equal(loaded.masks[rid], view_data.masks[rid])
    # B is unknown in run_a until its first event at 2.0 s
    b = loaded.var_names.index("B")
    assert loaded.masks["run_a"][:, b].tolist() == [True, True, True, True, False]


def test_transition_bins_flag_hidden_pulse():
    events = make_events(
        {"run_a": [(0.0, "A", "False", "Binary"), (1.2, "A", "True", "Binary"), (1.6, "A", "False", "Binary")]}
    )
    coarse = pp.transition_bins(events, 1.0).set_index("step")
    assert coarse.loc[2, "n_changes"] == 2
    assert not coarse.loc[2, "visible"] and coarse.loc[2, "masked"]

    fine = pp.transition_bins(events, 0.1)
    assert fine.visible.sum() == 2 and not fine.masked.any()


def test_edge_onset_delays_keep_effects_logged_before_causes():
    events = make_events(
        {
            "run_a": [
                (0.0, "U", "True", "Binary"),
                (0.0, "V", "False", "Binary"),
                (2.0, "U", "False", "Binary"),  # before cause_start_at: ignored
                (3.0, "U", "True", "Binary"),
                (9.8, "V", "True", "Binary"),
                (10.0, "U", "False", "Binary"),
            ]
        }
    )
    runs = pd.DataFrame({"recording_id": ["run_a"], "experiment_id": ["exp_1"], "cause_start_at": [5.0]})
    graph = nx.MultiDiGraph([("U", "V")])
    delays = data.edge_onset_delays(events, runs, graph)
    assert len(delays) == 1
    row = delays.iloc[0]
    assert (row.cause, row.effect) == ("U", "V")
    assert row.cause_onset_s == 10.0 and row.effect_onset_s == 9.8
    assert row.delay_s == pytest.approx(-0.2)


# --- graph projection, metrics and perturbations ----------------------------

from causal_rca import causal_discovery as cd
from causal_rca import counterfactuals as cfx
from causal_rca import graph_evaluation as ge
from causal_rca import interventions as itv
from causal_rca import root_cause_analysis as rca
from causal_rca import scm as sc


def test_compare_graphs_known_values():
    expert = nx.DiGraph([("a", "b"), ("b", "c")])
    predicted = nx.DiGraph()
    predicted.add_edge("a", "b", oriented=True)
    predicted.add_edge("c", "b", oriented=True)  # reversed
    predicted.add_edge("a", "c", oriented=False)  # extra, undirected
    m = ge.compare_graphs(predicted, expert)
    assert m["skeleton_precision"] == pytest.approx(2 / 3) and m["skeleton_recall"] == 1.0
    assert m["directed_precision"] == 0.5 and m["directed_recall"] == 0.5
    assert m["shd"] == 2  # one extra adjacency + one reversed mark
    assert m["orientation_accuracy"] == 0.5

    # restricting to testable pairs removes the extra edge from the comparison
    restricted = ge.compare_graphs(predicted, expert, pairs=[frozenset("ab"), frozenset("bc")])
    assert restricted["skeleton_precision"] == 1.0


def regime_result(variables, graph, p=None):
    p = np.zeros(graph.shape) if p is None else p
    return cd.RegimeResult(regime="r1", recording_ids=["x"], variables=variables, graph=graph,
                           val_matrix=np.ones(graph.shape), p_matrix=p, n_samples=10, runtime_s=0.0,
                           config=cd.DiscoveryConfig())


def test_links_table_lists_contemporaneous_pairs_once_and_orients_lagged_links():
    g = np.full((3, 3, 3), "", dtype="<U3")
    g[0, 1, 2] = "-->"  # a(t-2) -> b(t)
    g[1, 2, 0], g[2, 1, 0] = "o-o", "o-o"  # b - c unoriented
    g[0, 0, 1] = "-->"  # autodependency
    links = cd.links_table(regime_result(["a", "b", "c"], g))
    assert len(links) == 3
    contemp = links[links.kind == "contemporaneous"]
    assert len(contemp) == 1 and contemp.link.iloc[0] == "o-o"
    relations = cd.regime_relations(links)
    assert set(zip(relations.a, relations.b, relations.relation)) == {("a", "b", "directed"), ("b", "c", "undirected")}


def test_aggregation_counts_only_testable_regimes():
    relations = pd.DataFrame([
        {"regime": "r1", "a": "a", "b": "b", "relation": "directed", "basis": "lagged", "lags": [2]},
        {"regime": "r1", "a": "c", "b": "d", "relation": "undirected", "basis": "contemporaneous (o-o)", "lags": [0]},
    ])
    variables = {"r1": ["a", "b", "c", "d"], "r2": ["a", "b"]}
    pairs = cd.aggregate_relations(relations, variables, ["a", "b", "c", "d", "e"], threshold=0.5).set_index(["a", "b"])
    assert pairs.loc[("a", "b"), "support"] == 0.5 and pairs.loc[("a", "b"), "decision"] == "a -> b"
    assert pairs.loc[("c", "d"), "n_testable"] == 1 and pairs.loc[("c", "d"), "decision"] == "undirected"
    assert pairs.loc[("a", "e"), "decision"] == "untestable"
    # leaving r1 out removes all evidence
    assert cd.aggregate_relations(relations, variables, ["a", "b"], exclude=["r1"]).decision.iloc[0] == "absent"


def test_to_dag_drops_undirected_edges_and_breaks_cycles_at_the_weakest_edge():
    g = nx.DiGraph()
    g.add_edge("a", "b", oriented=True, support=1.0)
    g.add_edge("b", "a", oriented=True, support=0.5)
    g.add_edge("b", "c", oriented=False, support=1.0)
    dag, log = ge.to_dag(g)
    assert nx.is_directed_acyclic_graph(dag)
    assert list(dag.edges()) == [("a", "b")]
    assert set(log.action) == {"dropped (undirected)", "dropped (breaks cycle)"}


@pytest.mark.parametrize("kind", ["delete", "insert", "reverse"])
def test_perturbations_keep_graphs_acyclic(kind):
    base = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
    out = ge.perturb_graph(base, kind, 0.5, seed=1)
    assert nx.is_directed_acyclic_graph(out)
    changed = set(base.edges()) ^ set(out.edges())
    if kind == "delete":
        assert out.number_of_edges() == 2 and set(out.edges()) < set(base.edges())
    elif kind == "insert":
        assert out.number_of_edges() == 6 and set(base.edges()) < set(out.edges())
    else:
        assert out.number_of_edges() == 4 and len(changed) == 4  # two edges flipped
    assert ge.perturb_graph(base, kind, 0.5, seed=1).edges() == out.edges()


# --- synthetic scm with a known structure -----------------------------------


def view(series: dict[str, dict[str, list[float]]]) -> pp.ViewData:
    names = sorted(next(iter(series.values())))
    arrays = {r: np.column_stack([np.array(v[n], dtype=float) for n in names]) for r, v in series.items()}
    return pp.ViewData(var_names=names, arrays=arrays, masks={r: np.zeros_like(a, dtype=bool) for r, a in arrays.items()},
                       data_type=np.ones(len(names), dtype=int))


def synthetic_training(n_runs=6, length=200, seed=0):
    """r (root) stays 1; p switches randomly; c copies p in the same step; alarm a = 1 when r = 0."""
    rng = np.random.default_rng(seed)
    runs = {}
    for k in range(n_runs):
        p = (np.cumsum(rng.random(length) < 0.05) % 2).astype(float)
        runs[f"train_{k}"] = {"r": [1.0] * length, "p": list(p), "c": list(p), "a": [0.0] * length}
    return runs


def faulty_run(length=60, onset=30):
    r = [1.0] * onset + [0.0] * (length - onset)
    a = [0.0] * (onset + 1) + [1.0] * (length - onset - 1)
    return {"r": r, "p": [0.0] * length, "c": [0.0] * length, "a": a}


GRAPH = nx.DiGraph([("p", "c"), ("r", "a")])


def fitted_scm(extra_runs=None):
    runs = synthetic_training()
    # one training fault run teaches the alarm mechanism (r is manipulated there)
    runs["train_fault"] = faulty_run()
    vd = view({**runs, **(extra_runs or {})})
    manipulated = {"train_fault": {"r"}}
    return vd, sc.fit_scm(vd, GRAPH, list(runs), manipulated, interval_s=1.0)


def test_scm_learns_known_mechanisms():
    vd, model = fitted_scm()
    c = model.mechanisms["c"]
    # c copies p: switching on is likely exactly when p is 1 now
    assert c.prob_one(np.array([0.0]), np.array([[1, 0]]))[0] > 0.9
    assert c.prob_one(np.array([0.0]), np.array([[0, 0]]))[0] < 0.05
    # the root never changed outside the manipulated run: its change is very unlikely
    assert model.mechanisms["r"].prob_one(np.array([1.0]), np.zeros((1, 0)))[0] > 0.99
    assert not model.mechanisms["r"].supported and model.mechanisms["r"].has_data


def test_simulated_intervention_replaces_the_equation():
    _, model = fitted_scm()
    rng = np.random.default_rng(0)
    noise = {v: rng.uniform(size=(50, 40)) for v in model.mechanisms}
    initial = {"r": 1.0, "p": 0.0, "c": 0.0, "a": 0.0}
    sim = model.simulate(initial, noise, interventions={"r": (0.0, 10)})
    # from step 10 the equation is replaced exactly; before, r follows its own (persistent) mechanism
    assert (sim["r"][:, 10:] == 0).all() and (sim["r"][:, 1:10] == 1).mean() > 0.99
    # the learned alarm follows the broken root
    assert sim["a"][:, -1].mean() > 0.9


def test_rca_ranks_the_intervened_root_first():
    test = {"test_0": faulty_run()}
    test["test_0"]["p"] = [0.0] * 20 + [1.0] * 40  # p changes too, but c follows it as usual
    test["test_0"]["c"] = list(test["test_0"]["p"])
    vd, model = fitted_scm(extra_runs=test)
    s = sc.recording_series(vd, "test_0")
    window = rca.analysis_window(s, ["a"], 1.0, before_s=40, after_s=5)
    ranking = rca.score_candidates(model, s, rca.candidates(s, ["a", "c", "p", "r"], ["a"]), window)
    assert ranking.variable.iloc[0] == "r"
    assert "a" not in set(ranking.variable)


def test_ranking_metrics_with_multiple_causes():
    m = rca.ranking_metrics(["x", "a", "b", "y"], {"a", "b"}, k=3)
    assert m["top1"] == 0.0 and m["hit@3"] == 1.0 and m["recall@3"] == 1.0
    assert m["ap@3"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert m["rr"] == 0.5
    chance = rca.chance_metrics(["a", "x"], {"a"}, n_draws=4000, seed=1)
    assert chance["top1"] == pytest.approx(0.5, abs=0.05)


# --- interventions and counterfactuals --------------------------------------


def test_feasibility_rules():
    ok, problems = itv.check_feasibility([itv.Action("HP_Pump_Ok", 1.0, 5)], recorded={"HP_Pump_Ok"})
    assert ok and not problems
    for actions in (
        [itv.Action("HP_A_700304", 0.0, 5)],  # alarm
        [itv.Action("HP_Pump_isOff", 0.0, 5)],  # derived feedback
        [itv.Action("HP_Pump_Ok", 0.0, 5)],  # maintenance only towards healthy
    ):
        assert not itv.check_feasibility(actions)[0]
    assert not itv.check_feasibility([itv.Action("HP_Pump_Ok", 1.0, 5)], recorded={"LP_Pump_Ok"})[0]
    many = [itv.Action(v, itv.allowed_value(v), 5) for v in ["CLF_Filter_Ok", "F_Filter_Ok", "HP_Pump_Ok", "LP_Pump_Ok"]]
    assert not itv.check_feasibility(many)[0]
    table = itv.actionability_table(["HP_A_700304", "HP_Pump_Ok", "unknown_var"], {"HP_A_700304": "Alarm"})
    assert table.set_index("variable").actionable.to_dict() == {"HP_A_700304": False, "HP_Pump_Ok": True, "unknown_var": False}


def test_counterfactual_contract():
    vd, model = fitted_scm(extra_runs={"test_0": faulty_run()})
    s = sc.recording_series(vd, "test_0")
    rng = np.random.default_rng(0)
    # replaying the abducted noise without any action reproduces the observed run
    assert cfx.abduction_consistency(model, s, 20, rng) == 1.0

    # "r" is not in the coolant actionability table, so register it for this synthetic case
    itv.ACTIONABILITY["r"] = ("maintenance", 1.0, "synthetic root")
    try:
        result = cfx.counterfactual(model, s, [itv.Action("r", 1.0, 30)], ["a"], outcome_start=20, n_samples=100,
                                    rng=rng, recorded=set(s))
        assert result.feasible and result.valid()
        assert result.improvement > 0 and result.counterfactual_alarm_share < 0.05
        assert result.sparsity == 1 and result.proximity_s == 30.0
        ranking = pd.DataFrame({"variable": ["r", "p"], "rank": [1, 2]})
        best, tried = cfx.recommend(model, s, ranking, {"r": 30, "p": 25}, ["a"], 20, 100, rng, set(s))
        assert best is not None and [a.variable for a in best.actions] == ["r"]
        assert all(len(t.actions) >= 1 for t in tried)
    finally:
        del itv.ACTIONABILITY["r"]
    # acting on an alarm is never feasible
    bad = cfx.counterfactual(model, s, [itv.Action("a", 0.0, 30)], ["a"], 20, 20, rng, recorded=set(s))
    assert not bad.feasible and not bad.valid()


# --- audit corrections (stage 03) --------------------------------------------


def test_aggregation_reports_orientation_conflicts():
    relations = pd.DataFrame([
        {"regime": "r1", "a": "a", "b": "b", "relation": "directed", "basis": "lagged", "lags": [1]},
        {"regime": "r2", "a": "a", "b": "b", "relation": "directed", "basis": "lagged", "lags": [2]},
        {"regime": "r3", "a": "b", "b": "a", "relation": "directed", "basis": "lagged", "lags": [1]},
        {"regime": "r4", "a": "a", "b": "b", "relation": "undirected", "basis": "contemporaneous (o-o)", "lags": [0]},
    ])
    variables = {r: ["a", "b"] for r in ["r1", "r2", "r3", "r4"]}
    pair = cd.aggregate_relations(relations, variables, ["a", "b"], threshold=0.5).iloc[0]
    # majority direction is kept, but the opposite evidence is reported
    assert pair.decision == "a -> b" and pair.support_b_to_a == 0.25
    assert pair.orientation_conflict and pair.mixed_marks
    edges = cd.static_edges(pd.DataFrame([pair]))
    assert edges.orientation_conflict.tolist() == [True]


def test_links_table_marks_which_p_values_are_corrected():
    g = np.full((2, 2, 2), "", dtype="<U3")
    g[0, 1, 1] = "-->"
    g[0, 1, 0], g[1, 0, 0] = "o-o", "o-o"
    links = cd.links_table(regime_result(["a", "b"], g)).set_index("lag")
    # tigramite bh-corrects lagged p-values only
    assert bool(links.loc[1, "p_corrected"]) and not bool(links.loc[0, "p_corrected"])


def test_scm_rejects_undirected_edges():
    vd, _ = fitted_scm()
    g = nx.DiGraph()
    g.add_nodes_from(["a", "c", "p", "r"])
    g.add_edge("p", "c", oriented=False)
    with pytest.raises(ValueError):
        sc.fit_scm(vd, g, list(vd.arrays), {}, interval_s=1.0)
    # after to_dag the undirected edge is dropped, not oriented
    dag, log = ge.to_dag(g)
    assert dag.number_of_edges() == 0 and log.action.tolist() == ["dropped (undirected)"]


def test_first_activation_needs_a_rise_from_zero():
    traj = np.array([
        [0, 0, 1, 1, 0],  # rises at step 2
        [1, 1, 1, 1, 1],  # active from the start: never activated
        [1, 1, 0, 0, 1],  # active, clears, rises again at step 4
        [0, 0, 0, 0, 0],  # never on
    ], dtype=float)
    onset = itv.first_activation(traj, start=1)
    assert onset[0] == 2 and np.isnan(onset[1]) and onset[2] == 4 and np.isnan(onset[3])
    # a later start ignores earlier rises
    assert np.isnan(itv.first_activation(traj, start=3)[0])
    with pytest.raises(ValueError):
        itv.first_activation(traj, start=0)
