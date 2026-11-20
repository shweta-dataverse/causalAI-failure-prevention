import time
from collections import Counter
from dataclasses import dataclass, replace
from importlib import import_module

import networkx as nx
import numpy as np
import pandas as pd

from core import data, discovery, graphs
from core.graphs import GraphLayout
from utils import io
from utils.config import setting
from utils.logger import log

FIGURES = {"coolant": ("6.1", "Expert and discovered coolant graphs"),
           "probe": ("6.2", "Expert and discovered probe graphs")}

# node positions in cm, placed by hand so that no edge passes behind another node. both panels use them
LAYOUTS = {
    "coolant": GraphLayout(
        positions={
            "CLF_Filter_Ok": (0, 0), "CLF_A_700307": (9.3, 0),
            "CLT_Level_lt_Min": (0, -1), "CLT_A_700310": (9.3, -1),
            "F_Filter_Ok": (0, -2), "F_A_700313": (9.3, -2),
            "LP_Pump_Ok": (0, -3.5), "LP_A_700301": (9.3, -3), "LP_Pump_On": (4.65, -4),
            "HP_Pump_isOff": (9.3, -4.5),
            "HP_Pump_Ok": (0, -5.5), "HP_A_700304": (9.3, -5.5),
            "LT_Level_Ok": (0, -7), "LT_A_700317": (9.3, -6.5), "LT_Pump_Ok": (9.3, -7.5),
        },
        routes={
            frozenset({"CLF_A_700307", "LP_A_700301"}): ("east", "east", "out=0, in=0, looseness=1"),
            frozenset({"HP_A_700304", "LT_A_700317"}): ("east", "east", "out=0, in=0, looseness=1.2"),
        },
        y_scale=0.8, legend_gap=0.7),
    "probe": GraphLayout(
        positions={
            "MPA_toInitPos": (0, 0), "MPA_InitPos": (4.2, 0), "MPC_close": (8.4, 0), "MPC_Closed": (11.4, -1.0),
            "MPC_open": (8.4, -1.6), "MPC_isOpen": (8.4, -3.2),
            "MPA_A_701124": (2.6, -1.9), "MPA_A_701125": (5.6, -1.6),
            "MPA_toWorkPos": (0, -4.6), "MPA_WorkPos": (4.2, -4.6), "MP_Inactive": (8.4, -5.2),
        },
        routes={
            frozenset({"MPA_toWorkPos", "MP_Inactive"}): ("south", "south", "out=-90, in=-90, looseness=0.6"),
            frozenset({"MPC_close", "MPC_isOpen"}): ("east", "east", "out=0, in=0, looseness=0.7"),
            frozenset({"MPA_toWorkPos", "MPA_InitPos"}): ("north", "south west", "bend left=22"),
        },
        y_scale=0.8, legend_gap=1.9),
}

# step files start with their number, so they are imported by name
causal_discovery = import_module("pipeline.02_causal_discovery")


@dataclass
class Result:
    """agreement, stability and diagnostics of the discovered graphs of one subsystem."""

    context: causal_discovery.DiscoveryContext
    selected: discovery.StudyConfig
    original: discovery.StudyConfig
    graph: nx.DiGraph
    metrics: dict
    random: pd.Series
    dag_metrics: dict
    original_row: dict
    selected_row: pd.Series
    folds: pd.DataFrame
    bootstrap: dict
    missed: list[dict]
    extra_alarm_pairs: list[dict]


def run(cfg: dict, force: bool = False) -> None:
    splits = io.read_input("splits")
    settings = io.read_input("discovery_settings", keep_default_na=False)
    edges = io.read_input("discovered_graphs")
    started = time.time()
    results = {sub: evaluate(cfg, causal_discovery.DiscoveryContext(cfg, sub, splits), settings, edges, force, started)
               for sub in setting(cfg, "subsystems")}
    io.save_table(agreement_table(results), "6.1", "Agreement with the expert graphs")
    io.save_table(missed_table(results), "6.2", "Analysis of the missed coolant edges")
    io.save_table(settings_table(results), "A.1", "Original and selected discovery settings")
    for sub, result in results.items():
        save_graph_figures(sub, result)
    for result in results.values():
        io.report(metric_block(result, setting(cfg, "graph_evaluation", "stable_edge_share")))


# one subsystem


def evaluate(cfg: dict, context: causal_discovery.DiscoveryContext, settings: pd.DataFrame, edges: pd.DataFrame, force: bool,
             started: float) -> Result:
    study = setting(cfg, "discovery")
    grid = {c.config_id: c for c in discovery.grid_configs(study["grid"])}
    rows = settings[settings["subsystem"] == context.name].set_index("scope")
    selected = grid[rows.loc["all", "config_id"]]
    original = discovery.config_from_dict(study["original"])
    graph = scope_graph(edges, context, "all")
    metrics = graphs.compare_graphs(graph, context.expert)
    evaluation = setting(cfg, "graph_evaluation")
    random = graphs.random_reference(context.expert, metrics["n_pred_edges"], evaluation["random_draws"], evaluation["seed"])
    original_row, _ = discovery.evaluate_config(context.runs(original, context.all_regimes), original, context.nodes,
                                                context.expert)
    bootstrap = bootstrap_stability(cfg, context, [original, selected], force, started)
    traced = context.name in evaluation["missed_edges_for"]
    return Result(
        context=context, selected=selected, original=original, graph=graph,
        metrics=metrics, random=random.mean(numeric_only=True),
        dag_metrics=graphs.compare_graphs(graphs.to_dag(graph)[0], context.expert),
        original_row=original_row, selected_row=rows.loc["all"],
        folds=rows[[len(data.scope_held_out(scope)) == 1 for scope in rows.index]],
        bootstrap=bootstrap,
        missed=missed_edges(cfg, context, graph, selected, original, bootstrap[selected.config_id], force, started)
        if traced else [],
        extra_alarm_pairs=extra_alarm_pairs(context, graph) if traced else [],
    )


def scope_graph(edges: pd.DataFrame, context: causal_discovery.DiscoveryContext, scope: str) -> nx.DiGraph:
    rows = edges[(edges["subsystem"] == context.name) & (edges["scope"] == scope)]
    return discovery.edges_to_graph(rows[["source", "target", "oriented", "support"]], context.nodes)


def bootstrap_stability(cfg: dict, context: causal_discovery.DiscoveryContext, configs: list[discovery.StudyConfig], force: bool,
                        started: float) -> dict[str, dict]:
    """per configuration over resampled recordings: directed f1, jaccard and the share of draws per edge."""
    settings = setting(cfg, "discovery", "bootstrap")
    draws = discovery.bootstrap_regimes(context.all_regimes, settings["n_boot"], settings["seed"])
    causal_discovery.run_missing(cfg, context, [discovery.make_specs(context.name, context.view_name, context.view, draw, c)[0]
                               for c in configs for draw in draws], force, started)
    out = {}
    for config in configs:
        _, full = discovery.aggregate(context.runs(config, context.all_regimes), config, context.nodes)
        full_keys = discovery.edge_key(full)
        f1, jaccard, found, adjacent = [], [], Counter(), Counter()
        for draw in draws:
            row, drawn = discovery.evaluate_config(context.runs(config, draw), config, context.nodes, context.expert)
            if row["run_status"] != "ok":
                continue
            keys = discovery.edge_key(drawn)
            f1.append(row["directed_f1"])
            jaccard.append(discovery.jaccard(keys, full_keys))
            found.update(keys)
            adjacent.update({frozenset((a, b)) for a, b in zip(drawn["source"], drawn["target"])})
        out[config.config_id] = {"draws": len(f1), "directed_f1": float(np.mean(f1)), "jaccard": float(np.mean(jaccard)),
                                 "full_edges": full_keys, "edge_share": {key: n / len(f1) for key, n in found.items()},
                                 "adjacency_share": {pair: n / len(f1) for pair, n in adjacent.items()}}
    return out


# missed expert edges and extra alarm edges


def full_data_configs(cfg: dict, selected: discovery.StudyConfig,
                      original: discovery.StudyConfig) -> list[discovery.StudyConfig]:
    """every configuration run on all recordings: the grid, the conditioning caps and the long maximum lag."""
    shortlist = setting(cfg, "discovery", "shortlist")
    named = {"selected": selected, "original": original}
    configs = list(discovery.grid_configs(setting(cfg, "discovery", "grid")))
    configs += [replace(named[who], max_conds=(k, k, k)) for who in shortlist["caps_for"] for k in shortlist["conditioning_caps"]]
    configs += [replace(named[who], tau_max_s=float(shortlist["long_lag_s"])) for who in shortlist["long_lag_for"]]
    return list({c.run_id: c for c in configs}.values())


def missed_edges(cfg: dict, context: causal_discovery.DiscoveryContext, graph: nx.DiGraph, selected: discovery.StudyConfig,
                 original: discovery.StudyConfig, bootstrap: dict, force: bool, started: float) -> list[dict]:
    """per missed expert edge: where it was testable, which test removed it and what recovers it."""
    configs = full_data_configs(cfg, selected, original)
    causal_discovery.run_missing(cfg, context, [discovery.make_specs(context.name, context.view_name, context.view, context.all_regimes, c)[0]
                               for c in configs], force, started)
    found = {frozenset(e) for e in graph.edges}
    usable = {r: discovery.regime_variables(context.view, ids) for r, ids in context.all_regimes.items()}
    out = []
    for u, v in sorted(context.expert.edges):
        if frozenset((u, v)) in found:
            continue
        testable = [r for r in usable if {u, v} <= set(usable[r])]
        p_values, conditions = [], set()
        for r in testable:
            p_min, separations = discovery.pair_separation(context.view, context.all_regimes[r], usable[r], selected, u, v)
            p_values.append(p_min)
            conditions |= {c for s in separations for c in s["conditions"]}
        hits = []
        for config in configs:
            runs = {r: run for r, run in context.runs(config, context.all_regimes).items() if r in testable}
            links = discovery.pair_links(runs, u, v)
            if len(links):
                hits.append((config, links))
        out.append({"edge": (u, v), "testable": testable, "p_min": min(p_values), "conditions": sorted(conditions),
                    "selected_adjacencies": len(found),
                    "behaviour": fault_behaviour(context, testable, u, v, sorted(conditions)),
                    "hits": hits, "tradeoff": tradeoff(cfg, context, hits, u, v) if hits else None,
                    "bootstrap_share": bootstrap["adjacency_share"].get(frozenset((u, v)), 0.0)})
    return out


def fault_behaviour(context: causal_discovery.DiscoveryContext, testable: list[str], cause: str, effect: str,
                    conditions: list[tuple[str, int]]) -> dict:
    """how the effect and the conditioning variable behave while the cause is in its fault state."""
    intervals = changes = same_step = runs = 0
    values = set()
    j = context.view.var_names.index(effect)
    partner = conditions[0][0] if conditions else None
    for regime in testable:
        for rid in context.all_regimes[regime]:
            runs += 1
            effect_steps = [step for step, _ in discovery.state_changes(context.view, rid, effect)]
            for start, end in discovery.fault_intervals(context.view, rid, cause):
                intervals += 1
                segment = context.view.arrays[rid][start:end, j]
                values |= set(np.unique(segment).astype(int).tolist())
                changes += any(start - 1 <= step <= end for step in effect_steps)
            drop = first_drop(context, rid, cause)
            if partner is not None and drop is not None:
                same_step += drop == first_drop(context, rid, partner)
    return {"intervals": intervals, "intervals_with_change": changes, "values": sorted(values), "runs": runs,
            "partner": partner, "same_step": same_step}


def first_drop(context: causal_discovery.DiscoveryContext, recording_id: str, variable: str) -> int | None:
    return next((step for step, value in discovery.state_changes(context.view, recording_id, variable) if value == 0), None)


def tradeoff(cfg: dict, context: causal_discovery.DiscoveryContext, hits: list, u: str, v: str) -> dict:
    """best evaluated aggregation (selection order) of the runs that found the pair, among those that keep it."""
    removed_by_filter = all(not bool(s) for _, links in hits for s in links["significant"])
    thresholds = setting(cfg, "discovery", "grid", "support_threshold")
    rows, kept = [], {}
    for config, _ in hits:
        for threshold in thresholds:
            candidate = replace(config, support_threshold=threshold, apply_fdr_to_graph=not removed_by_filter)
            row, edges = discovery.evaluate_config(context.runs(candidate, context.all_regimes), candidate, context.nodes,
                                                   context.expert)
            if frozenset((u, v)) in {frozenset(e) for e in zip(edges["source"], edges["target"])}:
                rows.append(row)
                kept[candidate.config_id] = discovery.edges_to_graph(edges, context.nodes)
    best = discovery.rank_configs(pd.DataFrame(rows), causal_discovery.selection_order(setting(cfg, "discovery"))).iloc[0]
    metrics = graphs.compare_graphs(kept[best["config_id"]], context.expert)
    return {"alphas": sorted({c.pc_alpha for c, _ in hits}), "filter_off": removed_by_filter, "config": best,
            "found": metrics["skeleton_tp"], "reference": metrics["n_true_edges"], "adjacencies": metrics["n_pred_edges"]}


def extra_alarm_pairs(context: causal_discovery.DiscoveryContext, graph: nx.DiGraph) -> list[dict]:
    """discovered adjacencies between two alarms that the expert graph lacks, with runs that reset both in one step."""
    kinds = dict(context.expert.nodes(data="kind"))
    out = []
    for a, b in graph.edges:
        both_alarms = kinds[a] == "Alarm" and kinds[b] == "Alarm"
        if context.expert.has_edge(a, b) or context.expert.has_edge(b, a) or not both_alarms:
            continue
        regimes = [r for r, ids in context.all_regimes.items()
                   if {a, b} <= set(discovery.regime_variables(context.view, ids))]
        ids = [rid for r in regimes for rid in context.all_regimes[r]]
        same = sum(any(step_a == step_b for step_a, x in discovery.state_changes(context.view, rid, a) if x == 0
                       for step_b, y in discovery.state_changes(context.view, rid, b) if y == 0) for rid in ids)
        out.append({"pair": (a, b), "regimes": regimes, "runs": len(ids), "same_reset": same})
    return out


# tables and figures


def agreement_table(results: dict[str, Result]) -> pd.DataFrame:
    columns = {}
    for sub, r in results.items():
        m, rnd, dag = r.metrics, r.random, r.dag_metrics
        columns[f"{sub.capitalize()} discovered"] = [
            f"{m['n_pred_edges']} ({m['n_true_edges']})", f"{m['skeleton_precision']:.2f}", f"{m['skeleton_recall']:.2f}",
            f"{m['skeleton_f1']:.2f}", f"{m['directed_f1']:.2f}", f"{m['shd']}", f"{m['n_undirected_pred']}",
            f"{dag['n_pred_edges']}", f"{dag['skeleton_f1']:.2f}"]
        columns[f"{sub.capitalize()} random"] = [
            f"{m['n_pred_edges']}", f"{rnd['skeleton_precision']:.2f}", f"{rnd['skeleton_recall']:.2f}",
            f"{rnd['skeleton_f1']:.2f}", f"{rnd['directed_f1']:.2f}", f"{rnd['shd']:.1f}", io.DASH, io.DASH, io.DASH]
    metrics = ["Adjacencies (expert graph)", "Skeleton precision", "Skeleton recall", "Skeleton F1", "Directed F1", "SHD",
               "Undirected edges", "SCM graph: edges", "SCM graph: skeleton F1"]
    return pd.DataFrame({"Metric": metrics, **columns})


def missed_table(results: dict[str, Result]) -> pd.DataFrame:
    rows = []
    for r in results.values():
        for edge in r.missed:
            u, v = edge["edge"]
            rows.append({"Missing edge": f"{u} → {v}", "Testable": ", ".join(edge["testable"]),
                         "Lost in": lost_in(edge["conditions"]), "Evidence": evidence(edge, u, v),
                         "Change and trade-off": change_text(edge)})
    return pd.DataFrame(rows, columns=["Missing edge", "Testable", "Lost in", "Evidence", "Change and trade-off"])


def lost_in(conditions: list[tuple[str, int]]) -> str:
    if not conditions:
        return "Unconditional independence test"
    return "Test given " + ", ".join(f"{name}_t" if lag == 0 else f"{name}_t{lag}" for name, lag in conditions)


def evidence(edge: dict, cause: str, effect: str) -> str:
    b = edge["behaviour"]
    if b["intervals_with_change"] == 0 and len(b["values"]) == 1:
        state = "on" if b["values"][0] == 1 else "off"
        return (f"{effect} stays {state} in all {b['intervals']} fault intervals of {cause}; "
                f"smallest p-value {edge['p_min']:.2f}")
    return f"{cause} and {b['partner']} drop in the same step in {b['same_step']} of {b['runs']} runs"


def change_text(edge: dict) -> str:
    t = edge["tradeoff"]
    if t is None:
        return "Not recovered by any evaluated setting"
    alphas = ", ".join(f"{a:g}" for a in t["alphas"])
    condition = (f"α = {alphas}" if len(t["alphas"]) == 1 else f"α in {{{alphas}}}") + (
        " and no FDR filter" if t["filter_off"] else "")
    return (f"Kept only with {condition}: {t['found']} of {t['reference']} expert adjacencies, "
            f"but {t['adjacencies']} adjacencies instead of {edge['selected_adjacencies']}")


def settings_table(results: dict[str, Result]) -> pd.DataFrame:
    rows = []
    for sub, r in results.items():
        for label, config, row, ties in (("Original", r.original, r.original_row, io.DASH),
                                          ("Selected", r.selected, r.selected_row,
                                           str(r.selected_row["n_tied_objective_and_first_tiebreak"]))):
            boot = r.bootstrap[config.config_id]
            rows.append({"Subsystem": sub.capitalize(),
                         "(τmax, α, κ)": f"{label} ({config.tau_max_s:g}, {config.pc_alpha:g}, {config.support_threshold:g})",
                         "Dir. F1": f"{float(row['directed_f1']):.2f}", "Skel. F1": f"{float(row['skeleton_f1']):.2f}",
                         "SHD": str(int(row["shd"])), "Edges": str(int(row["n_adjacencies"])), "Ties": ties,
                         "Boot. F1": f"{boot['directed_f1']:.2f}", "Jacc.": f"{boot['jaccard']:.2f}"})
    return pd.DataFrame(rows)


def save_graph_figures(sub: str, result: Result) -> None:
    number, title = FIGURES[sub]
    layout = LAYOUTS[sub]
    expert = result.context.expert
    io.save_figure(graphs.draw_comparison(expert, result.graph, layout), number, title)


# metric block


def metric_block(r: Result, stable_share: float) -> list[str]:
    m, rnd, dag, c = r.metrics, r.random, r.dag_metrics, r.selected
    boot, original = r.bootstrap[c.config_id], r.bootstrap[r.original.config_id]
    folds = r.folds
    changed = int((folds["config_id"] != r.selected_row["config_id"]).sum())
    row = io.metric_row
    lines = [
        f"graph agreement, {r.context.name} (selected settings: max lag {c.tau_max_s:g} s, alpha {c.pc_alpha:g}, "
        f"support {c.support_threshold:g})",
        row("adjacencies", f"{m['n_pred_edges']}",
            f"expert graph {m['n_true_edges']}, {int(r.selected_row['reference_edges_testable'])} testable"),
        row("skeleton f1", f"{m['skeleton_f1']:.2f}", f"random graphs {rnd['skeleton_f1']:.2f}"),
        row("directed f1", f"{m['directed_f1']:.2f}", f"random graphs {rnd['directed_f1']:.2f}"),
        row("structural hamming distance", f"{m['shd']}", f"random graphs {rnd['shd']:.1f}"),
        row("scm graph edges", f"{dag['n_pred_edges']}", f"skeleton f1 {dag['skeleton_f1']:.2f}"),
        row("held-out fold adjacencies", span(folds["n_adjacencies"], "d"),
            f"skeleton f1 {span(folds['skeleton_f1'])}, directed f1 {span(folds['directed_f1'])}"),
        row("held-out folds with other settings", f"{changed} of {len(folds)}"),
        row("bootstrap directed f1", f"{boot['directed_f1']:.2f}", f"jaccard {boot['jaccard']:.2f} over {boot['draws']} draws"),
        row(f"edges in >= {stable_share:.0%} of the draws", f"{stable_edges(boot, stable_share)} of {len(boot['full_edges'])}",
            f"original settings {stable_edges(original, stable_share)} of {len(original['full_edges'])}"),
    ]
    for edge in r.missed:
        u, v = edge["edge"]
        lines.append(row(f"missed {u} → {v}", f"{lower_first(lost_in(edge['conditions']))}, "
                         f"{lower_first(change_text(edge))}, adjacent in {edge['bootstrap_share']:.0%} of the draws"))
    for pair in r.extra_alarm_pairs:
        lines.append(row(f"extra {pair['pair'][0]} – {pair['pair'][1]}",
                         f"both alarms reset in the same step in {pair['same_reset']} of {pair['runs']} runs"))
    lines.append(sentence(r))
    return lines


def stable_edges(bootstrap: dict, share: float) -> int:
    """edges of the full-data graph found with the same orientation in at least the given share of draws."""
    return sum(bootstrap["edge_share"].get(key, 0) >= share for key in bootstrap["full_edges"])


def lower_first(text: str) -> str:
    return text[0].lower() + text[1:]


def span(values: pd.Series, kind: str = ".2f") -> str:
    low, high = values.astype(float).min(), values.astype(float).max()
    if kind == "d":
        return f"{int(low)}–{int(high)}"
    return f"{low:{kind}}–{high:{kind}}"


def sentence(r: Result) -> str:
    """one plain sentence built from the computed agreement."""
    m = r.metrics
    oriented = [(u, v) for u, v, a in r.graph.edges(data=True) if a.get("oriented", True)]
    correct = sum(r.context.expert.has_edge(u, v) for u, v in oriented)
    orientation = (f"and orients all its {len(oriented)} directed edges correctly" if correct == len(oriented)
                   else f"but orients only {correct} of its {len(oriented)} directed edges correctly")
    return (f"the discovered {r.context.name} graph recovers {m['skeleton_tp']} of {m['n_true_edges']} expert connections "
            f"{orientation}, judged against the expert graph used for selection.")
