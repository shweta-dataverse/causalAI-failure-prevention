"""graph comparison with the expert reference, dag conversion, perturbations and drawing.

graphs are compared as static graphs over one node set; self-loops are ignored and duplicate
expert edges count once. undirected discovered edges (oriented=False) count for the skeleton,
never as a correct direction. agreement with the expert graph is agreement with expert
knowledge, not proof of the true causal structure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from core.plots import INK, TYPE_COLORS

Pair = frozenset


def simple_digraph(graph: nx.Graph, nodes: Sequence[str] | None = None) -> nx.DiGraph:
    """directed graph without self-loops or duplicate edges; keeps edge attributes of the first copy."""
    simple = nx.DiGraph()
    # keep node attributes (kind, io, ...) when no explicit node list is given
    simple.add_nodes_from(graph.nodes(data=True) if nodes is None else nodes)
    for u, v, attrs in graph.edges(data=True):
        if u != v and not simple.has_edge(u, v):
            simple.add_edge(u, v, **attrs)
    return simple


def _marks(graph: nx.DiGraph) -> dict[Pair, str]:
    """per unordered pair: 'a->b' style mark, 'undirected' or 'both'."""
    marks: dict[Pair, str] = {}
    for u, v, attrs in graph.edges(data=True):
        pair = Pair((u, v))
        if attrs.get("oriented", True) is False:
            marks[pair] = "undirected"
        elif pair in marks and marks[pair] not in ("undirected", f"{u}->{v}"):
            marks[pair] = "both"
        else:
            marks[pair] = f"{u}->{v}"
    return marks


def _prf(tp: int, n_pred: int, n_true: int) -> tuple[float, float, float]:
    precision = tp / n_pred if n_pred else np.nan
    recall = tp / n_true if n_true else np.nan
    f1 = 2 * precision * recall / (precision + recall) if n_pred and n_true and (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compare_graphs(
    predicted: nx.DiGraph, reference: nx.DiGraph, pairs: Iterable[Pair] | None = None
) -> dict[str, float]:
    """skeleton and directed precision/recall/f1, shd and orientation accuracy (optionally on given pairs)."""
    predicted, reference = simple_digraph(predicted), simple_digraph(reference)
    pred_marks, true_marks = _marks(predicted), _marks(reference)
    if pairs is not None:
        allowed = set(pairs)
        pred_marks = {p: m for p, m in pred_marks.items() if p in allowed}
        true_marks = {p: m for p, m in true_marks.items() if p in allowed}

    pred_skel, true_skel = set(pred_marks), set(true_marks)
    skel_tp = len(pred_skel & true_skel)
    sp, sr, sf = _prf(skel_tp, len(pred_skel), len(true_skel))

    # directed edges: a -> b counts once per direction; undirected edges never count
    pred_dir = {(u, v) for u, v, a in predicted.edges(data=True)
                if a.get("oriented", True) and Pair((u, v)) in pred_marks}
    true_dir = {(u, v) for u, v in reference.edges() if Pair((u, v)) in true_marks}
    dir_tp = len(pred_dir & true_dir)
    dp, dr, df = _prf(dir_tp, len(pred_dir), len(true_dir))

    common = pred_skel & true_skel
    wrong_marks = sum(pred_marks[p] != true_marks[p] for p in common)
    shd = len(pred_skel - true_skel) + len(true_skel - pred_skel) + wrong_marks
    single = [p for p in common if pred_marks[p] not in ("undirected", "both")]
    orientation_accuracy = (
        sum(pred_marks[p] == true_marks[p] for p in single) / len(single) if single else np.nan
    )
    return {
        "n_pred_edges": len(pred_skel),
        "n_true_edges": len(true_skel),
        "skeleton_tp": skel_tp,
        "skeleton_precision": sp,
        "skeleton_recall": sr,
        "skeleton_f1": sf,
        "n_pred_directed": len(pred_dir),
        "directed_tp": dir_tp,
        "directed_precision": dp,
        "directed_recall": dr,
        "directed_f1": df,
        "shd": shd,
        "n_undirected_pred": sum(m == "undirected" for m in pred_marks.values()),
        "orientation_accuracy": orientation_accuracy,
    }


def random_reference(
    reference: nx.DiGraph, n_edges: int, n_draws: int = 1000, seed: int = 0,
    pairs: Sequence[Pair] | None = None,
) -> pd.DataFrame:
    """metrics of random directed graphs with ``n_edges`` edges over the same nodes (chance level)."""
    rng = np.random.default_rng(seed)
    nodes = sorted(reference.nodes)
    candidates = [(u, v) for u in nodes for v in nodes if u != v]
    if pairs is not None:
        allowed = set(pairs)
        candidates = [(u, v) for u, v in candidates if Pair((u, v)) in allowed]
    rows = []
    for _ in range(n_draws):
        # one direction per pair, so the random graph has n_edges distinct pairs
        unordered = sorted({Pair(e) for e in candidates}, key=lambda p: sorted(p))
        chosen = rng.choice(len(unordered), size=min(n_edges, len(unordered)), replace=False)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for k in chosen:
            a, b = sorted(unordered[k])
            graph.add_edge(*((a, b) if rng.random() < 0.5 else (b, a)))
        rows.append(compare_graphs(graph, reference, pairs=pairs))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# turning a discovered graph into a dag for the scm
# ---------------------------------------------------------------------------


def to_dag(graph: nx.DiGraph) -> tuple[nx.DiGraph, pd.DataFrame]:
    """oriented edges only, cycles broken at the lowest-support edge; returns the dag and a removal log."""
    dag = nx.DiGraph()
    dag.add_nodes_from(graph.nodes)
    log = []
    for u, v, attrs in graph.edges(data=True):
        if attrs.get("oriented", True):
            dag.add_edge(u, v, **attrs)
        else:
            log.append({"source": u, "target": v, "action": "dropped (undirected)"})
    while True:
        try:
            cycle = nx.find_cycle(dag)
        except nx.NetworkXNoCycle:
            break
        weakest = min(cycle, key=lambda e: (dag.edges[e[0], e[1]].get("support", 0.0), e[0], e[1]))
        dag.remove_edge(weakest[0], weakest[1])
        log.append({"source": weakest[0], "target": weakest[1], "action": "dropped (breaks cycle)"})
    return dag, pd.DataFrame(log, columns=["source", "target", "action"])


# ---------------------------------------------------------------------------
# perturbations of the expert graph
# ---------------------------------------------------------------------------


def perturb_graph(graph: nx.DiGraph, kind: str, fraction: float, seed: int) -> nx.DiGraph:
    """copy of a dag with a share of edges deleted, inserted or reversed, kept acyclic."""
    if kind not in ("delete", "insert", "reverse"):
        raise ValueError(f"unknown perturbation {kind!r}")
    rng = np.random.default_rng(seed)
    base = simple_digraph(graph)
    if not nx.is_directed_acyclic_graph(base):
        raise ValueError("perturb_graph expects an acyclic graph")
    out = base.copy()
    edges = sorted(base.edges())
    n = int(round(fraction * len(edges)))
    if n == 0:
        return out

    if kind == "delete":
        for k in rng.choice(len(edges), size=min(n, len(edges)), replace=False):
            out.remove_edge(*edges[k])
    elif kind == "reverse":
        done = 0
        for k in rng.permutation(len(edges)):
            u, v = edges[k]
            out.remove_edge(u, v)
            out.add_edge(v, u)
            if nx.is_directed_acyclic_graph(out):
                done += 1
            else:
                out.remove_edge(v, u)
                out.add_edge(u, v)
            if done == n:
                break
    else:
        nodes = sorted(base.nodes)
        absent = [(u, v) for u in nodes for v in nodes if u != v and not base.has_edge(u, v) and not base.has_edge(v, u)]
        done = 0
        for k in rng.permutation(len(absent)):
            u, v = absent[k]
            out.add_edge(u, v)
            if nx.is_directed_acyclic_graph(out):
                done += 1
            else:
                out.remove_edge(u, v)
            if done == n:
                break
    return out


def scm_graph(edges: pd.DataFrame, nodes: Sequence[str]) -> nx.DiGraph:
    """scm graph from source/target rows; node order matters (it fixes the order of the random draws)."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(zip(edges["source"], edges["target"]))
    return graph


def edge_class(u: str, v: str, oriented: bool, reference: nx.DiGraph) -> str:
    """class of a discovered edge against the reference graph."""
    if not oriented:
        adjacent = reference.has_edge(u, v) or reference.has_edge(v, u)
        return "undirected, adjacency correct" if adjacent else "undirected, extra"
    if reference.has_edge(u, v):
        return "correct direction"
    if reference.has_edge(v, u):
        return "reversed"
    return "extra"


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def graph_layers(graph: nx.Graph) -> dict[Any, int] | None:
    """topological layer of each node, or None if the graph has a cycle."""
    simple = nx.DiGraph(graph)
    if not nx.is_directed_acyclic_graph(simple):
        return None
    return {node: layer for layer, nodes in enumerate(nx.topological_generations(simple)) for node in nodes}


def graph_layout(graph: nx.Graph, seed: int = 0) -> dict[Any, tuple[float, float]]:
    """left-to-right layers for acyclic graphs, a seeded spring layout otherwise."""
    simple = nx.DiGraph(graph)
    layers = graph_layers(simple)
    if layers is None:
        return nx.spring_layout(simple, seed=seed)
    nx.set_node_attributes(simple, layers, "layer")
    return nx.multipartite_layout(simple, subset_key="layer")


def draw_graph(
    graph: nx.Graph,
    ax: plt.Axes,
    node_types: dict[str, str] | None = None,
    highlight: Iterable[str] = (),
    title: str | None = None,
    node_size: int = 420,
    label_side: str = "alternate",
    font_size: float = 7,
    width_by_support: bool = False,
    edge_color: str | None = None,
    pos: dict | None = None,
) -> None:
    """draw a causal graph: node colour = variable type, dark ring = highlighted node.

    label_side: 'alternate' (above/below by column), 'above', or 'right'.
    """
    node_types = node_types or {}
    highlight = set(highlight)
    simple = nx.DiGraph(graph)
    # a fixed layout lets several graphs over the same nodes be compared side by side
    pos = pos if pos is not None else graph_layout(simple)
    nodes = list(simple.nodes)

    # oriented edges get arrows; edges marked oriented=False are dashed without arrows
    oriented = [(u, v) for u, v, a in simple.edges(data=True) if a.get("oriented", True)]
    unoriented = [(u, v) for u, v, a in simple.edges(data=True) if not a.get("oriented", True)]

    def widths(edges):
        if not width_by_support:
            return 1.2
        return [0.8 + 2.5 * simple.edges[e].get("support", 1.0) for e in edges]

    color = edge_color or INK["muted"]
    if oriented:
        nx.draw_networkx_edges(
            simple, pos, edgelist=oriented, ax=ax, edge_color=color, width=widths(oriented), arrows=True,
            arrowsize=12, node_size=node_size, connectionstyle="arc3,rad=0.08",
        )
    if unoriented:
        nx.draw_networkx_edges(
            simple, pos, edgelist=unoriented, ax=ax, edge_color=color, width=widths(unoriented), arrows=True,
            arrowstyle="-", style="dashed", node_size=node_size, connectionstyle="arc3,rad=0.08",
        )
    nx.draw_networkx_nodes(
        simple, pos, ax=ax, nodelist=nodes, node_size=node_size,
        node_color=[TYPE_COLORS.get(node_types.get(n), INK["muted"]) for n in nodes],
        edgecolors=[INK["primary"] if n in highlight else INK["surface"] for n in nodes],
        linewidths=[2.5 if n in highlight else 1.0 for n in nodes],
    )
    if label_side == "right":
        label_pos = {n: (x + 0.015, y) for n, (x, y) in pos.items()}
        nx.draw_networkx_labels(simple, label_pos, ax=ax, font_size=font_size, font_color=INK["primary"],
                                horizontalalignment="left")
    else:
        # odd columns of the drawn layout get their label below the node
        columns = sorted({round(x, 6) for x, _ in pos.values()})
        column_of = {n: columns.index(round(pos[n][0], 6)) for n in nodes}
        below = {n for n in nodes if label_side == "alternate" and column_of[n] % 2 == 1}
        for subset, dy, va in ((set(nodes) - below, 0.06, "bottom"), (below, -0.06, "top")):
            if subset:
                nx.draw_networkx_labels(
                    simple, {n: (pos[n][0], pos[n][1] + dy) for n in subset}, labels={n: n for n in subset},
                    ax=ax, font_size=font_size, font_color=INK["primary"], verticalalignment=va,
                )
    if title:
        ax.set_title(title)
    # extra vertical room so labels below the lowest nodes are not clipped
    ax.margins(0.12, 0.2)
    ax.axis("off")


def type_legend_handles(types: Iterable[str], highlight_label: str | None = None) -> list[Line2D]:
    """legend handles for node colours (and the highlight ring)."""
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=TYPE_COLORS[t],
               markeredgecolor=INK["surface"], label=t)
        for t in types
    ]
    if highlight_label:
        handles.append(
            Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=INK["surface"],
                   markeredgecolor=INK["primary"], markeredgewidth=2.5, label=highlight_label)
        )
    return handles
