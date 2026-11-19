"""graph comparison with the expert reference, dag conversion, perturbations and drawing.

graphs are compared as static graphs over one node set; self-loops are ignored and duplicate
expert edges count once. undirected discovered edges (oriented=False) count for the skeleton,
never as a correct direction. agreement with the expert graph is agreement with expert
knowledge, not proof of the true causal structure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from core.plots import COLORS, INK, TYPE_COLORS

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


# ---------------------------------------------------------------------------
# expert and discovered graph side by side (tikz and matplotlib)
# ---------------------------------------------------------------------------

# edge class: legend label, colour, line style
COMPARISON_CLASSES = {
    "correct direction": ("Correct direction", COLORS[0], "solid"),
    "reversed": ("Reversed direction", COLORS[2], "solid"),
    "extra": ("Directed, not in expert graph", COLORS[1], "solid"),
    "undirected, adjacency correct": ("Undirected, adjacent in expert graph", COLORS[0], "dashed"),
    "undirected, extra": ("Undirected, not in expert graph", COLORS[1], "dashed"),
    "missed": ("Expert edge not found", INK["muted"], "dotted"),
}
TIKZ_EDGE = {
    "expert": "->, draw=black!70, line width=0.7pt",
    "correct direction": "->, draw=figblue, line width=1.1pt",
    "reversed": "->, draw=figgreen, line width=1.1pt",
    "extra": "->, draw=figorange, line width=1.1pt",
    "undirected, adjacency correct": "draw=figblue, line width=1.1pt, dashed",
    "undirected, extra": "draw=figorange, line width=1.1pt, dashed",
    "missed": "->, draw=figgrey, line width=0.9pt, densely dotted",
}
TIKZ_HEADER = [
    r"\definecolor{figblue}{HTML}{2A78D6}\definecolor{figorange}{HTML}{EB6834}%",
    r"\definecolor{figgreen}{HTML}{1BAF7A}\definecolor{figgrey}{HTML}{898781}%",
    r"\begin{tikzpicture}[>=latex,",
    r"  st/.style={draw=figblue, fill=figblue!15, rounded corners=2pt, inner sep=2.5pt, font=\footnotesize\ttfamily},",
    r"  al/.style={draw=figorange, fill=figorange!15, rounded corners=2pt, inner sep=2.5pt, font=\footnotesize\ttfamily}]",
]
# matplotlib connection styles of the routed tikz edges
ROUTE_STYLES = {
    "out=0, in=0": "arc,angleA=0,angleB=0,armA={arm},armB={arm},rad=6",
    "out=-90, in=-90": "arc,angleA=-90,angleB=-90,armA={arm},armB={arm},rad=6",
    "bend left": "arc3,rad=-0.25",
}


@dataclass
class GraphLayout:
    """hand-placed node positions in cm (both panels), routed edges and the gap above the legend."""

    positions: dict[str, tuple[float, float]]
    routes: dict[frozenset, tuple[str, str, str]]  # pair -> (anchor of u, anchor of v, tikz options)
    y_scale: float
    legend_gap: float

    @property
    def height(self) -> float:
        return -self.y_scale * min(y for _, y in self.positions.values())

    @property
    def panel_shift(self) -> float:
        return -(self.height + 2.3)

    @property
    def centre(self) -> float:
        xs = [x for x, _ in self.positions.values()]
        return (min(xs) + max(xs)) / 2


def comparison_edges(discovered: nx.DiGraph, expert: nx.DiGraph) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """discovered edges with their class, then the expert edges that were not found, and the count per class."""
    edges, counts = [], {}
    for u, v, attrs in discovered.edges(data=True):
        cls = edge_class(u, v, attrs.get("oriented", True), expert)
        counts[cls] = counts.get(cls, 0) + 1
        edges.append((u, v, cls))
    missed = [(u, v, "missed") for u, v in expert.edges if not (discovered.has_edge(u, v) or discovered.has_edge(v, u))]
    counts["missed"] = len(missed)
    return edges + missed, counts


def comparison_titles(expert: nx.DiGraph, discovered: nx.DiGraph) -> tuple[str, str]:
    # a pair found in both directions is drawn as two arrows but counts as one adjacency
    adjacencies = len({frozenset(e) for e in discovered.edges})
    return (f"(a) Expert graph: {expert.number_of_edges()} edges",
            f"(b) Discovered summary graph, before DAG conversion: {adjacencies} adjacencies")


def _tex(name: str) -> str:
    return name.replace("_", r"\_")


def _tikz_panel(prefix: str, layout: GraphLayout, kinds: dict[str, str], edges: list[tuple[str, str, str]],
                title: str, top: float) -> list[str]:
    out = [rf"\node[anchor=south west, font=\normalsize] at (-1.7, {top + 0.55:.2f}) {{{title}}};"]
    for n, (x, y) in layout.positions.items():
        style = "al" if kinds[n] == "Alarm" else "st"
        out.append(rf"\node[{style}] ({prefix}{n.replace('_', '-')}) at ({x}, {layout.y_scale * y + top:.2f}) {{{_tex(n)}}};")
    for u, v, cls in edges:
        a, b = prefix + u.replace("_", "-"), prefix + v.replace("_", "-")
        route = layout.routes.get(frozenset({u, v}))
        if route:
            out.append(rf"\draw[{TIKZ_EDGE[cls]}] ({a}.{route[0]}) to[{route[2]}] ({b}.{route[1]});")
        else:
            out.append(rf"\draw[{TIKZ_EDGE[cls]}] ({a}) -- ({b});")
    return out


def comparison_tikz(expert: nx.DiGraph, discovered: nx.DiGraph, layout: GraphLayout, comment: str) -> str:
    """tikz picture: (a) the expert graph, (b) the discovered summary graph with edge classes, and a legend."""
    kinds = {n: expert.nodes[n]["kind"] for n in expert.nodes}
    edges, counts = comparison_edges(discovered, expert)
    title_a, title_b = comparison_titles(expert, discovered)
    lines = [f"% {comment}", *TIKZ_HEADER]
    lines += _tikz_panel("a-", layout, kinds, [(u, v, "expert") for u, v in expert.edges], title_a, 0)
    lines += _tikz_panel("b-", layout, kinds, edges, title_b, layout.panel_shift)
    rows = [rf"\tikz[baseline=-0.6ex]\draw[{TIKZ_EDGE[k]}] (0,0) -- (0.8,0); & {label} ({counts[k]})"
            for k, (label, _, _) in COMPARISON_CLASSES.items() if counts.get(k, 0) > 0]
    rows += [r"\tikz[baseline=-0.6ex]\node[st, minimum width=8mm] {};& State",
             r"\tikz[baseline=-0.6ex]\node[al, minimum width=8mm] {};& Alarm"]
    half = (len(rows) + 1) // 2
    legend_y = layout.panel_shift - layout.height - layout.legend_gap
    lines.append(rf"\node[anchor=north, font=\footnotesize] at ({layout.centre:.2f}, {legend_y:.2f}) "
                 r"{\begin{tabular}{@{}cl@{\hspace{6mm}}cl@{}}")
    lines += [rows[i] + " & " + (rows[i + half] if i + half < len(rows) else "&") + r" \\" for i in range(half)]
    lines += [r"\end{tabular}};", r"\end{tikzpicture}"]
    return "\n".join(lines) + "\n"


def draw_comparison(expert: nx.DiGraph, discovered: nx.DiGraph, layout: GraphLayout) -> plt.Figure:
    """the tikz picture drawn with matplotlib: same positions (1 data unit = 1 cm), edge classes and legend."""
    kinds = {n: expert.nodes[n]["kind"] for n in expert.nodes}
    edges, counts = comparison_edges(discovered, expert)
    titles = comparison_titles(expert, discovered)
    xs = [x for x, _ in layout.positions.values()]
    left, right = -1.8, max(xs) + 1.9
    bottom, top = layout.panel_shift - layout.height - layout.legend_gap - 0.3, 1.2
    fig = plt.figure(figsize=((right - left) / 2.54, (top - bottom) / 2.54))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.axis("off")
    for title, shift, panel in ((titles[0], 0.0, [(u, v, "expert") for u, v in expert.edges]),
                                (titles[1], layout.panel_shift, edges)):
        ax.text(-1.7, shift + 0.55, title, ha="left", va="bottom", fontsize=10)
        boxes = _draw_nodes(fig, ax, layout, kinds, shift)
        for u, v, cls in panel:
            _draw_edge(ax, boxes, layout.routes.get(frozenset({u, v})), u, v, cls)
    handles = [Line2D([], [], color=color, linestyle=style if style != "dotted" else (0, (1, 1.5)), linewidth=1.1,
                      label=f"{label} ({counts[k]})") for k, (label, color, style) in COMPARISON_CLASSES.items()
               if counts.get(k, 0) > 0]
    handles += [Line2D([], [], marker="s", markersize=9, linestyle="", markerfacecolor=_tint(TYPE_COLORS[kind]),
                       markeredgecolor=TYPE_COLORS[kind], label=label) for kind, label in (("Binary", "State"), ("Alarm", "Alarm"))]
    legend_y = layout.panel_shift - layout.height - layout.legend_gap
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(layout.centre, legend_y), bbox_transform=ax.transData,
              ncol=2, frameon=False, fontsize=8, handlelength=2.2, columnspacing=2.0)
    return fig


def _tint(color: str, share: float = 0.15) -> tuple[float, float, float]:
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(share * rgb + (1 - share))


def _draw_nodes(fig: plt.Figure, ax: plt.Axes, layout: GraphLayout, kinds: dict[str, str],
                shift: float) -> dict[str, tuple[float, float, float, float]]:
    """node boxes; returns their extent (x0, y0, x1, y1) in cm."""
    texts = {}
    for n, (x, y) in layout.positions.items():
        color = TYPE_COLORS["Alarm" if kinds[n] == "Alarm" else "Binary"]
        texts[n] = ax.text(x, layout.y_scale * y + shift, n, ha="center", va="center", fontsize=7.5, family="monospace",
                           bbox={"boxstyle": "round,pad=0.25,rounding_size=0.2", "facecolor": _tint(color),
                                 "edgecolor": color, "linewidth": 0.8})
    fig.canvas.draw()
    inverse = ax.transData.inverted()
    boxes = {}
    for n, text in texts.items():
        (x0, y0), (x1, y1) = inverse.transform(text.get_bbox_patch().get_window_extent())
        boxes[n] = (x0, y0, x1, y1)
    return boxes


def _anchor(box: tuple[float, float, float, float], name: str) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    xs = {"west": x0, "east": x1}
    ys = {"south": y0, "north": y1}
    x = next((xs[k] for k in xs if k in name), (x0 + x1) / 2)
    y = next((ys[k] for k in ys if k in name), (y0 + y1) / 2)
    return x, y


def _border_point(box: tuple[float, float, float, float], towards: tuple[float, float]) -> tuple[float, float]:
    """where the line from the box centre to a point leaves the box."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = towards[0] - cx, towards[1] - cy
    scale = min((x1 - cx) / abs(dx) if dx else np.inf, (y1 - cy) / abs(dy) if dy else np.inf)
    return cx + dx * scale, cy + dy * scale


def _draw_edge(ax: plt.Axes, boxes: dict, route: tuple[str, str, str] | None, u: str, v: str, cls: str) -> None:
    if cls == "expert":
        color, style, width = INK["secondary"], "solid", 0.7
    else:
        _, color, style = COMPARISON_CLASSES[cls]
        width = 0.9 if cls == "missed" else 1.1
    if route:
        start, end = _anchor(boxes[u], route[0]), _anchor(boxes[v], route[1])
        key = next(k for k in ROUTE_STYLES if route[2].startswith(k))
        looseness = float(route[2].split("looseness=")[1]) if "looseness=" in route[2] else 1.0
        connection = ROUTE_STYLES[key].format(arm=round(22 * looseness))
    else:
        centre_u = ((boxes[u][0] + boxes[u][2]) / 2, (boxes[u][1] + boxes[u][3]) / 2)
        centre_v = ((boxes[v][0] + boxes[v][2]) / 2, (boxes[v][1] + boxes[v][3]) / 2)
        start, end = _border_point(boxes[u], centre_v), _border_point(boxes[v], centre_u)
        connection = "arc3,rad=0"
    directed = not cls.startswith("undirected")
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>" if directed else "-", mutation_scale=8, color=color,
                                 linewidth=width, linestyle=style if style != "dotted" else (0, (1, 1.5)),
                                 connectionstyle=connection, shrinkA=0, shrinkB=0))
