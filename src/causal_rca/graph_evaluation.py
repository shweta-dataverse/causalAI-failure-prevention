"""compare graphs with the expert reference, random references, and graph perturbations.

conventions:

* graphs are compared as static graphs over one node set; self-loops are ignored and
  duplicate expert edges count once.
* a discovered edge is either oriented (a -> b) or undirected (oriented=False).
  undirected edges count for the skeleton but never as a correct direction.
* agreement with the expert graph is agreement with expert knowledge, not proof
  of the true causal structure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

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
    """skeleton and directed precision/recall/f1, shd and orientation accuracy.

    ``pairs`` restricts the comparison to a set of unordered node pairs (for example
    the pairs testable by discovery); None uses all pairs.
    """
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


def edge_comparison_table(predicted: nx.DiGraph, reference: nx.DiGraph) -> pd.DataFrame:
    """one row per pair in either graph: expert mark, predicted mark and outcome."""
    pred_marks, true_marks = _marks(simple_digraph(predicted)), _marks(simple_digraph(reference))
    rows = []
    for pair in sorted(set(pred_marks) | set(true_marks), key=lambda p: sorted(p)):
        a, b = sorted(pair)
        true, pred = true_marks.get(pair), pred_marks.get(pair)
        if true and pred:
            outcome = "correct direction" if true == pred else ("undirected" if pred == "undirected" else
                                                                ("both directions" if pred == "both" else "reversed"))
        elif true:
            outcome = "missed"
        else:
            outcome = "extra"
        rows.append({"a": a, "b": b, "expert": true, "discovered": pred, "outcome": outcome})
    return pd.DataFrame(rows)


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
    """keep oriented edges only and break directed cycles by dropping the lowest-support edge.

    undirected edges are not oriented by guesswork; they are dropped from the scm
    graph and listed in the returned log together with cycle-breaking removals.
    """
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
# perturbations of the expert graph (stage 8)
# ---------------------------------------------------------------------------


def perturb_graph(graph: nx.DiGraph, kind: str, fraction: float, seed: int) -> nx.DiGraph:
    """copy of a dag with a share of edges deleted, false edges inserted, or edges reversed.

    the result stays acyclic: inserted edges and reversals that would create a cycle
    are skipped. ``fraction`` is relative to the number of edges in ``graph``.
    """
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
