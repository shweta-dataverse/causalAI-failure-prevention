"""shared plot style, colours and saving helpers for the notebooks."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from causal_rca.data import RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

# categorical colours, always used in this fixed order (colour-blind checked)
COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

# ink and chrome for text, axes and grid
INK: dict[str, str] = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#ffffff",
}

# one sequential hue (blue), light -> dark, for magnitudes such as shares
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)

SUBSYSTEM_COLORS = {"coolant": COLORS[0], "hydraulics": COLORS[1], "probe": COLORS[2]}
SOURCE_COLORS = {"real_op": COLORS[0], "dig_twin": COLORS[1]}
SOURCE_LABELS = {"real_op": "normal operation", "dig_twin": "fault runs"}
TYPE_COLORS = {
    "Binary": COLORS[0],
    "Alarm": COLORS[1],
    "Continuous": COLORS[2],
    "Counter": COLORS[3],
    "Categorical": COLORS[4],
}


def set_plot_style() -> None:
    """quiet thesis style: white background, light grid, no top/right spines."""
    plt.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.labelcolor": INK["secondary"],
            "axes.edgecolor": INK["axis"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": plt.cycler(color=list(COLORS)),
            "grid.color": INK["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": INK["secondary"],
            "ytick.color": INK["secondary"],
            "text.color": INK["primary"],
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("sequential_blue", list(SEQUENTIAL_BLUE))


def save_figure(fig: plt.Figure, name: str, formats: Sequence[str] = ("pdf", "png"), dpi: int = 200) -> list[Path]:
    """save a figure to results/figures/<name>.<format>."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        paths.append(path)
    return paths


def save_table(table: pd.DataFrame, name: str) -> Path:
    """save a table to results/tables/<name>.csv; list and dict cells become json text."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, (list, tuple, set, dict))).any():
            out[col] = out[col].map(_json_cell)
    path = TABLES_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    return path


def _json_cell(value: Any) -> Any:
    if isinstance(value, set):
        value = sorted(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    return value


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

    label_side: 'alternate' (above/below by layer, avoids clashes in chains),
    'above', or 'right' (best for tall layers such as the full graph).
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
        # alternate columns of the drawn layout: odd columns get their label below the node
        # (uses positions, so it also works when a graph is drawn on another graph's layout)
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
# integrated framework evaluation (notebook 09)
# ---------------------------------------------------------------------------


def evaluate_graph(
    graph_getter,
    view_data,
    splits: pd.DataFrame,
    manipulated: dict[str, set[str]],
    alarms: Sequence[str],
    nodes: Sequence[str],
    interval_s: float,
    onset_steps: dict[tuple[str, str], int],
    n_samples: int = 100,
    seed: int = 0,
    counterfactuals: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    """run rca, the replay of real manipulations and counterfactual explanations with one graph.

    ``counterfactuals=False`` skips the explanation search (for subsystems without
    actionable variables).

    ``graph_getter(fold)`` gives the acyclic graph for a held-out scenario. returns a summary
    and one row per held-out run.
    """
    import numpy as np

    from causal_rca import counterfactuals as cfx
    from causal_rca import interventions as itv
    from causal_rca import root_cause_analysis as rca
    from causal_rca.scm import recording_series

    metrics, rankings, scms = rca.evaluate_rca(view_data, splits, manipulated, graph_getter, alarms, nodes,
                                               interval_s, n_chance=200, seed=seed)
    margins = {}
    for rid, group in rankings.groupby("recording_id"):
        others = group.loc[~group["is_cause"], "score"]
        margins[rid] = group.loc[group["is_cause"], "score"].min() - (others.max() if len(others) else 0.0)

    replay_rows = []
    for fold, model in scms.items():
        for rid in splits.loc[splits["fold"] == fold, "recording_id"]:
            series = recording_series(view_data, rid)
            onsets = {v: onset_steps[(rid, v)] for v in manipulated[rid] if (rid, v) in onset_steps}
            replay_rows.append(itv.replay_manipulation(model, series, onsets, alarms, n_samples,
                                                       np.random.default_rng(seed)).assign(recording_id=rid))
    replay = pd.concat(replay_rows, ignore_index=True)
    fired = replay[replay["observed_fired"]]

    if counterfactuals:
        explanations, _ = cfx.explain_runs(scms, view_data, splits, rankings, manipulated, alarms, interval_s,
                                           n_samples=n_samples, seed=seed)
    else:
        explanations = pd.DataFrame({"recording_id": metrics["recording_id"]})

    per_run = metrics[["fold", "recording_id", "ap@3", "top1", "hit@3"]].copy()
    per_run["margin"] = per_run["recording_id"].map(margins)
    per_run["replay_p_fires"] = per_run["recording_id"].map(fired.groupby("recording_id")["p_fires"].mean())
    keep = ["recording_id", "status", "actions", "validity_probability", "improvement", "n_actions", "action_recall"]
    # runs without actionable candidates have no explanation columns
    explanations = explanations.reindex(columns=sorted(set(explanations.columns) | set(keep)))
    per_run = per_run.merge(explanations[keep], on="recording_id", how="left")

    supported = fired[fired["alarm_supported"]]
    summary = {
        "map@3": metrics["ap@3"].mean(),
        "top1": metrics["top1"].mean(),
        "median_margin": float(np.nanmedian(list(margins.values()))),
        "replay_p_fires_all": fired["p_fires"].mean(),
        "replay_p_fires_supported": supported["p_fires"].mean(),
        "replay_delay_error_s": float(np.nanmedian(np.abs(supported["predicted_median_delay_steps"]
                                                          - supported["observed_delay_steps"]))) * interval_s,
        "cf_valid_share": (explanations["status"] == "valid").mean(),
        "cf_validity_mean": explanations["validity_probability"].mean(),
        "cf_improvement_mean": explanations["improvement"].mean(),
        "cf_action_recall": explanations["action_recall"].mean(),
        "cf_mean_actions": explanations["n_actions"].mean(),
    }
    if not counterfactuals:
        # not computed, not zero
        summary.update({k: np.nan for k in summary if k.startswith("cf_")})
    return summary, per_run
