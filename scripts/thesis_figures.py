"""thesis figures at print size, in sentence case, redrawn from the stage-15 results
(final evaluation with the selected discovery and scm settings).

the summary figures only read the stored stage-15 tables, so no number changes. the three
single-run figures (fault timeline, mechanism-break scores, counterfactual example) refit the
fold scm exactly as stage 15 does (selected smoothing and extra lag of that fold). the
counterfactual example reuses the independent random stream of stage 15, so it shows the same
2,000 conditional simulations as the results table.

usage: PYTHONPATH=src python scripts/thesis_figures.py
output: results/figures/thesis_*.pdf (and .png)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from causal_rca import counterfactuals as cfx
from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import graph_evaluation as ge
from causal_rca import interventions as itv
from causal_rca import preprocessing as pp
from causal_rca import root_cause_analysis as rca
from causal_rca import scm as sc

WIDTH = 5.5                  # text width of the thesis (14 cm), so 10 pt here is 10 pt in print
INTERVAL = 0.5
TABLES = data.RESULTS_DIR / "tables"
GRAPHS = ("expert", "discovered")
GRAPH_COLORS = {"expert": ev.COLORS[0], "discovered": ev.COLORS[1]}
GRAPH_LABELS = {"expert": "Expert graph", "discovered": "Discovered graph"}
BASELINE_COLOR = "#8A8A8A"          # grey for the non-causal baseline
CF_ENTROPY = 20261109        # independent stream of stage 14 and 15
CF_BATCHES = (10, 200)       # 2,000 simulations in 10 batches, as in stages 14 and 15
EXAMPLE_RUN = "coolant_exp_22_run_1"
EXAMPLE_FOLD = "exp_22"
STAGE = "15"                 # final evaluation with the selected settings
MUTED = ev.INK["muted"]


def style() -> None:
    ev.set_plot_style()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                         "axes.labelcolor": ev.INK["primary"], "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "legend.fontsize": 9.5, "xtick.color": ev.INK["primary"], "ytick.color": ev.INK["primary"]})


def save(fig: plt.Figure, name: str) -> None:
    ev.save_figure(fig, f"thesis_{name}")
    plt.close(fig)


def scenario_ticks(folds, counts=None) -> list[str]:
    if counts is None:
        return list(folds)
    return [f"{f}\nn = {int(counts[f])}" for f in folds]


def top_legend(fig: plt.Figure, handles, ncol: int) -> None:
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol, frameon=False)


# ---------------------------------------------------------------------------
# setting shared by the single-run figures
# ---------------------------------------------------------------------------


def coolant_setting() -> dict:
    view, _ = pp.load_view_data(data.PROCESSED_DIR, f"coolant_dt{INTERVAL:g}")
    splits = pd.read_csv(TABLES / "02_coolant_splits.csv")
    runs, experiments = data.load_fault_runs(), data.load_experiments()
    expert = ge.simple_digraph(data.load_expert_graph(subsystem="coolant"))
    nodes = sorted(expert.nodes)
    alarms = sorted(n for n in nodes if expert.nodes[n]["kind"] == "Alarm")
    return dict(view=view, splits=splits, runs=runs, experiments=experiments, expert=expert, nodes=nodes,
                alarms=alarms, manipulated=data.manipulated_by_recording(runs, experiments))


def fold_model(s: dict, fold: str) -> sc.TemporalSCM:
    # same fit as stage 15 (expert graph, smoothing and extra lag selected for this fold)
    sel = pd.read_csv(TABLES / f"{STAGE}_scm_selection.csv")
    row = sel[(sel.subsystem == "coolant") & (sel.graph == "expert") & (sel.fold == fold)].iloc[0]
    graph = s["expert"]
    lags = None if pd.isna(row.lag) else {(u, v): [int(row.lag)] for u, v in graph.edges if 3 * graph.in_degree(v) <= 20}
    return sc.fit_scm(s["view"], graph, data.fold_training_ids(s["splits"], fold), s["manipulated"], INTERVAL,
                      smoothing=float(row.alpha), lags=lags)


# ---------------------------------------------------------------------------
# figure 5.1: one fault run
# ---------------------------------------------------------------------------

TIME_LABELS = {"cause_start_at": "Fault start", "alarms_detected_at": "Alarms detected", "diagnosis_at": "Diagnosis",
               "cause_end_at": "Fault end", "alarms_resolved_at": "Alarms resolved"}


def figure_fault_timeline(s: dict) -> None:
    infos = [r for r in pp.discover_recordings(data.RAW_DIR) if r.recording_id == EXAMPLE_RUN]
    events = pp.mark_changes(pp.parse_values(pp.load_all_events(infos, data.RAW_DIR)))
    run = s["runs"].set_index("recording_id").loc[EXAMPLE_RUN]
    truth = s["experiments"].set_index(["subsystem", "experiment_id"]).loc[(run.subsystem, run.experiment_id)]
    types = {n: s["expert"].nodes[n]["kind"] for n in s["nodes"]}
    nodes = sorted(s["nodes"], key=lambda n: (types[n] == "Alarm", n), reverse=True)
    t_end = events.time_s.max()
    recorded = set(events.node)
    fig, ax = plt.subplots(figsize=(WIDTH, 5.0), layout="constrained")
    for i, node in enumerate(nodes):
        e = events[events.node == node].sort_values("time_s")
        if e.empty or e.value_num.isna().all():
            continue
        v = e.value_num.to_numpy(dtype=float)
        span = np.nanmax(v) - np.nanmin(v)
        v = (v - np.nanmin(v)) / span if span > 0 else np.zeros_like(v)
        color = ev.COLORS[1] if types[node] == "Alarm" else ev.COLORS[0]
        ax.step(np.append(e.time_s, t_end), i + 0.7 * np.append(v, v[-1]), where="post", color=color, linewidth=1.6)
    ax.set_yticks(np.arange(len(nodes)) + 0.35, [n if n in recorded else f"{n} (not recorded)" for n in nodes],
                  fontsize=8.5)
    for tick, node in zip(ax.get_yticklabels(), nodes):
        if node in truth.manipulated_vars:
            tick.set_fontweight("bold")
        if node not in recorded:
            tick.set_color(MUTED)
    # annotated times, numbered at the top; their names are in the legend
    for k, name in enumerate(data.CAUSE_TIMESTAMPS, start=1):
        ax.axvline(run[name], color=ev.INK["secondary"], linewidth=0.9, linestyle="--")
        ax.text(run[name], 1.005, str(k), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    ax.set_xlim(0, t_end)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time since recording start (s)")
    keys = [Line2D([], [], color="none", label=f"{k}: {TIME_LABELS[n]}") for k, n in enumerate(data.CAUSE_TIMESTAMPS, 1)]
    kinds = [Line2D([], [], color=ev.COLORS[0], label="State"), Line2D([], [], color=ev.COLORS[1], label="Alarm")]
    fig.legend(handles=kinds + keys, loc="outside lower center", ncol=4, frameon=False, fontsize=9,
               handlelength=1.2, columnspacing=1.2)
    save(fig, "fault_timeline")


# ---------------------------------------------------------------------------
# figures 6.1 and 6.2: graphs
# ---------------------------------------------------------------------------

EDGE_STYLE = {  # class: (label, colour, line style)
    "correct direction": ("Correct direction", ev.COLORS[0], "solid"),
    "reversed": ("Reversed direction", ev.COLORS[2], "solid"),
    "extra": ("Directed, not in expert graph", ev.COLORS[1], "solid"),
    "undirected, adjacency correct": ("Undirected, adjacent in expert graph", ev.COLORS[0], "dashed"),
    "undirected, extra": ("Undirected, not in expert graph", ev.COLORS[1], "dashed"),
    "missed": ("Expert edge not found", MUTED, "dotted"),
}


def edge_class(u: str, v: str, oriented: bool, ref: nx.DiGraph) -> str:
    if not oriented:
        return "undirected, adjacency correct" if (ref.has_edge(u, v) or ref.has_edge(v, u)) else "undirected, extra"
    if ref.has_edge(u, v):
        return "correct direction"
    if ref.has_edge(v, u):
        return "reversed"
    return "extra"


# node positions in cm, fixed by hand so that no edge passes behind another node; both panels
# of a subsystem use the same positions
POSITIONS = {
    "coolant": {
        "CLF_Filter_Ok": (0, 0), "CLF_A_700307": (9.3, 0),
        "CLT_Level_lt_Min": (0, -1), "CLT_A_700310": (9.3, -1),
        "F_Filter_Ok": (0, -2), "F_A_700313": (9.3, -2),
        "LP_Pump_Ok": (0, -3.5), "LP_A_700301": (9.3, -3), "LP_Pump_On": (4.65, -4),
        "HP_Pump_isOff": (9.3, -4.5),
        "HP_Pump_Ok": (0, -5.5), "HP_A_700304": (9.3, -5.5),
        "LT_Level_Ok": (0, -7), "LT_A_700317": (9.3, -6.5), "LT_Pump_Ok": (9.3, -7.5),
    },
    "probe": {
        "MPA_toInitPos": (0, 0), "MPA_InitPos": (4.2, 0), "MPC_close": (8.4, 0), "MPC_Closed": (11.4, -1.0),
        "MPC_open": (8.4, -1.6), "MPC_isOpen": (8.4, -3.2),
        "MPA_A_701124": (2.6, -1.9), "MPA_A_701125": (5.6, -1.6),
        "MPA_toWorkPos": (0, -4.6), "MPA_WorkPos": (4.2, -4.6), "MP_Inactive": (8.4, -5.2),
    },
}
Y_SCALE = 0.8               # vertical compression of the hand-placed rows, so both panels fit one page
# edges between nodes that would otherwise cross other nodes: (anchor u, anchor v, tikz options)
ROUTES = {
    frozenset({"CLF_A_700307", "LP_A_700301"}): ("east", "east", "out=0, in=0, looseness=1"),
    frozenset({"HP_A_700304", "LT_A_700317"}): ("east", "east", "out=0, in=0, looseness=1.2"),
    frozenset({"MPA_toWorkPos", "MP_Inactive"}): ("south", "south", "out=-90, in=-90, looseness=0.6"),
    frozenset({"MPC_close", "MPC_isOpen"}): ("east", "east", "out=0, in=0, looseness=0.7"),
    frozenset({"MPA_toWorkPos", "MPA_InitPos"}): ("north", "south west", "bend left=22"),
}
TIKZ_EDGE = {  # class: tikz style
    "expert": "->, draw=black!70, line width=0.7pt",
    "correct direction": "->, draw=figblue, line width=1.1pt",
    "reversed": "->, draw=figgreen, line width=1.1pt",
    "extra": "->, draw=figorange, line width=1.1pt",
    "undirected, adjacency correct": "draw=figblue, line width=1.1pt, dashed",
    "undirected, extra": "draw=figorange, line width=1.1pt, dashed",
    "missed": "->, draw=figgrey, line width=0.9pt, densely dotted",
}


def tex(name: str) -> str:
    return name.replace("_", r"\_")


def tikz_panel(prefix: str, pos: dict, kinds: dict, edges: list[tuple[str, str, str]], title: str, top: float) -> list[str]:
    out = [rf"\node[anchor=south west, font=\normalsize] at (-1.7, {top + 0.55:.2f}) {{{title}}};"]
    for n, (x, y) in pos.items():
        out.append(rf"\node[{'al' if kinds[n] == 'Alarm' else 'st'}] ({prefix}{n.replace('_', '-')}) at ({x}, {Y_SCALE * y + top:.2f}) {{{tex(n)}}};")
    for u, v, cls in edges:
        a, b = prefix + u.replace("_", "-"), prefix + v.replace("_", "-")
        route = ROUTES.get(frozenset({u, v}))
        if route:
            out.append(rf"\draw[{TIKZ_EDGE[cls]}] ({a}.{route[0]}) to[{route[2]}] ({b}.{route[1]});")
        else:
            out.append(rf"\draw[{TIKZ_EDGE[cls]}] ({a}) -- ({b});")
    return out


def figure_graphs(sub: str) -> None:
    """tikz code with the thesis fonts; edge classes are computed from the stored graphs."""
    from causal_rca import causal_discovery as cd
    expert = ge.simple_digraph(data.load_expert_graph(subsystem=sub))
    kinds = {n: expert.nodes[n]["kind"] for n in expert.nodes}
    table = pd.read_csv(TABLES / f"{STAGE}_{sub}_discovery_edges.csv")
    disc = nx.DiGraph(cd.edges_to_graph(table[table.graph_id == "discovered"], sorted(expert.nodes)))
    pos = POSITIONS[sub]
    assert set(pos) == set(expert.nodes)
    height = -Y_SCALE * min(y for _, y in pos.values())
    found, counts = [], {}
    for u, v, a in disc.edges(data=True):
        cls = edge_class(u, v, a.get("oriented", True), expert)
        counts[cls] = counts.get(cls, 0) + 1
        found.append((u, v, cls))
    missed = [(u, v, "missed") for u, v in expert.edges if not (disc.has_edge(u, v) or disc.has_edge(v, u))]
    counts["missed"] = len(missed)
    shift = -(height + 2.3)
    lines = [f"% generated by scripts/thesis_figures.py ({sub}); do not edit by hand",
             r"\definecolor{figblue}{HTML}{2A78D6}\definecolor{figorange}{HTML}{EB6834}%",
             r"\definecolor{figgreen}{HTML}{1BAF7A}\definecolor{figgrey}{HTML}{898781}%",
             r"\begin{tikzpicture}[>=latex,",
             r"  st/.style={draw=figblue, fill=figblue!15, rounded corners=2pt, inner sep=2.5pt, font=\footnotesize\ttfamily},",
             r"  al/.style={draw=figorange, fill=figorange!15, rounded corners=2pt, inner sep=2.5pt, font=\footnotesize\ttfamily}]"]
    lines += tikz_panel("a-", pos, kinds, [(u, v, "expert") for u, v in expert.edges],
                        f"(a) Expert graph: {expert.number_of_edges()} edges", 0)
    # a pair found in both directions is drawn as two arrows but counts as one adjacency
    adjacencies = len({frozenset((u, v)) for u, v in disc.edges})
    lines += tikz_panel("b-", pos, kinds, found + missed,
                        f"(b) Discovered summary graph, before DAG conversion: {adjacencies} adjacencies", shift)
    # legend: only classes that occur, with their counts
    labels = {k: lab for k, (lab, _, _) in EDGE_STYLE.items()}
    rows = [rf"\tikz[baseline=-0.6ex]\draw[{TIKZ_EDGE[k]}] (0,0) -- (0.8,0); & {labels[k]} ({counts[k]})"
            for k in EDGE_STYLE if counts.get(k, 0) > 0]
    rows += [r"\tikz[baseline=-0.6ex]\node[st, minimum width=8mm] {};& State",
             r"\tikz[baseline=-0.6ex]\node[al, minimum width=8mm] {};& Alarm"]
    half = (len(rows) + 1) // 2
    body = []
    for i in range(half):
        right = rows[i + half] if i + half < len(rows) else "&"
        body.append(rows[i] + " & " + right + r" \\")
    centre = (min(x for x, _ in pos.values()) + max(x for x, _ in pos.values())) / 2
    lines.append(rf"\node[anchor=north, font=\footnotesize] at ({centre:.2f}, {shift - height - (1.9 if sub == 'probe' else 0.7):.2f}) {{\begin{{tabular}}{{@{{}}cl@{{\hspace{{6mm}}}}cl@{{}}}}")
    lines += body
    lines.append(r"\end{tabular}};")
    lines.append(r"\end{tikzpicture}")
    path = data.RESULTS_DIR / "figures" / f"thesis_{sub}_graphs.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{sub}: {counts}")


# ---------------------------------------------------------------------------
# figure 6.3: state prediction on the normal recordings
# ---------------------------------------------------------------------------


def figure_scm() -> None:
    t = pd.read_csv(TABLES / f"{STAGE}_scm_prediction.csv")
    t = t[t.track == "N"]
    splits = ["train", "validation", "test"]
    names = {"train": "Training\n(in-sample)", "validation": "Validation", "test": "Test"}
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.0), layout="constrained")
    for ax, sub in zip(axes, ("coolant", "probe")):
        d = t[t.subsystem == sub]
        x = np.arange(3)
        for k, g in enumerate(GRAPHS):
            v = d[d.graph == g].set_index("split").loc[splits]
            bars = ax.bar(x + (k - 0.5) * 0.36, v.log_loss, width=0.36, color=GRAPH_COLORS[g])
            for rect, val in zip(bars, v.log_loss):
                ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
        n = d[d.graph == "expert"].set_index("split").loc[splits, "recordings"]
        ax.set_xticks(x, [f"{names[k]}\nn = {int(n[k])}" for k in splits])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, d.log_loss.max() * 1.22)
        ax.set_ylabel("Log loss (nats)")
        ax.set_title(f"({'ab'[('coolant', 'probe').index(sub)]}) {sub.capitalize()}", loc="left")
    top_legend(fig, [Patch(color=GRAPH_COLORS[g], label=GRAPH_LABELS[g]) for g in GRAPHS], 2)
    save(fig, "scm_normal")


# ---------------------------------------------------------------------------
# figure 6.4: mechanism-break scores in one run
# ---------------------------------------------------------------------------


def figure_surprise(s: dict, model: sc.TemporalSCM) -> None:
    series = sc.recording_series(s["view"], EXAMPLE_RUN)
    window = rca.analysis_window(series, s["alarms"], INTERVAL)
    cands = rca.candidates(series, s["nodes"], s["alarms"])
    causes = s["manipulated"][EXAMPLE_RUN]
    t = np.arange(1, len(series[cands[0]])) * INTERVAL
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9), layout="constrained")
    ax.axvspan(window[0] * INTERVAL, window[1] * INTERVAL, color=ev.INK["grid"], zorder=0)
    for k, v in enumerate(cands):
        cause = v in causes
        ax.plot(t, model.surprise(series, v), linestyle="-" if cause else ":", linewidth=2.0 if cause else 1.5,
                color=ev.COLORS[k % len(ev.COLORS)], label=v + (" (injected cause)" if cause else ""))
    onsets = rca.alarm_onsets(series, s["alarms"])
    for a, st in onsets.items():
        ax.axvline(st * INTERVAL, color=ev.INK["secondary"], linewidth=0.9, linestyle="--")
    ax.set_xlabel("Time since recording start (s)")
    ax.set_ylabel("Surprise (nats)")
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(color=ev.INK["grid"], label="Analysis window"),
                Line2D([], [], color=ev.INK["secondary"], linestyle="--", lw=0.9, label="First alarm activation")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    save(fig, "surprise_timeline")


# ---------------------------------------------------------------------------
# figure 6.5: rca per scenario
# ---------------------------------------------------------------------------


def figure_rca() -> None:
    per = pd.read_csv(TABLES / f"{STAGE}_rca_per_scenario.csv")
    # lstm + shap baseline (stage 16, main explanation variant), averaged over seeds per run, then per scenario
    lstm = pd.read_csv(TABLES / "16_lstm_shap_per_scenario.csv")
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.4), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1]})
    for ax, sub in zip(axes, ("coolant", "probe")):
        d = per[per.subsystem == sub]
        folds = sorted(d.fold.unique())
        x = np.arange(len(folds))
        for k, g in enumerate(GRAPHS):
            v = d[d.graph == g].set_index("fold").loc[folds]
            ax.bar(x + (k - 1) * 0.27, v.ap3, width=0.27, color=GRAPH_COLORS[g])
        b = lstm[lstm.subsystem == sub].set_index("fold").loc[folds]
        ax.bar(x + 0.27, b["ap@3"], width=0.27, color=BASELINE_COLOR)
        e = d[d.graph == "expert"].set_index("fold").loc[folds]
        ax.scatter(x, e.chance_ap3, marker="_", s=500, linewidths=2, color=ev.INK["primary"], zorder=3)
        ax.set_xticks(x, scenario_ticks(folds, e.runs))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("AP@3")
        ax.set_title(f"({'ab'[('coolant', 'probe').index(sub)]}) {sub.capitalize()}", loc="left")
    axes[1].set_xlabel("Held-out scenario")
    handles = [Patch(color=GRAPH_COLORS[g], label=f"Causal RCA, {GRAPH_LABELS[g].lower()}") for g in GRAPHS]
    handles.append(Patch(color=BASELINE_COLOR, label="LSTM + SHAP"))
    handles.append(Line2D([], [], marker="_", markersize=14, markeredgewidth=2, linestyle="", color=ev.INK["primary"],
                          label="Chance level"))
    top_legend(fig, handles, 2)
    save(fig, "rca_per_scenario")


def figure_shap_example() -> None:
    """shap attribution of the lstm anomaly score in one probe run (fixed rule: first run of exp_1, seed 0)."""
    imp = pd.read_csv(TABLES / "16_lstm_shap_importance.csv")
    rid = sorted(imp[(imp.subsystem == "probe") & (imp.fold == "exp_1")].recording_id.unique())[0]
    d = imp[(imp.recording_id == rid) & (imp.seed == 0)].sort_values("importance")
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3), layout="constrained")
    ax.barh([tex_free(v) for v in d.variable], d.importance,
            color=[ev.COLORS[1] if c else ev.COLORS[0] for c in d.is_cause])
    ax.set_xlabel("Summed absolute SHAP value of the anomaly score")
    ax.grid(axis="y", visible=False)
    top_legend(fig, [Patch(color=ev.COLORS[1], label="Injected cause"), Patch(color=ev.COLORS[0], label="Other candidate")], 2)
    save(fig, "shap_example")


def tex_free(v: str) -> str:
    return v


# ---------------------------------------------------------------------------
# figure 6.6: intervention analysis
# ---------------------------------------------------------------------------


def figure_interventions() -> None:
    rep = pd.read_csv(TABLES / f"{STAGE}_coolant_replay.csv")
    obs = rep[rep.observed_fired]
    folds = sorted(obs.fold.unique())
    x = np.arange(len(folds))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 7.6), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 0.75, 1.1]})

    # (a) modelled and unmodelled activations; identical for both graphs
    e = obs[obs.graph == "expert"]
    modelled = e[e.alarm_supported].groupby("fold").size().reindex(folds, fill_value=0)
    total = e.groupby("fold").size().reindex(folds)
    for g in GRAPHS:
        o = obs[obs.graph == g]
        assert (o[o.alarm_supported].groupby("fold").size().reindex(folds, fill_value=0) == modelled).all()
    ax = axes[0]
    ax.bar(x, modelled, width=0.5, color=ev.COLORS[0], label="Modelled alarm")
    ax.bar(x, total - modelled, bottom=modelled, width=0.5, color="white", edgecolor=MUTED, hatch="///",
           label="Unmodelled alarm")
    for i, f in enumerate(folds):
        ax.text(i, total[f] + 0.4, f"{modelled[f]} of {total[f]}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, total.max() * 1.25)
    ax.set_xticks(x, folds)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Activations")
    ax.set_title("(a) Observed alarm activations", loc="left")

    # (b) simulated activation probability of each modelled activation
    ax = axes[1]
    for k, g in enumerate(GRAPHS):
        d = obs[(obs.graph == g) & obs.alarm_supported]
        for i, f in enumerate(folds):
            v = d.loc[d.fold == f, "p_fires"]
            ax.scatter(np.full(len(v), i + (k - 0.5) * 0.3) + rng.uniform(-0.06, 0.06, len(v)), v, s=16,
                       color=GRAPH_COLORS[g], zorder=3)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks(x, scenario_ticks(folds, modelled))
    ax.set_ylabel("Probability")
    ax.set_title("(b) Simulated activation probability", loc="left")

    # (c) onset error of each modelled activation, with the median per scenario
    ax = axes[2]
    for k, g in enumerate(GRAPHS):
        d = obs[(obs.graph == g) & obs.alarm_supported & obs.abs_onset_error_s.notna()]
        for i, f in enumerate(folds):
            v = d.loc[d.fold == f, "abs_onset_error_s"]
            xo = i + (k - 0.5) * 0.3
            ax.scatter(np.full(len(v), xo) + rng.uniform(-0.06, 0.06, len(v)), v, s=16, color=GRAPH_COLORS[g],
                       alpha=0.8, zorder=3)
            ax.plot([xo - 0.12, xo + 0.12], [v.median()] * 2, color=ev.INK["primary"], linewidth=2, zorder=4)
    ax.axhline(INTERVAL, color=ev.INK["secondary"], linestyle="--", linewidth=1)
    ax.set_xticks(x, scenario_ticks(folds, modelled))
    ax.set_ylim(-0.15, 3.4)
    ax.set_ylabel("Onset error (s)")
    ax.set_xlabel("Held-out scenario")
    ax.set_title("(c) Onset error", loc="left")

    handles = [Patch(color=ev.COLORS[0], label="Modelled alarm"),
               Patch(facecolor="white", edgecolor=MUTED, hatch="///", label="Unmodelled alarm"),
               Line2D([], [], marker="o", linestyle="", color=GRAPH_COLORS["expert"], label=GRAPH_LABELS["expert"]),
               Line2D([], [], marker="o", linestyle="", color=GRAPH_COLORS["discovered"], label=GRAPH_LABELS["discovered"]),
               Line2D([], [], color=ev.INK["primary"], lw=2, label="Scenario median"),
               Line2D([], [], color=ev.INK["secondary"], lw=1, linestyle="--", label="One grid step (0.5 s)")]
    top_legend(fig, handles, 2)
    save(fig, "interventions")


# ---------------------------------------------------------------------------
# figure 6.7: counterfactual example with the 2,000 independent simulations of stage 15
# ---------------------------------------------------------------------------


def example_worlds(s: dict, model: sc.TemporalSCM, actions, recorded) -> tuple[dict, float, float]:
    run_ids = sorted(s["splits"].loc[s["splits"].split == "fault", "recording_id"])
    streams = dict(zip([(g, r) for g in GRAPHS for r in run_ids],
                       np.random.SeedSequence(CF_ENTROPY).spawn(len(GRAPHS) * len(run_ids))))
    series = sc.recording_series(s["view"], EXAMPLE_RUN)
    window = rca.analysis_window(series, s["alarms"], INTERVAL)
    outcome = [a for a in s["alarms"] if a in recorded and model.mechanisms[a].has_data]
    fixed = {v: x for v, x in cfx.unrecorded_context(model, series).items() if v not in {a.variable for a in actions}}
    interventions = {a.variable: (a.value, a.start_step) for a in actions}
    worlds = []
    # same batches and the same order of random draws as cfx.reevaluate_validity
    for child in streams[("expert", EXAMPLE_RUN)].spawn(CF_BATCHES[0]):
        rng = np.random.default_rng(child)
        itv.check_feasibility(actions, recorded=recorded)
        noise = model.abduct_noise(series, CF_BATCHES[1], rng)
        worlds.append(model.simulate(cfx.initial_state(model, series), noise, interventions=interventions, fixed=fixed))
    world = {v: np.concatenate([w[v] for w in worlds]) for v in worlds[0]}
    cf_any = np.max(np.stack([world[a][:, window[0]:] for a in outcome]), axis=0)
    observed = np.column_stack([np.nan_to_num(series[a][window[0]:]) for a in outcome])
    validity = float((cf_any.max(axis=1) == 0).mean())
    reduction = float(observed.max(axis=1).mean() - cf_any.mean())
    return world, validity, reduction


def figure_cf_example(s: dict, model: sc.TemporalSCM) -> None:
    cf = pd.read_csv(TABLES / f"{STAGE}_coolant_cf_runs.csv")
    row = cf[(cf.graph == "expert") & (cf.recording_id == EXAMPLE_RUN)].iloc[0]
    ranking = pd.read_csv(TABLES / f"{STAGE}_rca_rankings.csv")
    ranking = ranking[(ranking.subsystem == "coolant") & (ranking.graph == "expert") & (ranking.recording_id == EXAMPLE_RUN)]
    detection = dict(zip(ranking.variable, ranking.detection_step))
    chosen = [a.split("=")[0].removeprefix("do(") for a in row.actions.split(" + ")]
    actions = [itv.Action(v, itv.allowed_value(v), max(1, int(detection[v]))) for v in chosen]
    series = sc.recording_series(s["view"], EXAMPLE_RUN)
    recorded = {v for v in series if not np.isnan(series[v]).all()}
    world, validity, reduction = example_worlds(s, model, actions, recorded)
    assert abs(validity - row.validity_independent) < 1e-12, (validity, row.validity_independent)
    print(f"example: validity {validity:.4f} (table {row.validity_independent:.4f}), "
          f"reduction of alarm-active time {reduction:.4f}")
    alarms = [a for a in s["alarms"] if a in recorded]
    shown = chosen + alarms
    roles = {v: "action target" for v in chosen}
    roles.update({a: "modelled alarm" if model.mechanisms[a].has_data else "unmodelled alarm"
                  for a in alarms})
    t = np.arange(len(series[shown[0]])) * INTERVAL
    fig, axes = plt.subplots(len(shown), 1, figsize=(WIDTH, 1.35 * len(shown) + 1.0), sharex=True, layout="constrained")
    for ax, v in zip(axes, shown):
        ax.step(t, series[v], where="post", color=MUTED, linewidth=2.2)
        if model.mechanisms[v].has_data:
            ax.plot(t, world[v].mean(axis=0), color=ev.COLORS[0], linewidth=1.8)
        for a in actions:
            ax.axvline(a.start_step * INTERVAL, color=ev.COLORS[1], linewidth=1.4)
        ax.set_ylim(-0.1, 1.15)
        ax.set_yticks([0, 1])
        ax.set_title(f"{v} ({roles[v]})", loc="left", fontsize=9.5)
    axes[-1].set_xlabel("Time since recording start (s)")
    act = ", ".join(f"do({a.variable} = {int(a.value)}) at {a.start_step * INTERVAL:g} s" for a in actions)
    handles = [Line2D([], [], color=MUTED, lw=2.2, label="Observed state"),
               Line2D([], [], color=ev.COLORS[0], lw=1.8, label="Counterfactual probability of state 1"),
               Line2D([], [], color=ev.COLORS[1], lw=1.4, label=f"Action: {act}")]
    top_legend(fig, handles, 2)
    save(fig, "cf_example")


# ---------------------------------------------------------------------------
# figure 6.8: counterfactual validity and coverage
# ---------------------------------------------------------------------------


def figure_counterfactuals() -> None:
    cf = pd.read_csv(TABLES / f"{STAGE}_coolant_cf_runs.csv")
    folds = sorted(cf.fold.unique())
    x = np.arange(len(folds))
    rng = np.random.default_rng(0)
    counts = cf[cf.graph == "expert"].groupby("fold").size()
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.6), layout="constrained")
    ax = axes[0]
    for k, g in enumerate(GRAPHS):
        d = cf[cf.graph == g]
        for i, f in enumerate(folds):
            v = d.loc[d.fold == f, "validity_independent"].dropna()
            xo = i + (k - 0.5) * 0.3
            ax.scatter(np.full(len(v), xo) + rng.uniform(-0.06, 0.06, len(v)), v, s=18, color=GRAPH_COLORS[g], zorder=3)
            ax.plot([xo - 0.12, xo + 0.12], [v.mean()] * 2, color=ev.INK["primary"], linewidth=2, zorder=4)
    ax.axhline(0.9, color=ev.INK["secondary"], linestyle="--", linewidth=1)
    ax.set_ylim(0.78, 1.01)
    ax.set_xticks(x, scenario_ticks(folds, counts))
    ax.set_ylabel("Counterfactual validity")
    ax.set_title("(a) Validity", loc="left")
    ax = axes[1]
    for k, g in enumerate(GRAPHS):
        cov = cf[cf.graph == g].groupby("fold").alarm_coverage.mean().reindex(folds)
        ax.bar(x + (k - 0.5) * 0.36, cov, width=0.36, color=GRAPH_COLORS[g])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, scenario_ticks(folds, counts))
    ax.set_ylabel("Alarm coverage")
    ax.set_xlabel("Held-out scenario")
    ax.set_title("(b) Alarm coverage", loc="left")
    handles = [Line2D([], [], marker="o", linestyle="", color=GRAPH_COLORS[g], label=GRAPH_LABELS[g]) for g in GRAPHS]
    handles += [Line2D([], [], color=ev.INK["primary"], lw=2, label="Scenario mean"),
                Line2D([], [], color=ev.INK["secondary"], lw=1, linestyle="--", label="Threshold 0.90")]
    top_legend(fig, handles, 2)
    save(fig, "counterfactuals")


if __name__ == "__main__":
    style()
    s = coolant_setting()
    model = fold_model(s, EXAMPLE_FOLD)
    figure_fault_timeline(s)
    for sub in ("coolant", "probe"):
        figure_graphs(sub)
    figure_scm()
    figure_surprise(s, model)
    figure_rca()
    figure_shap_example()
    figure_interventions()
    figure_cf_example(s, model)
    figure_counterfactuals()
    print("done")
