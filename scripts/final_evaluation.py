"""final evaluation (stage 14): expert vs discovered graph under two evaluation tracks.

track n (normal prediction): one scm per subsystem and graph, fitted on the 119 normal
training recordings plus all fault runs of the subsystem; scored on normal train (in-sample),
validation and test recordings. its fault results are never reported as held-out.
track f (fault generalisation): outer leave-one-scenario-out. for held-out scenario k the
graph (discovered_without_k), the scm and every fill value use training data without k and
without the related partners of k's runs (configs/related_recordings.csv, stage 17a; a
retrospective correction of 17 nov 2026 after a recording-similarity check).
rca, intervention replay and counterfactuals are computed on k's runs with that fold model.

the discovered graphs are the inherited notebook-03/10 outputs (original discovery settings,
reproduced exactly by stage 12); they are not rediscovered here. the scm uses the default
first-order mechanisms with jeffreys smoothing (a = 0.5). empty and perturbed graphs are not
part of this evaluation (historical results stay in 06-11 tables).

aggregation (declared in the evaluation protocol of 9 nov 2026, after earlier results had
been inspected): scm scores are averaged over rows within each recording-variable pair, then
over variables within a recording; normal splits average recordings equally; fault results
average runs within a scenario, then scenarios equally. pooled row-weighted scores are kept as
a separately labelled supplement.

selected mode (stage 15, the main pipeline of the thesis): the discovered graphs and the scm
settings come from scripts/selected_settings.py instead. every graph is the stage-12 selection
of its own scope (all scenarios for track n, without k for fold k), and every scm uses the
smoothing and extra lag selected by held-out log loss (normal validation for track n, inner
leave-one-scenario-out without k for fold k). outputs start with 15_; the 14_ files stay as
the original-settings comparison.

usage: PYTHONPATH=src python scripts/final_evaluation.py [schema] [main] [selected]
  schema    replace the stored all-fault view spec by the predeclared node-list schema
            (verifies that the encoded arrays do not change)
  main      every table and figure of the evaluation (default: both)
  selected  use the selected discovery and scm settings and write 15_ outputs
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from causal_rca import causal_discovery as cd
from causal_rca import counterfactuals as cfx
from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import graph_evaluation as ge
from causal_rca import interventions as itv
from causal_rca import preprocessing as pp
from causal_rca import root_cause_analysis as rca
from causal_rca import scm as sc

INTERVAL = 0.5
SEED = 0
REPLAY_WORLDS = 200          # notebook 07
CF_WORLDS = 200              # notebook 08 (selection budget)
CF_INDEPENDENT = (10, 200)   # 2,000 fresh worlds in 10 batches
CF_ENTROPY = 20261109        # independent stream, never used during selection
N_REFITS_CF = 5              # bootstrap refits per fold for the validity robustness check
N_REFITS_N = 10              # bootstrap refits of the track-n model
THRESHOLD = 0.9
GRAPHS = ("expert", "discovered")
SUBSYSTEMS = ("coolant", "probe")
TABLES = data.RESULTS_DIR / "tables"
SELECTED = "selected" in sys.argv[1:]
OUT = "15" if SELECTED else "14"
GRAPH_COLORS = {"expert": ev.COLORS[0], "discovered": ev.COLORS[1]}
DISCOVERY_SETTINGS = {"algorithm": "PCMCI+ (tigramite 5.2.10.1), regime-wise", "ci_test": "G-squared (analytic)",
                      "interval_s": 0.5, "tau_min": 0, "tau_max_s": 10.0, "pc_alpha": 0.01,
                      "max_conds_dim/px/py": "3/3/3", "fdr": "BH on lagged p-values per regime + post-hoc filter",
                      "support_threshold": 0.5}

if SELECTED:
    DISCOVERY_SETTINGS = {**DISCOVERY_SETTINGS, "tau_max_s / pc_alpha / support_threshold":
                          "selected per scope by the stage-12 rule (15_{sub}_discovery_selection.csv)"}
    for k in ("tau_max_s", "pc_alpha", "support_threshold"):
        DISCOVERY_SETTINGS.pop(k)

check_rows: list[dict] = []


def same_rows(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """both models scored on the same (recording, variable, step) targets, in any order."""
    key = ["recording_id", "variable", "step"]
    return len(a) == len(b) and a[key].sort_values(key).reset_index(drop=True).equals(b[key].sort_values(key).reset_index(drop=True))


def check(name: str, ok: bool, detail: str = "") -> None:
    check_rows.append({"check": name, "passed": bool(ok), "detail": detail})
    print(("  ok  " if ok else "  FAIL") + f"  {name}" + (f" ({detail})" if detail else ""))


def save(table: pd.DataFrame, name: str) -> None:
    ev.save_table(table, f"{OUT}_{name}")


def step(t: float) -> int:
    return int(np.ceil(round(t / INTERVAL, 9)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# setting
# ---------------------------------------------------------------------------


def setting(sub: str) -> dict:
    prefix = "02" if sub == "coolant" else "10"
    view, spec = pp.load_view_data(data.PROCESSED_DIR, f"{sub}_dt{INTERVAL:g}")
    splits = pd.read_csv(TABLES / f"{prefix}_{sub}_splits.csv")
    runs, experiments = data.load_fault_runs(), data.load_experiments()
    manipulated = data.manipulated_by_recording(runs, experiments)
    expert = ge.simple_digraph(data.load_expert_graph(subsystem=sub))
    nodes = sorted(expert.nodes)
    types = {n: ("Alarm" if expert.nodes[n]["kind"] == "Alarm" else "Binary") for n in nodes}
    alarms = sorted(n for n in nodes if types[n] == "Alarm")
    # original settings: the notebook configuration aggregated on the same grouped scopes (stage 12)
    edges_file = TABLES / (f"15_{sub}_discovery_edges.csv" if SELECTED else f"12_{sub}_original_edges.csv")
    edges = pd.read_csv(edges_file)
    edges = edges[~edges.graph_id.str.contains("_and_")]  # inner-fold graphs are only used by stage 15a
    raw = {gid: cd.edges_to_graph(g, nodes) for gid, g in edges.groupby("graph_id")}
    dags = {gid: ge.to_dag(g) for gid, g in raw.items()}
    folds = sorted(splits.loc[splits.split == "fault", "fold"].unique())
    # scm settings (smoothing, extra lag) per graph and fold; the original pipeline uses the baseline everywhere
    scm_settings = {(g, f): (sc.SMOOTHING, None) for g in GRAPHS for f in ["all", *folds]}
    if SELECTED:
        sel = pd.read_csv(TABLES / "15_scm_selection.csv")
        sel = sel[sel.subsystem == sub]
        scm_settings = {(r.graph, r.fold): (float(r.alpha), None if pd.isna(r.lag) else int(r.lag)) for r in sel.itertuples()}
    return dict(sub=sub, view=view, spec=spec, splits=splits, runs=runs, experiments=experiments,
                manipulated=manipulated, expert=expert, nodes=nodes, types=types, alarms=alarms,
                raw=raw, dags=dags, folds=folds, edges_file=edges_file, scm_settings=scm_settings)


def fit_for(s: dict, name: str, fold: str | None, train_ids) -> sc.TemporalSCM:
    """scm of graph ``name`` for a fold (None = track n) with that fold's smoothing and extra lag."""
    graph = graph_for(s, name, fold)
    alpha, lag = s["scm_settings"][(name, "all" if fold is None else fold)]
    # extra lag on every edge whose child keeps at most 2^20 table configurations (as stage 13)
    lags = None if lag is None else {(u, v): [lag] for u, v in graph.edges if 3 * graph.in_degree(v) <= 20}
    return sc.fit_scm(s["view"], graph, train_ids, s["manipulated"], INTERVAL, smoothing=alpha, lags=lags)


def graph_for(s: dict, name: str, fold: str | None) -> nx.DiGraph:
    """scm graph: the expert graph, or the scm dag of the discovered graph of that fold."""
    if name == "expert":
        return s["expert"]
    return s["dags"]["discovered" if fold is None else f"discovered_without_{fold}"][0]


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def run_schema() -> None:
    print("schema")
    for sub in SUBSYSTEMS:
        s = setting(sub)
        view = pp.ModelView.from_schema(s["types"], s["nodes"], "discrete")
        spec = s["spec"]
        same = (spec["columns"] == view.columns and spec["column_types"] == view.column_types
                and not spec["means"] and not spec["stds"] and not spec["vocabularies"] and not spec["exclusions"]
                and list(s["view"].var_names) == view.columns)
        check(f"{sub}: stored view = predeclared node list, no fitted state", same,
              f"stored view had train_ids of {len(spec['train_ids'])} recordings; columns {len(view.columns)}")
        if not same:
            raise SystemExit("stored view differs from the schema; rebuild the view instead")
        # binary columns are encoded by value only, so the stored arrays are exactly what the schema gives
        path = data.PROCESSED_DIR / f"{sub}_dt{INTERVAL:g}.json"
        new = {**spec, **view.to_dict(), "schema": "predeclared expert node list; no data-dependent step"}
        path.write_text(json.dumps(new, indent=2), encoding="utf-8")
        check(f"{sub}: view spec rewritten as fixed schema", True, str(path.relative_to(data.PROJECT_ROOT)))


# ---------------------------------------------------------------------------
# manifests and boundaries
# ---------------------------------------------------------------------------


def changes_in(view, rid: str) -> int:
    a, m = view.arrays[rid], view.masks[rid]
    total = 0
    for j in range(a.shape[1]):
        x = a[~m[:, j], j]
        total += int(np.count_nonzero(np.diff(x)))
    return total


def manifests(s: dict) -> pd.DataFrame:
    sub, splits, view = s["sub"], s["splits"], s["view"]
    man = splits[["recording_id", "source", "split", "fold"]].copy()
    man["state_changes"] = [changes_in(view, r) for r in man.recording_id]
    man["zero_change"] = man.state_changes == 0
    digest = {r: hashlib.sha1(view.arrays[r].tobytes() + view.masks[r].tobytes()).hexdigest() for r in man.recording_id}
    man["input_hash"] = man.recording_id.map(digest)
    save(man, f"{sub}_split_manifest")

    rows = []
    normal = {k: set(man.loc[man.split == k, "recording_id"]) for k in ("train", "validation", "test")}
    full_train = data.fold_training_ids(splits, None)
    for r in full_train:
        rows.append({"track": "N", "fold": "all", "recording_id": r, "role": "fit"})
    for k in ("validation", "test"):
        rows += [{"track": "N", "fold": "all", "recording_id": r, "role": f"score ({k})"} for r in sorted(normal[k])]
    check(f"{sub}: track N fit excludes normal validation/test", not (set(full_train) & (normal["validation"] | normal["test"])))
    partners = data.related_partners()
    for fold in s["folds"]:
        train = data.fold_training_ids(splits, fold)
        test = splits.loc[splits.fold == fold, "recording_id"].tolist()
        related = sorted(data.protected_ids(test) - set(test))
        rows += [{"track": "F", "fold": fold, "recording_id": r, "role": "fit"} for r in train]
        rows += [{"track": "F", "fold": fold, "recording_id": r, "role": "outer test"} for r in test]
        rows += [{"track": "F", "fold": fold, "recording_id": r, "role": "excluded (related to an outer-test run)"} for r in related]
        check(f"{sub} {fold}: fold fit and outer test disjoint", not (set(train) & set(test)),
              f"{len(train)} fit, {len(test)} test")
        check(f"{sub} {fold}: no related partner of an outer-test run in the fit",
              not any(partners.get(r, set()) & set(train) for r in test),
              f"excluded {', '.join(related) if related else 'none'}")
        check(f"{sub} {fold}: fold fit excludes normal validation/test",
              not (set(train) & (normal["validation"] | normal["test"])))
        # the fold graph is discovered without regime k
        ok = f"discovered_without_{fold}" in s["raw"]
        check(f"{sub} {fold}: fold-specific discovered graph present", ok)
        if SELECTED:
            # the fold graph and the fold scm settings were chosen without scenario k
            dsel = pd.read_csv(TABLES / f"15_{sub}_discovery_selection.csv", keep_default_na=False).set_index("graph_id")
            ssel = pd.read_csv(TABLES / "15_scm_selection.csv")
            ssel = ssel[(ssel.subsystem == sub) & (ssel.fold == fold)]
            used = {j for txt in ssel.selection_data for j in txt.removeprefix("inner scenarios ").split(",")}
            check(f"{sub} {fold}: graph and scm settings selected without the outer scenario",
                  dsel.loc[f"discovered_without_{fold}", "excluded"] == fold and fold not in used and len(ssel) == len(GRAPHS),
                  f"discovery {dsel.loc[f'discovered_without_{fold}', 'config_id']}; scm inner {sorted(used)}")
            # related partners of k's runs: removed from k's discovery scopes and from every inner fit and score
            inner = pd.read_csv(TABLES / "15_scm_selection_inner.csv", keep_default_na=False)
            inner = inner[(inner.subsystem == sub) & (inner.track == "F") & (inner.outer_fold == fold)]
            in_inner = set(",".join(inner.inner_scored_runs).split(",")) - {""}
            scopes = dsel[(dsel.excluded == fold) | dsel.excluded.str.startswith(fold + ",")]
            seen = set()
            for txt in scopes.excluded:
                seen |= set(sum(cd.define_regimes(splits, tuple(txt.split(","))).values(), []))
            inner_fit = set()
            for j in inner.inner_fold.unique():
                inner_fit |= set(data.fold_training_ids(splits, (fold, j)))
            check(f"{sub} {fold}: outer test and its related partners absent from its discovery scopes, inner fits and inner scores",
                  not (data.protected_ids(test) & (seen | inner_fit | in_inner)),
                  f"{len(scopes)} discovery scopes, {inner.inner_fold.nunique()} inner folds")
    save(pd.DataFrame(rows), f"{sub}_fold_manifest")

    # identical encoded inputs across normal splits (zero-change recordings)
    nm = man[man.source == "real_op"]
    groups = nm.groupby("input_hash").split.agg(lambda x: sorted(set(x)))
    crossing = groups[groups.map(len) > 1]
    check(f"{sub}: identical encoded normal inputs across splits (disclosed, not acquisition duplicates)", True,
          f"{len(crossing)} identical-input groups span splits; zero-change recordings "
          + ", ".join(f"{k} {int(nm[(nm.split == k)].zero_change.sum())}" for k in ("train", "validation", "test")))
    return man


def data_table(s: dict, man: pd.DataFrame) -> list[dict]:
    rows = []
    for k in ("train", "validation", "test"):
        part = man[(man.source == "real_op") & (man.split == k)]
        rows.append({"subsystem": s["sub"], "data": f"normal {k}", "group": "recording", "recordings": len(part),
                     "zero_change_recordings": int(part.zero_change.sum()),
                     "role": {"train": "fit (tracks N and F); discovery regime 'normal'",
                              "validation": "track N scoring; selection of the track-N scm settings (stage 15a)",
                              "test": "track N scoring only"}[k]})
    for fold in s["folds"]:
        part = man[man.fold == fold]
        rows.append({"subsystem": s["sub"], "data": f"fault scenario {fold}", "group": "scenario (all runs together)",
                     "recordings": len(part), "zero_change_recordings": int(part.zero_change.sum()),
                     "role": "outer test in its own fold; fit in the other folds (without related partners "
                             "of the held-out runs) and in track N"})
    return rows


def config_record(settings: dict[str, dict]) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=data.PROJECT_ROOT).stdout.strip()
    import importlib.metadata as md
    files = {}
    for sub, s in settings.items():
        for suffix in ("npz", "json"):
            p = data.PROCESSED_DIR / f"{sub}_dt{INTERVAL:g}.{suffix}"
            files[str(p.relative_to(data.PROJECT_ROOT))] = sha256(p)
        files[str(s["edges_file"].relative_to(data.PROJECT_ROOT))] = sha256(s["edges_file"])
    return {
        "stage": "15 final evaluation, selected settings" if SELECTED else "14 final evaluation", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": "PYTHONPATH=src python scripts/" + " ".join(Path(a).name if i == 0 else a for i, a in enumerate(sys.argv)),
        "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": git("status", "--short").splitlines(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: md.version(p) for p in ("tigramite", "numpy", "pandas", "networkx", "scipy", "matplotlib")},
        "input_sha256": files,
        "evaluation protocol": "leave-one-scenario-out; related partners of held-out runs excluded from fitting, "
                               "discovery and selection (configs/related_recordings.csv)",
        "related_recordings_sha256": sha256(data.RELATED_PATH),
        "discovery_settings (inherited graphs)": DISCOVERY_SETTINGS,
        "scm": {"mechanism": "binary transition table p(x_t | x_{t-1}, parents_t, parents_{t-1})"
                             + (" plus the parent at the selected extra lag" if SELECTED else ""),
                "settings (graph, fold) -> (smoothing, extra lag)": {sub: {f"{g}/{f}": list(v) for (g, f), v in s["scm_settings"].items()}
                                                                    for sub, s in settings.items()},
                "graph": "expert, or discovered scm dag of the fold"},
        "rca": {"score": "max surprise in window", "window": "first observed alarm onset - 60 s to last onset + 5 s",
                "candidates": "recorded non-alarm variables", "ties": "max, then summed surprise, then name",
                "chance_draws": 2000, "seed": SEED},
        "replay": {"worlds": REPLAY_WORLDS, "seed": SEED, "mode": "sustained manipulation from observed onset"},
        "counterfactuals": {"selection_worlds": CF_WORLDS, "selection_seed": SEED, "top_k": 3, "max_actions": 3,
                            "threshold": THRESHOLD, "independent_worlds": CF_INDEPENDENT[0] * CF_INDEPENDENT[1],
                            "independent_entropy": CF_ENTROPY, "refits_per_fold": N_REFITS_CF},
        "track_n_refits": N_REFITS_N,
        "aggregation": "rows -> recording-variable -> recording; normal: recordings equal; fault: runs within scenario, then scenarios equal",
    }


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------


def testable_pairs(s: dict, exclude: str | None) -> set:
    regimes = cd.define_regimes(s["splits"], () if exclude is None else (exclude,))
    pairs = set()
    for ids in regimes.values():
        vs = cd.regime_variables(s["view"], ids)
        pairs |= {ge.Pair((a, b)) for i, a in enumerate(vs) for b in vs[i + 1:]}
    return pairs


def graph_tables(s: dict) -> tuple[list[dict], list[dict]]:
    rows, removed = [], []
    n_pairs = len(s["nodes"]) * (len(s["nodes"]) - 1) // 2
    for fold in [None, *s["folds"]]:
        gid = "discovered" if fold is None else f"discovered_without_{fold}"
        pairs = testable_pairs(s, fold)
        stages = {"discovered raw (PCMCI+ output)": s["raw"][gid], "discovered SCM DAG (undirected edges removed)": s["dags"][gid][0]}
        for stage, g in stages.items():
            for scope, allowed in (("all pairs", None), ("testable pairs", pairs)):
                m = ge.compare_graphs(g, s["expert"], allowed)
                rows.append({"subsystem": s["sub"], "graph_id": gid, "scope_scenarios": "all scenarios" if fold is None else f"without {fold}",
                             "graph_stage": stage, "pair_scope": scope, "n_pairs": n_pairs if allowed is None else len(allowed),
                             "reference_edges_in_scope": m["n_true_edges"], **m})
        log = s["dags"][gid][1]
        removed += [{"subsystem": s["sub"], "graph_id": gid, **r} for r in log.to_dict("records")]
    return rows, removed


def edge_class(u: str, v: str, oriented: bool, ref: nx.DiGraph) -> str:
    if not oriented:
        return "undirected, adjacency correct" if (ref.has_edge(u, v) or ref.has_edge(v, u)) else "undirected, extra"
    if ref.has_edge(u, v):
        return "correct direction"
    if ref.has_edge(v, u):
        return "reversed"
    return "extra"


EDGE_STYLE = {"correct direction": (ev.COLORS[0], "solid"), "reversed": (ev.COLORS[2], "solid"),
              "extra": (ev.COLORS[1], "solid"), "undirected, adjacency correct": (ev.COLORS[0], "dashed"),
              "undirected, extra": (ev.COLORS[1], "dashed"), "missed": (ev.INK["grid"], "dotted")}


def figure_graphs(s: dict) -> None:
    expert, disc = s["expert"], s["raw"]["discovered"]
    pos = ev.graph_layout(expert)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8.8), layout="constrained")
    ev.draw_graph(expert, axes[0], node_types=s["types"], pos=pos, node_size=300, font_size=8.5,
                  title=f"(a) expert graph: {expert.number_of_edges()} edges")
    # discovered graph on the expert layout, coloured by agreement; missed expert edges dotted
    ax = axes[1]
    simple = nx.DiGraph(disc)
    counts = {}
    for (u, v, a) in simple.edges(data=True):
        cls = edge_class(u, v, a.get("oriented", True), expert)
        counts[cls] = counts.get(cls, 0) + 1
        color, style = EDGE_STYLE[cls]
        nx.draw_networkx_edges(simple, pos, edgelist=[(u, v)], ax=ax, edge_color=color, width=2.0, style=style,
                               arrows=True, arrowstyle="-|>" if a.get("oriented", True) else "-", arrowsize=10,
                               node_size=260, connectionstyle="arc3,rad=0.08")
    missed = [(u, v) for u, v in expert.edges if not (simple.has_edge(u, v) or simple.has_edge(v, u))]
    counts["missed"] = len(missed)
    if missed:
        nx.draw_networkx_edges(expert, pos, edgelist=missed, ax=ax, edge_color=EDGE_STYLE["missed"][0], width=1.4,
                               style="dotted", arrows=True, arrowsize=8, node_size=260, connectionstyle="arc3,rad=0.08")
    nodes_only = nx.DiGraph()
    nodes_only.add_nodes_from(expert.nodes)
    ev.draw_graph(nodes_only, ax, node_types=s["types"], pos=pos, node_size=300, font_size=8.5,
                  title=f"(b) discovered graph (raw PCMCI+ output): {simple.number_of_edges()} edges")
    handles = [Line2D([0], [0], color=EDGE_STYLE[k][0], linestyle=EDGE_STYLE[k][1], lw=2, label=f"{k} ({counts.get(k, 0)})")
               for k in EDGE_STYLE]
    handles += ev.type_legend_handles(sorted(set(s["types"].values())))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    ev.save_figure(fig, f"{OUT}_{s['sub']}_graphs")
    plt.close(fig)


# ---------------------------------------------------------------------------
# scm predictions and aggregation
# ---------------------------------------------------------------------------


def predictions(model: sc.TemporalSCM, view, ids, manipulated: dict) -> pd.DataFrame:
    """one row per recording, variable and step with p, y, prev, manipulated and support flags."""
    frames = []
    for r in dict.fromkeys(ids):
        s = sc.recording_series(view, r)
        for v, mech in model.mechanisms.items():
            if v not in s:
                continue
            p = model.transition_probs(s, v)
            y, prev = s[v][1:], s[v][:-1]
            ok = ~np.isnan(p) & ~np.isnan(y)
            if not ok.any():
                continue
            feats = sc.parent_features(s, mech.parents, model.fills, mech.lags)[ok]
            codes = (feats.astype(int) @ (1 << np.arange(mech.width))) if mech.width else np.zeros(ok.sum(), dtype=int)
            pv = prev[ok].astype(int)
            seen = np.array([mech.counts[st][c] > 0 for st, c in zip(pv, codes)]) if mech.has_data else np.zeros(ok.sum(), bool)
            frames.append(pd.DataFrame({"recording_id": r, "variable": v, "step": np.flatnonzero(ok) + 1,
                                        "p": p[ok], "y": y[ok], "prev": prev[ok],
                                        "manipulated": v in manipulated.get(r, set()),
                                        "configuration_seen": seen, "mechanism_has_data": mech.has_data}))
    out = pd.concat(frames, ignore_index=True)
    out["log_loss"] = -np.log(np.where(out.y == 1, out.p, 1 - out.p))
    out["brier"] = (out.p - out.y) ** 2
    out["change"] = out.y != out.prev
    return out


def macro(pred: pd.DataFrame, scenario: dict | None) -> dict:
    """declared aggregation: rows -> recording-variable -> recording -> (scenario) -> overall."""
    def chain(rows: pd.DataFrame, col: str) -> tuple[float, int]:
        if rows.empty:
            return np.nan, 0
        rv = rows.groupby(["recording_id", "variable"])[col].mean()
        rec = rv.groupby(level="recording_id").mean()
        if scenario is None:
            return float(rec.mean()), len(rec)
        return float(rec.groupby(rec.index.map(scenario)).mean().mean()), len(rec)
    ll, n_rec = chain(pred, "log_loss")
    br, _ = chain(pred, "brier")
    cll, n_rec_change = chain(pred[pred.change], "log_loss")
    return {"log_loss": ll, "brier": br, "change_log_loss": cll, "recordings": n_rec,
            "recordings_with_changes": n_rec_change, "rows": len(pred), "change_rows": int(pred.change.sum()),
            "pooled_log_loss": float(pred.log_loss.mean()) if len(pred) else np.nan,
            "pooled_brier": float(pred.brier.mean()) if len(pred) else np.nan,
            "pooled_change_log_loss": float(pred.loc[pred.change, "log_loss"].mean()) if pred.change.any() else np.nan,
            "share_rows_unseen_configuration": float((~pred.configuration_seen).mean()) if len(pred) else np.nan,
            "share_rows_mechanism_without_data": float((~pred.mechanism_has_data).mean()) if len(pred) else np.nan}


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------


def fault_onsets(s: dict) -> dict:
    events = pp.mark_changes(pp.parse_values(pp.load_all_events(
        [r for r in pp.discover_recordings(data.RAW_DIR) if r.subsystem == s["sub"]], data.RAW_DIR)))
    events = events[events.source == "dig_twin"]
    changes = data.fault_variable_changes(events, s["runs"][s["runs"].subsystem == s["sub"]], s["experiments"])
    return {(r, v): step(t) for r, v, t in zip(changes.recording_id, changes.variable, changes.first_change_after_start_s)
            if not np.isnan(t)}


def activated_alarms(series: dict, alarms) -> list[str]:
    out = []
    for a in alarms:
        x = series.get(a)
        if x is None:
            continue
        ok = ~np.isnan(x)
        if ((x[1:] == 1) & (x[:-1] == 0) & ok[1:] & ok[:-1]).any():
            out.append(a)
    return out


def run_main() -> None:
    t_start = time.perf_counter()
    settings = {sub: setting(sub) for sub in SUBSYSTEMS}
    record = config_record(settings)
    data_rows, graph_rows, removed_rows = [], [], []
    scm_rows, refit_rows, scm_var_rows, rca_runs, rca_rank = [], [], [], [], []
    models = {}
    for sub, s in settings.items():
        print(f"\n== {sub}")
        man = manifests(s)
        data_rows += data_table(s, man)
        g_rows, rem = graph_tables(s)
        graph_rows += g_rows
        removed_rows += rem
        figure_graphs(s)
        scenario = dict(zip(s["splits"].recording_id, s["splits"].fold))

        # track n: one full-fit model per graph, scored per normal split
        full_train = data.fold_training_ids(s["splits"], None)
        normal = {k: s["splits"].loc[s["splits"].split == k, "recording_id"].tolist() for k in ("train", "validation", "test")}
        preds_n = {}
        for g in GRAPHS:
            model = fit_for(s, g, None, full_train)
            for k, ids in normal.items():
                p = predictions(model, s["view"], ids, s["manipulated"])
                preds_n[(g, k)] = p
                scm_rows.append({"subsystem": sub, "track": "N", "graph": g, "split": k,
                                 "status": "in-sample" if k == "train" else "held out from fit",
                                 "variables": "all", **macro(p, None)})
                per_var = p.groupby("variable").agg(rows=("p", "size"), changes=("change", "sum"),
                                                    log_loss=("log_loss", "mean"), brier=("brier", "mean"))
                scm_var_rows += [{"subsystem": sub, "track": "N", "graph": g, "split": k, **r}
                                 for r in per_var.reset_index().to_dict("records")]
            # refit stability with multiplicity, scored on normal validation and test only
            if g == "expert":
                rng = np.random.default_rng(SEED)
                for b in range(N_REFITS_N):
                    sample = list(rng.choice(full_train, size=len(full_train), replace=True))
                    refit = fit_for(s, g, None, sample)
                    for k in ("validation", "test"):
                        refit_rows.append({"subsystem": sub, "graph": g, "refit": b, "split": k,
                                           "unique_recordings_in_sample": len(set(sample)),
                                           **macro(predictions(refit, s["view"], normal[k], s["manipulated"]), None)})
        for k in normal:
            a, b = preds_n[("expert", k)], preds_n[("discovered", k)]
            check(f"{sub} track N {k}: expert and discovered scored on identical rows",
                  same_rows(a, b), f"{len(a)} rows")
        # reproduction of the notebook-05/10 per-variable held-out fit (pooled validation + test)
        if sub == "coolant" and not SELECTED:
            old = pd.read_csv(TABLES / "05_coolant_fit_normal.csv")
            if "graph" in old.columns:
                pooled = pd.concat([preds_n[("expert", "validation")], preds_n[("expert", "test")]])
                mine = pooled.groupby("variable").log_loss.mean()
                o = old[old.graph == "expert"].set_index("variable").log_loss
                common = o.index.intersection(mine.index)
                diff = float((mine[common] - o[common]).abs().max()) if len(common) else np.nan
                check("coolant: notebook-05 normal held-out log loss per variable reproduced (pooled)", diff < 1e-9, f"max diff {diff:.2e}")

        # track f: fold models per graph (rca refits them exactly as notebook 06/10)
        for g in GRAPHS:
            metrics, rankings, scms = rca.evaluate_rca(s["view"], s["splits"], s["manipulated"],
                                                       lambda fold, g=g: graph_for(s, g, fold), s["alarms"], s["nodes"],
                                                       INTERVAL, seed=SEED,
                                                       fit_model=lambda fold, train, g=g: fit_for(s, g, fold, train))
            models[(sub, g)] = (scms, rankings)
            rca_runs.append(metrics.assign(subsystem=sub, graph=g, model_id=lambda d: f"{sub}-{g}-without_" + d.fold))
            rca_rank.append(rankings.assign(subsystem=sub, graph=g))
            if not SELECTED:
                # only folds whose training data is not changed by the related-recording exclusion can reproduce
                old_name = f"06_coolant_run_metrics_{g}" if sub == "coolant" else f"10_probe_rca_runs_{g}"
                old = pd.read_csv(TABLES / f"{old_name}.csv").set_index("recording_id")
                same = [f for f in s["folds"] if data.fold_training_ids(s["splits"], f) == data.fold_training_ids(s["splits"], f, exclude_related=False)]
                m = metrics[metrics.fold.isin(same)]
                diff = float((m.set_index("recording_id")["ap@3"] - old.loc[m.recording_id, "ap@3"].to_numpy()).abs().max())
                check(f"{sub} {g}: rca per-run ap@3 reproduces {old_name} in the folds unchanged by the exclusion",
                      diff < 1e-12, f"max diff {diff:.2e}; folds {', '.join(same)}")
            for fold, model in scms.items():
                test = s["splits"].loc[s["splits"].fold == fold, "recording_id"].tolist()
                p = predictions(model, s["view"], test, s["manipulated"])
                for part, rows in (("non-manipulated (mechanism prediction)", p[~p.manipulated]),
                                   ("manipulated (mechanism overridden)", p[p.manipulated])):
                    scm_rows.append({"subsystem": sub, "track": "F", "graph": g, "split": f"held-out {fold}",
                                     "status": "outer test", "variables": part, **macro(rows, None)})
                models.setdefault((sub, g, "pred"), []).append(p.assign(fold=fold))
        for g in GRAPHS:
            p = pd.concat(models[(sub, g, "pred")], ignore_index=True)
            for part, rows in (("non-manipulated (mechanism prediction)", p[~p.manipulated]),
                               ("manipulated (mechanism overridden)", p[p.manipulated])):
                scm_rows.append({"subsystem": sub, "track": "F", "graph": g, "split": "all held-out scenarios (scenario macro)",
                                 "status": "outer test", "variables": part, **macro(rows, scenario)})
        a = pd.concat(models[(sub, "expert", "pred")]); b = pd.concat(models[(sub, "discovered", "pred")])
        check(f"{sub} track F: expert and discovered scored on identical rows",
              same_rows(a, b),
              f"{len(a)} rows")

    save(pd.DataFrame(data_rows), "data_splits")
    save(pd.DataFrame(graph_rows), "graph_agreement")
    save(pd.DataFrame(removed_rows), "dag_conversion_removals")
    scm_table = pd.DataFrame(scm_rows)
    save(scm_table, "scm_prediction")
    save(pd.DataFrame(scm_var_rows), "scm_prediction_per_variable")
    save(pd.DataFrame(refit_rows), "scm_refit_stability")
    runs_table = pd.concat(rca_runs, ignore_index=True)
    save(runs_table, "rca_runs")
    save(pd.concat(rca_rank, ignore_index=True), "rca_rankings")
    rca_summary(runs_table)
    figure_scm(scm_table)

    # coolant: interventions and counterfactuals
    s = settings["coolant"]
    interventions(s, models)
    counterfactuals(s, models)

    figure_split()
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    record["runtime_s"] = round(time.perf_counter() - t_start, 1)
    (TABLES / f"{OUT}_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    save(pd.DataFrame(check_rows), "checks")
    print(f"\nfinished in {record['runtime_s']} s; {sum(not r['passed'] for r in check_rows)} failed checks")


# ---------------------------------------------------------------------------
# rca summary
# ---------------------------------------------------------------------------


def rca_summary(runs: pd.DataFrame) -> None:
    scored = runs[runs.scored]
    per_scen = (scored.groupby(["subsystem", "graph", "fold"])
                .agg(runs=("recording_id", "size"), candidates=("n_candidates", "mean"), causes=("n_causes", "mean"),
                     top1=("top1", "mean"), ap3=("ap@3", "mean"), recall3=("recall@3", "mean"),
                     chance_top1=("chance_top1", "mean"), chance_ap3=("chance_ap@3", "mean")).reset_index())
    save(per_scen, "rca_per_scenario")
    rows = []
    for (sub, g), d in scored.groupby(["subsystem", "graph"]):
        ps = per_scen[(per_scen.subsystem == sub) & (per_scen.graph == g)]
        lo, hi = rca.bootstrap_ci(d["ap@3"], seed=SEED)
        rows.append({"subsystem": sub, "graph": g, "runs_scored": len(d),
                     "runs_total": int(((runs.subsystem == sub) & (runs.graph == g)).sum()),
                     "scenarios": len(ps),
                     "top1_run_mean": d.top1.mean(), "map3_run_mean": d["ap@3"].mean(),
                     "map3_run_mean_ci_low (runs, ignores scenario dependence)": lo, "map3_run_mean_ci_high": hi,
                     "recall3_run_mean": d["recall@3"].mean(), "hit3_run_mean": d["hit@3"].mean(), "mrr_run_mean": d.rr.mean(),
                     "top1_scenario_macro": ps.top1.mean(), "ap3_scenario_macro": ps.ap3.mean(),
                     "chance_top1_run_mean": d.chance_top1.mean(), "chance_map3_run_mean": d["chance_ap@3"].mean(),
                     "chance_ap3_scenario_macro": ps.chance_ap3.mean(),
                     "candidates_min": int(d.n_candidates.min()), "candidates_max": int(d.n_candidates.max())})
    save(pd.DataFrame(rows), "rca_summary")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained", gridspec_kw={"width_ratios": [4, 6]})
    for ax, sub in zip(axes, SUBSYSTEMS):
        d = per_scen[per_scen.subsystem == sub]
        folds = sorted(d.fold.unique())
        x = np.arange(len(folds))
        for k, g in enumerate(GRAPHS):
            v = d[d.graph == g].set_index("fold").loc[folds]
            ax.bar(x + (k - 0.5) * 0.38, v.ap3, width=0.38, color=GRAPH_COLORS[g], label=f"{g} graph")
        chance = d[d.graph == "expert"].set_index("fold").loc[folds]
        ax.scatter(x, chance.chance_ap3, marker="_", s=600, color=ev.INK["primary"], zorder=3, label="chance (same candidates)")
        n = d[d.graph == "expert"].set_index("fold").loc[folds, "runs"]
        ax.set_xticks(x, [f"{f.removeprefix('exp_')}\nn={int(n[f])}" for f in folds])
        ax.set_ylim(0, 1.05)
        summ = pd.DataFrame(rows).set_index(["subsystem", "graph"])
        ax.set_title(f"({'ab'[SUBSYSTEMS.index(sub)]}) {sub}", loc="left")
    axes[0].set_ylabel("AP@3 (mean over runs of the scenario)")
    axes[1].legend(loc="upper left", fontsize=7.5, markerscale=0.35)
    axes[0].set_xlabel("held-out scenario (exp_)"); axes[1].set_xlabel("held-out scenario (exp_)")
    ev.save_figure(fig, f"{OUT}_rca_per_scenario")
    plt.close(fig)


def figure_scm(table: pd.DataFrame) -> None:
    t = table[table.track == "N"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained", sharey=False)
    splits = ["train", "validation", "test"]
    for ax, sub in zip(axes, SUBSYSTEMS):
        d = t[t.subsystem == sub]
        x = np.arange(3)
        for k, g in enumerate(GRAPHS):
            v = d[d.graph == g].set_index("split").loc[splits]
            bars = ax.bar(x + (k - 0.5) * 0.38, v.log_loss, width=0.38, color=GRAPH_COLORS[g], label=f"{g} graph")
            for rect, val in zip(bars, v.log_loss):
                ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        n = d[d.graph == "expert"].set_index("split").loc[splits, "recordings"]
        ax.set_xticks(x, [f"{k}{' (in-sample)' if k == 'train' else ''}\nn={int(n[k])}" for k in splits])
        ax.set_title(f"({'ab'[SUBSYSTEMS.index(sub)]}) {sub}", loc="left")
    axes[0].set_ylabel("log loss [nats]")
    axes[0].legend(loc="upper left", fontsize=7.5)
    ev.save_figure(fig, f"{OUT}_scm_normal_splits")
    plt.close(fig)


# ---------------------------------------------------------------------------
# interventions (coolant)
# ---------------------------------------------------------------------------


def interventions(s: dict, models: dict) -> None:
    print("\n== coolant interventions")
    onsets = fault_onsets(s)
    rows = []
    for g in GRAPHS:
        scms, _ = models[("coolant", g)]
        for fold, model in scms.items():
            for rid in s["splits"].loc[s["splits"].fold == fold, "recording_id"]:
                series = sc.recording_series(s["view"], rid)
                on = {v: onsets[(rid, v)] for v in s["manipulated"][rid] if (rid, v) in onsets}
                res = itv.replay_manipulation(model, series, on, s["alarms"], REPLAY_WORLDS, np.random.default_rng(SEED))
                rows.append(res.assign(graph=g, fold=fold, recording_id=rid, model_id=f"coolant-{g}-without_{fold}",
                                       worlds=REPLAY_WORLDS))
    rep = pd.concat(rows, ignore_index=True)
    rep["observed_delay_s"] = rep.observed_delay_steps * INTERVAL
    rep["predicted_delay_s"] = rep.predicted_median_delay_steps * INTERVAL
    rep["abs_onset_error_s"] = (rep.predicted_delay_s - rep.observed_delay_s).abs()
    rep["never_activated_in_simulation"] = rep.p_fires == 0
    save(rep, "coolant_replay")
    if not SELECTED:
        old = pd.read_csv(TABLES / "07_coolant_replay.csv")
        m = rep.merge(old, on=["graph", "recording_id", "alarm"], suffixes=("", "_old"))
        diff = float((m.p_fires - m.p_fires_old).abs().max())
        check("coolant: replay p_fires reproduces notebook 07", diff < 1e-12 and len(m) == len(rep), f"{len(m)} pairs, max diff {diff:.2e}")
    obs = rep[rep.observed_fired]
    neg = rep[~rep.observed_fired]
    check("coolant: negative replay examples (recorded alarms that did not activate)", True,
          f"{len(neg)} negatives: false-alarm specificity {'cannot be assessed' if len(neg) == 0 else 'assessable'}")
    summary = []
    for (g, scope), d in [((g, "all scenarios"), obs[obs.graph == g]) for g in GRAPHS] + \
                         [((g, f), obs[(obs.graph == g) & (obs.fold == f)]) for g in GRAPHS for f in s["folds"]]:
        sup = d[d.alarm_supported]
        scored = sup[sup.p_fires > 0]
        summary.append({"graph": g, "scope": scope, "observed_activations": len(d), "supported": len(sup),
                        "unsupported (no mechanism data)": len(d) - len(sup),
                        "activation_probability_supported_mean": sup.p_fires.mean() if len(sup) else np.nan,
                        "miss_probability_supported_mean": (1 - sup.p_fires).mean() if len(sup) else np.nan,
                        "supported_never_activated": int((sup.p_fires == 0).sum()),
                        "onset_error_scored_pairs": int(scored.predicted_delay_s.notna().sum()),
                        "onset_abs_error_median_s": float(np.nanmedian(scored.abs_onset_error_s)) if len(scored) else np.nan,
                        "onset_abs_error_mean_s": float(np.nanmean(scored.abs_onset_error_s)) if len(scored) else np.nan,
                        "negatives (not activated in recording)": int(((rep.graph == g) & ~rep.observed_fired & (True if scope == "all scenarios" else rep.fold == scope)).sum())})
    summ = pd.DataFrame(summary)
    save(summ, "coolant_replay_summary")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
    folds = s["folds"]
    x = np.arange(len(folds))
    for k, g in enumerate(GRAPHS):
        d = summ[(summ.graph == g) & (summ.scope != "all scenarios")].set_index("scope").loc[folds]
        off = (k - 0.5) * 0.38
        # share of observed activations: predicted by a supported mechanism (weighted by activation
        # probability) and not modelled (alarm without training data, never simulated)
        predicted = d.supported / d.observed_activations * d.activation_probability_supported_mean
        unsupported = (d.observed_activations - d.supported) / d.observed_activations
        axes[0].bar(x + off, predicted, width=0.38, color=GRAPH_COLORS[g], label=f"{g} graph: supported, simulated")
        axes[0].bar(x + off, unsupported, bottom=predicted, width=0.38, color=ev.INK["grid"], edgecolor=GRAPH_COLORS[g],
                    hatch="//", label="no mechanism data (not simulated)" if k == 0 else None)
        for i, (_, r) in enumerate(d.iterrows()):
            axes[0].text(i + off, 1.02, f"{r.supported}/{r.observed_activations}", ha="center", fontsize=7)
        err = d.onset_abs_error_median_s.fillna(0)
        axes[1].bar(x + off, err, width=0.38, color=GRAPH_COLORS[g], label=f"{g} graph")
        for i, (_, r) in enumerate(d.iterrows()):
            axes[1].text(i + off, err.iloc[i] + 0.03, f"{r.onset_abs_error_median_s:.1f}\nn={r.onset_error_scored_pairs}",
                         ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x, folds); axes[1].set_xticks(x, folds)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("share of observed alarm activations")
    axes[0].set_title("(a) observed alarm activations", loc="left")
    axes[1].set_ylim(0, max(1.6, float(summ.onset_abs_error_median_s.max()) + 0.5))
    axes[1].set_ylabel("median |simulated - recorded onset| [s]")
    axes[1].set_title("(b) onset error, supported activations", loc="left")
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    axes[1].legend(loc="upper left", fontsize=7.5)
    ev.save_figure(fig, f"{OUT}_coolant_interventions")
    plt.close(fig)


# ---------------------------------------------------------------------------
# counterfactuals (coolant)
# ---------------------------------------------------------------------------


def parse_actions(text: str, ranking: pd.DataFrame) -> list[itv.Action]:
    detection = dict(zip(ranking.variable, ranking.detection_step))
    names = [a.split("=")[0].removeprefix("do(") for a in text.split(" + ")]
    return [itv.Action(v, itv.allowed_value(v), max(1, int(detection[v]))) for v in names]


def counterfactuals(s: dict, models: dict) -> None:
    print("\n== coolant counterfactuals")
    root = np.random.SeedSequence(CF_ENTROPY)
    run_ids = sorted(s["splits"].loc[s["splits"].split == "fault", "recording_id"])
    streams = dict(zip([(g, r) for g in GRAPHS for r in run_ids], root.spawn(len(GRAPHS) * len(run_ids))))
    rows, refit_rows = [], []
    for g in GRAPHS:
        scms, rankings = models[("coolant", g)]
        expl, tried = cfx.explain_runs(scms, s["view"], s["splits"], rankings, s["manipulated"], s["alarms"], INTERVAL,
                                       n_samples=CF_WORLDS, seed=SEED)
        if not SELECTED:
            old = pd.read_csv(TABLES / f"08_coolant_explanations_{g}.csv").set_index("recording_id")
            same = (expl.set_index("recording_id").actions.fillna("") == old.loc[expl.recording_id, "actions"].fillna("").to_numpy()).all()
            diff = float((expl.set_index("recording_id").validity_probability - old.loc[expl.recording_id, "validity_probability"].to_numpy()).abs().max())
            check(f"coolant {g}: counterfactual selection reproduces notebook 08", bool(same) and diff < 1e-12, f"max validity diff {diff:.2e}")
        save(tried.assign(graph=g), f"coolant_cf_tried_{g}")
        for _, r in expl.iterrows():
            fold, rid = r.fold, r.recording_id
            model = scms[fold]
            series = sc.recording_series(s["view"], rid)
            ranking = rankings[rankings.recording_id == rid].sort_values("rank")
            window = rca.analysis_window(series, s["alarms"], INTERVAL)
            recorded = {v for v in series if not np.isnan(series[v]).all()}
            activated = activated_alarms(series, s["alarms"])
            covered = [a for a in activated if model.mechanisms[a].has_data]
            base = {"graph": g, "fold": fold, "recording_id": rid, "model_id": f"coolant-{g}-without_{fold}",
                    "status_selection": r.status, "actions": r.actions, "n_actions": r.get("n_actions", 0),
                    "selection_worlds": CF_WORLDS, "validity_selection": r.get("validity_probability", np.nan),
                    "improvement_selection": r.get("improvement", np.nan),
                    "factual_alarm_share": r.get("factual_alarm_share", np.nan),
                    "action_precision": r.action_precision, "action_recall": r.action_recall, "true_causes": r.true_causes,
                    "activated_alarms": ", ".join(activated), "covered_alarms": ", ".join(covered),
                    "uncovered_alarms": ", ".join(a for a in activated if a not in covered),
                    "alarm_coverage": len(covered) / len(activated) if activated else np.nan,
                    "rca_top3": " > ".join(ranking.variable.head(3)),
                    "window_start_s": window[0] * INTERVAL if window else np.nan}
            if not isinstance(r.actions, str):
                rows.append({**base, "independent_worlds": 0})
                continue
            acts = parse_actions(r.actions, ranking)
            outcome = [a for a in s["alarms"] if a in recorded and model.mechanisms[a].has_data]
            feasible, _ = itv.check_feasibility(acts, recorded=recorded)
            res = cfx.reevaluate_validity(model, series, acts, outcome, window[0], recorded, streams[(g, rid)],
                                          n_batches=CF_INDEPENDENT[0], batch_size=CF_INDEPENDENT[1])
            # alarm-time improvement on the same fresh worlds is not returned; re-estimate with one fresh batch
            imp = cfx.counterfactual(model, series, acts, outcome, window[0], CF_INDEPENDENT[1],
                                     np.random.default_rng(streams[(g, rid)].spawn(1)[0]), recorded)
            rows.append({**base, "independent_worlds": res["n_worlds"], "validity_independent": res["validity_independent"],
                         "wilson_low": res["wilson_low"], "wilson_high": res["wilson_high"], "feasible": feasible,
                         "valid_independent": bool(feasible and res["validity_independent"] >= THRESHOLD),
                         "borderline (interval contains 0.90)": bool(res["wilson_low"] < THRESHOLD <= res["wilson_high"]),
                         "improvement_fresh": imp.improvement,
                         "action_start_s": "; ".join(f"{a.variable} at {a.start_step * INTERVAL:g} s" for a in acts)})
            # robustness to mechanism estimation: bootstrap refits of the fold training data (with multiplicity)
            train = data.fold_training_ids(s["splits"], fold)
            for b in range(N_REFITS_CF):
                sample = list(np.random.default_rng(100 + b).choice(train, size=len(train), replace=True))
                refit = fit_for(s, g, fold, sample)
                out_refit = [a for a in s["alarms"] if a in recorded and refit.mechanisms[a].has_data]
                res_b = cfx.counterfactual(refit, series, acts, out_refit, window[0], CF_WORLDS,
                                           np.random.default_rng(1000 + b), recorded)
                refit_rows.append({"graph": g, "fold": fold, "recording_id": rid, "refit": b,
                                   "unique_recordings": len(set(sample)), "draws": len(sample),
                                   "worlds": CF_WORLDS, "validity": res_b.validity_probability})
    cf = pd.DataFrame(rows)
    save(cf, "coolant_cf_runs")
    refits = pd.DataFrame(refit_rows)
    save(refits, "coolant_cf_refits")
    scen = dict(zip(s["splits"].recording_id, s["splits"].fold))
    summ = []
    for g in GRAPHS:
        d = cf[cf.graph == g]
        rf = refits[refits.graph == g]
        rf_run = rf.groupby("recording_id").validity.min()
        summ.append({"graph": g, "runs": len(d), "runs_with_actions": int(d.actions.notna().sum()),
                     "valid_share_selection (200 worlds)": (d.status_selection == "valid").mean(),
                     "valid_share_independent (2000 fresh worlds)": d.valid_independent.fillna(False).astype(bool).mean(),
                     "validity_independent_mean": d.validity_independent.mean(),
                     "validity_independent_scenario_macro": d.groupby("fold").validity_independent.mean().mean(),
                     "borderline_runs": int(d["borderline (interval contains 0.90)"].fillna(False).astype(bool).sum()),
                     "alarm_coverage_run_mean": d.alarm_coverage.mean(),
                     "alarm_coverage_scenario_macro": d.groupby("fold").alarm_coverage.mean().mean(),
                     "alarms_covered_pooled": f"{sum(len(x.split(', ')) for x in d.covered_alarms if x)}/{sum(len(x.split(', ')) for x in d.activated_alarms if x)}",
                     "runs_all_alarms_covered": int((d.alarm_coverage == 1).sum()),
                     "actions_mean": d.n_actions.mean(), "improvement_mean (fresh worlds)": d.improvement_fresh.mean(),
                     "action_precision_mean": d.action_precision.mean(), "action_recall_mean": d.action_recall.mean(),
                     "refit_validity_mean": rf.validity.mean(), "refit_validity_min": rf.validity.min(),
                     "runs_valid_in_all_refits": int((rf_run >= THRESHOLD).sum())})
    save(pd.DataFrame(summ), "coolant_cf_summary")
    examples(cf[cf.graph == "expert"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    folds = s["folds"]
    x = np.arange(len(folds))
    rng = np.random.default_rng(SEED)
    for k, g in enumerate(GRAPHS):
        d = cf[cf.graph == g]
        off = (k - 0.5) * 0.38
        means = d.groupby("fold").validity_independent.mean().reindex(folds)
        axes[0].bar(x + off, means, width=0.38, color=GRAPH_COLORS[g], alpha=0.55, label=f"{g} graph (scenario mean)")
        for i, f in enumerate(folds):
            v = d.loc[d.fold == f, "validity_independent"].dropna()
            axes[0].scatter(np.full(len(v), i + off) + rng.uniform(-0.08, 0.08, len(v)), v, s=14, color=GRAPH_COLORS[g], zorder=3)
        cov = d.groupby("fold").alarm_coverage.mean().reindex(folds)
        axes[1].bar(x + off, cov, width=0.38, color=GRAPH_COLORS[g], label=f"{g} graph")
    axes[0].axhline(THRESHOLD, color=ev.INK["primary"], linewidth=1, linestyle="--", label="declared threshold 0.90")
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_xticks(x, folds); axes[1].set_xticks(x, folds)
    axes[0].set_ylabel("validity (2,000 fresh worlds)")
    axes[0].set_title("(a) validity of the selected actions", loc="left")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("supported / activated alarms")
    axes[1].set_title("(b) alarm coverage per run", loc="left")
    axes[0].legend(loc="lower left", fontsize=7)
    ev.save_figure(fig, f"{OUT}_coolant_counterfactuals")
    plt.close(fig)


def examples(cf: pd.DataFrame) -> None:
    """three runs chosen by a fixed rule, not by appearance: best-covered valid run, the least
    covered run, and the run with the lowest independent validity."""
    d = cf.copy()
    picks = []
    valid = d[d.valid_independent.fillna(False).astype(bool)].sort_values(["alarm_coverage", "recording_id"], ascending=[False, True])
    if len(valid):
        picks.append(("success: valid, highest alarm coverage", valid.iloc[0]))
    low = d.sort_values(["alarm_coverage", "recording_id"]).iloc[0]
    picks.append(("limitation: lowest alarm coverage (unsupported alarms)", low))
    worst = d.sort_values(["validity_independent", "recording_id"]).iloc[0]
    picks.append(("weakest: lowest independent validity", worst))
    cols = ["fold", "recording_id", "activated_alarms", "rca_top3", "true_causes", "actions", "action_start_s",
            "validity_independent", "wilson_low", "wilson_high", "covered_alarms", "uncovered_alarms",
            "alarm_coverage", "improvement_fresh"]
    save(pd.DataFrame([{"example": label, **row[cols].to_dict()} for label, row in picks]), "coolant_cf_examples")


# ---------------------------------------------------------------------------
# data-separation diagram
# ---------------------------------------------------------------------------


def figure_split() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    ax.axis("off")
    c_tr, c_va, c_te, c_out, c_in = ev.COLORS[0], ev.COLORS[1], ev.COLORS[2], ev.INK["primary"], ev.INK["muted"]

    def box(x, y, w, h, color, text, alpha=0.9, tc="white"):
        ax.add_patch(plt.Rectangle((x, y), w, h, color=color, alpha=alpha))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color=tc)

    ax.text(0, 9.4, "Track N - normal prediction (170 normal recordings, same split for both subsystems)",
            fontsize=8.5, weight="bold")
    box(0, 7.6, 7.0, 1.2, c_tr, "normal train: 119 recordings\nfitted with all fault runs")
    box(7.1, 7.6, 1.55, 1.2, c_va, "validation\n26")
    box(8.75, 7.6, 1.5, 1.2, c_te, "test\n25")
    ax.text(10.4, 8.2, "scored with the same fitted model;\nits fault results are not held out", fontsize=7, va="center")
    ax.text(0, 6.4, "Track F - fault generalisation (leave-one-scenario-out; coolant 4, probe 6 scenarios)",
            fontsize=8.5, weight="bold")
    scen = ["S1", "S2", "S3", "S4"]
    for i in range(4):
        y = 5.2 - i * 1.15
        box(0, y, 3.4, 0.9, c_tr, "normal train (119)", alpha=0.75)
        for j, sname in enumerate(scen):
            color, text = (c_out, f"{sname}: outer test") if j == i else (c_in, f"{sname}: fit")
            box(3.5 + j * 1.75, y, 1.65, 0.9, color, text, alpha=0.95 if j == i else 0.55)
        ax.text(10.6, y + 0.45, f"fold {i + 1}: graph, scm, fills and any selection\nwithout the outer-test scenario",
                fontsize=7, va="center")
    ax.text(0, -0.3, "Inner validation (only when settings are selected on fault outcomes): inside each outer fold,\n"
            "one further scenario is held out, so inner fits exclude both the outer and the inner scenario.", fontsize=7)
    ax.set_xlim(0, 14); ax.set_ylim(-0.9, 9.8)
    ev.save_figure(fig, f"{OUT}_data_separation")
    plt.close(fig)


if __name__ == "__main__":
    ev.set_plot_style()
    # print-size figures (about the text width), so fonts stay readable in the thesis
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})
    stages = [a for a in sys.argv[1:] if a != "selected"] or (["main"] if SELECTED else ["schema", "main"])
    if "schema" in stages:
        run_schema()
    if "main" in stages:
        run_main()
    if "schema" in stages and "main" not in stages:
        save(pd.DataFrame(check_rows), "checks_schema")
