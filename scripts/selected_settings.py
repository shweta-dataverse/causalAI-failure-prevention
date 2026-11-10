"""stage 15a: selected discovery and scm settings for the main pipeline.

discovery: the stage-12 selection rule (directed f1, then skeleton f1, shorter lag, smaller
alpha, stricter support, config id) is applied separately in every scope: all scenarios
(track n), without the outer scenario k (track f), and without k and an inner scenario j
(inner folds of the scm selection). all regime runs come from the stage-12 cache, so nothing
is rediscovered; only the aggregation and the ranking are repeated per scope. a held-out
scenario therefore never influences the choice of its own graph.

scm: smoothing alpha in {0.01, 0.1, 0.5, 1.0} x extra lag in {none, 2, 4} steps (as stage 13).
one rule for both graphs: the lowest held-out one-step log loss of the non-manipulated variables.
  * track f, outer scenario k: inner leave-one-scenario-out over the other scenarios j; each
    inner fit uses normal train + fault runs except k and j, with the graph of scope (k, j);
    the pooled log loss over all inner runs decides.
  * track n: fit on normal train + all fault runs, scored on the normal validation recordings.
ties go to the earlier baseline (alpha 0.5, no lag), then no lag, then alpha closest to 0.5.
the normal test recordings and the outer scenario are never used for any choice.

outputs: results/tables/15_{sub}_discovery_selection.csv, 15_{sub}_discovery_edges.csv,
15_scm_selection_inner.csv, 15_scm_selection.csv.

usage: PYTHONPATH=src .venv/bin/python scripts/selected_settings.py [discovery] [scm]
"""

from __future__ import annotations

import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from causal_rca import data  # noqa: E402
from causal_rca import discovery_study as ds  # noqa: E402
from causal_rca import evaluation as ev  # noqa: E402
from causal_rca import graph_evaluation as ge  # noqa: E402
from causal_rca import scm as sc  # noqa: E402
import run_discovery_study as rds  # noqa: E402

INTERVAL = 0.5
TABLES = data.RESULTS_DIR / "tables"
ALPHAS = (0.01, 0.1, 0.5, 1.0)
LAGS = (None, 2, 4)
BASELINE = (0.5, None)
SETTINGS = [(a, l) for a in ALPHAS for l in LAGS]
SUBSYSTEMS = ("coolant", "probe")


def scope_id(exclude: tuple[str, ...]) -> str:
    return "discovered" if not exclude else "discovered_without_" + "_and_".join(exclude)


def preference(setting):
    """tie-break order: baseline first, then no lag before lag, then alpha closest to 0.5."""
    alpha, lag = setting
    return (setting != BASELINE, lag is not None, lag or 0, abs(np.log(alpha / 0.5)))


def lags_for(graph, lag):
    """the extra lag on every edge whose child keeps a table of at most 2^20 configurations (stage 13)."""
    if lag is None:
        return None
    return {(u, v): [lag] for u, v in graph.edges if 3 * graph.in_degree(v) <= 20}


def fit(view, graph, train_ids, manipulated, setting):
    alpha, lag = setting
    return sc.fit_scm(view, graph, train_ids, manipulated, INTERVAL, smoothing=alpha, lags=lags_for(graph, lag))


def nll(model, view, ids, manipulated):
    """summed one-step negative log likelihood and row count of the non-manipulated variables."""
    total, rows = 0.0, 0
    for r in ids:
        s = sc.recording_series(view, r)
        for v in model.mechanisms:
            if v in manipulated.get(r, set()) or v not in s:
                continue
            p = model.transition_probs(s, v)
            y = s[v][1:]
            ok = ~np.isnan(p) & ~np.isnan(y)
            if ok.any():
                total += float(-np.log(np.where(y[ok] == 1, p[ok], 1 - p[ok])).sum())
                rows += int(ok.sum())
    return total, rows


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def discovery(sub: str, study: dict) -> None:
    s = rds.Setting(sub)
    order = rds.selection_order(study)
    grid = ds.grid_configs(study["grid"])
    runs = {c.run_id: s.runs(c) for c in grid}
    scopes = [()] + [(k,) for k in s.folds] + [tuple(p) for p in permutations(s.folds, 2)]
    sel_rows, edge_rows = [], []
    for exclude in scopes:
        rows, edges = [], {}
        for c in grid:
            row, e = ds.evaluate_config(runs[c.run_id], c, s.nodes, s.expert, exclude=exclude)
            rows.append({"subsystem": sub, "scope": scope_id(exclude), **row})
            edges[c.config_id] = e
        best, ties = ds.select(pd.DataFrame(rows), order)
        sel_rows.append({"subsystem": sub, "graph_id": scope_id(exclude), "excluded": ",".join(exclude),
                         **best.drop(["subsystem", "scope", "rank"]).to_dict(), **ties})
        edge_rows.append(edges[best.config_id].assign(graph_id=scope_id(exclude), config_id=best.config_id))
    sel = pd.DataFrame(sel_rows)
    # the all-scenario and single-exclusion choices must equal the stage-12 selection
    old = pd.read_csv(TABLES / f"12_{sub}_selection.csv")
    old_ids = dict(zip(old.scope.map(lambda x: "discovered" if x == "all scenarios" else
                                     "discovered_without_" + x.removeprefix("without ")), old.config_id))
    mine = dict(zip(sel.graph_id, sel.config_id))
    bad = {k: (v, mine.get(k)) for k, v in old_ids.items() if mine.get(k) != v}
    if bad:
        raise SystemExit(f"{sub}: selection differs from stage 12: {bad}")
    print(f"{sub}: {len(scopes)} scopes; all-scenario and fold choices equal stage 12", flush=True)
    ev.save_table(sel, f"15_{sub}_discovery_selection")
    ev.save_table(pd.concat(edge_rows, ignore_index=True), f"15_{sub}_discovery_edges")


# ---------------------------------------------------------------------------
# scm settings
# ---------------------------------------------------------------------------


def scm_selection() -> None:
    manipulated = data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments())
    inner_rows, sel_rows = [], []
    for sub in SUBSYSTEMS:
        s = rds.Setting(sub)
        edges = pd.read_csv(TABLES / f"15_{sub}_discovery_edges.csv")
        from causal_rca import causal_discovery as cd
        dags = {gid: ge.to_dag(cd.edges_to_graph(g, s.nodes))[0] for gid, g in edges.groupby("graph_id")}
        splits = s.splits
        normal_train = splits.loc[splits.split == "train", "recording_id"].tolist()
        validation = splits.loc[splits.split == "validation", "recording_id"].tolist()

        def train_without(*held):
            return normal_train + splits.loc[(splits.split == "fault") & (~splits.fold.isin(held)), "recording_id"].tolist()

        def runs_of(fold):
            return splits.loc[splits.fold == fold, "recording_id"].tolist()

        def graph(kind, exclude):
            return s.expert if kind == "expert" else dags[scope_id(tuple(exclude))]

        for kind in ("expert", "discovered"):
            # track n: normal validation log loss
            scores = {}
            for st in SETTINGS:
                total, rows = nll(fit(s.view, graph(kind, ()), train_without(), manipulated, st), s.view, validation, manipulated)
                scores[st] = total / rows
                inner_rows.append({"subsystem": sub, "graph": kind, "track": "N", "outer_fold": "", "inner_fold": "normal validation",
                                   "alpha": st[0], "lag": st[1], "nll": total, "rows": rows})
            best = min(scores.values())
            chosen = min((st for st in SETTINGS if np.isclose(scores[st], best, rtol=1e-12, atol=0)), key=preference)
            sel_rows.append({"subsystem": sub, "graph": kind, "track": "N", "fold": "all", "alpha": chosen[0], "lag": chosen[1],
                             "selection_log_loss": scores[chosen], "baseline_log_loss": scores[BASELINE],
                             "selection_data": "normal validation recordings"})
            # track f: inner leave-one-scenario-out within the outer training scenarios
            for k in s.folds:
                acc = {st: [0.0, 0] for st in SETTINGS}
                for j in s.folds:
                    if j == k:
                        continue
                    for st in SETTINGS:
                        total, rows = nll(fit(s.view, graph(kind, (k, j)), train_without(k, j), manipulated, st),
                                          s.view, runs_of(j), manipulated)
                        acc[st][0] += total
                        acc[st][1] += rows
                        inner_rows.append({"subsystem": sub, "graph": kind, "track": "F", "outer_fold": k, "inner_fold": j,
                                           "alpha": st[0], "lag": st[1], "nll": total, "rows": rows})
                scores = {st: t / n for st, (t, n) in acc.items()}
                best = min(scores.values())
                chosen = min((st for st in SETTINGS if np.isclose(scores[st], best, rtol=1e-12, atol=0)), key=preference)
                sel_rows.append({"subsystem": sub, "graph": kind, "track": "F", "fold": k, "alpha": chosen[0], "lag": chosen[1],
                                 "selection_log_loss": scores[chosen], "baseline_log_loss": scores[BASELINE],
                                 "selection_data": "inner scenarios " + ",".join(j for j in s.folds if j != k)})
                print(sub, kind, k, chosen, flush=True)
    ev.save_table(pd.DataFrame(inner_rows), "15_scm_selection_inner")
    ev.save_table(pd.DataFrame(sel_rows), "15_scm_selection")


if __name__ == "__main__":
    t0 = time.perf_counter()
    stages = sys.argv[1:] or ["discovery", "scm"]
    if "discovery" in stages:
        study = rds.load_study()
        for sub in SUBSYSTEMS:
            discovery(sub, study)
    if "scm" in stages:
        scm_selection()
    print(f"finished in {time.perf_counter() - t0:.0f} s")
