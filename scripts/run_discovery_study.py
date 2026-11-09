"""stage 12: focused causal-discovery hyperparameter study (coolant and probe).

stages (each one reads the run cache, so re-running a stage only computes what is missing):

  timing     expert-free timing evidence and sample availability per regime and maximum lag
  grid       pcmci+ for the 9 alpha x lag settings per subsystem, 27 aggregations each, scored
             on all scenarios and on every leave-one-scenario-out fold; fdr filter off as a
             sensitivity (no rerun); reproduction check against notebooks 03 / 10
  select     the pre-declared selection rule per scope; shortlist; selected graphs per fold
  staged     conditioning caps 2 and 4, a 60 s maximum lag and the cmisymb permutation test
             on the selected (and original) configuration
  bootstrap  recording-level resampling of the shortlisted configurations
  report     combined thesis tables and heatmaps

the objective and every rule come from configs/discovery_study.yaml. scores of selected
configurations are reference-guided tuning performance on the causRCA subsystem, not
independent test accuracy. existing result files are never written; outputs start with 12_.

usage: PYTHONPATH=src .venv/bin/python scripts/run_discovery_study.py [stage ...|all] [--workers N]
"""

from __future__ import annotations

import os

# one thread per worker process (set before numpy is imported)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from causal_rca import causal_discovery as cd
from causal_rca import data
from causal_rca import discovery_study as ds
from causal_rca import evaluation as ev
from causal_rca import graph_evaluation as ge
from causal_rca import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "discovery_study.yaml"
TABLES = data.RESULTS_DIR / "tables"
LOG_PATH = data.RESULTS_DIR / "logs" / "12_discovery_study.log"
CACHE = ds.RunCache(ROOT / "data" / "cache" / "discovery_study")
MANIFEST = TABLES / "12_study_manifest.jsonl"  # one json line per invocation, appended
EVENTS = TABLES / "12_run_events.csv"
EVENT_COLUMNS = ["subsystem", "regime", "run_id", "key", "status", "error", "wall_s"]
STAGES = ["timing", "grid", "select", "staged", "bootstrap", "report"]
TIMING_BINS = {"lag0": (0, 0), "0.5-10s": (1, 20), "10.5-20s": (21, 40), "20.5-60s": (41, 120)}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_study() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def save(table: pd.DataFrame, name: str) -> None:
    ev.save_table(table, name)
    log(f"  wrote results/tables/{name}.csv ({len(table)} rows)")


# ---------------------------------------------------------------------------
# setting of one subsystem
# ---------------------------------------------------------------------------


class Setting:
    def __init__(self, sub: str):
        self.sub = sub
        self.prefix = "02" if sub == "coolant" else "10"
        self.view_name = f"{sub}_dt0.5"
        self.view, _ = pp.load_view_data(data.PROCESSED_DIR, self.view_name)
        self.splits = pd.read_csv(TABLES / f"{self.prefix}_{sub}_splits.csv")
        self.regimes = cd.define_regimes(self.splits)
        self.expert = ge.simple_digraph(data.load_expert_graph(subsystem=sub))
        # node schema: the subsystem's variable list (expert node table), independent of held-out values
        self.nodes = sorted(self.expert.nodes)
        if set(self.nodes) != set(self.view.var_names):
            raise RuntimeError(f"{sub}: view variables differ from the subsystem node list")
        self.folds = [r for r in self.regimes if r != "normal"]
        self.scopes = {"all scenarios": ()} | {f"without {f}": (f,) for f in self.folds}

    def specs(self, config: ds.StudyConfig, regimes: dict[str, list[str]] | None = None):
        return ds.make_specs(self.sub, self.view_name, self.view, regimes or self.regimes, config)

    def runs(self, config: ds.StudyConfig, regimes=None) -> dict[str, ds.RegimeRun]:
        specs, _ = self.specs(config, regimes)
        return ds.load_runs(CACHE, specs)

    def training_ids(self, scope: str) -> dict[str, list[str]]:
        exclude = set(self.scopes[scope])
        return {r: ids for r, ids in self.regimes.items() if r not in exclude}


def execute(settings_list: list[Setting], configs_by_sub: dict[str, list[ds.StudyConfig]], workers: int,
            timeout_s: float | None, regimes_by_sub: dict[str, list[dict]] | None = None) -> None:
    """run every missing regime run of the given configurations (all subsystems in one pool)."""
    specs, skipped = [], []
    for s in settings_list:
        for config in configs_by_sub.get(s.sub, []):
            for regimes in (regimes_by_sub or {}).get(s.sub, [s.regimes]):
                sp, sk = s.specs(config, regimes)
                specs += sp
                skipped += [{**k, "subsystem": s.sub} for k in sk]
    events = ds.run_specs(specs, CACHE, data.PROCESSED_DIR, workers=workers, timeout_s=timeout_s, log=log)
    if skipped:
        events += [{"subsystem": k["subsystem"], "regime": k["regime"], "run_id": k["run_id"], "key": "",
                    "status": k["status"], "error": k["reason"], "wall_s": 0.0} for k in skipped]
    if events:
        # appended, so two study processes can log at the same time
        pd.DataFrame(events, columns=EVENT_COLUMNS).to_csv(EVENTS, mode="a", header=not EVENTS.exists(), index=False)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def write_manifest(stages: list[str]) -> None:
    """append one line per invocation (appending keeps concurrent study processes from clobbering it)."""
    files = [data.PROCESSED_DIR / f"{s}_dt0.5.{ext}" for s in ("coolant", "probe") for ext in ("npz", "json")]
    files += [TABLES / "02_coolant_splits.csv", TABLES / "10_probe_splits.csv"]
    entry = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "stages": stages,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": git("status", "--porcelain").splitlines(),
        "environment": ds.environment(),
        "study_config_sha256": ds.file_sha256(CONFIG_PATH),
        "data_sha256": {str(p.relative_to(ROOT)): ds.file_sha256(p) for p in files},
    }
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# stage: timing evidence and sample availability
# ---------------------------------------------------------------------------


def stage_timing(settings: list[Setting], study: dict) -> None:
    log("stage timing")
    taus = sorted(set(study["grid"]["tau_max_s"]) | {study["shortlist"]["long_lag_s"]})
    runs, experiments = data.load_fault_runs(), data.load_experiments()
    manipulated = data.manipulated_by_recording(runs, experiments)
    for s in settings:
        rows, profiles = [], []
        for regime, ids in s.regimes.items():
            variables = cd.regime_variables(s.view, ids)
            for tau_s in taus:
                rows.append({"subsystem": s.sub, "regime": regime, "tau_max_s": tau_s, "tau_max_steps": int(tau_s / 0.5),
                             "variables": len(variables),
                             **ds.sample_availability(s.view, ids, variables, int(round(tau_s / 0.5)))})
            prof = ds.co_change_profile(s.view, ids, variables, max_lag=120, interval_s=0.5, bins=TIMING_BINS)
            profiles.append(prof.assign(subsystem=s.sub, regime=regime))
        save(pd.DataFrame(rows), f"12_{s.sub}_sample_availability")
        prof = pd.concat(profiles, ignore_index=True)
        save(prof, f"12_{s.sub}_timing_evidence")
        # injection annotation -> first observed change of a manipulated variable (ground truth, report only)
        delays = []
        sub_runs = runs[runs["recording_id"].isin(s.splits.loc[s.splits.split == "fault", "recording_id"])]
        for row in sub_runs.itertuples(index=False):
            x, m = s.view.arrays[row.recording_id], s.view.masks[row.recording_id]
            first = np.inf
            for v in manipulated.get(row.recording_id, set()):
                j = s.view.var_names.index(v)
                ch = np.flatnonzero((np.diff(x[:, j]) != 0) & ~m[1:, j] & ~m[:-1, j]) + 1
                ch = ch[ch * 0.5 >= row.cause_start_at]
                if len(ch):
                    first = min(first, ch[0] * 0.5)
            delays.append({"subsystem": s.sub, "recording_id": row.recording_id, "scenario": row.experiment_id,
                           "cause_start_at_s": row.cause_start_at,
                           "first_manipulated_change_s": first if np.isfinite(first) else np.nan,
                           "annotation_to_change_s": first - row.cause_start_at if np.isfinite(first) else np.nan,
                           "recording_length_s": len(x) * 0.5})
        save(pd.DataFrame(delays), f"12_{s.sub}_annotation_delay")


# ---------------------------------------------------------------------------
# stage: grid
# ---------------------------------------------------------------------------


def original_config(study: dict) -> ds.StudyConfig:
    return ds.config_from_dict(study["original"])


def grid_rows(s: Setting, configs: list[ds.StudyConfig], stage: str) -> tuple[pd.DataFrame, dict]:
    rows, edges = [], {}
    run_cache: dict[str, dict] = {}
    for config in configs:
        if config.run_id not in run_cache:
            run_cache[config.run_id] = s.runs(config)
        runs = run_cache[config.run_id]
        for scope, exclude in s.scopes.items():
            row, e = ds.evaluate_config(runs, config, s.nodes, s.expert, exclude=exclude)
            rows.append({"subsystem": s.sub, "stage": stage, "scope": scope, **row})
            edges[(config.config_id, scope)] = e
    return pd.DataFrame(rows), edges


def stage_grid(settings: list[Setting], study: dict, workers: int) -> None:
    log("stage grid")
    grid = ds.grid_configs(study["grid"])
    execute(settings, {s.sub: grid for s in settings}, workers, study["shortlist"]["run_timeout_s"])
    for s in settings:
        table, _ = grid_rows(s, grid, "a: grid")
        off, _ = grid_rows(s, [replace(c, apply_fdr_to_graph=False) for c in grid], "c: fdr filter off")
        save(table, f"12_{s.sub}_grid")
        save(off, f"12_{s.sub}_fdr_off")
        # fdr bookkeeping per regime for every alpha x lag setting
        diag = pd.concat([ds.fdr_diagnostics(s.runs(c), c) for c in {c.run_id: c for c in grid}.values()],
                         ignore_index=True)
        save(diag.assign(subsystem=s.sub), f"12_{s.sub}_fdr_diagnostics")
        reproduction_check(s, study)


def reproduction_check(s: Setting, study: dict) -> None:
    """the original configuration rebuilt from the cache must give the saved graphs of notebooks 03 / 10."""
    saved = pd.read_csv(TABLES / ("03_coolant_edges.csv" if s.sub == "coolant" else "10_probe_edges.csv"))
    config = original_config(study)
    runs = s.runs(config)
    rows = []
    for scope, exclude in s.scopes.items():
        gid = "discovered" if not exclude else f"discovered_without_{exclude[0]}"
        _, edges = ds.aggregate(runs, config, s.nodes, exclude=exclude)
        same = cd.edge_key(edges) == cd.edge_key(saved[saved.graph_id == gid])
        rows.append({"subsystem": s.sub, "scope": scope, "saved_graph_id": gid, "identical_edge_set": same,
                     "n_edges_rebuilt": len(edges), "n_edges_saved": int((saved.graph_id == gid).sum())})
    table = pd.DataFrame(rows)
    save(table, f"12_{s.sub}_reproduction_check")
    if not table.identical_edge_set.all():
        log(f"WARNING: {s.sub}: rebuilt original graphs differ from the saved notebook graphs")


# ---------------------------------------------------------------------------
# stage: selection
# ---------------------------------------------------------------------------


def selection_order(study: dict) -> list[tuple[str, str]]:
    order = [tuple(next(iter(d.items()))) for d in study["selection"]["order"]]
    if tuple(order) != ds.SELECTION_ORDER:
        raise RuntimeError("selection order in the yaml differs from discovery_study.SELECTION_ORDER")
    return order


def selections(s: Setting, study: dict) -> pd.DataFrame:
    table = pd.read_csv(TABLES / f"12_{s.sub}_grid.csv")
    order = selection_order(study)
    rows = []
    for scope in s.scopes:
        best, ties = ds.select(table[table.scope == scope], order)
        train = s.training_ids(scope)
        rows.append({"subsystem": s.sub, "scope": scope, **best.drop(["subsystem", "stage", "scope", "rank"]).to_dict(),
                     **ties, "training_regimes": ",".join(train),
                     "training_recordings": sum(len(v) for v in train.values())})
    return pd.DataFrame(rows)


def shortlist(s: Setting, study: dict) -> list[ds.StudyConfig]:
    """top 3 distinct configurations on all scenarios, plus the original."""
    table = pd.read_csv(TABLES / f"12_{s.sub}_grid.csv")
    ranked = ds.rank_configs(table[table.scope == "all scenarios"], selection_order(study))
    grid = {c.config_id: c for c in ds.grid_configs(study["grid"])}
    chosen = [grid[c] for c in ranked.config_id.head(3)]
    original = original_config(study)
    if original.config_id not in [c.config_id for c in chosen]:
        chosen.append(original)
    return chosen


def selected_config(s: Setting, study: dict, scope: str = "all scenarios") -> ds.StudyConfig:
    sel = pd.read_csv(TABLES / f"12_{s.sub}_selection.csv")
    cid = sel.loc[sel.scope == scope, "config_id"].iloc[0]
    return {c.config_id: c for c in ds.grid_configs(study["grid"])}[cid]


def stage_select(settings: list[Setting], study: dict) -> None:
    log("stage select")
    for s in settings:
        sel = selections(s, study)
        save(sel, f"12_{s.sub}_selection")
        grid = {c.config_id: c for c in ds.grid_configs(study["grid"])}
        # graphs of the selected configuration of every scope, in the edge format of notebook 03
        edges = []
        for row in sel.itertuples(index=False):
            config = grid[row.config_id]
            exclude = s.scopes[row.scope]
            _, e = ds.aggregate(s.runs(config), config, s.nodes, exclude=exclude)
            gid = "discovered" if not exclude else f"discovered_without_{exclude[0]}"
            edges.append(e.assign(graph_id=gid, config_id=row.config_id))
        save(pd.concat(edges, ignore_index=True), f"12_{s.sub}_selected_edges")
        ids = [{"subsystem": s.sub, "scope": scope, "regime": r, "recording_id": rid}
               for scope in s.scopes for r, rids in s.training_ids(scope).items() for rid in rids]
        save(pd.DataFrame(ids), f"12_{s.sub}_scope_training_ids")
        short = shortlist(s, study)
        save(pd.DataFrame([{"subsystem": s.sub, "position": k + 1, **c.row()} for k, c in enumerate(short)]),
             f"12_{s.sub}_shortlist")


# ---------------------------------------------------------------------------
# stage: staged comparisons (caps, 60 s lag, cmisymb)
# ---------------------------------------------------------------------------


def staged_configs(s: Setting, study: dict) -> list[tuple[str, str, ds.StudyConfig, str]]:
    """(stage, variant_of, config, regimes) of the staged comparisons; regimes is 'all' or 'fault'."""
    sh = study["shortlist"]
    named = {"selected": selected_config(s, study), "original": original_config(study)}
    out = []
    for who in sh["caps_for"]:
        for k in sh["conditioning_caps"]:
            out.append(("c: conditioning caps", who, replace(named[who], max_conds=(k, k, k)), "all"))
    for who in sh["long_lag_for"]:
        out.append(("c: 60 s maximum lag", who, replace(named[who], tau_max_s=float(sh["long_lag_s"])), "all"))
    alt = sh["alternative_ci_test"]
    for who in alt["for"]:
        # g-squared on the same regimes as the comparison baseline (its runs are cached from the grid)
        out.append(("c: ci test, fault regimes only", who, named[who], alt["regimes"]))
        out.append(("c: ci test, fault regimes only", who, replace(named[who], ci_test=alt["test"],
                    sig_samples=alt["sig_samples"], seed=alt["seed"]), alt["regimes"]))
    unique, seen = [], set()
    for item in out:
        if (item[2].config_id, item[3]) not in seen:
            seen.add((item[2].config_id, item[3]))
            unique.append(item)
    return unique


def scoped_regimes(s: Setting, which: str) -> dict[str, list[str]]:
    return s.regimes if which == "all" else {r: ids for r, ids in s.regimes.items() if r != "normal"}


def stage_staged(settings: list[Setting], study: dict, workers: int, part: str = "all") -> None:
    """part: 'all', 'nonci' (caps and 60 s lag) or 'ci' (the slow cmisymb runs only)."""
    log(f"stage staged ({part})")
    plan = {s.sub: staged_configs(s, study) for s in settings}
    wanted = lambda c: part == "all" or (part == "ci") == (c.ci_test != "gsq")
    for which in ("all", "fault"):
        execute(settings, {sub: [c for _, _, c, w in items if w == which and wanted(c)] for sub, items in plan.items()}, workers,
                study["shortlist"]["run_timeout_s"],
                regimes_by_sub={s.sub: [scoped_regimes(s, which)] for s in settings})
    for s in settings:
        rows, per_regime = [], []
        for stage, who, config, which in plan[s.sub]:
            runs = s.runs(config, scoped_regimes(s, which))
            row, _ = ds.evaluate_config(runs, config, s.nodes, s.expert)
            rows.append({"subsystem": s.sub, "stage": stage, "scope": "all scenarios" if which == "all"
                         else "all scenarios, fault regimes only", "variant_of": who, **row,
                         "regime_runtimes_s": json.dumps({r: round(v.runtime_s, 1) for r, v in runs.items()}),
                         "regime_status": json.dumps({r: v.status for r, v in runs.items()})})
            # per-regime relations, to compare tests regime by regime
            rel = ds.regime_relations(runs, config.apply_fdr_to_graph)
            for regime, run in runs.items():
                r = rel[rel.regime == regime]
                per_regime.append({"subsystem": s.sub, "stage": stage, "config_id": config.config_id,
                                   "regime": regime, "status": run.status, "runtime_s": run.runtime_s,
                                   "relations": json.dumps(sorted(f"{a}->{b}" if t == "directed" else f"{a}--{b}"
                                                                  for a, b, t in zip(r.a, r.b, r.relation)))})
        save(pd.DataFrame(rows), f"12_{s.sub}_staged")
        save(pd.DataFrame(per_regime), f"12_{s.sub}_staged_regimes")


# ---------------------------------------------------------------------------
# stage: bootstrap stability
# ---------------------------------------------------------------------------


def stage_bootstrap(settings: list[Setting], study: dict, workers: int) -> None:
    log("stage bootstrap")
    b = study["bootstrap"]
    short = {s.sub: shortlist(s, study) for s in settings}
    draws = {s.sub: ds.bootstrap_regimes(s.regimes, b["n_boot"], b["seed"]) for s in settings}
    execute(settings, short, workers, study["shortlist"]["run_timeout_s"], regimes_by_sub=draws)
    order = selection_order(study)
    for s in settings:
        rows, edge_rows = [], []
        full = {}
        for config in short[s.sub]:
            _, e = ds.aggregate(s.runs(config), config, s.nodes)
            full[config.config_id] = cd.edge_key(e)
        for k, regimes in enumerate(draws[s.sub]):
            for config in short[s.sub]:
                runs = s.runs(config, regimes)
                row, e = ds.evaluate_config(runs, config, s.nodes, s.expert)
                keys = cd.edge_key(e) if row["run_status"] == "ok" else set()
                rows.append({"subsystem": s.sub, "draw": k, **row,
                             "jaccard_with_full_data_graph": cd.jaccard(keys, full[config.config_id])
                             if row["run_status"] == "ok" else np.nan})
                edge_rows += [{"config_id": config.config_id, "draw": k, "edge": edge_name(a, b_, o)} for a, b_, o in keys]
        per_draw = pd.DataFrame(rows)
        # which shortlisted configuration the selection rule would pick in each draw
        winners = []
        for k, g in per_draw.groupby("draw"):
            try:
                best, ties = ds.select(g.assign(scope="all scenarios"), order)
                winners.append({"draw": k, "winner": best.config_id, **ties})
            except ValueError:
                winners.append({"draw": k, "winner": "none eligible"})
        save(per_draw, f"12_{s.sub}_bootstrap_draws")
        save(pd.DataFrame(winners).assign(subsystem=s.sub), f"12_{s.sub}_bootstrap_winners")
        er = pd.DataFrame(edge_rows)
        if len(er):
            freq = (er.groupby(["config_id", "edge"]).draw.nunique() / b["n_boot"]).rename("bootstrap_frequency").reset_index()
            freq["in_full_data_graph"] = [edge in {edge_name(*x) for x in full[c]} for c, edge in zip(freq.config_id, freq.edge)]
            freq["reference_outcome"] = freq.edge.map(lambda e: reference_outcome(e, s.expert))
            save(freq.assign(subsystem=s.sub), f"12_{s.sub}_bootstrap_edges")
        summary = (per_draw[per_draw.run_status == "ok"].groupby("config_id")
                   .agg(draws_ok=("draw", "nunique"), directed_f1_mean=("directed_f1", "mean"),
                        directed_f1_sd=("directed_f1", "std"), directed_f1_min=("directed_f1", "min"),
                        directed_f1_max=("directed_f1", "max"), skeleton_f1_mean=("skeleton_f1", "mean"),
                        skeleton_f1_sd=("skeleton_f1", "std"), shd_mean=("shd", "mean"),
                        jaccard_mean=("jaccard_with_full_data_graph", "mean")).reset_index())
        wins = pd.DataFrame(winners).winner.value_counts()
        summary["selected_in_draws"] = summary.config_id.map(wins).fillna(0).astype(int)
        save(summary.assign(subsystem=s.sub), f"12_{s.sub}_bootstrap_summary")
        if s.sub == "coolant":
            check_bootstrap_reproduction(study, er)


def edge_name(a: str, b: str, oriented: bool) -> str:
    return f"{a} → {b}" if oriented else f"{min(a, b)} — {max(a, b)}"


def reference_outcome(edge: str, expert) -> str:
    if " → " in edge:
        a, b = edge.split(" → ")
        return "correct direction" if expert.has_edge(a, b) else ("reversed" if expert.has_edge(b, a) else "extra")
    a, b = edge.split(" — ")
    return "undirected (adjacency correct)" if expert.has_edge(a, b) or expert.has_edge(b, a) else "extra"


def check_bootstrap_reproduction(study: dict, er: pd.DataFrame) -> None:
    """the original configuration's draws must reproduce notebook 03's bootstrap frequencies."""
    saved = pd.read_csv(TABLES / "03_coolant_bootstrap_stability.csv")
    cid = original_config(study).config_id
    mine = (er[er.config_id == cid].groupby("edge").draw.nunique() / study["bootstrap"]["n_boot"])
    same = all(abs(mine.get(e, 0.0) - f) < 1e-9 for e, f in zip(saved.edge, saved.bootstrap_frequency)) and \
        set(mine.index) == set(saved.edge)
    log(f"  coolant bootstrap of the original configuration reproduces notebook 03: {same}")
    pd.DataFrame([{"check": "coolant original bootstrap = 03_coolant_bootstrap_stability", "identical": same}]).to_csv(
        TABLES / "12_coolant_bootstrap_reproduction_check.csv", index=False)


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "subsystem", "stage", "config_id", "scope", "ci_test", "interval_s", "tau_max_s", "tau_max_steps", "pc_alpha",
    "conditioning_caps", "fdr_method", "fdr_filter", "support_threshold", "skeleton_precision", "skeleton_recall",
    "skeleton_f1", "directed_precision", "directed_recall", "directed_f1", "shd", "n_adjacencies",
    "n_directed_edges", "n_undirected_edges", "n_both_directions", "unresolved_orientations",
    "testable_pair_share", "reference_edge_coverage", "runtime_s", "run_status", "selected",
]


def stage_report(settings: list[Setting], study: dict) -> None:
    log("stage report")
    ev.set_plot_style()
    all_rows, compact = [], []
    for s in settings:
        grid = pd.read_csv(TABLES / f"12_{s.sub}_grid.csv")
        parts = [grid, pd.read_csv(TABLES / f"12_{s.sub}_fdr_off.csv")]
        staged_path = TABLES / f"12_{s.sub}_staged.csv"
        if staged_path.exists():
            parts.append(pd.read_csv(staged_path).drop(columns=["variant_of", "regime_runtimes_s", "regime_status"]))
        table = pd.concat(parts, ignore_index=True)
        sel = pd.read_csv(TABLES / f"12_{s.sub}_selection.csv")
        chosen = set(zip(sel.config_id, sel.scope))
        table["selected"] = [(c, sc) in chosen and st == "a: grid" for c, sc, st in zip(table.config_id, table.scope, table.stage)]
        all_rows.append(table)
        original = original_config(study).config_id
        for scope in s.scopes:
            g = grid[grid.scope == scope]
            for label, cid in (("original", original), ("selected", sel.loc[sel.scope == scope, "config_id"].iloc[0])):
                r = g[g.config_id == cid].iloc[0]
                compact.append({"subsystem": s.sub, "scope": scope, "configuration": label, "config_id": cid,
                                **{k: r[k] for k in ("tau_max_s", "pc_alpha", "support_threshold", "skeleton_precision",
                                                     "skeleton_recall", "skeleton_f1", "directed_precision",
                                                     "directed_recall", "directed_f1", "shd", "n_adjacencies",
                                                     "n_directed_edges", "unresolved_orientations")},
                                "n_tied_objective": int(sel.loc[sel.scope == scope, "n_tied_objective"].iloc[0])
                                if label == "selected" else np.nan})
        heatmap(s, grid, sel)
    results = pd.concat(all_rows, ignore_index=True)
    save(results[RESULT_COLUMNS], "12_results_table")
    save(pd.DataFrame(compact), "12_original_vs_selected")


def heatmap(s: Setting, grid: pd.DataFrame, sel: pd.DataFrame) -> None:
    g = grid[grid.scope == "all scenarios"]
    alphas = sorted(g.pc_alpha.unique())
    taus = sorted(g.tau_max_s.unique())
    thresholds = sorted(g.support_threshold.unique())
    best = sel.loc[sel.scope == "all scenarios"].iloc[0]
    fig, axes = plt.subplots(1, len(thresholds), figsize=(4.3 * len(thresholds), 3.8), layout="constrained", sharey=True)
    cmap = ev.sequential_cmap()
    for ax, thr in zip(np.atleast_1d(axes), thresholds):
        sub = g[g.support_threshold == thr]
        val = np.array([[sub[(sub.pc_alpha == a) & (sub.tau_max_s == t)].directed_f1.iloc[0] for t in taus] for a in alphas])
        skel = np.array([[sub[(sub.pc_alpha == a) & (sub.tau_max_s == t)].skeleton_f1.iloc[0] for t in taus] for a in alphas])
        image = ax.imshow(val, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, a in enumerate(alphas):
            for j, t in enumerate(taus):
                mark = best.pc_alpha == a and best.tau_max_s == t and best.support_threshold == thr
                ax.text(j, i, f"{val[i, j]:.2f}\n({skel[i, j]:.2f})" + (" *" if mark else ""), ha="center", va="center",
                        fontsize=8, color=ev.INK["surface"] if val[i, j] > 0.55 else ev.INK["primary"],
                        fontweight="bold" if mark else "normal")
        ax.set_xticks(range(len(taus)), [f"{t:g} s\n({int(t / 0.5)})" for t in taus])
        ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
        ax.set_xlabel("maximum lag (steps)")
        ax.set_title(f"support threshold {thr:g}")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("pc_alpha")
    fig.colorbar(image, ax=axes, shrink=0.8, label="directed F1 vs expert reference")
    fig.suptitle(f"{s.sub}: directed F1 (skeleton F1), all scenarios; * = selected "
                 f"(reference-guided tuning performance)", x=0.01, ha="left", fontsize=10)
    ev.save_figure(fig, f"12_{s.sub}_heatmap_directed_f1")
    plt.close(fig)
    log(f"  wrote results/figures/12_{s.sub}_heatmap_directed_f1.pdf/png")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="*", default=["all"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--subsystems", nargs="*", default=None)
    parser.add_argument("--staged-part", choices=["all", "nonci", "ci"], default="all")
    args = parser.parse_args()
    stages = STAGES if args.stages in (["all"], []) else args.stages
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}")
    study = load_study()
    workers = args.workers or int(study["runtime"]["workers"])
    write_manifest(stages)
    log(f"study stages {stages}, workers {workers}")
    settings = [Setting(sub) for sub in (args.subsystems or study["subsystems"])]
    for stage in stages:
        start = time.time()
        if stage == "timing":
            stage_timing(settings, study)
        elif stage == "grid":
            stage_grid(settings, study, workers)
        elif stage == "select":
            stage_select(settings, study)
        elif stage == "staged":
            stage_staged(settings, study, workers, args.staged_part)
        elif stage == "bootstrap":
            stage_bootstrap(settings, study, workers)
        elif stage == "report":
            stage_report(settings, study)
        log(f"stage {stage} finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
