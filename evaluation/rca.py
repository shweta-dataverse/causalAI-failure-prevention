import hashlib
import json
import pickle
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core import data, lstm_shap, rca, scm
from core.plots import BASELINE_COLOR, COLORS, GRAPH_COLORS, GRAPH_LABELS, INK, PAGE_WIDTH, set_print_style
from evaluation.scm import OriginalGraphs
from pipeline.root_cause_analysis import ranking_of
from pipeline.scm_fitting import Subsystem, baseline, chosen_setting, fit, load_subsystem
from utils import io
from utils.config import load_config, setting
from utils.logger import log


# lstm + shap baseline


_SUBSYSTEMS: dict[str, Subsystem] = {}


def _subsystem(name: str) -> Subsystem:
    """data of a subsystem, loaded once per worker process."""
    if name not in _SUBSYSTEMS:
        _SUBSYSTEMS[name] = load_subsystem(load_config(), name)
    return _SUBSYSTEMS[name]


def _windows(sub: Subsystem, ids: list[str], fills: dict, lookback: int, weighted: bool = False) -> dict:
    """look-back windows; with weighted, manipulated variables of a fault run are left out of the loss."""
    arrays = [lstm_shap.run_arrays(sub.view, r, fills, sub.manipulated.get(r, set()) if weighted else frozenset())
              for r in ids]
    return lstm_shap.make_windows(arrays, lookback)


def _train(lc: dict, train_win: dict, val_win: dict, n_vars: int, hidden: int, seed: int):
    return lstm_shap.train(train_win, val_win, n_vars, hidden, seed, lc["learning_rate"], lc["batch_size"],
                           lc["max_epochs"], lc["patience"], lc["min_improvement"])


def _tuning_job(args: tuple) -> dict:
    """one grid point, trained on the normal training recordings and scored on normal validation."""
    name, hidden, lookback = args
    cfg = load_config()
    lc = setting(cfg, "lstm")
    torch.set_num_threads(lc["threads_per_worker"])
    sub = _subsystem(name)
    train_ids = sub.ids("train")
    fills = scm.fill_values(sub.view, train_ids)
    _, loss, history = _train(lc, _windows(sub, train_ids, fills, lookback),
                              _windows(sub, sub.ids("validation"), fills, lookback),
                              len(sub.view.var_names), hidden, lc["seeds"][0])
    return {"subsystem": name, "hidden": hidden, "lookback_steps": lookback,
            "lookback_s": lookback * sub.interval_s, "normal_validation_loss": loss, "epochs": len(history)}


def _fold_job(args: tuple) -> tuple[list[dict], list[dict]]:
    """train the chosen setting without the held-out scenario and rank the candidates of its runs."""
    name, fold, seed, hidden, lookback, exclude_related = args
    cfg = load_config()
    lc = setting(cfg, "lstm")
    torch.set_num_threads(lc["threads_per_worker"])
    sub = _subsystem(name)
    train_ids = data.fold_training_ids(sub.splits, fold, exclude_related=exclude_related)
    fills = scm.fill_values(sub.view, train_ids)
    model, loss, history = _train(lc, _windows(sub, train_ids, fills, lookback, weighted=True),
                                  _windows(sub, sub.ids("validation"), fills, lookback),
                                  len(sub.view.var_names), hidden, seed)
    background = lstm_shap.background_windows(_windows(sub, sub.ids("train"), fills, lookback),
                                              lc["shap_background"], lc["shap_background_seed"])
    main = tuple(lc["main"])
    rows, importance_rows = [], []
    for rid in sub.runs_of(fold):
        base = {"subsystem": name, "fold": fold, "recording_id": rid, "seed": seed}
        result = lstm_shap.explain(model, sub.view, rid, fills, sub.alarms, sub.nodes, background, lookback,
                                   sub.interval_s, lc["shap_samples"], lc["step_variants"])
        if result is None:
            rows.append({**base, "scored": False})
            continue
        cands, importance, _ = result
        truth = set(sub.manipulated[rid])
        chance = rca.chance_metrics(cands, truth, setting(cfg, "rca", "top_k"), setting(cfg, "rca", "chance_draws"),
                                    setting(cfg, "rca", "seed"))
        for (steps, aggregation), vector in importance.items():
            ranking, scores = lstm_shap.importance_ranking(vector, sub.view.var_names, cands)
            rows.append({**base, "scored": True, "steps": steps, "aggregation": aggregation,
                         "n_candidates": len(cands), "n_causes": len(truth), "epochs": len(history),
                         "fold_validation_loss": loss, "ranking": " > ".join(ranking),
                         **rca.ranking_metrics(ranking, truth, setting(cfg, "rca", "top_k")),
                         **{f"chance_{m}": x for m, x in chance.items()}})
            if (steps, aggregation) == main:
                importance_rows += [{**base, "variable": v, "importance": scores[v], "is_cause": v in truth}
                                    for v in cands]
    return rows, importance_rows


def _pool_map(cfg: dict, job, jobs: list):
    """results of the jobs in order, each as soon as it and the jobs before it are done."""
    with get_context("spawn").Pool(setting(cfg, "lstm", "workers")) as pool:
        yield from pool.imap(job, jobs)


def lstm_tuning(cfg: dict, force: bool = False) -> pd.DataFrame:
    """normal validation loss of every (hidden units, look-back) pair, cached in data/cache/lstm."""
    file = io.cache_dir("lstm") / "tuning.csv"
    if file.exists() and not force:
        return pd.read_csv(file)
    lc = setting(cfg, "lstm")
    jobs = [(sub, h, lb) for sub in setting(cfg, "subsystems") for h in lc["hidden"] for lb in lc["lookback_steps"]]
    log.info(f"tuning the lstm ({len(jobs)} settings)")
    table = pd.DataFrame(list(_pool_map(cfg, _tuning_job, jobs)))
    io.write_csv(table, file)
    return table


def lstm_choice(tuning: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """lowest validation loss per subsystem (ties: fewer hidden units, shorter look-back)."""
    chosen = {}
    for sub, part in tuning.groupby("subsystem"):
        best = part.sort_values(["normal_validation_loss", "hidden", "lookback_steps"]).iloc[0]
        chosen[sub] = (int(best.hidden), int(best.lookback_steps))
    return chosen


def _fold_file(job: tuple) -> Path:
    """cache file of one fold model, keyed by its training recordings (folds without near-duplicates are shared)."""
    name, fold, seed, hidden, lookback, exclude_related = job
    train_ids = data.fold_training_ids(_subsystem(name).splits, fold, exclude_related=exclude_related)
    digest = hashlib.sha1(json.dumps([hidden, lookback, train_ids]).encode()).hexdigest()[:12]
    return io.cache_dir("lstm") / "folds" / f"{name}_{fold}_seed{seed}_{digest}.pkl"


def lstm_results(cfg: dict, exclude_related: bool = True, force: bool = False) -> dict:
    """tuning, chosen settings, rankings and attributions of the lstm + shap baseline; folds cached in data/cache/lstm."""
    tuning = lstm_tuning(cfg, force)
    chosen = lstm_choice(tuning)
    jobs = [(sub, fold, seed, *chosen[sub], exclude_related)
            for sub in setting(cfg, "subsystems") for fold in _subsystem(sub).folds
            for seed in setting(cfg, "lstm", "seeds")]
    missing = [job for job in jobs if force or not _fold_file(job).exists()]
    if missing:
        log.info(f"training the lstm + shap baseline ({len(missing)} fold models, the slowest step of the "
                 f"evaluation); results are cached in {io.relative(io.cache_dir('lstm'))}")
        for job, result in zip(missing, _pool_map(cfg, _fold_job, missing)):
            _fold_file(job).parent.mkdir(parents=True, exist_ok=True)
            _fold_file(job).write_bytes(pickle.dumps(result))
    results = [pickle.loads(_fold_file(job).read_bytes()) for job in jobs]
    return {"tuning": tuning, "chosen": chosen,
            "runs": pd.DataFrame([r for rows, _ in results for r in rows]),
            "importance": pd.DataFrame([r for _, rows in results for r in rows])}


def lstm_per_run(cfg: dict, runs: pd.DataFrame, steps: str, aggregation: str) -> pd.DataFrame:
    """metrics of one explanation variant, averaged over the seeds of each run."""
    k = setting(cfg, "rca", "top_k")
    part = runs[runs.scored.astype(bool) & (runs.steps == steps) & (runs.aggregation == aggregation)]
    return part.groupby(["subsystem", "fold", "recording_id"])[metric_names(k) + [f"chance_ap@{k}"]].mean().reset_index()


def lstm_summaries(cfg: dict, results: dict) -> dict[str, pd.DataFrame]:
    """per-scenario results and summary of the fixed variant, and every variant for the sensitivity check."""
    k = setting(cfg, "rca", "top_k")
    columns = metric_names(k) + [f"chance_ap@{k}"]
    main = tuple(setting(cfg, "lstm", "main"))
    runs = results["runs"]
    variants = sorted(runs.dropna(subset=["steps"]).groupby(["steps", "aggregation"]).groups)
    per_run = {v: lstm_per_run(cfg, runs, *v) for v in variants}
    per_scenario = {v: part.groupby(["subsystem", "fold"])[columns].mean().reset_index() for v, part in per_run.items()}
    sensitivity = [{"subsystem": sub, "steps": v[0], "aggregation": v[1], "main": v == main,
                    "top1_run_mean": part.top1.mean(), f"map{k}_run_mean": part[f"ap@{k}"].mean(),
                    f"ap{k}_scenario_macro": per_scenario[v].loc[per_scenario[v].subsystem == sub, f"ap@{k}"].mean()}
                   for v in variants for sub, part in per_run[v].groupby("subsystem")]
    main_runs = runs[(runs.steps == main[0]) & (runs.aggregation == main[1])]
    seeds = main_runs.groupby(["subsystem", "seed"])[["top1", f"ap@{k}"]].mean()
    summary = []
    for sub, part in per_run[main].groupby("subsystem"):
        hidden, lookback = results["chosen"][sub]
        summary.append({"subsystem": sub, "hidden": hidden, "lookback_s": lookback * setting(cfg, "model_interval_s"),
                        "runs": len(part), "top1_run_mean": part.top1.mean(),
                        f"hit{k}_run_mean": part[f"hit@{k}"].mean(), f"recall{k}_run_mean": part[f"recall@{k}"].mean(),
                        f"map{k}_run_mean": part[f"ap@{k}"].mean(), "mrr_run_mean": part.rr.mean(),
                        f"ap{k}_scenario_macro": per_scenario[main].loc[per_scenario[main].subsystem == sub,
                                                                        f"ap@{k}"].mean(),
                        f"chance_map{k}_run_mean": part[f"chance_ap@{k}"].mean(),
                        f"map{k}_seed_min": seeds.loc[sub, f"ap@{k}"].min(), f"map{k}_seed_max": seeds.loc[sub, f"ap@{k}"].max(),
                        "top1_seed_min": seeds.loc[sub, "top1"].min(), "top1_seed_max": seeds.loc[sub, "top1"].max()})
    return {"summary": pd.DataFrame(summary), "per_scenario": per_scenario[main], "sensitivity": pd.DataFrame(sensitivity)}


# causal rca metrics


def metric_names(k: int) -> list[str]:
    return ["top1", f"hit@{k}", f"recall@{k}", f"ap@{k}", "rr"]


def causal_per_scenario(cfg: dict, runs: pd.DataFrame) -> pd.DataFrame:
    k = setting(cfg, "rca", "top_k")
    scored = runs[runs.scored.astype(bool)]
    return (scored.groupby(["subsystem", "graph", "fold"])
            .agg(runs=("recording_id", "size"), candidates=("n_candidates", "mean"), causes=("n_causes", "mean"),
                 top1=("top1", "mean"), **{f"ap{k}": (f"ap@{k}", "mean"), f"recall{k}": (f"recall@{k}", "mean")},
                 chance_top1=("chance_top1", "mean"), **{f"chance_ap{k}": (f"chance_ap@{k}", "mean")})
            .reset_index())


def causal_summary(cfg: dict, runs: pd.DataFrame, per_scenario: pd.DataFrame) -> pd.DataFrame:
    """run means with a bootstrap interval of map@k, scenario means and chance levels per subsystem and graph."""
    rc = setting(cfg, "rca")
    k = rc["top_k"]
    rows = []
    for (sub, graph), part in runs[runs.scored.astype(bool)].groupby(["subsystem", "graph"]):
        scenarios = per_scenario[(per_scenario.subsystem == sub) & (per_scenario.graph == graph)]
        low, high = rca.bootstrap_ci(part[f"ap@{k}"], rc["bootstrap_draws"], rc["seed"])
        rows.append({"subsystem": sub, "graph": graph, "runs_scored": len(part),
                     "runs_total": int(((runs.subsystem == sub) & (runs.graph == graph)).sum()),
                     "scenarios": len(scenarios), "top1_run_mean": part.top1.mean(),
                     f"map{k}_run_mean": part[f"ap@{k}"].mean(),
                     f"map{k}_run_mean_ci_low (runs, ignores scenario dependence)": low,
                     f"map{k}_run_mean_ci_high": high,
                     f"recall{k}_run_mean": part[f"recall@{k}"].mean(), f"hit{k}_run_mean": part[f"hit@{k}"].mean(),
                     "mrr_run_mean": part.rr.mean(), "top1_scenario_macro": scenarios.top1.mean(),
                     f"ap{k}_scenario_macro": scenarios[f"ap{k}"].mean(),
                     "chance_top1_run_mean": part.chance_top1.mean(), f"chance_map{k}_run_mean": part[f"chance_ap@{k}"].mean(),
                     f"chance_ap{k}_scenario_macro": scenarios[f"chance_ap{k}"].mean(),
                     "candidates_min": int(part.n_candidates.min()), "candidates_max": int(part.n_candidates.max())})
    return pd.DataFrame(rows)


def score_gaps(rankings: pd.DataFrame) -> pd.DataFrame:
    """per run: lowest score of an injected cause minus the highest score of another candidate."""
    rows = []
    for (sub, graph, fold, rid), part in rankings.groupby(["subsystem", "graph", "fold", "recording_id"]):
        causes, others = part[part.is_cause].score, part[~part.is_cause].score
        rows.append({"subsystem": sub, "graph": graph, "fold": fold, "recording_id": rid,
                     "gap": causes.min() - others.max() if len(causes) and len(others) else np.nan})
    return pd.DataFrame(rows)


def distinct_orders(rankings: pd.DataFrame) -> int:
    return rankings.sort_values(["recording_id", "rank"]).groupby("recording_id").variable.apply(tuple).nunique()


# graph and scm settings of the probe ranking


def decomposition(cfg: dict, sub: Subsystem, settings: pd.DataFrame) -> pd.DataFrame:
    """fold models of every graph version with the default and the selected scm settings."""
    k = setting(cfg, "rca", "top_k")
    original = OriginalGraphs(cfg, sub, exclude_related=True)
    chosen = settings.set_index(["subsystem", "graph", "scope"])
    max_features = setting(cfg, "scm", "max_table_features")

    def graph_of(kind: str, version: str, fold: str):
        if kind == "expert":
            return sub.expert
        if version == "original":
            return original.get("discovered", (fold,))
        return scm.load_fitted(sub.name, "discovered", data.scope_name(fold)).graph

    rows = []
    for kind, version in (("expert", "-"), ("discovered", "original"), ("discovered", "selected")):
        for scm_version in ("default", "selected"):
            metrics, rankings = [], []
            for fold in sub.folds:
                choice = (baseline(cfg) if scm_version == "default"
                          else chosen_setting(chosen.loc[(sub.name, kind, data.scope_name(fold))]))
                model = fit(sub, graph_of(kind, version, fold), data.fold_training_ids(sub.splits, fold), choice,
                            max_features)
                for rid in sub.runs_of(fold):
                    ranking = ranking_of(cfg, sub, model, scm.recording_series(sub.view, rid))
                    if ranking is None:
                        continue
                    metrics.append({"fold": fold, **rca.ranking_metrics(ranking["variable"], sub.manipulated[rid], k)})
                    rankings.append(ranking.assign(recording_id=rid))
            metrics = pd.DataFrame(metrics)
            per_fold = metrics.groupby("fold")[f"ap@{k}"].mean()
            rows.append({"graph": kind, "discovered_graph": version, "scm_settings": scm_version,
                         "top1_run_mean": metrics.top1.mean(), f"map{k}_run_mean": metrics[f"ap@{k}"].mean(),
                         f"ap{k}_scenario_macro": per_fold.mean(),
                         "distinct_rankings": distinct_orders(pd.concat(rankings, ignore_index=True)),
                         **{f"ap{k} {f}": v for f, v in per_fold.items()}})
    return pd.DataFrame(rows)


# tables


def ranking_table(cfg: dict, summary: pd.DataFrame, runs: pd.DataFrame, per_scenario: pd.DataFrame,
                  lstm: pd.DataFrame) -> pd.DataFrame:
    """run means of the five metrics and the scenario-average ap@k per subsystem and method, with chance."""
    k = setting(cfg, "rca", "top_k")
    columns = ["Top-1", f"Hit@{k}", f"Recall@{k}", f"MAP@{k}", "MRR", f"Scenario avg. AP@{k}"]
    fields = ["top1_run_mean", f"hit{k}_run_mean", f"recall{k}_run_mean", f"map{k}_run_mean", "mrr_run_mean",
              f"ap{k}_scenario_macro"]
    rows = []
    for sub in setting(cfg, "subsystems"):
        causal = summary[summary.subsystem == sub].set_index("graph")
        values = {g: [io.half_up(causal.loc[g, f], 2) for f in fields] for g in setting(cfg, "graphs")}
        if len({tuple(v) for v in values.values()}) == 1:
            rows.append((sub, "Causal, both graphs", next(iter(values.values()))))
        else:
            rows += [(sub, f"Causal, {g}", v) for g, v in values.items()]
        baseline_row = lstm[lstm.subsystem == sub].iloc[0]
        rows.append((sub, "LSTM + SHAP", [io.half_up(baseline_row[f], 2) for f in fields]))
        graph = setting(cfg, "graphs")[0]
        scored = runs[(runs.subsystem == sub) & (runs.graph == graph) & runs.scored.astype(bool)]
        chance_scenarios = per_scenario[(per_scenario.subsystem == sub) & (per_scenario.graph == graph)]
        chance = [scored[f"chance_{m}"].mean() for m in metric_names(k)] + [chance_scenarios[f"chance_ap{k}"].mean()]
        rows.append((sub, "Chance", [io.half_up(v, 2) for v in chance]))
    return pd.DataFrame([{"Subsystem": s.capitalize(), "Method": m, **dict(zip(columns, v))} for s, m, v in rows])


def decomposition_table(cfg: dict, table: pd.DataFrame) -> pd.DataFrame:
    """top-1, map@k, scenario-average ap@k and distinct orders per graph version and scm settings."""
    k = setting(cfg, "rca", "top_k")
    labels = {("expert", "-"): "Expert", ("discovered", "original"): "Discovered, original settings",
              ("discovered", "selected"): "Discovered, selected settings"}
    rows = []
    for (kind, version), part in table.groupby(["graph", "discovered_graph"], sort=False):
        printed = [{"Graph": labels[(kind, version)], "SCM settings": r.scm_settings.capitalize(),
                    "Top-1": io.half_up(r.top1_run_mean, 2), f"MAP@{k}": io.half_up(getattr(r, f"map{k}_run_mean"), 2),
                    f"Scenario avg. AP@{k}": io.half_up(getattr(r, f"ap{k}_scenario_macro"), 2),
                    "Orders": int(r.distinct_rankings)} for r in part.itertuples()]
        # a graph that ranks the same with both settings is shown once
        same = all({key: v for key, v in p.items() if key != "SCM settings"} ==
                   {key: v for key, v in printed[0].items() if key != "SCM settings"} for p in printed)
        rows += printed[:1] if same else printed
    return pd.DataFrame(rows)


def tuning_table(cfg: dict, tuning: pd.DataFrame) -> pd.DataFrame:
    """normal validation loss per hidden units and look-back, and the settings fixed in advance."""
    lc = setting(cfg, "lstm")
    interval = setting(cfg, "model_interval_s")
    columns = {(sub, lb): f"{sub.capitalize()} ℓ = {lb * interval:g} s"
               for sub in setting(cfg, "subsystems") for lb in lc["lookback_steps"]}
    losses = tuning.set_index(["subsystem", "hidden", "lookback_steps"]).normal_validation_loss
    rows = [{"Hidden units H": str(h), **{name: f"{losses[(s, h, lb)]:.5f}" for (s, lb), name in columns.items()}}
            for h in lc["hidden"]]
    main_steps, main_aggregation = lc["main"]
    attribution = {"abs": "absolute", "pos": "positive"}[main_aggregation]
    fixed = [("Optimiser; learning rate; batch size", f"Adam; {lc['learning_rate']:g}; {lc['batch_size']}"),
             (f"Epochs (early stopping, patience {lc['patience']})", f"at most {lc['max_epochs']}"),
             ("Seeds per fold", str(len(lc["seeds"]))),
             ("Explained steps; attribution", f"{lc['step_variants'][main_steps]} most anomalous; {attribution}"),
             ("SHAP method; background; samples", f"expected gradients; {lc['shap_background']}; {lc['shap_samples']}")]
    first = next(iter(columns.values()))
    rows.append({"Hidden units H": "Fixed in advance"})
    rows += [{"Hidden units H": label, first: value} for label, value in fixed]
    return pd.DataFrame(rows, columns=["Hidden units H", *columns.values()]).fillna("")


def sensitivity_table(cfg: dict, sensitivity: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    """map@k of one subsystem by explained steps and aggregation of the attributions."""
    k = setting(cfg, "rca", "top_k")
    part = sensitivity[sensitivity.subsystem == subsystem].set_index(["steps", "aggregation"])
    labels = {name: ("All steps of the window" if n is None else f"{n} most anomalous")
              for name, n in setting(cfg, "lstm", "step_variants").items()}
    names = {"abs": "Absolute attribution", "pos": "Positive attribution"}
    return pd.DataFrame([{"Explained steps": label,
                          **{names[a]: io.half_up(part.loc[(steps, a), f"map{k}_run_mean"], 2)
                             for a in setting(cfg, "lstm", "aggregations")}}
                         for steps, label in labels.items()])


# figures


def scenario_figure(cfg: dict, per_scenario: pd.DataFrame, lstm: pd.DataFrame) -> plt.Figure:
    """ap@k per test scenario for causal rca with both graphs and for lstm + shap, with chance marks."""
    k = setting(cfg, "rca", "top_k")
    kinds, subsystems = setting(cfg, "graphs"), setting(cfg, "subsystems")
    set_print_style()
    fig, axes = plt.subplots(len(subsystems), 1, figsize=(PAGE_WIDTH, 5.4), layout="constrained")
    for i, (ax, sub) in enumerate(zip(axes, subsystems)):
        part = per_scenario[per_scenario.subsystem == sub]
        folds = sorted(part.fold.unique())
        x = np.arange(len(folds))
        for j, graph in enumerate(kinds):
            values = part[part.graph == graph].set_index("fold").loc[folds]
            ax.bar(x + (j - 1) * 0.27, values[f"ap{k}"], width=0.27, color=GRAPH_COLORS[graph])
        ax.bar(x + 0.27, lstm[lstm.subsystem == sub].set_index("fold").loc[folds, f"ap@{k}"], width=0.27,
               color=BASELINE_COLOR)
        reference = part[part.graph == kinds[0]].set_index("fold").loc[folds]
        ax.scatter(x, reference[f"chance_ap{k}"], marker="_", s=500, linewidths=2, color=INK["primary"], zorder=3)
        ax.set_xticks(x, [f"{f}\nn = {int(reference.runs[f])}" for f in folds])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(f"AP@{k}")
        ax.set_title(f"({'abcdefgh'[i]}) {sub.capitalize()}", loc="left")
    axes[-1].set_xlabel("Test scenario")
    handles = [Patch(color=GRAPH_COLORS[g], label=f"Causal RCA, {GRAPH_LABELS[g].lower()}") for g in kinds]
    handles += [Patch(color=BASELINE_COLOR, label="LSTM + SHAP"),
                Line2D([], [], marker="_", markersize=14, markeredgewidth=2, linestyle="", color=INK["primary"],
                       label="Chance level")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    return fig


def attribution_figure(cfg: dict, importance: pd.DataFrame) -> plt.Figure:
    """shap score of every candidate in the first run of the example probe scenario (first seed)."""
    fold = setting(cfg, "figure_examples", "probe_fold")
    seed = setting(cfg, "lstm", "seeds")[0]
    rid = sorted(importance[(importance.subsystem == "probe") & (importance.fold == fold)].recording_id.unique())[0]
    part = importance[(importance.recording_id == rid) & (importance.seed == seed)].sort_values("importance")
    set_print_style()
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.3), layout="constrained")
    ax.barh(part.variable, part.importance, color=[COLORS[1] if c else COLORS[0] for c in part.is_cause])
    ax.set_xlabel("SHAP score")
    ax.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(color=COLORS[1], label="Injected cause"), Patch(color=COLORS[0], label="Other candidate")],
               loc="outside upper center", ncol=2, frameon=False)
    return fig


# messages


def ranking_message(cfg: dict, table: pd.DataFrame, summary: pd.DataFrame, lstm: dict,
                    gaps: pd.DataFrame, orders: dict) -> list[str]:
    k = setting(cfg, "rca", "top_k")
    graphs = setting(cfg, "graphs")
    low = f"map{k}_run_mean_ci_low (runs, ignores scenario dependence)"
    row = io.metric_row
    cell = 7
    lines = ["root-cause ranking (means over runs; scenario avg. weights scenarios equally)",
             row("", "".join(f"{h:<{cell}}" for h in ("top-1", f"hit@{k}", f"rec@{k}", f"map@{k}", "mrr", "scen.")))]
    for r in table.itertuples(index=False):
        lines.append(row(f"{r[0]} {r[1]}".lower(), "".join(f"{v:<{cell}.2f}" for v in r[2:])))
    ci = summary.set_index(["subsystem", "graph"])
    sensitivity = lstm["sensitivity"]
    seeds = lstm["summary"].set_index("subsystem")
    expert_gaps = gaps[gaps.graph == graphs[0]]
    for sub in setting(cfg, "subsystems"):
        variants = sensitivity[sensitivity.subsystem == sub][f"map{k}_run_mean"]
        lines += [
            row(f"{sub} map@{k} 95% run interval", ", ".join(f"{g} [{ci.loc[(sub, g), low]:.2f}, "
                                                          f"{ci.loc[(sub, g), f'map{k}_run_mean_ci_high']:.2f}]"
                                                          for g in graphs)),
            row(f"{sub} distinct rankings", ", ".join(f"{g} {orders[(sub, g)]}" for g in graphs)),
            row(f"{sub} lstm + shap map@{k} range", f"{seeds.loc[sub, f'map{k}_seed_min']:.2f}{io.DASH}"
                f"{seeds.loc[sub, f'map{k}_seed_max']:.2f} over seeds, {variants.min():.2f}{io.DASH}{variants.max():.2f} "
                "over explanation settings")]
        part = expert_gaps[expert_gaps.subsystem == sub]
        if len(part) and (part.gap > 0).all():
            per_fold = part.groupby("fold").gap.median()
            lines.append(row(f"{sub} cause score gap ({graphs[0]})", f"median {part.gap.median():.2f} nats, "
                             f"scenarios {per_fold.min():.2f}{io.DASH}{per_fold.max():.2f}"))
    lines.append(ranking_sentence(cfg, table, ci, low))
    return lines


def ranking_sentence(cfg: dict, table: pd.DataFrame, ci: pd.DataFrame, low: str) -> str:
    """which method ranks best per subsystem, built from the printed map@k."""
    k = setting(cfg, "rca", "top_k")
    printed = table.set_index(["Subsystem", "Method"])
    clauses = []
    for sub in setting(cfg, "subsystems"):
        maps = printed.loc[sub.capitalize()][f"MAP@{k}"].drop("Chance")
        if (maps == 1).all():
            clauses.append(f"every method ranks all {sub} causes on top")
            continue
        best = maps.idxmax()
        clause = f"for {sub} {method_name(best)} ranks best"
        bounds = ci.loc[sub]
        if best.startswith("Causal") and bounds[low].max() < bounds[f"map{k}_run_mean_ci_high"].min():
            clause += ", though the run intervals of the two graphs overlap"
        clauses.append(clause)
    return "; ".join(clauses) + " (retrospective diagnosis of injected faults)."


def method_name(method: str) -> str:
    """'Causal, expert' -> 'causal rca with the expert graph', other methods in lower case."""
    if method.startswith("Causal, "):
        return f"causal rca with the {method.removeprefix('Causal, ')} graph"
    return method.lower()


def decomposition_message(cfg: dict, table: pd.DataFrame) -> list[str]:
    k = setting(cfg, "rca", "top_k")
    row = io.metric_row
    lines = ["probe ranking by graph and scm settings (fold models; means over runs)",
             row("", "top-1", f"map@{k}", "scenario avg.", "rankings")]
    graph_labels = {"Expert": "expert", "Discovered, original settings": "discovered (original)",
                    "Discovered, selected settings": "discovered (selected)"}
    for r in table.itertuples(index=False):
        lines.append(row(f"{graph_labels[r[0]]}, {r[1].lower()} scm", f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}",
                         f"{r[5]}"))
    maps = table.set_index(["Graph", "SCM settings"])[f"MAP@{k}"]
    selected = "Discovered, selected settings"
    settings_gain = maps[(selected, "Selected")] - maps[(selected, "Default")]
    graph_gain = maps[(selected, "Default")] - maps[("Discovered, original settings", "Default")]
    if settings_gain > graph_gain:
        effect = "lowers" if graph_gain < 0 else "raises less" if graph_gain > 0 else "leaves"
        lines.append(f"the gain over the original configuration comes from the selected scm settings, not from the new "
                     f"discovered graph, which alone {effect} map@{k}{' unchanged' if graph_gain == 0 else ''}.")
    else:
        lines.append(f"the gain over the original configuration comes from the new discovered graph more than from the "
                     f"selected scm settings, judged by map@{k} of the fold models.")
    return lines


def run(cfg: dict, force: bool = False) -> None:
    runs = io.read_input("rca_runs")
    rankings = io.read_input("rca_rankings")
    settings = io.read_input("scm_settings")
    per_scenario = causal_per_scenario(cfg, runs)
    summary = causal_summary(cfg, runs, per_scenario)
    gaps = score_gaps(rankings)
    orders = {(s, g): distinct_orders(part) for (s, g), part in rankings.groupby(["subsystem", "graph"])}
    focus = setting(cfg, "lstm", "sensitivity_subsystem")
    log.info(f"decomposing the {focus} ranking by graph and scm settings")
    parts = decomposition(cfg, load_subsystem(cfg, focus), settings)
    results = lstm_results(cfg, force=force)
    lstm = lstm_summaries(cfg, results)

    io.save_result(per_scenario, "rca_per_scenario")
    io.save_result(summary, "rca_summary")
    io.save_result(gaps, "rca_score_gaps")
    io.save_result(parts, "rca_decomposition")
    io.save_result(results["tuning"], "lstm_shap_tuning")
    io.save_result(results["runs"], "lstm_shap_runs")
    io.save_result(results["importance"], "lstm_shap_importance")
    for name, table in lstm.items():
        io.save_result(table, f"lstm_shap_{name}")

    ranking = ranking_table(cfg, summary, runs, per_scenario, lstm["summary"])
    printed_parts = decomposition_table(cfg, parts)
    io.save_table(tuning_table(cfg, results["tuning"]), "5.8", "Settings of the LSTM + SHAP baseline", "evaluation")
    io.save_table(ranking, "6.4", "Root-cause ranking performance", "evaluation")
    io.save_table(printed_parts, "6.5", "Probe ranking by graph and SCM settings", "evaluation")
    io.save_table(sensitivity_table(cfg, lstm["sensitivity"], focus), "6.6", "Sensitivity of the LSTM + SHAP baseline",
                  "evaluation")
    io.save_figure(scenario_figure(cfg, per_scenario, lstm["per_scenario"]), "6.5",
                   "Root-cause ranking by fault scenario", "evaluation")
    io.save_figure(attribution_figure(cfg, results["importance"]), "6.6", "SHAP attribution in a probe fault run",
                   "evaluation")
    io.report(ranking_message(cfg, ranking, summary, lstm, gaps, orders))
    io.report(decomposition_message(cfg, printed_parts))
