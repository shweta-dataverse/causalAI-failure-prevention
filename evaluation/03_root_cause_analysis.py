from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from core import data, lstm_shap, rca, scm
from core.data import Subsystem, load_subsystem
from core.plots import BASELINE_COLOR, COLORS, GRAPH_COLORS, GRAPH_LABELS, INK, PAGE_WIDTH, set_print_style
from utils import io
from utils.config import setting
from utils.logger import log

# step files start with their number, so they are imported by name
scm_evaluation = import_module("evaluation.02_scm")
root_cause_analysis = import_module("pipeline.04_root_cause_analysis")
scm_fitting = import_module("pipeline.03_scm_fitting")


# causal rca metrics


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
    original = scm_evaluation.OriginalGraphs(cfg, sub, exclude_related=True)
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
                choice = (scm_fitting.baseline(cfg) if scm_version == "default"
                          else scm_fitting.chosen_setting(chosen.loc[(sub.name, kind, data.scope_name(fold))]))
                model = scm_fitting.fit(sub, graph_of(kind, version, fold), data.fold_training_ids(sub.splits, fold), choice,
                            max_features)
                for rid in sub.runs_of(fold):
                    ranking = root_cause_analysis.ranking_of(cfg, sub, model, scm.recording_series(sub.view, rid))
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
        chance = [scored[f"chance_{m}"].mean() for m in rca.metric_names(k)] + [chance_scenarios[f"chance_ap{k}"].mean()]
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
    results = lstm_shap.lstm_results(cfg, force=force)
    lstm = lstm_shap.lstm_summaries(cfg, results)

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
    io.save_table(tuning_table(cfg, results["tuning"]), "5.8", "Settings of the LSTM + SHAP baseline")
    io.save_table(ranking, "6.4", "Root-cause ranking performance")
    io.save_table(printed_parts, "6.5", "Probe ranking by graph and SCM settings")
    io.save_table(sensitivity_table(cfg, lstm["sensitivity"], focus), "6.6", "Sensitivity of the LSTM + SHAP baseline")
    io.save_figure(scenario_figure(cfg, per_scenario, lstm["per_scenario"]), "6.5",
                   "Root-cause ranking by fault scenario")
    io.save_figure(attribution_figure(cfg, results["importance"]), "6.6", "SHAP attribution in a probe fault run")
    io.report(ranking_message(cfg, ranking, summary, lstm, gaps, orders))
    io.report(decomposition_message(cfg, printed_parts))
