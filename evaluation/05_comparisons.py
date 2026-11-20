import time
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pandas as pd

from core import data, discovery
from core.data import Subsystem, load_subsystem
from core.lstm_shap import lstm_per_run, lstm_results, lstm_summaries
from utils import io
from utils.config import setting
from utils.errors import DataError, MissingInputError
from utils.logger import log

DISCOVERED = "discovered"
NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

# step files start with their number, so they are imported by name
counterfactual_evaluation = import_module("evaluation.04_counterfactuals")
rca_evaluation = import_module("evaluation.03_root_cause_analysis")
scm_evaluation = import_module("evaluation.02_scm")
counterfactuals = import_module("pipeline.06_counterfactuals")
interventions = import_module("pipeline.05_interventions")
root_cause_analysis = import_module("pipeline.04_root_cause_analysis")
scm_fitting = import_module("pipeline.03_scm_fitting")
causal_discovery = import_module("pipeline.02_causal_discovery")


# alternative runs of the pipeline, saved in data/results/comparisons


def variant_dir(variant: str) -> Path:
    return io.results_dir("comparisons") / variant


def save_variant(tables: dict[str, pd.DataFrame], variant: str) -> None:
    for name, table in tables.items():
        io.write_csv(table, variant_dir(variant) / f"{name}.csv")


def fixed_settings(cfg: dict, subsystem: str, scopes: pd.Series) -> pd.DataFrame:
    """the baseline pseudo-count without extra lag for both graphs and every scope."""
    alpha, lag = scm_fitting.baseline(cfg)
    table = pd.DataFrame([{"subsystem": subsystem, "graph": graph, "scope": scope, "alpha": alpha, "lag": lag}
                          for graph in setting(cfg, "graphs") for scope in scopes])
    table["lag"] = table["lag"].astype("Int64")
    return table


def rank_both_graphs(cfg: dict, sub: Subsystem, models: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = [root_cause_analysis.rank_subsystem(cfg, sub, graph, models) for graph in setting(cfg, "graphs")]
    return (pd.concat([runs for runs, _ in results], ignore_index=True),
            pd.concat([rankings for _, rankings in results], ignore_index=True))


def original_configuration(cfg: dict, splits: pd.DataFrame, force: bool) -> dict[str, pd.DataFrame]:
    """every stage with the original discovery settings and first-order scms with the baseline pseudo-count."""
    variant = "original"
    models = variant_dir(variant) / "scm"
    config = discovery.config_from_dict(setting(cfg, "discovery", "original"))
    order = causal_discovery.selection_order(setting(cfg, "discovery"))
    started = time.time()
    parts: dict[str, list[pd.DataFrame]] = {}
    for name in setting(cfg, "subsystems"):
        log.info(f"running the {name} pipeline with the original settings")
        context = causal_discovery.DiscoveryContext(cfg, name, splits)
        scopes = [()] + [(k,) for k in context.folds]
        edges, discovery_settings = causal_discovery.discover(cfg, context, [config], order, scopes, force, started)
        sub = load_subsystem(cfg, name)
        dags, _ = scm_fitting.discovered_dags(sub, edges)
        scm_settings = fixed_settings(cfg, name, discovery_settings["scope"])
        scm_fitting.fit_selected(cfg, sub, dags, scm_settings, directory=models)
        normal_rows, _ = scm_evaluation.normal_prediction(cfg, sub, models)
        fault_rows, _ = scm_evaluation.fault_prediction(cfg, sub, models)
        runs, rankings = rank_both_graphs(cfg, sub, models)
        for key, table in (("discovered_graphs", edges), ("discovery_settings", discovery_settings),
                           ("scm_graphs", scm_fitting.scm_graph_rows(sub, dags)), ("scm_settings", scm_settings),
                           ("scm_prediction", pd.DataFrame(normal_rows + fault_rows)), ("rca_runs", runs),
                           ("rca_rankings", rankings)):
            parts.setdefault(key, []).append(table)
    tables = {key: pd.concat(frames, ignore_index=True) for key, frames in parts.items()}

    s = replace(load_subsystem(cfg, setting(cfg, "interventions", "subsystem")), scm_dir=models)
    tables["replay"] = interventions.replay_runs(cfg, s)
    s = replace(load_subsystem(cfg, setting(cfg, "counterfactuals", "subsystem")), scm_dir=models)
    rankings = tables["rca_rankings"]
    tables["recommendations"], _, tables["recommendation_refits"] = counterfactuals.explain_all(
        cfg, s, rankings[rankings["subsystem"] == s.name], tables["scm_settings"])
    save_variant(tables, variant)
    return tables


def without_exclusion(cfg: dict, splits: pd.DataFrame, name: str, force: bool) -> dict[str, pd.DataFrame]:
    """discovery and scm selection, fitting and rca of one subsystem without removing near-duplicate partners."""
    log.info(f"running the {name} pipeline without the near-duplicate exclusion")
    variant = "without_exclusion"
    models = variant_dir(variant) / "scm"
    study = setting(cfg, "discovery")
    context = causal_discovery.DiscoveryContext(cfg, name, splits, exclude_related=False)
    edges, discovery_settings = causal_discovery.discover(
        cfg, context, discovery.grid_configs(study["grid"]), causal_discovery.selection_order(study),
        context.scopes(), force, time.time())
    sub = load_subsystem(cfg, name)
    dags, _ = scm_fitting.discovered_dags(sub, edges)
    _, chosen = scm_fitting.select_settings(cfg, sub, dags, exclude_related=False)
    scm_settings = pd.DataFrame(chosen)
    scm_settings["lag"] = scm_settings["lag"].astype("Int64")
    scm_fitting.fit_selected(cfg, sub, dags, scm_settings, exclude_related=False, directory=models)
    fault_rows, _ = scm_evaluation.fault_prediction(cfg, sub, models)
    runs, rankings = rank_both_graphs(cfg, sub, models)
    tables = {"discovered_graphs": edges, "discovery_settings": discovery_settings, "scm_settings": scm_settings,
              "scm_prediction": pd.DataFrame(fault_rows), "rca_runs": runs, "rca_rankings": rankings}
    save_variant(tables, variant)
    return tables


def near_duplicate_subsystem(cfg: dict, splits: pd.DataFrame) -> str:
    """the subsystem whose recordings include near-duplicates."""
    related = set(data.related_partners())
    names = [s for s in setting(cfg, "subsystems")
             if splits.loc[splits["subsystem"] == s, "recording_id"].isin(related).any()]
    if len(names) != 1:
        raise DataError(f"expected near-duplicates in one subsystem, found {names or 'none'} "
                        f"({io.relative(io.related_recordings_path())})")
    return names[0]


# results of the main pipeline


def evaluation_result(name: str, part: str) -> pd.DataFrame:
    file = io.results_dir("evaluation") / f"{name}.csv"
    if not file.exists():
        raise MissingInputError(f"{io.relative(file)} not found: run \"python main.py evaluate {part}\" first")
    return pd.read_csv(file)


def selected_configuration() -> dict[str, pd.DataFrame]:
    tables = {name: io.read_input(name)
              for name in ("scm_graphs", "rca_runs", "replay", "recommendations", "recommendation_refits")}
    tables["discovery_settings"] = io.read_input("discovery_settings", keep_default_na=False)
    tables["scm_prediction"] = evaluation_result("scm_prediction", "scm")
    return tables


def with_rca_summary(cfg: dict, tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    runs = tables["rca_runs"]
    summary = rca_evaluation.causal_summary(cfg, runs, rca_evaluation.causal_per_scenario(cfg, runs)).set_index(["subsystem", "graph"])
    return {**tables, "rca_summary": summary}


# table cells


def numbers(text: str) -> list[float]:
    """'0.8 s/3.0 s' -> [0.8, 3.0]."""
    return [float(part.removesuffix(" s")) for part in text.split("/")]


def joined(values: list[float], decimals: int) -> str:
    return "/".join(io.printed(v, decimals) for v in values)


def prediction(tables: dict, sub: str, graph: str, column: str, normal: bool) -> float:
    """normal test recordings (normal) or all held-out scenarios, non-manipulated variables."""
    t = tables["scm_prediction"]
    if normal:
        rows = (t["track"] == "N") & (t["split"] == "test")
    else:
        rows = (t["split"] == scm_evaluation.ALL_SCENARIOS) & (t["variables"] == scm_evaluation.NON_MANIPULATED)
    return float(t.loc[rows & (t["subsystem"] == sub) & (t["graph"] == graph), column].iloc[0])


def discovered_f1(tables: dict, sub: str) -> float:
    settings = tables["discovery_settings"]
    return float(settings.loc[(settings["subsystem"] == sub) & (settings["scope"] == "all"), "directed_f1"].iloc[0])


def scm_edges(tables: dict, sub: str) -> int:
    edges = tables["scm_graphs"]
    return int(((edges["subsystem"] == sub) & (edges["graph"] == DISCOVERED) & (edges["scope"] == "all")).sum())


def activation_error(cfg: dict, tables: dict) -> str:
    """mean and maximum onset error of the replayed activations (once when both graphs agree)."""
    replay = tables["replay"]
    texts = {}
    for graph in setting(cfg, "graphs"):
        part = replay[replay["graph"] == graph]
        mean = counterfactual_evaluation.replay_summary(part, [], [graph])["onset_abs_error_mean_s"].iloc[0]
        maximum = counterfactual_evaluation.replay_stats(part, setting(cfg, "model_interval_s"))["max"]
        texts[graph] = f"{io.printed(mean, 1)} s/{io.printed(maximum, 1)} s"
    if len(set(texts.values())) == 1:
        return next(iter(texts.values()))
    return "; ".join(f"{text} ({graph})" for graph, text in texts.items())


def original_metric(cfg: dict, tables: dict, sub: str, metric: str) -> str:
    """one cell of the original/selected table for one configuration."""
    k = setting(cfg, "rca", "top_k")
    graphs = setting(cfg, "graphs")
    recommendations = tables["recommendations"]
    refits = tables["recommendation_refits"]
    per_graph = {
        "normal_change_log_loss": lambda g: prediction(tables, sub, g, "change_log_loss", normal=True),
        "fault_log_loss": lambda g: prediction(tables, sub, g, "log_loss", normal=False),
        "fault_change_log_loss": lambda g: prediction(tables, sub, g, "change_log_loss", normal=False),
        "top1": lambda g: tables["rca_summary"].loc[(sub, g), "top1_run_mean"],
        "map": lambda g: tables["rca_summary"].loc[(sub, g), f"map{k}_run_mean"],
        "scenario_ap": lambda g: tables["rca_summary"].loc[(sub, g), f"ap{k}_scenario_macro"],
        "valid_recommendations": lambda g: counterfactual_evaluation.valid_mask(recommendations[recommendations["graph"] == g]).sum(),
        "mean_validity": lambda g: recommendations.loc[recommendations["graph"] == g, "validity_independent"].mean(),
        "lowest_refit_validity": lambda g: refits.loc[refits["graph"] == g, "validity"].min(),
    }
    decimals = {"fault_log_loss": 3, "valid_recommendations": 0, "mean_validity": 3}
    if metric in per_graph:
        return joined([per_graph[metric](g) for g in graphs], decimals.get(metric, 2))
    if metric == "directed_f1":
        return io.printed(discovered_f1(tables, sub), 2)
    if metric == "directed_f1_scm_edges":
        return f"{io.printed(discovered_f1(tables, sub), 2)}/{scm_edges(tables, sub)}"
    if metric == "activation_error":
        return activation_error(cfg, tables)
    raise DataError(f"unknown row {metric!r} in comparisons.original_rows of config/config.yaml")


def original_labels(k: int) -> dict[str, str]:
    return {
        "directed_f1": "Discovered graph, directed F1",
        "directed_f1_scm_edges": "Discovered graph, directed F1/SCM edges",
        "normal_change_log_loss": "LL at changes, normal test (E/D)",
        "fault_log_loss": "Log loss, fault (E/D)",
        "fault_change_log_loss": "LL at changes, fault (E/D)",
        "top1": "Top-1 (E/D)",
        "map": f"MAP@{k} (E/D)",
        "scenario_ap": f"Scenario-average AP@{k} (E/D)",
        "activation_error": "Activation-time error, mean/maximum",
        "valid_recommendations": "Valid recommendations (E/D)",
        "mean_validity": "Mean validity (E/D)",
        "lowest_refit_validity": "Lowest validity, resampled SCMs (E/D)",
    }


def original_table(cfg: dict, original: dict, selected: dict) -> pd.DataFrame:
    labels = original_labels(setting(cfg, "rca", "top_k"))
    rows = []
    for sub, metrics in setting(cfg, "comparisons", "original_rows").items():
        for metric in metrics:
            rows.append({"Subsystem": sub.capitalize(), "Result": labels[metric],
                         "Original": original_metric(cfg, original, sub, metric),
                         "Selected": original_metric(cfg, selected, sub, metric), "_metric": metric})
    return pd.DataFrame(rows)


def protocol_values(cfg: dict, tables: dict, lstm: dict, sub: str) -> dict:
    """expert, discovered and lstm + shap values of the ranking metrics, overall and by near-duplicate status."""
    k = setting(cfg, "rca", "top_k")
    related = set(data.related_partners())
    runs = tables["rca_runs"]
    runs = runs[(runs["subsystem"] == sub) & runs["scored"].astype(bool)]
    baseline_runs = lstm_per_run(cfg, lstm["runs"], *setting(cfg, "lstm", "main"))
    baseline_runs = baseline_runs[baseline_runs["subsystem"] == sub]
    per_run = [runs[runs["graph"] == g] for g in setting(cfg, "graphs")] + [baseline_runs]
    summaries = [tables["rca_summary"].loc[(sub, g)] for g in setting(cfg, "graphs")]
    summaries.append(lstm_summaries(cfg, lstm)["summary"].set_index("subsystem").loc[sub])
    return {
        "top1": [s["top1_run_mean"] for s in summaries],
        "map": [s[f"map{k}_run_mean"] for s in summaries],
        "scenario_ap": [s[f"ap{k}_scenario_macro"] for s in summaries],
        "related_ap": [r.loc[r["recording_id"].isin(related), f"ap@{k}"].mean() for r in per_run],
        "other_ap": [r.loc[~r["recording_id"].isin(related), f"ap@{k}"].mean() for r in per_run],
        "related_runs": int(per_run[0]["recording_id"].isin(related).sum()),
        "other_runs": int((~per_run[0]["recording_id"].isin(related)).sum()),
        "change_log_loss": [prediction(tables, sub, g, "change_log_loss", normal=False)
                            for g in setting(cfg, "graphs")],
    }


def changed_settings(without: pd.DataFrame, main: pd.DataFrame, sub: str) -> tuple[int, int]:
    """outer folds whose selected discovery configuration differs, and the number of outer folds."""
    def outer(table: pd.DataFrame) -> pd.Series:
        part = table[(table["subsystem"] == sub) & table["scope"].map(lambda s: len(data.scope_held_out(s)) == 1)]
        return part.set_index("scope")["config_id"]

    before, after = outer(without), outer(main)
    return int((before != after.loc[before.index]).sum()), len(after)


def word(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)


def exclusion_table(cfg: dict, without: dict, main: dict, changed: tuple[int, int]) -> pd.DataFrame:
    k = setting(cfg, "rca", "top_k")
    rows = [("LL at changes (E/D)", "change_log_loss"), ("Top-1 (E/D/L)", "top1"), (f"MAP@{k} (E/D/L)", "map"),
            (f"Scenario-average AP@{k} (E/D/L)", "scenario_ap"),
            (f"AP@{k} of the {word(main['related_runs'])} runs with a near-duplicate (E/D/L)", "related_ap"),
            (f"AP@{k} of the other {main['other_runs']} runs (E/D/L)", "other_ap")]
    table = [{"Result": label, "Without": joined(without[key], 2), "With": joined(main[key], 2), "_metric": key}
             for label, key in rows]
    table.append({"Result": "Discovered fold graphs with changed settings", "Without": io.DASH,
                  "With": f"{changed[0]} of {changed[1]}", "_metric": "changed_settings"})
    return pd.DataFrame(table)


# messages


def direction(before: str, after: str) -> str:
    """'raise', 'keep', 'lower' or 'change' for two printed cells such as '0.48/0.56'."""
    old, new = numbers(before), numbers(after)
    if old == new:
        return "keep"
    if all(b >= a for a, b in zip(old, new)):
        return "raise"
    if all(b <= a for a, b in zip(old, new)):
        return "lower"
    return "change"


def and_list(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"


def original_clause(cells: pd.DataFrame) -> str:
    """'raise graph agreement and diagnosis' for the original and selected cells of one subsystem."""
    names = {"directed_f1": "graph agreement", "map": "diagnosis", "valid_recommendations": "valid recommendations"}
    pairs = {metric: (row.Original, row.Selected) for metric, row in cells.iterrows()}
    # the graph row may also hold the scm edges. its first number is the directed f1
    graph_row = next(metric for metric in pairs if metric.startswith("directed_f1"))
    pairs["directed_f1"] = tuple(text.split("/")[0] for text in pairs[graph_row])
    verbs: dict[str, list[str]] = {}
    for metric, name in names.items():
        if metric in pairs:
            verbs.setdefault(direction(*pairs[metric]), []).append(name)
    order = ("raise", "keep", "lower", "change")
    return " and ".join(f"{verb} {and_list(verbs[verb])}" for verb in order if verb in verbs)


def original_message(table: pd.DataFrame) -> list[str]:
    row = io.metric_row
    lines = ["original and selected configuration (e/d = expert/discovered graph)", row("", "original", "selected")]
    for sub, rows in table.groupby("Subsystem", sort=False):
        lines.append(f"  {sub.lower()}")
        lines += [row(r.Result.lower(), r.Original, r.Selected) for r in rows.itertuples()]
    clauses = [f"for {sub.lower()} " + ("the selected settings " if i == 0 else "they ")
               + original_clause(rows.set_index("_metric"))
               for i, (sub, rows) in enumerate(table.groupby("Subsystem", sort=False))]
    lines.append(", and ".join(clauses) + ". agreement is judged against the expert graph used for selection.")
    return lines


def exclusion_message(cfg: dict, table: pd.DataFrame, without: dict, main: dict, sub: str) -> list[str]:
    k = setting(cfg, "rca", "top_k")
    widths = {"label_width": 54, "cell_width": 18}
    lines = [f"{sub} results without and with the near-duplicate exclusion (e/d/l = expert, discovered, lstm + shap)",
             io.metric_row("", "without", "with", **widths)]
    lines += [io.metric_row(r.Result.lower(), r.Without, r.With, **widths) for r in table.itertuples()]
    methods = ("the expert graph", "the discovered graph", "lstm + shap")
    maps = [(name, float(io.printed(before, 2)), float(io.printed(after, 2)))
            for name, before, after in zip(methods, without["map"], main["map"])]
    unchanged = [name for name, before, after in maps if before == after]
    moved = [name for name, before, after in maps if before != after]
    shift = max((abs(after - before) for _, before, after in maps), default=0.0)
    if not moved:
        effect = f"leaves map@{k} of every method unchanged"
    elif not unchanged:
        effect = f"moves map@{k} of every method by at most {shift:.2f}"
    else:
        effect = f"leaves map@{k} of {and_list(unchanged)} unchanged and moves {and_list(moved)} by at most {shift:.2f}"
    lines.append(f"excluding the near-duplicates {effect}. the shift mixes recording dependence, less training data "
                 "and changed settings.")
    return lines


def printed_columns(table: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns="_metric")


def run(cfg: dict, force: bool = False) -> None:
    splits = io.read_input("splits")
    selected = with_rca_summary(cfg, selected_configuration())
    original = with_rca_summary(cfg, original_configuration(cfg, splits, force))
    sub = near_duplicate_subsystem(cfg, splits)
    without = with_rca_summary(cfg, without_exclusion(cfg, splits, sub, force))
    lstm_without = lstm_results(cfg, exclude_related=False, force=force)
    lstm_with = lstm_results(cfg)

    comparison = original_table(cfg, original, selected)
    changed = changed_settings(without["discovery_settings"], selected["discovery_settings"], sub)
    before = protocol_values(cfg, without, lstm_without, sub)
    after = protocol_values(cfg, selected, lstm_with, sub)
    exclusion = exclusion_table(cfg, before, after, changed)
    io.save_table(printed_columns(comparison), "A.3", "Results with original and selected settings")
    io.save_table(printed_columns(exclusion), "A.5", "Results before and after near-duplicate exclusion")
    io.report(original_message(comparison))
    io.report(exclusion_message(cfg, exclusion, before, after, sub))
