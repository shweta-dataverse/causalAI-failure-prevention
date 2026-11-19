from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from core import data, discovery, graphs, rca, scm
from core.plots import GRAPH_COLORS, GRAPH_LABELS, PAGE_WIDTH, set_print_style
from pipeline.root_cause_analysis import ranking_of
from pipeline.scm_fitting import Subsystem, baseline, chosen_setting, fit, grid, load_subsystem, preference
from utils import io
from utils.config import setting
from utils.errors import MissingInputError
from utils.logger import log

NORMAL_SPLITS = ("train", "validation", "test")
NON_MANIPULATED = "non-manipulated (mechanism prediction)"
MANIPULATED = "manipulated (mechanism overridden)"
ALL_SCENARIOS = "all held-out scenarios (scenario macro)"


# one-step predictions and their aggregation


def predictions(model: scm.TemporalSCM, sub: Subsystem, ids: list[str]) -> pd.DataFrame:
    """one row per recording, variable and step with p, y, prev, manipulated and support flags."""
    frames = []
    for r in dict.fromkeys(ids):
        series = scm.recording_series(sub.view, r)
        for v, mech in model.mechanisms.items():
            if v not in series:
                continue
            p = model.transition_probs(series, v)
            y, prev = series[v][1:], series[v][:-1]
            ok = ~np.isnan(p) & ~np.isnan(y)
            if not ok.any():
                continue
            features = scm.parent_features(series, mech.parents, model.fills, mech.lags)[ok]
            codes = features.astype(int) @ (1 << np.arange(mech.width)) if mech.width else np.zeros(ok.sum(), dtype=int)
            seen = (np.array([mech.counts[state][c] > 0 for state, c in zip(prev[ok].astype(int), codes)])
                    if mech.has_data else np.zeros(ok.sum(), bool))
            frames.append(pd.DataFrame({"recording_id": r, "variable": v, "step": np.flatnonzero(ok) + 1,
                                        "p": p[ok], "y": y[ok], "prev": prev[ok],
                                        "manipulated": v in sub.manipulated.get(r, set()),
                                        "configuration_seen": seen, "mechanism_has_data": mech.has_data}))
    out = pd.concat(frames, ignore_index=True)
    out["log_loss"] = -np.log(np.where(out.y == 1, out.p, 1 - out.p))
    out["brier"] = (out.p - out.y) ** 2
    out["change"] = out.y != out.prev
    return out


def macro(pred: pd.DataFrame, scenario: dict | None = None) -> dict:
    """rows -> recording-variable -> recording (-> scenario) means, plus pooled row means."""
    def chain(rows: pd.DataFrame, column: str) -> tuple[float, int]:
        if rows.empty:
            return np.nan, 0
        per_recording = rows.groupby(["recording_id", "variable"])[column].mean().groupby(level="recording_id").mean()
        if scenario is None:
            return float(per_recording.mean()), len(per_recording)
        return float(per_recording.groupby(per_recording.index.map(scenario)).mean().mean()), len(per_recording)

    def pooled(rows: pd.DataFrame, column: str) -> float:
        return float(rows[column].mean()) if len(rows) else np.nan

    log_loss, recordings = chain(pred, "log_loss")
    brier, _ = chain(pred, "brier")
    change_log_loss, with_changes = chain(pred[pred.change], "log_loss")
    return {"log_loss": log_loss, "brier": brier, "change_log_loss": change_log_loss, "recordings": recordings,
            "recordings_with_changes": with_changes, "rows": len(pred), "change_rows": int(pred.change.sum()),
            "pooled_log_loss": pooled(pred, "log_loss"), "pooled_brier": pooled(pred, "brier"),
            "pooled_change_log_loss": pooled(pred[pred.change], "log_loss"),
            "share_rows_unseen_configuration": float((~pred.configuration_seen).mean()) if len(pred) else np.nan,
            "share_rows_mechanism_without_data": float((~pred.mechanism_has_data).mean()) if len(pred) else np.nan}


def normal_prediction(cfg: dict, sub: Subsystem, directory: Path | None = None) -> tuple[list[dict], list[dict]]:
    """the model without a held-out scenario scored on every normal split, overall and per variable."""
    rows, per_variable = [], []
    for graph in setting(cfg, "graphs"):
        model = scm.load_fitted(sub.name, graph, "all", directory)
        for split in NORMAL_SPLITS:
            pred = predictions(model, sub, sub.ids(split))
            rows.append({"subsystem": sub.name, "track": "N", "graph": graph, "split": split,
                         "status": "in-sample" if split == "train" else "held out from fit", "variables": "all",
                         **macro(pred)})
            variables = pred.groupby("variable").agg(rows=("p", "size"), changes=("change", "sum"),
                                                     log_loss=("log_loss", "mean"), brier=("brier", "mean"))
            per_variable += [{"subsystem": sub.name, "track": "N", "graph": graph, "split": split, **r}
                             for r in variables.reset_index().to_dict("records")]
    return rows, per_variable


def fault_prediction(cfg: dict, sub: Subsystem, directory: Path | None = None) -> tuple[list[dict], pd.DataFrame]:
    """every held-out scenario predicted by the model of its fold, then all scenarios with equal weight."""
    scenario = dict(zip(sub.splits.recording_id, sub.splits.fold))
    rows, frames = [], []
    for graph in setting(cfg, "graphs"):
        for fold in sub.folds:
            model = scm.load_fitted(sub.name, graph, data.scope_name(fold), directory)
            pred = predictions(model, sub, sub.runs_of(fold))
            frames.append(pred.assign(subsystem=sub.name, graph=graph, fold=fold))
            for label, part in ((NON_MANIPULATED, pred[~pred.manipulated]), (MANIPULATED, pred[pred.manipulated])):
                rows.append({"subsystem": sub.name, "track": "F", "graph": graph, "split": f"held-out {fold}",
                             "status": "outer test", "variables": label, **macro(part)})
    pooled = pd.concat(frames, ignore_index=True)
    for graph in setting(cfg, "graphs"):
        pred = pooled[pooled.graph == graph]
        for label, part in ((NON_MANIPULATED, pred[~pred.manipulated]), (MANIPULATED, pred[pred.manipulated])):
            rows.append({"subsystem": sub.name, "track": "F", "graph": graph, "split": ALL_SCENARIOS,
                         "status": "outer test", "variables": label, **macro(part, scenario)})
    return rows, pooled


# diagnostics of the normal-prediction model


def refit_stability(cfg: dict, sub: Subsystem, settings: pd.DataFrame) -> list[dict]:
    """expert model refitted on bootstrap samples of its training recordings, scored on validation and test."""
    choice = chosen_setting(settings.set_index(["subsystem", "graph", "scope"]).loc[(sub.name, "expert", "all")])
    train = data.fold_training_ids(sub.splits, None)
    rng = np.random.default_rng(setting(cfg, "scm", "seed"))
    rows = []
    for b in range(setting(cfg, "scm", "normal_refits")):
        sample = list(rng.choice(train, size=len(train), replace=True))
        model = fit(sub, sub.expert, sample, choice, setting(cfg, "scm", "max_table_features"))
        for split in ("validation", "test"):
            rows.append({"subsystem": sub.name, "graph": "expert", "refit": b, "split": split,
                         "unique_recordings_in_sample": len(set(sample)),
                         **macro(predictions(model, sub, sub.ids(split)))})
    return rows


def normal_diagnostics(cfg: dict, sub: Subsystem) -> pd.DataFrame:
    """quantile residuals on normal validation and test, simulated against observed behaviour on normal test."""
    model = scm.load_fitted(sub.name, "expert", "all")
    held = sub.splits.loc[sub.splits.split.isin(["validation", "test"]), "recording_id"].tolist()
    residuals = scm.quantile_residuals(scm.held_out_predictions(model, sub.view, held),
                                       np.random.default_rng(setting(cfg, "scm", "residual_seed")))
    rows = [{"quantity": "quantile residual sd (validation + test)", "value": float(residuals.std())},
            {"quantity": "share of |residual| > 3", "value": float((np.abs(residuals) > 3).mean())}]
    behaviour = scm.simulated_behaviour(model, sub.view, sub.ids("test"),
                                        np.random.default_rng(setting(cfg, "scm", "seed")),
                                        n_samples=setting(cfg, "scm", "behaviour_samples"))
    for v, part in behaviour.groupby("variable"):
        if part.observed_changes_per_min.sum() == 0:
            continue
        for column in ("observed_share_on", "simulated_share_on", "observed_changes_per_min", "simulated_changes_per_min"):
            rows.append({"quantity": f"{v}: {column}", "value": float(part[column].mean())})
    return pd.DataFrame(rows)


# table and figure


def prediction_table(cfg: dict, table: pd.DataFrame) -> pd.DataFrame:
    """log loss, brier score and log loss at changes per subsystem, data and graph, rounded as printed."""
    picks = {"Validation": (table.track == "N") & (table.split == "validation"),
             "Test": (table.track == "N") & (table.split == "test"),
             "Fault scenarios": (table.split == ALL_SCENARIOS) & (table.variables == NON_MANIPULATED)}
    rows = []
    for sub in setting(cfg, "subsystems"):
        for label, mask in picks.items():
            part = table[mask & (table.subsystem == sub)].set_index("graph")
            row = {"Subsystem": sub.capitalize(), "Data": label}
            for graph in setting(cfg, "graphs"):
                name = graph.capitalize()
                row[f"{name} LL"] = io.half_up(part.loc[graph, "log_loss"], 3)
                row[f"{name} Brier"] = io.half_up(part.loc[graph, "brier"], 3)
                row[f"{name} LL changes"] = io.half_up(part.loc[graph, "change_log_loss"], 2)
            rows.append(row)
    return pd.DataFrame(rows)


def prediction_figure(cfg: dict, table: pd.DataFrame) -> plt.Figure:
    """log loss on the normal training, validation and test recordings per subsystem and graph."""
    normal = table[table.track == "N"]
    names = {"train": "Training", "validation": "Validation", "test": "Test"}
    subsystems, kinds = setting(cfg, "subsystems"), setting(cfg, "graphs")
    set_print_style()
    fig, axes = plt.subplots(len(subsystems), 1, figsize=(PAGE_WIDTH, 5.0), layout="constrained")
    for i, (ax, sub) in enumerate(zip(axes, subsystems)):
        part = normal[normal.subsystem == sub]
        x = np.arange(len(NORMAL_SPLITS))
        for k, graph in enumerate(kinds):
            values = part[part.graph == graph].set_index("split").loc[list(NORMAL_SPLITS)]
            bars = ax.bar(x + (k - 0.5) * 0.36, values.log_loss, width=0.36, color=GRAPH_COLORS[graph])
            for rect, value in zip(bars, values.log_loss):
                ax.text(rect.get_x() + rect.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)
        counts = part[part.graph == kinds[0]].set_index("split").loc[list(NORMAL_SPLITS), "recordings"]
        ax.set_xticks(x, [f"{names[k]}\nn = {int(counts[k])}" for k in NORMAL_SPLITS])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, part.log_loss.max() * 1.22)
        ax.set_ylabel("Log loss (nats)")
        ax.set_title(f"({'abcdefgh'[i]}) {sub.capitalize()}", loc="left")
    fig.legend(handles=[Patch(color=GRAPH_COLORS[g], label=GRAPH_LABELS[g]) for g in kinds],
               loc="outside upper center", ncol=len(kinds), frameon=False)
    return fig


# selection objective of the scm settings


class OriginalGraphs:
    """expert, empty and original discovered graphs, aggregated from the cached regime runs of each scope."""

    def __init__(self, cfg: dict, sub: Subsystem, exclude_related: bool):
        self.sub = sub
        self.exclude_related = exclude_related
        self.config = discovery.config_from_dict(setting(cfg, "discovery", "original"))
        self.run_cache = discovery.RunCache(io.cache_dir("discovery_study"))
        self.empty = nx.DiGraph()
        self.empty.add_nodes_from(sub.nodes)
        self.dags: dict[tuple[str, ...], nx.DiGraph] = {}

    def get(self, kind: str, held: tuple[str, ...]) -> nx.DiGraph:
        if kind == "expert":
            return self.sub.expert
        if kind == "empty":
            return self.empty
        if held not in self.dags:
            regimes = discovery.define_regimes(self.sub.splits, held, exclude_related=self.exclude_related)
            specs, _ = discovery.make_specs(self.sub.name, io.view_name(self.sub.name, self.sub.interval_s), self.sub.view,
                                            regimes, self.config)
            runs = discovery.load_runs(self.run_cache, specs)
            missing = sorted(r for r, run in runs.items() if run.status != "ok")
            if missing:
                raise MissingInputError(f"{self.sub.name}: discovery runs of the original settings missing for "
                                        f"{missing}: run \"python main.py run discovery\" first")
            _, edges = discovery.aggregate(runs, self.config, self.sub.nodes, exclude=held)
            self.dags[held] = graphs.to_dag(discovery.edges_to_graph(edges, self.sub.nodes))[0]
        return self.dags[held]


def score_runs(cfg: dict, sub: Subsystem, model: scm.TemporalSCM, ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ranking metrics per run and one-step predictions of the non-manipulated variables."""
    metrics = []
    for rid in ids:
        ranking = ranking_of(cfg, sub, model, scm.recording_series(sub.view, rid))
        if ranking is not None:
            metrics.append({"recording_id": rid, **rca.ranking_metrics(ranking["variable"], set(sub.manipulated[rid]),
                                                                       setting(cfg, "rca", "top_k"))})
    pred = predictions(model, sub, ids)
    return pd.DataFrame(metrics), pred[~pred.manipulated]


def fit_scores(pred: pd.DataFrame) -> dict:
    log_loss = pred.log_loss.to_numpy()
    change = pred.change.to_numpy()
    return {"rows": len(pred), "changes": int(change.sum()), "log_loss": float(log_loss.mean()),
            "change_log_loss": float(log_loss[change].mean()) if change.any() else np.nan,
            "stay_log_loss": float(log_loss[~change].mean())}


def best_setting_loose(scores: dict, base: tuple, highest: bool = False) -> tuple:
    """best score with numpy's default closeness for ties, as in the objective study."""
    best = max(scores.values()) if highest else min(scores.values())
    return min((c for c in scores if np.isclose(scores[c], best)), key=lambda c: preference(c, base))


def objective_study(cfg: dict, sub: Subsystem) -> tuple[list[dict], list[dict], list[dict]]:
    """nested selection by inner ap@k and by inner log loss, then one outer fit per rule and scenario."""
    ap_key = f"ap@{setting(cfg, 'rca', 'top_k')}"
    base = baseline(cfg)
    original = OriginalGraphs(cfg, sub, exclude_related=False)
    inner_rows, selection_rows, outer_rows = [], [], []
    for kind in setting(cfg, "scm", "objective_study", "graphs"):
        choices = [c for c in grid(cfg) if kind != "empty" or c[1] is None]
        for outer in sub.folds:
            rows, ap, log_loss = inner_scores(cfg, sub, original, kind, outer, choices)
            inner_rows += rows
            chosen = {"baseline": base,
                      f"selected by {ap_key.upper()}": best_setting_loose(ap, base, highest=True),
                      "selected by log loss": best_setting_loose(log_loss, base)}
            head = {"subsystem": sub.name, "graph": kind, "outer_fold": outer}
            selection_rows += [{**head, "rule": rule, "alpha": c[0], "lag": c[1], f"inner_{ap_key}": ap[c],
                                "inner_log_loss": log_loss[c]} for rule, c in chosen.items()]
            outer_rows += outer_scores(cfg, sub, original.get(kind, (outer,)), outer, chosen, head)
            log.debug(f"{sub.name} {kind} {outer}: {chosen}")
    return inner_rows, selection_rows, outer_rows


def inner_scores(cfg: dict, sub: Subsystem, original: OriginalGraphs, kind: str, outer: str,
                 choices: list[tuple]) -> tuple[list[dict], dict, dict]:
    """inner rows, pooled ap@k and pooled log loss of every setting over the scenarios other than the outer one."""
    ap_key = f"ap@{setting(cfg, 'rca', 'top_k')}"
    max_features = setting(cfg, "scm", "max_table_features")
    pooled = {c: {"ap": [], "nll": 0.0, "rows": 0} for c in choices}
    rows = []
    for j in sub.folds:
        if j == outer:
            continue
        train = data.fold_training_ids(sub.splits, (outer, j), exclude_related=False)
        graph = original.get(kind, (outer, j))
        for c in choices:
            metrics, pred = score_runs(cfg, sub, fit(sub, graph, train, c, max_features), sub.runs_of(j))
            total = float(pred.log_loss.to_numpy().sum())
            pooled[c]["ap"].extend(metrics[ap_key].tolist())
            pooled[c]["nll"] += total
            pooled[c]["rows"] += len(pred)
            rows.append({"subsystem": sub.name, "graph": kind, "outer_fold": outer, "inner_fold": j,
                         "alpha": c[0], "lag": c[1], ap_key: metrics[ap_key].mean(), "nll": total, "rows": len(pred)})
    ap = {c: float(np.mean(v["ap"])) for c, v in pooled.items()}
    log_loss = {c: v["nll"] / v["rows"] for c, v in pooled.items()}
    return rows, ap, log_loss


def outer_scores(cfg: dict, sub: Subsystem, graph: nx.DiGraph, outer: str, chosen: dict, head: dict) -> list[dict]:
    """ranking metrics per run and fit scores of the outer model of every selection rule."""
    train = data.fold_training_ids(sub.splits, outer, exclude_related=False)
    max_features = setting(cfg, "scm", "max_table_features")
    rows = []
    for rule, c in chosen.items():
        metrics, pred = score_runs(cfg, sub, fit(sub, graph, train, c, max_features), sub.runs_of(outer))
        settings = {**head, "rule": rule, "alpha": c[0], "lag": c[1]}
        rows += [{**settings, **row} for row in metrics.to_dict("records")]
        rows.append({**settings, "recording_id": "__fit__", **fit_scores(pred)})
    return rows


def setting_label(alpha: float, lag) -> str:
    return f"{alpha:g}, {io.DASH if pd.isna(lag) else int(lag)}"


def objective_settings_table(cfg: dict, selection: pd.DataFrame) -> pd.DataFrame:
    """settings selected per outer scenario by inner ap@3 and by inner log loss."""
    ap_rule = f"selected by AP@{setting(cfg, 'rca', 'top_k')}"
    chosen = selection.set_index(["subsystem", "graph", "outer_fold", "rule"])
    rows = []
    for sub in setting(cfg, "subsystems"):
        for fold in sorted(selection.loc[selection.subsystem == sub, "outer_fold"].unique()):
            row = {"Subsystem": sub.capitalize(), "Scenario": fold}
            for kind, name in (("expert", "Expert"), ("discovered", "Discovered (original)")):
                for rule, label in ((ap_rule, "AP@3"), ("selected by log loss", "Log loss")):
                    pick = chosen.loc[(sub, kind, fold, rule)]
                    row[f"{name} {label}"] = setting_label(pick.alpha, pick.lag)
            rows.append(row)
    return pd.DataFrame(rows)


def objective_scenarios_table(cfg: dict, outer: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    """map@3 per test scenario by selection rule (expert) and with the original discovered graph (default)."""
    ap_key = f"ap@{setting(cfg, 'rca', 'top_k')}"
    runs = outer[(outer.subsystem == subsystem) & (outer.recording_id != "__fit__")]
    columns = {"Default": ("expert", "baseline"), "AP@3": ("expert", f"selected by {ap_key.upper()}"),
               "Log loss": ("expert", "selected by log loss"), "Discovered": ("discovered", "baseline")}
    per_run = {name: runs[(runs.graph == g) & (runs.rule == r)].set_index("outer_fold")[ap_key]
               for name, (g, r) in columns.items()}
    per_scenario = {name: values.groupby(level=0).mean() for name, values in per_run.items()}
    counts = runs[(runs.graph == "expert") & (runs.rule == "baseline")].groupby("outer_fold").size()
    rows = [{"Scenario": f, "Runs": int(counts[f]), **{n: io.half_up(v[f], 2) for n, v in per_scenario.items()}}
            for f in counts.index]
    rows.append({"Scenario": "Scenario mean", "Runs": "", **{n: io.half_up(v.mean(), 2) for n, v in per_scenario.items()}})
    rows.append({"Scenario": "Run mean", "Runs": int(counts.sum()), **{n: io.half_up(v.mean(), 2) for n, v in per_run.items()}})
    return pd.DataFrame(rows)


# messages


def better_graph(expert: float, discovered: float, decimals: int) -> str | None:
    """the graph with the lower printed loss, None when both print the same."""
    if io.half_up(expert, decimals) == io.half_up(discovered, decimals):
        return None
    return "expert" if expert < discovered else "discovered"


def winners_clause(winners: dict[str, str | None]) -> str:
    """'the discovered graph does better for coolant and the expert graph for probe'."""
    by_graph: dict[str | None, list[str]] = {}
    for sub, graph in winners.items():
        by_graph.setdefault(graph, []).append(sub)
    parts = [f"the {graph} graph" + (" does better" if i == 0 else "") + f" for {' and '.join(subs)}"
             if graph else f"the graphs tie for {' and '.join(subs)}"
             for i, (graph, subs) in enumerate(by_graph.items())]
    return " and ".join(parts)


def prediction_message(cfg: dict, table: pd.DataFrame) -> list[str]:
    printed = table.set_index(["Subsystem", "Data"])
    row = io.metric_row
    lines = ["state prediction in nats (recordings and scenarios weighted equally)",
             row("", "log loss", "", "log loss at changes"),
             row("", "expert", "discovered", "expert", "discovered")]
    for (sub, label), values in printed.iterrows():
        lines.append(row(f"{sub.lower()} {label.lower()}", f"{values['Expert LL']:.3f}", f"{values['Discovered LL']:.3f}",
                         f"{values['Expert LL changes']:.2f}", f"{values['Discovered LL changes']:.2f}"))
    subsystems = setting(cfg, "subsystems")
    normal = {sub: better_graph(*printed.loc[(sub.capitalize(), "Test"), ["Expert LL", "Discovered LL"]], 3)
              for sub in subsystems}
    fault = {sub: better_graph(*printed.loc[(sub.capitalize(), "Fault scenarios"),
                                            ["Expert LL changes", "Discovered LL changes"]], 2)
             for sub in subsystems}
    normal_text = ("both graphs predict normal test states equally well" if not any(normal.values())
                   else f"on normal test states {winners_clause(normal)}")
    fault_text = ("both graphs predict state changes in unseen faults equally well" if not any(fault.values())
                  else f"at state changes in unseen faults {winners_clause(fault)}")
    lines.append(f"{normal_text}; {fault_text}, in model-based one-step predictions.")
    return lines


def diagnostics_message(diagnostics: pd.DataFrame, refits: pd.DataFrame, fault: pd.DataFrame, sub: str) -> list[str]:
    values = dict(zip(diagnostics.quantity, diagnostics.value))
    spread = refits[refits.subsystem == sub].groupby("split").change_log_loss.std()
    without_data = fault[(fault.subsystem == sub) & (fault.graph == "expert") & (fault.split == ALL_SCENARIOS)
                         & (fault.variables == NON_MANIPULATED)].share_rows_mechanism_without_data.iloc[0]
    residual_sd = values["quantile residual sd (validation + test)"]
    row = io.metric_row
    lines = [f"diagnostics of the {sub} normal-prediction model (expert graph)",
             row("quantile residual sd", f"{residual_sd:.2f}", "standard normal 1"),
             row("share of |residual| > 3", f"{100 * values['share of |residual| > 3']:.2f}%", "standard normal 0.27%"),
             row("sd of log loss at changes over refits", f"{spread.min():.2f}{io.DASH}{spread.max():.2f} nats"),
             row("fault steps of untrained mechanisms", f"{100 * without_data:.1f}%"),
             row("", "share on", "", "changes per minute"),
             row("", "observed", "simulated", "observed", "simulated")]
    behaviour = sorted({q.split(":")[0] for q in values if ": " in q})
    for v in behaviour:
        lines.append(row(v, *(f"{values[f'{v}: {q}']:.2f}" for q in ("observed_share_on", "simulated_share_on",
                                                                      "observed_changes_per_min",
                                                                      "simulated_changes_per_min"))))
    under = [v for v in behaviour
             if values[f"{v}: simulated_changes_per_min"] < values[f"{v}: observed_changes_per_min"] - 0.05]
    calibrated = abs(residual_sd - 1) < 0.05
    lines.append(f"the one-step probabilities are {'well' if calibrated else 'not well'} calibrated on normal data"
                 + (f", but simulated runs switch less often than recorded for {' and '.join(under)}." if under
                    else ", and simulated runs switch about as often as recorded."))
    return lines


def objective_message(cfg: dict, scenarios: pd.DataFrame, outer: pd.DataFrame, subsystem: str) -> list[str]:
    k = setting(cfg, "rca", "top_k")
    means = scenarios.set_index("Scenario").loc["Scenario mean"]
    fits = outer[(outer.subsystem == subsystem) & (outer.graph == "expert") & (outer.recording_id == "__fit__")]
    # log loss at changes pooled over the changes of all held-out runs
    totals = fits.assign(summed=fits.change_log_loss * fits.changes).groupby("rule")[["summed", "changes"]].sum()
    change = totals.summed / totals.changes
    ap_rule, loss_rule = f"selected by AP@{k}", "selected by log loss"
    row = io.metric_row
    lines = [f"scm selection objective, {subsystem} expert graph (original discovered graphs, no near-duplicate exclusion)",
             row("", "default", f"by ap@{k}", "by log loss"),
             row(f"scenario-average map@{k}", *(f"{means[c]:.2f}" for c in ("Default", f"AP@{k}", "Log loss"))),
             row("log loss at changes (nats)", *(f"{change[r]:.2f}" for r in ("baseline", ap_rule, loss_rule)))]
    gain = change[ap_rule] - change[loss_rule]
    rca_text = (f"selecting by ap@{k} diagnoses better but needs the injected causes" if means[f"AP@{k}"] > means["Log loss"]
                else f"selecting by ap@{k} does not diagnose better and needs the injected causes")
    fit_text = (f"while log loss predicts state changes {'slightly ' if gain < 0.1 else ''}better without them" if gain > 0
                else "while log loss selects without them")
    lines.append(f"{rca_text}, {fit_text}, so the pipeline keeps log loss.")
    return lines


def run(cfg: dict, force: bool = False) -> None:
    settings = io.read_input("scm_settings")
    focus = setting(cfg, "scm", "diagnostics_subsystem")
    prediction_rows, per_variable, refits, fault_frames = [], [], [], []
    inner_rows, selection_rows, outer_rows = [], [], []
    for name in setting(cfg, "subsystems"):
        sub = load_subsystem(cfg, name)
        log.info(f"scoring the {name} scms")
        normal_rows, variables = normal_prediction(cfg, sub)
        fault_rows, pooled = fault_prediction(cfg, sub)
        prediction_rows += normal_rows + fault_rows
        per_variable += variables
        fault_frames.append(pooled)
        refits += refit_stability(cfg, sub, settings)
        if name == focus:
            diagnostics = normal_diagnostics(cfg, sub)
        log.info(f"running the scm objective study for {name}")
        inner, selection, outer = objective_study(cfg, sub)
        inner_rows += inner
        selection_rows += selection
        outer_rows += outer

    table, refit_table = pd.DataFrame(prediction_rows), pd.DataFrame(refits)
    io.save_result(table, "scm_prediction")
    io.save_result(pd.DataFrame(per_variable), "scm_prediction_per_variable")
    io.save_result(refit_table, "scm_refit_stability")
    io.save_result(diagnostics, "scm_normal_diagnostics")
    fault_variables = (pd.concat(fault_frames).groupby(["subsystem", "graph", "fold", "variable", "manipulated"])
                       .agg(rows=("p", "size"), changes=("change", "sum"), summed_log_loss=("log_loss", "sum"))
                       .reset_index())
    io.save_result(fault_variables, "scm_fault_prediction_per_variable")
    inner, selection, outer = pd.DataFrame(inner_rows), pd.DataFrame(selection_rows), pd.DataFrame(outer_rows)
    io.save_result(inner, "scm_objective_inner")
    io.save_result(selection, "scm_objective_selection")
    io.save_result(outer, "scm_objective_outer")

    printed = prediction_table(cfg, table)
    io.save_table(printed, "6.3", "State-prediction performance", "evaluation")
    io.save_figure(prediction_figure(cfg, table), "6.3", "State-prediction loss across data splits", "evaluation")
    io.save_table(objective_settings_table(cfg, selection), "A.8", "SCM settings by selection objective", "evaluation")
    study_subsystem = setting(cfg, "scm", "objective_study", "table_subsystem")
    scenarios = objective_scenarios_table(cfg, outer, study_subsystem)
    io.save_table(scenarios, "A.9", "Probe diagnosis by selection objective", "evaluation")

    io.report(prediction_message(cfg, printed))
    io.report(diagnostics_message(diagnostics, refit_table, table, focus))
    io.report(objective_message(cfg, scenarios, outer, study_subsystem))
