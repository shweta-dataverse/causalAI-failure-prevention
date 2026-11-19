from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from core import data, discovery, graphs, scm
from core.preprocessing import ViewData
from utils import io
from utils.config import setting
from utils.logger import log


@dataclass
class Subsystem:
    """data of one subsystem shared by the scm, rca and evaluation steps."""

    name: str
    view: ViewData
    splits: pd.DataFrame
    manipulated: dict[str, set[str]]
    expert: nx.DiGraph
    interval_s: float
    # fitted scms, data/cache/scm when not set
    scm_dir: Path | None = None

    @property
    def nodes(self) -> list[str]:
        return sorted(self.expert.nodes)

    @property
    def alarms(self) -> list[str]:
        return sorted(n for n in self.nodes if self.expert.nodes[n]["kind"] == "Alarm")

    @property
    def folds(self) -> list[str]:
        return sorted(self.splits.loc[self.splits.split == "fault", "fold"].unique())

    def ids(self, split: str) -> list[str]:
        return self.splits.loc[self.splits.split == split, "recording_id"].tolist()

    def runs_of(self, fold: str) -> list[str]:
        return self.splits.loc[self.splits.fold == fold, "recording_id"].tolist()

    def fold_model(self, graph: str, fold: str) -> scm.TemporalSCM:
        return scm.load_fitted(self.name, graph, data.scope_name(fold), self.scm_dir)


def load_subsystem(cfg: dict, name: str) -> Subsystem:
    """model view, splits, ground truth and expert graph of one subsystem."""
    interval = setting(cfg, "model_interval_s")
    view = io.load_view(io.view_name(name, interval))
    splits = io.read_input("splits")
    # keep the row order: bootstrap draws depend on it
    splits = splits[splits.subsystem == name].reset_index(drop=True)
    manipulated = data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments())
    # node-table order of the expert graph fixes the order of the mechanisms
    expert = graphs.simple_digraph(data.load_expert_graph(subsystem=name))
    return Subsystem(name, view, splits, manipulated, expert, interval)


# scm graphs


def discovered_dags(sub: Subsystem, discovered: pd.DataFrame) -> tuple[dict[str, nx.DiGraph], pd.DataFrame]:
    """dag of the discovered graph of every scope and the edges removed on the way."""
    dags, removals = {}, []
    for scope, edges in discovered[discovered.subsystem == sub.name].groupby("scope", sort=False):
        dag, removed = graphs.to_dag(discovery.edges_to_graph(edges, sub.nodes))
        dags[scope] = dag
        removals.append(removed.assign(subsystem=sub.name, scope=scope))
    removal_table = pd.concat(removals, ignore_index=True)[["subsystem", "scope", "source", "target", "action"]]
    return dags, removal_table


def scm_graph_rows(sub: Subsystem, dags: dict[str, nx.DiGraph]) -> pd.DataFrame:
    rows = []
    for scope, dag in dags.items():
        for kind, graph in (("expert", sub.expert), ("discovered", dag)):
            rows += [{"subsystem": sub.name, "graph": kind, "scope": scope, "source": u, "target": v}
                     for u, v in graph.edges]
    return pd.DataFrame(rows)


def graph_for(sub: Subsystem, dags: dict[str, nx.DiGraph], kind: str, held: tuple[str, ...]) -> nx.DiGraph:
    return sub.expert if kind == "expert" else dags[data.scope_name(held)]


# selection of the scm settings


def grid(cfg: dict) -> list[tuple[float, int | None]]:
    """smoothing x extra lag, smoothing-major."""
    return [(float(a), lag) for a in setting(cfg, "scm", "alphas") for lag in setting(cfg, "scm", "lags")]


def baseline(cfg: dict) -> tuple[float, int | None]:
    base = setting(cfg, "scm", "baseline")
    return float(base["alpha"]), base["lag"]


def preference(choice: tuple[float, int | None], base: tuple[float, int | None]) -> tuple:
    """tie-break order: the baseline, then no lag, then the shorter lag, then smoothing closest to the baseline."""
    alpha, lag = choice
    return choice != base, lag is not None, lag or 0, abs(np.log(alpha / base[0]))


def best_setting(scores: dict, base: tuple[float, int | None]) -> tuple[float, int | None]:
    lowest = min(scores.values())
    tied = [s for s, v in scores.items() if np.isclose(v, lowest, rtol=1e-12, atol=0)]
    return min(tied, key=lambda s: preference(s, base))


def fit(sub: Subsystem, graph: nx.DiGraph, train_ids: list[str], choice: tuple[float, int | None],
        max_features: int) -> scm.TemporalSCM:
    alpha, lag = choice
    return scm.fit_scm(sub.view, graph, train_ids, sub.manipulated, sub.interval_s, smoothing=alpha,
                       lags=scm.extra_lags(graph, lag, max_features))


def nll(model: scm.TemporalSCM, view: ViewData, ids: list[str], manipulated: dict[str, set[str]]) -> tuple[float, int]:
    """summed one-step negative log likelihood and row count of the non-manipulated variables."""
    total, rows = 0.0, 0
    for r in ids:
        series = scm.recording_series(view, r)
        for v in model.mechanisms:
            if v in manipulated.get(r, set()) or v not in series:
                continue
            p = model.transition_probs(series, v)
            y = series[v][1:]
            ok = ~np.isnan(p) & ~np.isnan(y)
            if ok.any():
                total += float(-np.log(np.where(y[ok] == 1, p[ok], 1 - p[ok])).sum())
                rows += int(ok.sum())
    return total, rows


def select_normal(cfg: dict, sub: Subsystem, graph: nx.DiGraph, kind: str) -> tuple[list[dict], dict]:
    """settings scored on the normal validation recordings by a model fitted without any held-out scenario."""
    max_features = setting(cfg, "scm", "max_table_features")
    train, validation = data.fold_training_ids(sub.splits, None), sub.ids("validation")
    inner, scores = [], {}
    for choice in grid(cfg):
        total, rows = nll(fit(sub, graph, train, choice, max_features), sub.view, validation, sub.manipulated)
        scores[choice] = total / rows
        inner.append({"subsystem": sub.name, "graph": kind, "track": "N", "outer_fold": "",
                      "inner_fold": "normal validation", "alpha": choice[0], "lag": choice[1], "nll": total, "rows": rows})
    return inner, scores


def select_fold(cfg: dict, sub: Subsystem, dags: dict, kind: str, outer: str,
                exclude_related: bool = True) -> tuple[list[dict], dict]:
    """inner leave-one-scenario-out over the scenarios other than the outer one, pooled log loss."""
    max_features = setting(cfg, "scm", "max_table_features")
    totals = {choice: [0.0, 0] for choice in grid(cfg)}
    protected = data.protected_ids(sub.runs_of(outer)) if exclude_related else set()
    inner = []
    for j in sub.folds:
        if j == outer:
            continue
        train = data.fold_training_ids(sub.splits, (outer, j), exclude_related=exclude_related)
        # near-duplicates of the outer runs are never scored
        scored = [r for r in sub.runs_of(j) if r not in protected]
        left_out = sorted(set(data.fold_training_ids(sub.splits, None)) - set(train) - set(sub.runs_of(outer))
                          - set(sub.runs_of(j)) | set(sub.runs_of(j)) - set(scored))
        graph = graph_for(sub, dags, kind, (outer, j))
        for choice in grid(cfg):
            total, rows = nll(fit(sub, graph, train, choice, max_features), sub.view, scored, sub.manipulated)
            totals[choice][0] += total
            totals[choice][1] += rows
            inner.append({"subsystem": sub.name, "graph": kind, "track": "F", "outer_fold": outer, "inner_fold": j,
                          "alpha": choice[0], "lag": choice[1], "nll": total, "rows": rows,
                          "inner_train_recordings": len(train), "inner_scored_runs": ",".join(scored),
                          "related_left_out": ",".join(left_out)})
    return inner, {choice: total / rows for choice, (total, rows) in totals.items()}


def select_settings(cfg: dict, sub: Subsystem, dags: dict,
                    exclude_related: bool = True) -> tuple[list[dict], list[dict]]:
    base = baseline(cfg)
    inner_rows, setting_rows = [], []
    for kind in setting(cfg, "graphs"):
        inner, scores = select_normal(cfg, sub, graph_for(sub, dags, kind, ()), kind)
        inner_rows += inner
        scopes = [("all", scores, "normal validation recordings")]
        for outer in sub.folds:
            inner, scores = select_fold(cfg, sub, dags, kind, outer, exclude_related)
            inner_rows += inner
            others = ",".join(j for j in sub.folds if j != outer)
            scopes.append((data.scope_name(outer), scores, f"inner scenarios {others}"))
        for scope, scores, used in scopes:
            choice = best_setting(scores, base)
            log.debug(f"{sub.name} {kind} {scope}: smoothing {choice[0]:g}, extra lag {choice[1]}")
            setting_rows.append({"subsystem": sub.name, "graph": kind, "scope": scope, "alpha": choice[0],
                                 "lag": choice[1], "selection_log_loss": scores[choice],
                                 "baseline_log_loss": scores[base], "selection_data": used})
    return inner_rows, setting_rows


# fitted models


def chosen_setting(row) -> tuple[float, int | None]:
    return float(row.alpha), None if pd.isna(row.lag) else int(row.lag)


def fit_selected(cfg: dict, sub: Subsystem, dags: dict, settings: pd.DataFrame, exclude_related: bool = True,
                 directory: Path | None = None) -> int:
    """the model of every graph and scope with its selected settings, saved to data/cache/scm by default."""
    max_features = setting(cfg, "scm", "max_table_features")
    saved = 0
    for row in settings[settings.subsystem == sub.name].itertuples():
        held = data.scope_held_out(row.scope)
        train = data.fold_training_ids(sub.splits, held, exclude_related=exclude_related)
        model = fit(sub, graph_for(sub, dags, row.graph, held), train, chosen_setting(row), max_features)
        scm.save_fitted(model, sub.name, row.graph, row.scope, directory)
        saved += 1
    return saved


# tables


def number(value: float) -> str:
    return f"{value:g}"


def per_subsystem(values: dict[str, str]) -> str:
    """one value when the subsystems agree, else 'coolant / probe'."""
    distinct = list(dict.fromkeys(values.values()))
    return distinct[0] if len(distinct) == 1 else " / ".join(values.values())


def most_common(values: list) -> object:
    return Counter(values).most_common(1)[0][0]


def configuration_table(cfg: dict, discovery_settings: pd.DataFrame, settings: pd.DataFrame) -> pd.DataFrame:
    """parameters, candidates and the values selected without a held-out scenario."""
    whole = discovery_settings[discovery_settings.scope == "all"].set_index("subsystem")
    subsystems = setting(cfg, "subsystems")
    grid_cfg = setting(cfg, "discovery", "grid")
    fixed = grid_cfg["fixed"]
    scm_all = settings[settings.scope == "all"]
    alpha = most_common(scm_all.alpha.tolist())
    lag = most_common([None if pd.isna(v) else int(v) for v in scm_all.lag])
    cf = setting(cfg, "counterfactuals")
    conditioning = sorted(set(fixed["max_conds"]))
    chance = {setting(cfg, "rca", "chance_draws"), setting(cfg, "graph_evaluation", "random_draws")}
    samples = {setting(cfg, "scm", "behaviour_samples"), setting(cfg, "scm", "abduction_samples")}

    def selected(column: str, unit: str = "") -> str:
        return per_subsystem({s: number(whole.loc[s, column]) for s in subsystems}) + unit

    def each(values: set) -> str:
        return f"{values.pop():,} each" if len(values) == 1 else "; ".join(f"{v:,}" for v in sorted(values))

    rows = [
        ("Preprocessing and discovery", "", "", "", ""),
        ("Δ", "Grid interval", "Time step of the state series", io.DASH, f"{number(fixed['interval_s'])} s"),
        ("τmax", "Maximum lag", "Longest delay tested", ", ".join(map(number, grid_cfg["tau_max_s"])) + " s",
         selected("tau_max_s", " s")),
        ("α", "Significance level", "Level of each independence test", ", ".join(map(number, grid_cfg["pc_alpha"])),
         selected("pc_alpha")),
        (io.DASH, "Conditioning limit", "Maximum conditions per test", io.DASH, "/".join(map(str, conditioning))),
        (io.DASH, "FDR filter", "BH filter on lagged links", io.DASH, "Applied" if fixed["apply_fdr_to_graph"] else "Not applied"),
        ("κ", "Support threshold", "Min. share of supporting regimes",
         ", ".join(map(number, grid_cfg["support_threshold"])), selected("support_threshold")),
        ("SCM and RCA", "", "", "", ""),
        ("a", "Pseudo-count", "Smoothing of probabilities", ", ".join(map(number, setting(cfg, "scm", "alphas"))),
         number(alpha)),
        ("L", "Extra lag", "Memory beyond one step",
         ", ".join("none" if v is None else str(v) for v in setting(cfg, "scm", "lags")) + " steps",
         "none" if lag is None else f"{lag} steps"),
        ("b, e", "Window bounds", "Window around alarm onsets", io.DASH,
         f"{number(setting(cfg, 'rca', 'window_before_s'))} s, {number(setting(cfg, 'rca', 'window_after_s'))} s"),
        ("Counterfactual explanations", "", "", "", ""),
        ("k", "Candidate pool", "Top-ranked actionable variables", io.DASH, str(cf["top_k"])),
        ("m", "Maximum actions", "Largest action set examined", io.DASH, str(cf["max_actions"])),
        ("γ", "Validity threshold", "Minimum estimated validity", io.DASH, f"{cf['validity_threshold']:.2f}"),
        ("N", "Search simulations", "Simulations per candidate set", io.DASH, str(cf["selection_worlds"])),
        ("Evaluation budgets", "", "", "", ""),
        ("", "Independent re-evaluation of the selected action sets", "", "",
         f"{cf['independent_batches'] * cf['independent_batch_size']:,}"),
        ("", "Simulations per run in the intervention analysis", "", "",
         f"{setting(cfg, 'interventions', 'replay_worlds'):,}"),
        ("", "SCMs trained on resampled recordings: per fold; normal model", "", "",
         f"{cf['refits_per_fold']}; {setting(cfg, 'scm', 'normal_refits')}"),
        ("", "Bootstrap draws of recordings (discovery)", "", "", str(setting(cfg, "discovery", "bootstrap", "n_boot"))),
        ("", "Random rankings; random graphs (chance levels)", "", "", each(chance)),
        ("", "Samples for simulated behaviour; abduction consistency", "", "", each(samples)),
    ]
    return pd.DataFrame(rows, columns=["Symbol", "Parameter", "Meaning", "Candidates", "Value"])


def scm_label(row) -> str:
    alpha, lag = chosen_setting(row)
    return f"{number(alpha)}, {io.DASH if lag is None else lag}"


def fold_settings_table(cfg: dict, discovery_settings: pd.DataFrame, settings: pd.DataFrame) -> pd.DataFrame:
    """discovery (tau_max, alpha, kappa) and scm (a, lag) per subsystem and held-out scenario."""
    chosen = settings.set_index(["subsystem", "scope", "graph"])
    rows = []
    for sub in setting(cfg, "subsystems"):
        part = discovery_settings[(discovery_settings.subsystem == sub)
                                  & ~discovery_settings.held_out.astype(str).str.contains(",")]
        for d in part.itertuples():
            rows.append({"Subsystem": sub.capitalize(), "Held out": d.held_out or "None",
                         "Discovery": f"({number(d.tau_max_s)}, {number(d.pc_alpha)}, {number(d.support_threshold)})",
                         "SCM expert": scm_label(chosen.loc[(sub, d.scope, "expert")]),
                         "SCM discovered": scm_label(chosen.loc[(sub, d.scope, "discovered")])})
    return pd.DataFrame(rows)


def run(cfg: dict, force: bool = False) -> None:
    discovered = io.read_input("discovered_graphs")
    discovery_settings = io.read_input("discovery_settings", keep_default_na=False)
    graph_tables, removal_tables, inner_rows, setting_rows = [], [], [], []
    subsystems, dags = {}, {}
    for name in setting(cfg, "subsystems"):
        sub = subsystems[name] = load_subsystem(cfg, name)
        dags[name], removals = discovered_dags(sub, discovered)
        graph_tables.append(scm_graph_rows(sub, dags[name]))
        removal_tables.append(removals)
        log.info(f"selecting scm settings for {name} ({len(sub.folds)} outer folds)")
        inner, chosen = select_settings(cfg, sub, dags[name])
        inner_rows += inner
        setting_rows += chosen

    settings = pd.DataFrame(setting_rows)
    settings["lag"] = settings["lag"].astype("Int64")
    io.write_output(pd.concat(graph_tables, ignore_index=True), "scm_graphs")
    io.write_output(pd.concat(removal_tables, ignore_index=True), "dag_removals")
    io.write_output(pd.DataFrame(inner_rows), "scm_selection_inner")
    io.write_output(settings, "scm_settings")

    saved = sum(fit_selected(cfg, sub, dags[name], settings) for name, sub in subsystems.items())
    io.save_table(configuration_table(cfg, discovery_settings, settings), "5.7",
                  "Model parameters and selected settings", "pipeline")
    io.save_table(fold_settings_table(cfg, discovery_settings, settings), "A.2",
                  "Selected settings by evaluation fold", "pipeline")
    log.info(f"scm fitting done. {saved} fitted scms saved to {io.relative(io.cache_dir('scm'))}, "
             f"settings in {io.relative(io.contract_path('scm_settings'))}")
