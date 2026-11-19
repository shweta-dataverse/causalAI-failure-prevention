import time
from itertools import permutations

import pandas as pd

from core import data, discovery, graphs
from core import preprocessing as pp
from utils import io
from utils.config import setting
from utils.errors import DataError, MissingInputError
from utils.logger import log

SETTINGS_DROPPED = ["subsystem", "scope", "rank", "runtime_s", "run_status"]


def run(cfg: dict, force: bool = False) -> None:
    splits = io.read_input("splits")
    study = setting(cfg, "discovery")
    grid = discovery.grid_configs(study["grid"])
    order = selection_order(study)
    started = time.time()
    edges, settings = [], []
    for sub in setting(cfg, "subsystems"):
        context = Subsystem(cfg, sub, splits)
        scopes = context.scopes()
        # the support threshold only aggregates, so one pcmci+ run serves three configurations
        runs_needed = {c.run_id: c for c in grid}.values()
        run_missing(cfg, context, [discovery.make_specs(sub, context.view_name, context.view, context.regimes(held), c)[0]
                                   for held in scopes for c in runs_needed], force, started)
        for held in scopes:
            best_row, best_edges = select_scope(context, held, grid, order)
            settings.append(best_row)
            edges.append(best_edges)
        log.debug(f"{sub}: {len(scopes)} scopes selected")
    edges, settings = pd.concat(edges, ignore_index=True), pd.DataFrame(settings)
    io.write_output(edges, "discovered_graphs")
    io.write_output(settings, "discovery_settings")
    counts = settings.groupby("subsystem", sort=False).size()
    log.info("causal discovery done. " + " and ".join(f"{n} {sub}" for sub, n in counts.items())
             + f" graphs saved to {io.relative(io.contract_path('discovered_graphs'))}, their settings to "
             + io.relative(io.contract_path("discovery_settings")))


class Subsystem:
    """state view, splits, expert graph and regimes of one subsystem."""

    def __init__(self, cfg: dict, subsystem: str, splits: pd.DataFrame):
        self.name = subsystem
        self.view_name = f"{subsystem}_dt{setting(cfg, 'model_interval_s'):g}"
        self.view = load_view(self.view_name)
        # row order matters: the bootstrap draws recordings by position
        self.splits = splits[splits["subsystem"] == subsystem].reset_index(drop=True)
        self.expert = graphs.simple_digraph(data.load_expert_graph(subsystem=subsystem))
        self.nodes = sorted(self.expert.nodes)
        if set(self.nodes) != set(self.view.var_names):
            raise DataError(f"{self.view_name}: the view variables differ from the expert node list")
        self.all_regimes = discovery.define_regimes(self.splits)
        self.folds = [r for r in self.all_regimes if r != "normal"]
        self._runs: dict[tuple, dict[str, discovery.RegimeRun]] = {}

    def scopes(self) -> list[tuple[str, ...]]:
        """nothing held out, each scenario, and each ordered (outer, inner) pair of scenarios."""
        return [()] + [(k,) for k in self.folds] + [tuple(p) for p in permutations(self.folds, 2)]

    def regimes(self, held: tuple[str, ...]) -> dict[str, list[str]]:
        """regimes without the held-out scenarios and without the near-duplicates of their runs."""
        return discovery.define_regimes(self.splits, held) if held else self.all_regimes

    def runs(self, config: discovery.StudyConfig, regimes: dict[str, list[str]]) -> dict[str, discovery.RegimeRun]:
        """cached regime runs of a configuration (support thresholds share them, so they are read once)."""
        key = (config.run_id, tuple((r, tuple(ids)) for r, ids in regimes.items()))
        if key not in self._runs:
            specs, _ = discovery.make_specs(self.name, self.view_name, self.view, regimes, config)
            self._runs[key] = discovery.load_runs(run_cache(), specs)
        return self._runs[key]

    def related_removed(self, held: tuple[str, ...]) -> str:
        """recordings of the remaining regimes left out as near-duplicates of the held-out runs."""
        kept = {r for ids in self.regimes(held).values() for r in ids}
        everything = {r for ids in self.all_regimes.values() for r in ids}
        held_runs = {r for k in held for r in self.all_regimes[k]}
        return ",".join(sorted(everything - kept - held_runs))


def load_view(name: str) -> pp.ViewData:
    directory = io.processed_dir()
    if not (directory / f"{name}.npz").exists():
        raise MissingInputError(f"{io.relative(directory / name)}.npz not found: run \"python main.py run preprocessing\" first")
    return pp.load_view_data(directory, name)[0]


def selection_order(study: dict) -> list[tuple[str, str]]:
    return [tuple(next(iter(rule.items()))) for rule in study["selection"]["order"]]


def run_cache() -> discovery.RunCache:
    return discovery.RunCache(io.cache_dir("discovery_study"))


def run_missing(cfg: dict, context: Subsystem, spec_lists: list[list[discovery.RunSpec]], force: bool,
                started: float) -> None:
    """pcmci+ for every regime run not in the cache (with force: every run not written since the start)."""
    specs = [spec for specs in spec_lists for spec in specs]
    cache = discovery.RefreshCache(io.cache_dir("discovery_study"), started) if force else run_cache()
    unique = {spec.key() for spec in specs}
    missing = {spec.key() for spec in specs if not cache.has(spec)}
    log.info(f"{context.name}: {len(unique)} pcmci+ regime runs, {len(unique) - len(missing)} from "
             f"{io.relative(cache.directory)}, {len(missing)} to run")
    if missing:
        discovery.run_specs(specs, cache, io.processed_dir(), workers=setting(cfg, "discovery", "runtime", "workers"),
                            timeout_s=setting(cfg, "discovery", "shortlist", "run_timeout_s"), log=log.info)


def select_scope(context: Subsystem, held: tuple[str, ...], grid: list[discovery.StudyConfig],
                 order: list[tuple[str, str]]) -> tuple[dict, pd.DataFrame]:
    """the selected configuration of one scope (its settings row) and its aggregated graph."""
    regimes = context.regimes(held)
    rows, edges = [], {}
    for config in grid:
        runs = context.runs(config, regimes)
        bad = sorted(r for r, regime_run in runs.items() if regime_run.status != "ok")
        if bad:
            raise DataError(f"{context.name} {config.run_id}: regime runs {bad} did not finish; "
                            "rerun \"python main.py run discovery --force\"")
        row, edges[config.config_id] = discovery.evaluate_config(runs, config, context.nodes, context.expert, exclude=held)
        rows.append({"subsystem": context.name, "scope": data.scope_name(held), **row})
    best, ties = discovery.select(pd.DataFrame(rows), order)
    scope = data.scope_name(held)
    settings = {"subsystem": context.name, "scope": scope, "held_out": ",".join(held),
                "related_removed": context.related_removed(held), **best.drop(SETTINGS_DROPPED).to_dict(), **ties}
    graph = edges[best["config_id"]]
    graph.insert(0, "scope", scope)
    graph.insert(0, "subsystem", context.name)
    return settings, graph
