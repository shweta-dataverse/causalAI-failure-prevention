from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from core import data, discovery, graphs, rca, scm
from core import preprocessing as pp
from core.plots import COLORS, INK, PAGE_WIDTH, set_print_style
from utils import io
from utils.config import setting
from utils.errors import DataError, MissingInputError
from utils.logger import log

# every subsystem variable is binary or an alarm, so every state is discrete
VIEW_KIND = "discrete"
ROLE_ORDER = ("Health state", "Control command", "Feedback state", "Alarm")
TIME_LABELS = {"cause_start_at": "Fault start", "alarms_detected_at": "Alarms detected", "diagnosis_at": "Diagnosis",
               "cause_end_at": "Fault end", "alarms_resolved_at": "Alarms resolved"}


def run(cfg: dict, force: bool = False) -> None:
    raw = io.raw_dir()
    if not raw.is_dir():
        raise MissingInputError(f"{io.relative(raw)} not found: put the causRCA dataset (version 1.0.0) there first")
    infos = pp.discover_recordings(raw)
    recordings = pd.DataFrame([asdict(info) for info in infos])
    runs, experiments = data.load_fault_runs(), data.load_experiments()
    subsystems = setting(cfg, "subsystems")
    interval = setting(cfg, "model_interval_s")
    events = {sub: subsystem_events(infos, runs, sub) for sub in subsystems}

    splits = build_splits(cfg, recordings)
    io.write_output(splits, "splits")
    views = {sub: build_states(sub, interval, events[sub], recordings, runs) for sub in subsystems}
    pairs = check_near_duplicates(cfg, runs, experiments)
    save_thesis_items(cfg, raw, runs, experiments, splits, events, views, pairs)
    log.info(f"preprocessing done. state tables of {len(subsystems)} subsystems and the splits of {len(splits)} "
             f"recordings saved to {io.relative(io.processed_dir())}")


# events, splits and state tables


def subsystem_events(infos: list[pp.RecordingInfo], runs: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    """parsed events of the normal recordings and the fault runs of one subsystem."""
    keep = set(normal_ids(infos)) | set(fault_ids(runs, subsystem))
    events = pp.parse_values(pp.load_all_events([i for i in infos if i.recording_id in keep], io.raw_dir()))
    events, _ = pp.resolve_duplicates(events)
    return pp.mark_changes(events)


def normal_ids(infos: list[pp.RecordingInfo]) -> list[str]:
    return [i.recording_id for i in infos if i.source == "real_op"]


def fault_ids(runs: pd.DataFrame, subsystem: str) -> list[str]:
    return runs.loc[runs["subsystem"] == subsystem, "recording_id"].tolist()


def expert_nodes(subsystem: str) -> list[str]:
    return sorted(data.load_expert_graph(subsystem=subsystem).nodes)


def build_splits(cfg: dict, recordings: pd.DataFrame) -> pd.DataFrame:
    """normal recordings by number and fault runs by scenario, one block per subsystem."""
    fractions = tuple(setting(cfg, "split_fractions"))
    return pd.concat([data.make_splits(recordings, sub, fractions).assign(subsystem=sub) for sub in setting(cfg, "subsystems")],
                     ignore_index=True)


def build_states(subsystem: str, interval: float, events: pd.DataFrame, recordings: pd.DataFrame,
                 runs: pd.DataFrame) -> pp.ViewData:
    """state of every subsystem variable on the time grid, checked cell by cell against the events, then saved."""
    nodes = expert_nodes(subsystem)
    ids = recordings.loc[recordings["source"] == "real_op", "recording_id"].tolist() + fault_ids(runs, subsystem)
    states = pp.reconstruct_states(events, interval, nodes=nodes)
    check = pp.verify_states(events, states, interval)
    if not check.passed:
        raise DataError(f"{subsystem}: {check.n_mismatches} state cells differ from the raw events")
    log.debug(f"{subsystem}: {check.n_cells_checked} state cells match the raw events")
    view = pp.ModelView.from_schema(pp.node_types(events), nodes, VIEW_KIND).transform(states, ids)
    pp.write_states(view, io.states_path(subsystem), interval)
    return view


# near-duplicate recordings


def check_near_duplicates(cfg: dict, runs: pd.DataFrame, experiments: pd.DataFrame) -> list[dict]:
    """pairs of recordings that share most raw rows; the groups must equal the frozen manifest."""
    threshold = setting(cfg, "near_duplicates", "jaccard_threshold")
    rows = data.raw_rows_after_start()
    similarity = data.recording_similarity(rows)
    groups = data.near_duplicate_groups(similarity, threshold)
    frozen = pd.read_csv(io.related_recordings_path())
    if groups != sorted((set(g["recording_id"]) for _, g in frozen.groupby("group_id")), key=sorted):
        raise DataError(f"the near-duplicate groups of the raw recordings differ from {io.relative(io.related_recordings_path())}")
    close = similarity["jaccard"] >= threshold
    rest = similarity[~close]
    log.info(f"near-duplicate check: {int(close.sum())} pairs in {len(groups)} groups, as in "
             f"{io.relative(io.related_recordings_path())}; every other pair shares at most {rest['jaccard'].max():.1%} "
             f"of its rows ({rest.loc[rest['shared_rows'] > 1, 'jaccard'].max():.1%} with more than one shared row)")
    manipulated = data.manipulated_by_recording(runs, experiments)
    pairs = []
    for row in similarity[close].itertuples(index=False):
        single, double = sorted((row.a, row.b), key=lambda r: len(manipulated[r]))
        pairs.append({"single": single, "double": double, "shared": row.shared_rows,
                      "rows": max(row.rows_a, row.rows_b),
                      "only_single": sorted(rows[single] - rows[double]), "only_double": sorted(rows[double] - rows[single])})
    return sorted(pairs, key=lambda p: p["single"])


# tables and figure


def save_thesis_items(cfg: dict, raw: Path, runs: pd.DataFrame, experiments: pd.DataFrame, splits: pd.DataFrame,
                      events: dict[str, pd.DataFrame], views: dict[str, pp.ViewData], pairs: list[dict]) -> None:
    example = setting(cfg, "example")
    sub = runs.set_index("recording_id").loc[example["recording"], "subsystem"]
    sub_splits = splits[splits["subsystem"] == sub].reset_index(drop=True)
    interval = setting(cfg, "model_interval_s")
    io.save_table(subsystem_table(cfg, runs, experiments), "5.1", "Analysed subsystems of the causRCA dataset")
    io.save_table(scenario_table(cfg, runs, experiments, views), "5.2", "Fault scenarios and injected causes")
    io.save_table(variable_table(sub, events[sub], experiments), "5.3", "Coolant variables and their roles")
    io.save_table(raw_event_table(raw, example, sub), "5.4", "Raw coolant event log")
    io.save_table(state_table(views[sub], sub_splits, example, interval), "5.5", "Reconstructed coolant states")
    io.save_table(regime_table(views[sub], sub_splits), "5.6", "Coolant discovery regimes")
    io.save_table(near_duplicate_table(pairs, runs), "A.4", "Near-duplicate probe recordings")
    io.save_figure(fault_timeline(raw, example["recording"], runs, experiments, sub), "5.1",
                   "Machine states during a coolant fault")


def subsystem_table(cfg: dict, runs: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sub in setting(cfg, "subsystems"):
        expert = graphs.simple_digraph(data.load_expert_graph(subsystem=sub))
        scenarios = experiments[experiments["subsystem"] == sub]
        counts = scenarios["n_manipulated"]
        rows.append({"Subsystem": sub.capitalize(), "Nodes": expert.number_of_nodes(), "Edges": expert.number_of_edges(),
                     "Scenarios": len(scenarios), "Runs": int((runs["subsystem"] == sub).sum()),
                     "Manipulated variables per scenario": f"{counts.min()}{io.DASH}{counts.max()} "
                                                           f"({setting(cfg, 'manipulated_kinds', sub)})"})
    return pd.DataFrame(rows)


def scenario_table(cfg: dict, runs: pd.DataFrame, experiments: pd.DataFrame, views: dict[str, pp.ViewData]) -> pd.DataFrame:
    """manipulated variables and expected alarms per scenario; a dagger marks an alarm of a single scenario."""
    interval = setting(cfg, "model_interval_s")
    manipulated = data.manipulated_by_recording(runs, experiments)
    rows = []
    for sub in setting(cfg, "subsystems"):
        scenarios = experiments[experiments["subsystem"] == sub]
        scenarios_of_alarm = scenarios.explode("alarms").groupby("alarms")["experiment_id"].nunique()
        alarm_nodes = sorted(n for n, kind in data.load_expert_graph(subsystem=sub).nodes(data="kind") if kind == "Alarm")
        for scenario in scenarios.itertuples(index=False):
            alarms = ", ".join(a + ("†" if scenarios_of_alarm[a] == 1 else "") for a in sorted(scenario.alarms))
            if sub == "probe":
                ids = runs.loc[(runs["subsystem"] == sub) & (runs["experiment_id"] == scenario.experiment_id)]
                later = sum(alarm_after_first_change(views[sub], run, manipulated[run.recording_id], alarm_nodes, interval)
                            for run in ids.itertuples(index=False))
                alarms += f" ({later}/{len(ids)})"
            rows.append({"Subsystem": sub.capitalize(), "Scenario": scenario.experiment_id, "Runs": scenario.n_runs,
                         "Manipulated variables": ", ".join(sorted(scenario.manipulated_vars)),
                         "Expected alarms": alarms})
    return pd.DataFrame(rows)


def alarm_after_first_change(view: pp.ViewData, run, injected: set[str], alarms: list[str], interval: float) -> bool:
    """the first alarm onset is not earlier than the first observed change of an injected variable."""
    series = scm.recording_series(view, run.recording_id)
    changes = [step for variable in injected for step, _ in discovery.state_changes(view, run.recording_id, variable)
               if step * interval >= run.cause_start_at]
    onsets = rca.alarm_onsets(series, alarms)
    return bool(changes and onsets) and min(onsets.values()) >= min(changes)


def variable_table(subsystem: str, events: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    """role, share of normal recordings with a change, degree and manipulating scenarios of every variable."""
    expert = graphs.simple_digraph(data.load_expert_graph(subsystem=subsystem))
    scenarios = experiments[experiments["subsystem"] == subsystem].explode("manipulated_vars")
    manipulated_in = scenarios.groupby("manipulated_vars")["experiment_id"].apply(sorted).to_dict()
    normal = events[events["source"] == "real_op"]
    n_normal = normal["recording_id"].nunique()
    changed = normal[normal["is_change"]].groupby("node")["recording_id"].nunique()
    recorded = set(normal["node"])
    rows = []
    for node, attrs in expert.nodes(data=True):
        role = ("Alarm" if attrs["kind"] == "Alarm" else "Control command" if attrs["io"] == "output"
                else "Health state" if node in manipulated_in else "Feedback state")
        share = f"{100 * changed.get(node, 0) / n_normal:.1f}" if node in recorded else "n.r."
        rows.append({"Variable": node, "Role": role, "Normal": share,
                     "In/out": f"{expert.in_degree(node)}/{expert.out_degree(node)}",
                     "Manipulated in": ", ".join(manipulated_in.get(node, [])) or io.DASH})
    table = pd.DataFrame(rows)
    table["order"] = table["Role"].map(ROLE_ORDER.index)
    return table.sort_values(["order", "Variable"]).drop(columns="order").reset_index(drop=True)


def raw_event_table(raw: Path, example: dict, subsystem: str) -> pd.DataFrame:
    """the first events of the subsystem variables in the example recording, as raw strings."""
    log_rows = pd.read_csv(raw / example["source_file"], dtype=str, keep_default_na=False)
    return log_rows[log_rows["node"].isin(expert_nodes(subsystem))].head(example["steps"]).reset_index(drop=True)


def state_table(view: pp.ViewData, splits: pd.DataFrame, example: dict, interval: float) -> pd.DataFrame:
    """the first grid steps of the example recording, restricted to the variables usable in its regime."""
    usable = discovery.regime_variables(view, discovery.define_regimes(splits)[example["fold"]])
    steps = example["steps"]
    states = pd.DataFrame(view.arrays[example["recording"]][:steps], columns=view.var_names).astype("Int64")
    states = states.mask(pd.DataFrame(view.masks[example["recording"]][:steps], columns=view.var_names))
    states.insert(0, "t_s", [k * interval for k in range(steps)])
    states.insert(0, "step", range(steps))
    return states[["step", "t_s", *usable]]


def regime_table(view: pp.ViewData, splits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, ids in discovery.define_regimes(splits).items():
        rows.append({"Regime": "Normal operation (training)" if regime == "normal" else f"Scenario {regime}",
                     "Recordings": len(ids), "Time steps": sum(len(view.arrays[r]) for r in ids),
                     "Usable variables": len(discovery.regime_variables(view, ids))})
    return pd.DataFrame(rows)


def near_duplicate_table(pairs: list[dict], runs: pd.DataFrame) -> pd.DataFrame:
    by_id = runs.set_index("recording_id")

    def label(recording_id: str) -> str:
        return f"{by_id.loc[recording_id, 'experiment_id']}, {by_id.loc[recording_id, 'run_id'].replace('_', ' ')}"

    rows = []
    for pair in pairs:
        differing = "; ".join(
            f"{node} at {float(t_single):.1f} or {float(t_double):.1f} s"
            for (t_single, node, _), (t_double, _, _) in zip(pair["only_single"], pair["only_double"]))
        rows.append({"Single-sensor run": label(pair["single"]), "Two-sensor run": label(pair["double"]),
                     "Rows": f"{pair['shared']} of {pair['rows']}", "Differing row": differing})
    return pd.DataFrame(rows)


def fault_timeline(raw: Path, recording_id: str, runs: pd.DataFrame, experiments: pd.DataFrame,
                   subsystem: str) -> plt.Figure:
    """state of every subsystem variable over one fault run, with the annotated times of the dataset."""
    set_print_style()
    infos = [i for i in pp.discover_recordings(raw) if i.recording_id == recording_id]
    events = pp.mark_changes(pp.parse_values(pp.load_all_events(infos, raw)))
    run = runs.set_index("recording_id").loc[recording_id]
    injected = experiments.set_index(["subsystem", "experiment_id"]).loc[(run.subsystem, run.experiment_id),
                                                                         "manipulated_vars"]
    expert = data.load_expert_graph(subsystem=subsystem)
    kinds = dict(expert.nodes(data="kind"))
    nodes = sorted(expert.nodes, key=lambda n: (kinds[n] == "Alarm", n), reverse=True)
    t_end = events["time_s"].max()
    recorded = set(events["node"])
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 5.0), layout="constrained")
    for i, node in enumerate(nodes):
        node_events = events[events["node"] == node].sort_values("time_s")
        if node_events.empty or node_events["value_num"].isna().all():
            continue
        values = node_events["value_num"].to_numpy(dtype=float)
        span = np.nanmax(values) - np.nanmin(values)
        values = (values - np.nanmin(values)) / span if span > 0 else np.zeros_like(values)
        ax.step(np.append(node_events["time_s"], t_end), i + 0.7 * np.append(values, values[-1]), where="post",
                color=COLORS[1] if kinds[node] == "Alarm" else COLORS[0], linewidth=1.6)
    ax.set_yticks(np.arange(len(nodes)) + 0.35, [n if n in recorded else f"{n} (not recorded)" for n in nodes],
                  fontsize=8.5)
    for tick, node in zip(ax.get_yticklabels(), nodes):
        if node in injected:
            tick.set_fontweight("bold")
        if node not in recorded:
            tick.set_color(INK["muted"])
    for k, name in enumerate(data.CAUSE_TIMESTAMPS, start=1):
        ax.axvline(run[name], color=INK["secondary"], linewidth=0.9, linestyle="--")
        ax.text(run[name], 1.005, str(k), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    ax.set_xlim(0, t_end)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time since recording start (s)")
    keys = [Line2D([], [], color="none", label=f"{k}: {TIME_LABELS[n]}") for k, n in enumerate(data.CAUSE_TIMESTAMPS, 1)]
    kinds_legend = [Line2D([], [], color=COLORS[0], label="State"), Line2D([], [], color=COLORS[1], label="Alarm")]
    fig.legend(handles=kinds_legend + keys, loc="outside lower center", ncol=4, frameon=False, fontsize=9,
               handlelength=1.2, columnspacing=1.2)
    return fig
