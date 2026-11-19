"""metadata, ground truth, expert graphs, splits and scopes of the causRCA dataset."""

from __future__ import annotations

import ast
import json
from collections import Counter
from itertools import chain, combinations
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from core import preprocessing as pp
from utils import io

RAW_DIR = io.raw_dir()
# frozen groups of near-duplicate recordings
RELATED_PATH = io.related_recordings_path()

SUBSYSTEMS: tuple[str, ...] = ("coolant", "hydraulics", "probe")
CAUSE_TIMESTAMPS: tuple[str, ...] = (
    "cause_start_at",
    "alarms_detected_at",
    "diagnosis_at",
    "cause_end_at",
    "alarms_resolved_at",
)

# counts stated in README_DATASET.md
README_COUNTS: dict[str, dict[str, int]] = {
    "full": {"nodes": 92, "edges": 104, "normal_recordings": 170, "fault_runs": 100,
             "scenarios": 19, "diagnoses": 10, "causing_variables": 14},
    "coolant": {"nodes": 15, "edges": 10, "fault_runs": 25, "scenarios": 4, "diagnoses": 2, "causing_variables": 6},
    "hydraulics": {"nodes": 17, "edges": 18, "fault_runs": 41, "scenarios": 9, "diagnoses": 5, "causing_variables": 6},
    "probe": {"nodes": 11, "edges": 15, "fault_runs": 34, "scenarios": 6, "diagnoses": 3, "causing_variables": 2},
}

NODE_COLUMNS = {"id", "path", "label", "type", "io", "datatype", "examples", "comment"}
EDGE_COLUMNS = {"source_id", "target_id", "source_label", "target_label", "edge_name", "comment"}
DESCRIPTION_KEYS = {"exp_id", "group", "scenario", "manipulatedVars", "alarms", "diagnoses"}
DIAGNOSIS_KEYS = {"id", "name", "variablesStates"}


# expert graphs


def _graph_paths(raw_dir: Path, subsystem: str | None) -> tuple[Path, Path, Path]:
    """nodes csv, edges csv and gml file of the full graph or one subsystem."""
    if subsystem is None:
        base = Path(raw_dir) / "expert_graph"
        return base / "all_nodes.csv", base / "all_edges.csv", base / "expert_graph.gml"
    if subsystem not in SUBSYSTEMS:
        raise ValueError(f"unknown subsystem {subsystem!r}; expected one of {SUBSYSTEMS}")
    base = Path(raw_dir) / "dig_twin" / f"exp_{subsystem}"
    return base / f"{subsystem}_nodes.csv", base / f"{subsystem}_edges.csv", base / f"{subsystem}_expert_graph.gml"


def load_graph_tables(raw_dir: Path = RAW_DIR, subsystem: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """node and edge tables of the expert graph, checked for internal consistency."""
    nodes_path, edges_path, _ = _graph_paths(raw_dir, subsystem)
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    for name, table, required in (("nodes", nodes, NODE_COLUMNS), ("edges", edges, EDGE_COLUMNS)):
        missing = required - set(table.columns)
        if missing:
            raise pp.SchemaError(f"{name} table of {subsystem or 'full'} graph lacks columns {sorted(missing)}")
    if nodes["label"].duplicated().any():
        raise pp.SchemaError(f"duplicate node labels in {nodes_path.name}")

    # edge labels must agree with the node ids they reference
    label_of = nodes.set_index("id")["label"]
    for end in ("source", "target"):
        expected = edges[f"{end}_id"].map(label_of)
        bad = edges[expected != edges[f"{end}_label"]]
        if not bad.empty:
            raise pp.SchemaError(f"{edges_path.name}: {end} ids and labels disagree for {bad['edge_name'].tolist()}")
    return nodes, edges


def _parse_list_text(value: Any) -> Any:
    """turn gml strings like "['Variable']" into "Variable"."""
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        if isinstance(parsed, list):
            return parsed[0] if len(parsed) == 1 else parsed
    return value


def load_gml_graph(raw_dir: Path = RAW_DIR, subsystem: str | None = None) -> nx.MultiDiGraph:
    """expert graph from its gml file, nodes keyed by label."""
    graph = nx.read_gml(_graph_paths(raw_dir, subsystem)[2], label="label")
    for _, attrs in graph.nodes(data=True):
        for key, value in attrs.items():
            attrs[key] = _parse_list_text(value)
    for _, _, attrs in graph.edges(data=True):
        for key, value in attrs.items():
            attrs[key] = _parse_list_text(value)
    return graph


def load_expert_graph(raw_dir: Path = RAW_DIR, subsystem: str | None = None) -> nx.MultiDiGraph:
    """expert graph from the csv tables, nodes keyed by label in node-table order."""
    nodes, edges = load_graph_tables(raw_dir, subsystem)
    gml = load_gml_graph(raw_dir, subsystem)

    graph = nx.MultiDiGraph(name=subsystem or "full")
    for row in nodes.itertuples(index=False):
        graph.add_node(
            row.label,
            node_id=int(row.id),
            path=row.path,
            kind=row.type,
            io=None if pd.isna(row.io) else row.io,
            datatype=row.datatype,
            comment=row.comment,
            variabletype=gml.nodes[row.label].get("variabletype") if row.label in gml else None,
        )
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source_label, row.target_label, key=row.edge_name, comment=row.comment)
    return graph


def edge_pairs(graph: nx.MultiDiGraph) -> Counter:
    """multiset of (cause, effect) pairs."""
    return Counter((u, v) for u, v in graph.edges())


# fault metadata


def load_experiments(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """one row per fault scenario from the exp_*_description.json files."""
    rows = []
    paths = sorted((Path(raw_dir) / "dig_twin").glob("exp_*/exp_*/exp_*_description.json"))
    if not paths:
        raise FileNotFoundError(f"no experiment description files under {raw_dir}")
    for path in paths:
        exp_dir = path.parent
        content = json.loads(path.read_text(encoding="utf-8"))
        missing = DESCRIPTION_KEYS - set(content)
        if missing:
            raise pp.SchemaError(f"{path.name} lacks keys {sorted(missing)}")
        for diagnosis in content["diagnoses"]:
            if DIAGNOSIS_KEYS - set(diagnosis):
                raise pp.SchemaError(f"{path.name}: diagnosis lacks keys {sorted(DIAGNOSIS_KEYS - set(diagnosis))}")

        scenario = content["scenario"]
        rows.append(
            {
                "subsystem": exp_dir.parent.name.removeprefix("exp_"),
                "experiment_id": exp_dir.name,
                # first number in the folder name, only used for sorting
                "experiment_number": int(exp_dir.name.split("_")[1]),
                "json_exp_id": content["exp_id"],
                "json_group": content["group"],
                "n_runs": len(list(exp_dir.glob("run_*"))),
                "manipulated_vars": list(content["manipulatedVars"]),
                "alarms": list(content["alarms"]),
                "diagnosis_ids": [d["id"] for d in content["diagnoses"]],
                "diagnosis_names": [d["name"].strip() for d in content["diagnoses"]],
                "expected_states": {d["id"]: d["variablesStates"] for d in content["diagnoses"]},
                "scenario_init": scenario.get("init"),
                "expected_result": scenario.get("expected_result"),
                "instead_failure": scenario.get("instead_failure"),
            }
        )
    experiments = pd.DataFrame(rows).sort_values(["subsystem", "experiment_number"], ignore_index=True)
    experiments["n_manipulated"] = experiments["manipulated_vars"].str.len()
    experiments["multi_cause"] = experiments["n_manipulated"] > 1
    experiments["multi_diagnosis"] = experiments["diagnosis_ids"].str.len() > 1
    return experiments


def load_fault_runs(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """one row per fault run with the causes.json timestamps (seconds since recording start)."""
    rows = []
    for info in pp.discover_recordings(raw_dir):
        if info.source != "dig_twin":
            continue
        path = Path(raw_dir) / Path(info.source_file).parent / "causes.json"
        content = json.loads(path.read_text(encoding="utf-8"))
        missing = set(CAUSE_TIMESTAMPS) - set(content)
        if missing:
            raise pp.SchemaError(f"{path} lacks keys {sorted(missing)}")
        rows.append(
            {
                "recording_id": info.recording_id,
                "subsystem": info.subsystem,
                "experiment_id": info.experiment_id,
                "run_id": info.run_id,
                **{key: float(content[key]) for key in CAUSE_TIMESTAMPS},
            }
        )
    runs = pd.DataFrame(rows)
    # true when the five timestamps are in the order listed in CAUSE_TIMESTAMPS
    runs["timestamps_ordered"] = (np.diff(runs[list(CAUSE_TIMESTAMPS)].to_numpy(), axis=1) >= 0).all(axis=1)
    return runs


def run_ground_truth(runs: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    """fault runs with their ground-truth causes, diagnoses and expected alarms."""
    columns = ["subsystem", "experiment_id", "manipulated_vars", "diagnosis_ids", "alarms"]
    return runs.merge(experiments[columns], on=["subsystem", "experiment_id"], how="left", validate="many_to_one")


def check_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """consistency checks of graphs and fault metadata against each other and the readme."""
    rows: list[dict[str, Any]] = []

    def add(scope: str, check: str, expected: Any, observed: Any) -> None:
        rows.append({"scope": scope, "check": check, "expected": str(expected),
                     "observed": str(observed), "passed": expected == observed})

    full = load_expert_graph(raw_dir)
    full_gml = load_gml_graph(raw_dir)
    experiments = load_experiments(raw_dir)
    runs = load_fault_runs(raw_dir)
    recordings = pp.discover_recordings(raw_dir)
    counts = README_COUNTS["full"]

    add("full", "nodes (csv)", counts["nodes"], full.number_of_nodes())
    add("full", "edges (csv)", counts["edges"], full.number_of_edges())
    add("full", "gml nodes equal csv nodes", True, set(full_gml.nodes) == set(full.nodes))
    add("full", "gml edges equal csv edges", True, edge_pairs(full_gml) == edge_pairs(full))
    add("full", "normal recordings", counts["normal_recordings"], sum(r.source == "real_op" for r in recordings))
    add("full", "fault runs", counts["fault_runs"], len(runs))
    add("full", "scenarios", counts["scenarios"], len(experiments))
    add("full", "distinct diagnoses", counts["diagnoses"], len(set(chain(*experiments["diagnosis_ids"]))))
    add("full", "distinct causing variables", counts["causing_variables"],
        len(set(chain(*experiments["manipulated_vars"]))))
    add("full", "runs with ordered causes.json timestamps", len(runs), int(runs["timestamps_ordered"].sum()))
    add("full", "description exp_id equals folder", [], experiments.query("json_exp_id != experiment_id")["experiment_id"].tolist())
    add("full", "description group equals subsystem", [], experiments.query("json_group != subsystem")["experiment_id"].tolist())

    for subsystem in SUBSYSTEMS:
        graph = load_expert_graph(raw_dir, subsystem)
        gml = load_gml_graph(raw_dir, subsystem)
        exp = experiments[experiments["subsystem"] == subsystem]
        counts = README_COUNTS[subsystem]
        nodes = set(graph.nodes)

        add(subsystem, "nodes (csv)", counts["nodes"], graph.number_of_nodes())
        add(subsystem, "edges (csv)", counts["edges"], graph.number_of_edges())
        add(subsystem, "gml nodes equal csv nodes", True, set(gml.nodes) == nodes)
        add(subsystem, "gml edges equal csv edges", True, edge_pairs(gml) == edge_pairs(graph))
        add(subsystem, "nodes missing from full graph", [], sorted(nodes - set(full.nodes)))
        add(subsystem, "edges missing from full graph", [], sorted(set(edge_pairs(graph)) - set(edge_pairs(full))))
        add(subsystem, "fault runs", counts["fault_runs"], int((runs["subsystem"] == subsystem).sum()))
        add(subsystem, "scenarios", counts["scenarios"], len(exp))
        add(subsystem, "distinct diagnoses", counts["diagnoses"], len(set(chain(*exp["diagnosis_ids"]))))
        add(subsystem, "distinct causing variables", counts["causing_variables"],
            len(set(chain(*exp["manipulated_vars"]))))
        add(subsystem, "manipulated vars missing from graph", [],
            sorted(set(chain(*exp["manipulated_vars"])) - nodes))
        add(subsystem, "alarms missing from graph", [], sorted(set(chain(*exp["alarms"])) - nodes))
        state_vars = {var for states in exp["expected_states"] for d in states.values() for var in d}
        add(subsystem, "diagnosis state vars missing from graph", [], sorted(state_vars - nodes))
    return pd.DataFrame(rows)


# eda helpers on parsed events


def fault_variable_changes(
    events: pd.DataFrame, runs: pd.DataFrame, experiments: pd.DataFrame, column: str = "manipulated_vars"
) -> pd.DataFrame:
    """per fault run and listed variable (manipulated_vars or alarms): does it change after cause_start_at, and when."""
    if column not in ("manipulated_vars", "alarms"):
        raise ValueError("column must be 'manipulated_vars' or 'alarms'")
    pairs = (
        run_ground_truth(runs, experiments)[["recording_id", "subsystem", "experiment_id", *CAUSE_TIMESTAMPS, column]]
        .explode(column)
        .rename(columns={column: "variable"})
        .reset_index(drop=True)
    )
    fault_events = events[events["source"] == "dig_twin"]

    changes = fault_events.loc[fault_events["is_change"], ["recording_id", "node", "time_s"]].merge(
        pairs[["recording_id", "variable", "cause_start_at"]],
        left_on=["recording_id", "node"],
        right_on=["recording_id", "variable"],
    )
    after = changes["time_s"] >= changes["cause_start_at"]
    summary = (
        changes.assign(before_start=~after, time_after_start=changes["time_s"].where(after))
        .groupby(["recording_id", "variable"])
        .agg(
            n_changes=("time_s", "size"),
            n_changes_before_start=("before_start", "sum"),
            first_change_after_start_s=("time_after_start", "min"),
        )
    )
    initial = fault_events[fault_events["is_initial"]].set_index(["recording_id", "node"])["raw_value"]
    observed = set(zip(fault_events["recording_id"], fault_events["node"]))

    pairs = pairs.join(summary, on=["recording_id", "variable"])
    pairs["initial_value"] = [initial.get((r, v)) for r, v in zip(pairs["recording_id"], pairs["variable"])]
    pairs["observed"] = [(r, v) in observed for r, v in zip(pairs["recording_id"], pairs["variable"])]
    for col in ("n_changes", "n_changes_before_start"):
        pairs[col] = pairs[col].fillna(0).astype(int)
    pairs["changes_after_start"] = pairs["first_change_after_start_s"].notna()
    pairs["delay_from_cause_start_s"] = pairs["first_change_after_start_s"] - pairs["cause_start_at"]
    return pairs


def variable_suitability(
    stats: pd.DataFrame,
    experiments: pd.DataFrame,
    graphs: dict[str, nx.MultiDiGraph],
    min_share_changing: float = 0.0,
) -> pd.DataFrame:
    """per subsystem variable: how often it changes in normal and fault recordings (label-free usable_* flags)."""
    normal = stats[stats["source"] == "real_op"]
    n_normal = normal["recording_id"].nunique()
    causes = {(s, v) for s, vs in zip(experiments["subsystem"], experiments["manipulated_vars"]) for v in vs}
    alarms = {(s, v) for s, vs in zip(experiments["subsystem"], experiments["alarms"]) for v in vs}

    rows = []
    for subsystem, graph in graphs.items():
        fault = stats[(stats["source"] == "dig_twin") & (stats["subsystem"] == subsystem)]
        n_fault = fault["recording_id"].nunique()
        for node in sorted(graph.nodes):
            in_normal = normal[normal["node"] == node]
            in_fault = fault[fault["node"] == node]
            declared = pd.concat([in_normal["type"], in_fault["type"]])
            rows.append(
                {
                    "subsystem": subsystem,
                    "variable": node,
                    "declared_type": declared.iloc[0] if len(declared) else None,
                    "graph_kind": graph.nodes[node].get("kind"),
                    "io": graph.nodes[node].get("io"),
                    "in_degree": graph.in_degree(node),
                    "out_degree": graph.out_degree(node),
                    "normal_recordings_observed": in_normal["recording_id"].nunique(),
                    "normal_share_changing": (in_normal["n_changes"] > 0).sum() / n_normal if n_normal else np.nan,
                    "normal_median_changes": in_normal["n_changes"].median(),
                    "fault_runs_observed": in_fault["recording_id"].nunique(),
                    "fault_share_changing": (in_fault["n_changes"] > 0).sum() / n_fault if n_fault else np.nan,
                    "fault_median_changes": in_fault["n_changes"].median(),
                    "is_ground_truth_cause": (subsystem, node) in causes,
                    "is_expected_alarm": (subsystem, node) in alarms,
                }
            )
    table = pd.DataFrame(rows)
    table["varies_in_normal"] = table["normal_share_changing"] > min_share_changing
    table["varies_in_faults"] = table["fault_share_changing"] > min_share_changing
    table["is_alarm"] = table["declared_type"] == "Alarm"
    # label-free suitability for later fitting
    table["usable_normal_only"] = table["varies_in_normal"]
    table["usable_with_fault_runs"] = table["varies_in_normal"] | table["varies_in_faults"]
    return table


def timestamp_batches(events: pd.DataFrame) -> pd.DataFrame:
    """events per (recording, timestamp) after t = 0; more than one means nodes published together."""
    after_start = events[events["time_s"] > 0]
    return (
        after_start.groupby(["recording_id", "time_s"])
        .agg(source=("source", "first"), n_events=("node", "size"))
        .reset_index()
    )


def state_durations(events: pd.DataFrame, types: tuple[str, ...] = ("Binary",)) -> pd.DataFrame:
    """how long each state lasted before the node changed again; the last state is censored."""
    starts = events[(events["is_initial"] | events["is_change"]) & events["type"].isin(types)]
    starts = starts.sort_values(["recording_id", "node", "time_s", "event_seq"])
    ends = events.groupby("recording_id")["time_s"].max()
    next_start = starts.groupby(["recording_id", "node"])["time_s"].shift(-1)
    censored = next_start.isna()
    end_time = next_start.fillna(starts["recording_id"].map(ends))
    return pd.DataFrame(
        {
            "recording_id": starts["recording_id"],
            "source": starts["source"],
            "subsystem": starts["subsystem"],
            "node": starts["node"],
            "type": starts["type"],
            "value_num": starts["value_num"],
            "start_s": starts["time_s"],
            "duration_s": end_time - starts["time_s"],
            "censored": censored,
        }
    ).reset_index(drop=True)


# splits and scopes


def make_splits(
    recordings: pd.DataFrame, subsystem: str, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> pd.DataFrame:
    """split labels of one subsystem: normal recordings by number, fault runs by scenario."""
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0) or min(fractions) < 0:
        raise ValueError(f"fractions must be three non-negative numbers summing to 1, got {fractions}")
    columns = ["recording_id", "source", "subsystem", "experiment_id", "split", "fold"]

    normal = recordings[recordings["source"] == "real_op"].copy()
    normal["number"] = normal["recording_id"].str.rsplit("_", n=1).str[-1].astype(int)
    normal = normal.sort_values("number")
    n_train = int(round(fractions[0] * len(normal)))
    n_validation = int(round(fractions[1] * len(normal)))
    n_test = len(normal) - n_train - n_validation
    normal["split"] = ["train"] * n_train + ["validation"] * n_validation + ["test"] * n_test
    normal["fold"] = None

    fault = recordings[(recordings["source"] == "dig_twin") & (recordings["subsystem"] == subsystem)].copy()
    if fault.empty:
        raise ValueError(f"no fault runs for subsystem {subsystem!r}")
    fault["split"] = "fault"
    fault["fold"] = fault["experiment_id"]
    return pd.concat([normal[columns], fault[columns]], ignore_index=True)


def manipulated_by_recording(runs: pd.DataFrame, experiments: pd.DataFrame) -> dict[str, set[str]]:
    """recording id -> variables manipulated in that fault run (ground truth)."""
    truth = run_ground_truth(runs, experiments)
    return {r: set(v) for r, v in zip(truth["recording_id"], truth["manipulated_vars"])}


def related_partners(path: Path = RELATED_PATH) -> dict[str, set[str]]:
    """recording id -> the other recordings of its related group (frozen manifest)."""
    groups = pd.read_csv(path)
    out: dict[str, set[str]] = {}
    for _, g in groups.groupby("group_id"):
        ids = set(g["recording_id"])
        for r in ids:
            out[r] = ids - {r}
    return out


def raw_rows_after_start(raw_dir: Path = RAW_DIR) -> dict[str, set[tuple[str, str, str]]]:
    """recording id -> its raw (time_s, node, value) rows after t = 0, as the original strings."""
    rows = {}
    for info in pp.discover_recordings(raw_dir):
        raw = pd.read_csv(Path(raw_dir) / info.source_file, dtype=str, keep_default_na=False)
        raw = raw[raw["time_s"].astype(float) > 0]
        rows[info.recording_id] = set(zip(raw["time_s"], raw["node"], raw["value"]))
    return rows


def recording_similarity(rows: dict[str, set[tuple[str, str, str]]]) -> pd.DataFrame:
    """every pair of recordings that shares at least one row, with the jaccard share of its rows."""
    holders: dict[tuple[str, str, str], list[str]] = {}
    for recording_id, recording_rows in rows.items():
        for row in recording_rows:
            holders.setdefault(row, []).append(recording_id)
    shared = Counter(pair for ids in holders.values() for pair in combinations(ids, 2))
    table = pd.DataFrame(
        [{"a": a, "b": b, "rows_a": len(rows[a]), "rows_b": len(rows[b]), "shared_rows": n,
          "jaccard": n / len(rows[a] | rows[b])} for (a, b), n in shared.items()],
        columns=["a", "b", "rows_a", "rows_b", "shared_rows", "jaccard"],
    )
    return table.sort_values(["jaccard", "a", "b"], ascending=[False, True, True], ignore_index=True)


def near_duplicate_groups(similarity: pd.DataFrame, threshold: float) -> list[set[str]]:
    """connected groups of recordings whose pairwise jaccard reaches the threshold."""
    graph = nx.Graph()
    close = similarity[similarity["jaccard"] >= threshold]
    graph.add_edges_from(zip(close["a"], close["b"]))
    return sorted((set(g) for g in nx.connected_components(graph)), key=sorted)


def protected_ids(ids: Sequence[str]) -> set[str]:
    """the given recordings together with their related partners."""
    partners = related_partners()
    return set(ids) | {p for r in ids for p in partners.get(r, ())}


def held_tuple(held_out: str | Sequence[str] | None) -> tuple[str, ...]:
    """held-out scenarios as a tuple (none, one scenario or several)."""
    if held_out is None:
        return ()
    return (held_out,) if isinstance(held_out, str) else tuple(held_out)


def fold_training_ids(splits: pd.DataFrame, held_out: str | Sequence[str] | None,
                      exclude_related: bool = True) -> list[str]:
    """normal training recordings and the runs of the other scenarios, without near-duplicates."""
    held = held_tuple(held_out)
    normal = splits.loc[splits["split"] == "train", "recording_id"].tolist()
    fault = splits.loc[(splits["split"] == "fault") & (~splits["fold"].isin(held)), "recording_id"].tolist()
    if exclude_related and held:
        drop = protected_ids(splits.loc[splits["fold"].isin(held), "recording_id"].tolist())
        normal = [r for r in normal if r not in drop]
        fault = [r for r in fault if r not in drop]
    return normal + fault


def scope_name(held_out: str | Sequence[str] | None) -> str:
    """'all', 'without_exp_22' or, for an inner fold (outer first), 'without_exp_22_exp_27'."""
    held = held_tuple(held_out)
    return "all" if not held else "without_" + "_".join(held)


def scope_held_out(scope: str) -> tuple[str, ...]:
    """inverse of scope_name; scenario ids all start with 'exp_' and may contain underscores."""
    if scope == "all":
        return ()
    if not scope.startswith("without_exp_"):
        raise ValueError(f"unknown scope {scope!r}")
    body = scope.removeprefix("without_")
    return tuple("exp_" + part for part in body.removeprefix("exp_").split("_exp_"))
