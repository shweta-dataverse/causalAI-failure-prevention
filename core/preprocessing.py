"""event-log preprocessing: parsing, duplicates, zero-order-hold states and state tables."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from utils.logger import log as logger

VARIABLE_TYPES: tuple[str, ...] = ("Binary", "Alarm", "Continuous", "Counter", "Categorical")
BOOLEAN_TYPES = frozenset({"Binary", "Alarm"})
NUMERIC_TYPES = frozenset({"Continuous", "Counter"})
DISCRETE_TYPES = frozenset({"Binary", "Alarm", "Categorical"})

EVENT_COLUMNS: tuple[str, ...] = ("time_s", "node", "value", "type")
ID_COLUMNS: tuple[str, ...] = (
    "recording_id",
    "source",
    "subsystem",
    "experiment_id",
    "run_id",
    "source_file",
)
GRID_COLUMNS: tuple[str, ...] = ("step", "t_s")

_TRUE_STRINGS = frozenset({"true", "1", "1.0"})
_FALSE_STRINGS = frozenset({"false", "0", "0.0"})
# rounding t / interval to 9 decimals removes float noise before the ceiling
_GRID_DECIMALS = 9
_GRID_TOLERANCE = 1e-9

_REAL_OP_RE = re.compile(r"dataset_real_op_(\d+)\.csv")
_FAULT_RE = re.compile(r"faultDataset_([a-z]+)_exp([0-9_]+)_run_(\d+)\.csv")

DuplicatePolicy = Literal["last", "first", "error"]
ViewKind = Literal["continuous", "discrete", "mixed"]
VIEW_KINDS: tuple[str, ...] = ("continuous", "discrete", "mixed")


class PreprocessingError(Exception):
    """base class of preprocessing failures."""


class SchemaError(PreprocessingError):
    """raw files or event tables do not have the expected structure."""


class ValueParseError(PreprocessingError):
    """a value cannot be parsed according to its declared variable type."""


class DuplicateEventError(PreprocessingError):
    """several events share (recording, node, time_s)."""


# recording discovery


@dataclass(frozen=True)
class RecordingInfo:
    recording_id: str
    source: str  # "real_op" (normal operation) or "dig_twin" (fault run)
    subsystem: str | None  # coolant / hydraulics / probe, None for normal operation
    experiment_id: str | None  # experiment folder name, e.g. "exp_2_3"
    run_id: str | None  # e.g. "run_1"
    source_file: str  # posix path relative to the raw dataset directory
    naming_note: str | None = None


def discover_recordings(raw_dir: Path) -> list[RecordingInfo]:
    """every recording of the raw dataset in a fixed order. ids come from the folders, not the file names."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    infos: list[RecordingInfo] = []
    for path in sorted((raw_dir / "real_op").glob("*.csv")):
        match = _REAL_OP_RE.fullmatch(path.name)
        if match is None:
            raise SchemaError(f"Unexpected normal-operation file name: {path.name}")
        infos.append(
            RecordingInfo(
                recording_id=f"real_op_{match.group(1)}",
                source="real_op",
                subsystem=None,
                experiment_id=None,
                run_id=None,
                source_file=path.relative_to(raw_dir).as_posix(),
            )
        )
    for path in sorted((raw_dir / "dig_twin").glob("exp_*/exp_*/run_*/*.csv")):
        infos.append(_fault_recording_info(path, raw_dir))

    if not infos:
        raise FileNotFoundError(f"No recordings found under {raw_dir}")
    ids = [info.recording_id for info in infos]
    if len(set(ids)) != len(ids):
        raise SchemaError("Recording IDs are not unique")
    return infos


def _fault_recording_info(path: Path, raw_dir: Path) -> RecordingInfo:
    run_dir, exp_dir = path.parent, path.parent.parent
    subsystem = exp_dir.parent.name.removeprefix("exp_")
    match = _FAULT_RE.fullmatch(path.name)
    if match is None:
        raise SchemaError(f"Unexpected fault recording file name: {path}")
    file_subsystem, file_experiment, file_run = match.groups()
    if file_subsystem != subsystem or f"run_{file_run}" != run_dir.name:
        raise SchemaError(f"File name {path.name} contradicts its folder {run_dir}")
    note = None
    if f"exp_{file_experiment}" != exp_dir.name:
        note = f"file name refers to exp{file_experiment}, folder {exp_dir.name} is used as experiment_id"
    return RecordingInfo(
        recording_id=f"{subsystem}_{exp_dir.name}_{run_dir.name}",
        source="dig_twin",
        subsystem=subsystem,
        experiment_id=exp_dir.name,
        run_id=run_dir.name,
        source_file=path.relative_to(raw_dir).as_posix(),
        naming_note=note,
    )


# loading and parsing


def load_events(info: RecordingInfo, raw_dir: Path) -> pd.DataFrame:
    """one recording with every raw field kept as an unmodified string."""
    raw = pd.read_csv(Path(raw_dir) / info.source_file, dtype=str, keep_default_na=False)
    return prepare_events(raw, info)


def load_all_events(infos: Sequence[RecordingInfo], raw_dir: Path) -> pd.DataFrame:
    frames = [load_events(info, raw_dir) for info in infos]
    logger.debug("loaded %d events from %d recordings", sum(map(len, frames)), len(frames))
    return pd.concat(frames, ignore_index=True)


def prepare_events(raw: pd.DataFrame, info: RecordingInfo) -> pd.DataFrame:
    """validate a raw time_s,node,value,type table and attach the recording ids."""
    if tuple(raw.columns) != EVENT_COLUMNS:
        raise SchemaError(f"{info.source_file}: expected columns {EVENT_COLUMNS}, got {tuple(raw.columns)}")
    if raw.empty:
        raise SchemaError(f"{info.source_file}: recording has no events")

    time_s = pd.to_numeric(raw["time_s"], errors="coerce")
    if time_s.isna().any():
        bad = raw.loc[time_s.isna(), "time_s"].unique()[:5].tolist()
        raise SchemaError(f"{info.source_file}: unparsable timestamps {bad}")
    if (time_s < 0).any():
        raise SchemaError(f"{info.source_file}: negative timestamps")
    unknown = sorted(set(raw["type"]) - set(VARIABLE_TYPES))
    if unknown:
        raise SchemaError(f"{info.source_file}: unknown variable types {unknown}")

    n = len(raw)
    return pd.DataFrame(
        {
            "recording_id": [info.recording_id] * n,
            "source": [info.source] * n,
            "subsystem": [info.subsystem] * n,
            "experiment_id": [info.experiment_id] * n,
            "run_id": [info.run_id] * n,
            "source_file": [info.source_file] * n,
            "event_seq": np.arange(n, dtype=np.int64),
            "time_s": time_s.to_numpy(dtype=float),
            "time_s_raw": raw["time_s"].to_numpy(),
            "node": raw["node"].to_numpy(),
            "type": raw["type"].to_numpy(),
            "raw_value": raw["value"].to_numpy(),
        }
    )


def parse_values(events: pd.DataFrame, strict_binary: bool = True) -> pd.DataFrame:
    """add value_num (binary/alarm 0/1, continuous/counter float) and value_cat (categorical strings)."""
    conflicting = events.groupby("node")["type"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        raise SchemaError(f"Nodes with more than one declared type: {conflicting.index.tolist()}")
    unknown = sorted(set(events["type"]) - set(VARIABLE_TYPES))
    if unknown:
        raise SchemaError(f"Unknown variable types {unknown}")

    out = events.copy()
    raw = out["raw_value"].astype(str)
    value_num = pd.Series(np.nan, index=out.index, dtype=float)

    is_bool = out["type"].isin(BOOLEAN_TYPES)
    normalized = raw[is_bool].str.strip().str.lower()
    parsed = pd.Series(np.nan, index=normalized.index, dtype=float)
    parsed[normalized.isin(_TRUE_STRINGS)] = 1.0
    parsed[normalized.isin(_FALSE_STRINGS)] = 0.0
    bad = parsed.index[parsed.isna()]
    if len(bad):
        examples = _examples(out.loc[bad])
        if strict_binary:
            raise ValueParseError(f"{len(bad)} Binary/Alarm values are not True/False/1/0, e.g. {examples}")
        logger.warning("%d Binary/Alarm values set to NA, e.g. %s", len(bad), examples)
    value_num[is_bool] = parsed

    is_numeric = out["type"].isin(NUMERIC_TYPES)
    numbers = pd.to_numeric(raw[is_numeric].str.strip(), errors="coerce")
    bad = numbers.index[numbers.isna() | ~np.isfinite(numbers)]
    if len(bad):
        raise ValueParseError(
            f"{len(bad)} Continuous/Counter values are not finite numbers, e.g. {_examples(out.loc[bad])}"
        )
    value_num[is_numeric] = numbers

    out["value_num"] = value_num
    out["value_cat"] = raw.where(out["type"] == "Categorical").astype("string")
    return out


def _examples(rows: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    cols = ["recording_id", "node", "raw_value"]
    return rows[cols].drop_duplicates().head(limit).to_dict("records")


def node_types(events: pd.DataFrame) -> dict[str, str]:
    """declared variable type per node."""
    return events.drop_duplicates("node").set_index("node")["type"].to_dict()


# duplicates and change marking


def resolve_duplicates(
    events: pd.DataFrame, policy: DuplicatePolicy = "last"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """sort events by (recording, time, file order) and apply the duplicate policy. returns events and a log."""
    if policy not in ("last", "first", "error"):
        raise ValueError(f"Unknown duplicate policy: {policy!r}")
    ordered = events.sort_values(["recording_id", "time_s", "event_seq"], kind="mergesort").reset_index(drop=True)
    key = ["recording_id", "node", "time_s"]
    in_group = ordered.duplicated(key, keep=False)
    if not in_group.any():
        empty_log = ordered.iloc[0:0].assign(kept=pd.Series(dtype=bool), policy=pd.Series(dtype=str))
        return ordered, empty_log
    if policy == "error":
        raise DuplicateEventError(f"{int(in_group.sum())} events share (recording, node, time_s)")

    dropped = ordered.duplicated(key, keep=policy)
    log = ordered.loc[in_group].assign(kept=~dropped[in_group], policy=policy)
    logger.debug("duplicate policy %r dropped %d of %d duplicated events", policy, int(dropped.sum()), len(log))
    return ordered.loc[~dropped].reset_index(drop=True), log


def _value_key(events: pd.DataFrame) -> pd.Series:
    """one comparable string per event value, whatever the variable type."""
    numeric = events["value_num"].map(repr)
    return numeric.where(events["type"] != "Categorical", events["value_cat"].astype(object))


def mark_changes(events: pd.DataFrame) -> pd.DataFrame:
    """flag each event as first event (is_initial), value change (is_change) or repeat (is_repeat)."""
    ordered = events.sort_values(["recording_id", "node", "time_s", "event_seq"], kind="mergesort")
    key = _value_key(ordered)
    previous = key.groupby([ordered["recording_id"], ordered["node"]]).shift()
    has_previous = previous.notna()
    marked = ordered.assign(
        is_initial=~has_previous,
        is_change=has_previous & (key != previous),
        is_repeat=has_previous & (key == previous),
    )
    return marked.sort_values(["recording_id", "time_s", "event_seq"], kind="mergesort").reset_index(drop=True)


# state reconstruction


def grid_index(time_s: np.ndarray | pd.Series, interval_s: float) -> np.ndarray:
    """index k of the first grid point k * interval_s that is >= time_s."""
    scaled = np.round(np.asarray(time_s, dtype=float) / interval_s, _GRID_DECIMALS)
    return np.ceil(scaled).astype(np.int64)


def state_dtype(variable_type: str) -> str:
    if variable_type in BOOLEAN_TYPES:
        return "Int8"
    if variable_type in NUMERIC_TYPES:
        return "float64"
    return "string"


def state_node_columns(states: pd.DataFrame) -> list[str]:
    reserved = set(ID_COLUMNS) | set(GRID_COLUMNS)
    return [col for col in states.columns if col not in reserved]


def reconstruct_states(
    events: pd.DataFrame, interval_s: float, nodes: Sequence[str] | None = None
) -> pd.DataFrame:
    """zero-order-hold state table: one row per recording and grid time, na before a node's first event."""
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError(f"interval_s must be positive, got {interval_s}")
    required = {"recording_id", "time_s", "event_seq", "node", "type", "value_num", "value_cat"}
    missing = required - set(events.columns)
    if missing:
        raise SchemaError(f"Events lack columns {sorted(missing)}. Run parse_values first")
    if events.duplicated(["recording_id", "node", "time_s"]).any():
        raise DuplicateEventError("Resolve duplicate events before reconstructing states")

    types = node_types(events)
    nodes = sorted(types) if nodes is None else list(nodes)
    unknown = [node for node in nodes if node not in types]
    if unknown:
        raise KeyError(f"No events for nodes {unknown}")

    ordered = events.sort_values(["recording_id", "time_s", "event_seq"], kind="mergesort")
    frames = [
        _reconstruct_recording(group, interval_s, nodes, types)
        for _, group in ordered.groupby("recording_id", sort=True)
    ]
    return pd.concat(frames, ignore_index=True)


def _reconstruct_recording(
    events: pd.DataFrame, interval_s: float, nodes: list[str], types: dict[str, str]
) -> pd.DataFrame:
    steps = grid_index(events["time_s"], interval_s)
    grid = pd.RangeIndex(int(steps.max()) + 1)
    # events are in time order, so the last row per (node, step) is the state at the end of the step
    last = events.assign(_step=steps).groupby(["node", "_step"], sort=False).tail(1)

    observed: dict[str, pd.Series] = {}
    for node, group in last.groupby("node", sort=False):
        if types[node] == "Categorical":
            values = group["value_cat"].to_numpy(dtype=object)
        else:
            values = group["value_num"].to_numpy(dtype=float)
        series = pd.Series(values, index=group["_step"].to_numpy())
        observed[node] = series.reindex(grid).ffill()

    columns: dict[str, pd.Series] = {}
    for node in nodes:
        dtype = state_dtype(types[node])
        if node in observed:
            columns[node] = observed[node].astype(dtype)
        else:
            columns[node] = pd.Series(np.nan, index=grid).astype(dtype)

    first = events.iloc[0]
    meta = {col: np.repeat(np.array([first[col]], dtype=object), len(grid)) for col in ID_COLUMNS}
    meta["step"] = grid.to_numpy(dtype=np.int64)
    meta["t_s"] = np.round(grid.to_numpy() * interval_s, _GRID_DECIMALS)
    return pd.concat([pd.DataFrame(meta, index=grid), pd.DataFrame(columns, index=grid)], axis=1)


# verification against raw events


@dataclass
class VerificationResult:
    interval_s: float
    n_recordings: int = 0
    n_cells_checked: int = 0
    n_mismatches: int = 0
    n_grid_errors: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.n_mismatches == 0 and self.n_grid_errors == 0


def verify_states(
    events: pd.DataFrame, states: pd.DataFrame, interval_s: float, max_examples: int = 20
) -> VerificationResult:
    """re-derive every state cell from the events by binary search and compare it with the state table."""
    result = VerificationResult(interval_s=interval_s)
    types = node_types(events)
    node_columns = state_node_columns(states)
    events_by_recording = dict(tuple(events.groupby("recording_id", sort=False)))
    state_ids = set(states["recording_id"].unique())

    for recording_id in sorted(set(events_by_recording) - state_ids):
        result.n_grid_errors += 1
        _add_example(result, max_examples, recording_id=recording_id, problem="recording missing from states")

    for recording_id, rows in states.groupby("recording_id", sort=True):
        result.n_recordings += 1
        recording_events = events_by_recording.get(recording_id)
        if recording_events is None:
            result.n_grid_errors += 1
            _add_example(result, max_examples, recording_id=recording_id, problem="no events for recording")
            continue
        rows = rows.sort_values("step")
        expected_rows = int(grid_index([recording_events["time_s"].max()], interval_s)[0]) + 1
        steps = rows["step"].to_numpy()
        grid_ok = (
            len(rows) == expected_rows
            and np.array_equal(steps, np.arange(expected_rows))
            and np.allclose(rows["t_s"].to_numpy(dtype=float), steps * interval_s)
        )
        if not grid_ok:
            result.n_grid_errors += 1
            _add_example(result, max_examples, recording_id=recording_id, problem="unexpected grid")
            continue

        t_grid = rows["t_s"].to_numpy(dtype=float)
        by_node = dict(tuple(recording_events.groupby("node", sort=False)))
        for node in node_columns:
            node_events = by_node.get(node)
            if node_events is None:
                expected = np.full(len(rows), None, dtype=object)
            else:
                node_events = node_events.sort_values(["time_s", "event_seq"])
                times = node_events["time_s"].to_numpy(dtype=float)
                is_cat = types[node] == "Categorical"
                values = (node_events["value_cat"] if is_cat else node_events["value_num"]).to_numpy(dtype=object)
                position = np.searchsorted(times, t_grid + _GRID_TOLERANCE, side="right") - 1
                expected = np.where(position >= 0, values[np.clip(position, 0, None)], None)
            mismatch = _mismatches(rows[node], expected, categorical=types.get(node) == "Categorical")
            result.n_cells_checked += len(rows)
            if mismatch.any():
                result.n_mismatches += int(mismatch.sum())
                first = int(np.flatnonzero(mismatch)[0])
                _add_example(
                    result,
                    max_examples,
                    recording_id=recording_id,
                    node=node,
                    t_s=float(t_grid[first]),
                    state=_jsonable(rows[node].iloc[first]),
                    expected=_jsonable(expected[first]),
                )
    return result


def _mismatches(actual: pd.Series, expected: np.ndarray, categorical: bool) -> np.ndarray:
    expected_na = pd.isna(expected)
    if categorical:
        actual_values = actual.to_numpy(dtype=object)
        actual_na = pd.isna(actual_values)
        same = np.array(
            [a == e for a, e in zip(np.where(actual_na, "", actual_values), np.where(expected_na, "", expected))],
            dtype=bool,
        )
    else:
        actual_values = actual.to_numpy(dtype=float, na_value=np.nan)
        actual_na = np.isnan(actual_values)
        expected_values = np.where(expected_na, np.nan, expected).astype(float)
        same = (actual_values == expected_values) | (actual_na & expected_na)
    return (actual_na != expected_na) | ~same


def _add_example(result: VerificationResult, limit: int, **example: Any) -> None:
    if len(result.examples) < limit:
        result.examples.append(example)


def _jsonable(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


# diagnostics


def event_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """per recording and node: event and change counts, gaps, t = 0 snapshot and counter resets."""
    marked = events if "is_change" in events.columns else mark_changes(events)
    ordered = marked.sort_values(["recording_id", "node", "time_s", "event_seq"], kind="mergesort")
    grouped = ordered.groupby(["recording_id", "node"])
    ordered = ordered.assign(
        _gap=grouped["time_s"].diff(),
        _reset=(ordered["type"] == "Counter") & (grouped["value_num"].diff() < 0),
    )
    stats = (
        ordered.groupby(["recording_id", "node"], sort=True)
        .agg(
            source=("source", "first"),
            subsystem=("subsystem", "first"),
            experiment_id=("experiment_id", "first"),
            type=("type", "first"),
            n_events=("time_s", "size"),
            n_changes=("is_change", "sum"),
            n_repeats=("is_repeat", "sum"),
            first_event_s=("time_s", "min"),
            last_event_s=("time_s", "max"),
            min_gap_s=("_gap", "min"),
            median_gap_s=("_gap", "median"),
            max_gap_s=("_gap", "max"),
            n_counter_resets=("_reset", "sum"),
        )
        .reset_index()
    )
    stats["has_t0_snapshot"] = stats["first_event_s"] == 0.0
    return stats


def sampling_sensitivity(events: pd.DataFrame, intervals_s: Sequence[float]) -> pd.DataFrame:
    """per interval, source and type: how many state changes a regular grid keeps, hides or cannot order."""
    marked = events if "is_change" in events.columns else mark_changes(events)
    ordered = marked.sort_values(["recording_id", "time_s", "event_seq"], kind="mergesort")
    ordered = ordered.assign(_key=_value_key(ordered))
    durations = ordered.groupby("recording_id").agg(source=("source", "first"), t_end=("time_s", "max"))

    rows: list[dict[str, Any]] = []
    for interval_s in intervals_s:
        bins = (
            ordered.assign(_step=grid_index(ordered["time_s"], interval_s))
            .groupby(["recording_id", "node", "_step"], sort=True)
            .agg(
                source=("source", "first"),
                type=("type", "first"),
                key=("_key", "last"),
                n_changes=("is_change", "sum"),
            )
            .reset_index()
        )
        previous = bins.groupby(["recording_id", "node"])["key"].shift()
        bins["visible"] = previous.notna() & (bins["key"] != previous)
        bins["masked"] = (bins["n_changes"] > 0) & ~bins["visible"]
        bins["lost"] = bins["n_changes"] - bins["visible"].astype(int)

        changed_nodes = bins[bins["visible"]].groupby(["recording_id", "_step"]).agg(
            source=("source", "first"), n_nodes=("node", "size")
        )
        n_rows = grid_index(durations["t_end"], interval_s) + 1
        median_rows = pd.Series(n_rows, index=durations.index).groupby(durations["source"]).median()

        def summary(group: pd.DataFrame, source: str, variable_type: str) -> dict[str, Any]:
            true_changes = int(group["n_changes"].sum())
            visible = int(group["visible"].sum())
            return {
                "interval_s": interval_s,
                "source": source,
                "variable_type": variable_type,
                "n_true_changes": true_changes,
                "n_visible_transitions": visible,
                "transition_retention": visible / true_changes if true_changes else np.nan,
                "n_lost_changes": int(group["lost"].sum()),
                "n_bins_with_masked_change": int(group["masked"].sum()),
                "multi_change_step_fraction": np.nan,
                "median_rows_per_recording": float(median_rows.get(source, np.nan)),
            }

        for (source, variable_type), group in bins.groupby(["source", "type"], sort=True):
            rows.append(summary(group, source, variable_type))
        for source, group in bins.groupby("source", sort=True):
            row = summary(group, source, "ALL")
            steps = changed_nodes[changed_nodes["source"] == source]["n_nodes"]
            row["multi_change_step_fraction"] = float((steps > 1).mean()) if len(steps) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# model-ready views


@dataclass
class ViewData:
    """per-recording arrays for an estimator. lags never cross recordings."""

    var_names: list[str]
    arrays: dict[str, np.ndarray]  # recording_id -> (T, N), unknown cells hold 0
    masks: dict[str, np.ndarray]  # recording_id -> (T, N), True where the state is unknown
    data_type: np.ndarray  # (N,): 0 continuous, 1 discrete (tigramite convention)


@dataclass
class ModelView:
    """column selection and encodings (continuous, discrete or mixed) fitted on training recordings."""

    kind: str
    train_ids: list[str]
    columns: list[str]
    column_types: dict[str, str]
    sources: dict[str, str]  # view column -> state column
    means: dict[str, float]
    stds: dict[str, float]
    vocabularies: dict[str, list[str]]
    exclusions: list[dict[str, str]]
    counter_first_differences: bool = False

    @classmethod
    def from_schema(cls, types: dict[str, str], nodes: Sequence[str], kind: ViewKind) -> ModelView:
        """view from a predeclared node list (binary/alarm only), without any data-dependent step."""
        if kind not in VIEW_KINDS:
            raise ValueError(f"Unknown view kind {kind!r}. Expected one of {VIEW_KINDS}")
        exclusions, columns, column_types, sources = [], [], {}, {}
        for node in nodes:
            if node not in types:
                exclusions.append({"column": node, "reason": "no_type"})
                continue
            reason = _type_exclusion(types[node], kind, False)
            if reason:
                exclusions.append({"column": node, "reason": reason})
                continue
            if types[node] not in BOOLEAN_TYPES:
                raise ValueError(f"{node}: type {types[node]!r} is not binary")
            columns.append(node)
            column_types[node] = types[node]
            sources[node] = node
        return cls(kind=kind, train_ids=[], columns=columns, column_types=column_types, sources=sources,
                   means={}, stds={}, vocabularies={}, exclusions=exclusions)

    @staticmethod
    def _source_frame(states: pd.DataFrame, sources: dict[str, str], column_types: dict[str, str]) -> pd.DataFrame:
        ordered = states.sort_values(["recording_id", "step"], kind="mergesort")
        frame = {"recording_id": ordered["recording_id"].to_numpy()}
        for name, node in sources.items():
            if column_types[name] == "Counter":
                frame[name] = ordered.groupby("recording_id")[node].diff().to_numpy(dtype=float)
            else:
                frame[name] = ordered[node].to_numpy(dtype=object)
        return pd.DataFrame(frame)

    @property
    def data_type(self) -> np.ndarray:
        return np.array([int(self.column_types[c] in DISCRETE_TYPES) for c in self.columns], dtype=int)

    def transform(self, states: pd.DataFrame, recording_ids: Sequence[str]) -> ViewData:
        absent = set(recording_ids) - set(states["recording_id"])
        if absent:
            raise KeyError(f"Recordings missing from states: {sorted(absent)}")
        subset = states[states["recording_id"].isin(recording_ids)]
        frame = self._source_frame(subset, self.sources, self.column_types)
        arrays: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        for recording_id, rows in frame.groupby("recording_id", sort=False):
            data = np.zeros((len(rows), len(self.columns)), dtype=float)
            mask = np.zeros_like(data, dtype=bool)
            for j, name in enumerate(self.columns):
                values = rows[name]
                unknown = values.isna().to_numpy()
                encoded = self._encode(name, values[~unknown])
                data[~unknown, j] = encoded
                mask[:, j] = unknown
            arrays[recording_id] = data
            masks[recording_id] = mask
        return ViewData(
            var_names=list(self.columns),
            arrays={rid: arrays[rid] for rid in recording_ids},
            masks={rid: masks[rid] for rid in recording_ids},
            data_type=self.data_type,
        )

    def _encode(self, name: str, values: pd.Series) -> np.ndarray:
        column_type = self.column_types[name]
        if column_type == "Categorical":
            vocabulary = self.vocabularies[name]
            codes = {value: code for code, value in enumerate(vocabulary)}
            return values.astype(str).map(codes).fillna(len(vocabulary)).to_numpy(dtype=float)
        if column_type in NUMERIC_TYPES:
            return (values.to_numpy(dtype=float) - self.means[name]) / self.stds[name]
        return values.to_numpy(dtype=float)


def states_file(directory: Path, subsystem: str) -> Path:
    return Path(directory) / f"{subsystem}_states.csv"


def write_states(view_data: ViewData, file: Path, interval_s: float) -> Path:
    """one row per recording and grid step, one column per variable. unknown states are left empty."""
    frames = []
    for recording_id, values in view_data.arrays.items():
        frame = pd.DataFrame(values, columns=view_data.var_names).astype("Int64")
        frame = frame.mask(pd.DataFrame(view_data.masks[recording_id], columns=view_data.var_names))
        frame.insert(0, "t_s", np.arange(len(frame)) * interval_s)
        frame.insert(0, "step", range(len(frame)))
        frame.insert(0, "recording_id", recording_id)
        frames.append(frame)
    file.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(file, index=False)
    return file


def read_states(file: Path) -> ViewData:
    """read what write_states wrote. every variable is binary or an alarm, so all are discrete."""
    table = pd.read_csv(file, dtype={"recording_id": str})
    var_names = [c for c in table.columns if c not in ("recording_id", "step", "t_s")]
    arrays, masks = {}, {}
    for recording_id, rows in table.groupby("recording_id", sort=False):
        values = rows[var_names]
        masks[recording_id] = values.isna().to_numpy()
        arrays[recording_id] = values.fillna(0).to_numpy(dtype=float)
    return ViewData(var_names=var_names, arrays=arrays, masks=masks, data_type=np.ones(len(var_names), dtype=int))


def _type_exclusion(variable_type: str, kind: str, counter_first_differences: bool) -> str | None:
    if variable_type == "Counter" and kind != "discrete" and not counter_first_differences:
        return "cumulative_counter_nonstationary"
    if kind == "continuous" and variable_type in DISCRETE_TYPES:
        return "discrete_type_not_in_continuous_view"
    if kind == "discrete" and variable_type in NUMERIC_TYPES:
        return "numeric_type_not_in_discrete_view"
    return None
