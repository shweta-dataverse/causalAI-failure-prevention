"""Event-log preprocessing for the causRCA dataset.

Every step works one recording at a time and never mixes recordings:

1. ``discover_recordings``  RecordingInfo per raw CSV, IDs taken from folders
2. ``load_events``          raw OPC UA change events, values kept as strings
3. ``parse_values``         typed values according to the *declared* type
4. ``resolve_duplicates``   deterministic policy for equal (recording, node, time_s)
5. ``reconstruct_states``   zero-order-hold state table on a regular grid
6. ``verify_states``        independent re-check of the state table against events
7. diagnostics              event statistics, sampling sensitivity, quality report
8. ``ModelView``            estimator-specific views fitted on training recordings

Policies (also written into the data-quality report):

* Timestamps are OPC UA *publish* times. Events sharing a timestamp have no
  reliable order; the file row order (``event_seq``) only breaks ties
  deterministically and is never evidence of causal order.
* Duplicate (recording, node, time_s) events: by default the last row in the
  file wins (``duplicate_policy="last"``).
* The state at grid time t is the value of the last event with time <= t
  (forward fill). Before a node's first event its state is unknown (NA) and is
  never backfilled. Rows at t = 0 are the recorder's initial snapshot.
* Binary/Alarm values are parsed strictly. Categorical values stay strings even
  when they look numeric (e.g. ``Prog_LineNo``).
* Nothing here reads fault annotations (causes.json, experiment descriptions),
  so no label information can enter the processed data.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

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
# timestamps have millisecond resolution; rounding t / interval to 9 decimals
# removes floating-point noise before taking the ceiling.
_GRID_DECIMALS = 9
_GRID_TOLERANCE = 1e-9

_REAL_OP_RE = re.compile(r"dataset_real_op_(\d+)\.csv")
_FAULT_RE = re.compile(r"faultDataset_([a-z]+)_exp([0-9_]+)_run_(\d+)\.csv")

DuplicatePolicy = Literal["last", "first", "error"]
ViewKind = Literal["continuous", "discrete", "mixed"]
VIEW_KINDS: tuple[str, ...] = ("continuous", "discrete", "mixed")


class PreprocessingError(Exception):
    """Base class for preprocessing failures."""


class SchemaError(PreprocessingError):
    """Raw files or event tables do not have the expected structure."""


class ValueParseError(PreprocessingError):
    """A value cannot be parsed according to its declared variable type."""


class DuplicateEventError(PreprocessingError):
    """Several events share (recording, node, time_s)."""


class VerificationError(PreprocessingError):
    """The reconstructed state table disagrees with the raw events."""


# ---------------------------------------------------------------------------
# configuration and recording discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingConfig:
    raw_dir: Path = Path("data/raw/causRCA")
    output_dir: Path = Path("data/processed")
    report_dir: Path = Path("results/tables/preprocessing")
    sampling_interval_s: float = 1.0
    sensitivity_intervals_s: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
    duplicate_policy: DuplicatePolicy = "last"
    strict_binary: bool = True
    counter_first_differences: bool = False

    def __post_init__(self) -> None:
        intervals = (self.sampling_interval_s, *self.sensitivity_intervals_s)
        if any(not np.isfinite(dt) or dt <= 0 for dt in intervals):
            raise ValueError(f"Sampling intervals must be positive, got {intervals}")
        if self.duplicate_policy not in ("last", "first", "error"):
            raise ValueError(f"Unknown duplicate_policy: {self.duplicate_policy!r}")

    @classmethod
    def from_yaml(cls, path: Path) -> PreprocessingConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown preprocessing config keys: {sorted(unknown)}")
        for key in ("raw_dir", "output_dir", "report_dir"):
            if key in data:
                data[key] = Path(data[key])
        if "sensitivity_intervals_s" in data:
            data["sensitivity_intervals_s"] = tuple(float(x) for x in data["sensitivity_intervals_s"])
        if "sampling_interval_s" in data:
            data["sampling_interval_s"] = float(data["sampling_interval_s"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = value.as_posix()
            elif isinstance(value, tuple):
                data[key] = list(value)
        return data


@dataclass(frozen=True)
class RecordingInfo:
    recording_id: str
    source: str  # "real_op" (normal operation) or "dig_twin" (fault run)
    subsystem: str | None  # coolant / hydraulics / probe; None for normal operation
    experiment_id: str | None  # experiment folder name, e.g. "exp_2_3"
    run_id: str | None  # e.g. "run_1"
    source_file: str  # POSIX path relative to the raw dataset directory
    naming_note: str | None = None


def discover_recordings(raw_dir: Path) -> list[RecordingInfo]:
    """List every recording in the raw dataset, in a deterministic order.

    IDs come from the folder structure. Where a file name disagrees with its
    folder (``exp_2_3/`` holds ``faultDataset_probe_exp2_*``) the folder wins
    and the disagreement is kept in ``naming_note``.
    """
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
        note = f"file name refers to exp{file_experiment}; folder {exp_dir.name} is used as experiment_id"
    return RecordingInfo(
        recording_id=f"{subsystem}_{exp_dir.name}_{run_dir.name}",
        source="dig_twin",
        subsystem=subsystem,
        experiment_id=exp_dir.name,
        run_id=run_dir.name,
        source_file=path.relative_to(raw_dir).as_posix(),
        naming_note=note,
    )


# ---------------------------------------------------------------------------
# loading and parsing
# ---------------------------------------------------------------------------


def load_events(info: RecordingInfo, raw_dir: Path) -> pd.DataFrame:
    """Read one recording; every raw field is kept as an unmodified string."""
    raw = pd.read_csv(Path(raw_dir) / info.source_file, dtype=str, keep_default_na=False)
    return prepare_events(raw, info)


def load_all_events(infos: Sequence[RecordingInfo], raw_dir: Path) -> pd.DataFrame:
    frames = [load_events(info, raw_dir) for info in infos]
    logger.info("Loaded %d events from %d recordings", sum(map(len, frames)), len(frames))
    return pd.concat(frames, ignore_index=True)


def prepare_events(raw: pd.DataFrame, info: RecordingInfo) -> pd.DataFrame:
    """Validate a raw ``time_s,node,value,type`` table and attach recording IDs."""
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
    """Add ``value_num`` (Binary/Alarm as 0/1, Continuous/Counter as float) and
    ``value_cat`` (Categorical values as strings, never converted to numbers)."""
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
    """Declared variable type per node."""
    return events.drop_duplicates("node").set_index("node")["type"].to_dict()


# ---------------------------------------------------------------------------
# duplicates and change marking
# ---------------------------------------------------------------------------


def resolve_duplicates(
    events: pd.DataFrame, policy: DuplicatePolicy = "last"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort events by (recording, time, file order) and apply the duplicate policy.

    Returns the resolved events and a log of every event involved in a
    duplicate group, with ``kept`` marking the survivor.
    """
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
    logger.info("Duplicate policy %r dropped %d of %d duplicated events", policy, int(dropped.sum()), len(log))
    return ordered.loc[~dropped].reset_index(drop=True), log


def _value_key(events: pd.DataFrame) -> pd.Series:
    """One comparable string per event value, regardless of variable type."""
    numeric = events["value_num"].map(repr)
    return numeric.where(events["type"] != "Categorical", events["value_cat"].astype(object))


def mark_changes(events: pd.DataFrame) -> pd.DataFrame:
    """Flag each event as the node's first event (``is_initial``), a value
    change (``is_change``) or a re-publication of the same value (``is_repeat``)."""
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


# ---------------------------------------------------------------------------
# state reconstruction
# ---------------------------------------------------------------------------


def grid_index(time_s: np.ndarray | pd.Series, interval_s: float) -> np.ndarray:
    """Index k of the first grid point k * interval_s that is >= time_s."""
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
    """Archival state table: one row per recording and grid time ``t = k * interval_s``.

    Each node column holds the value of that node's last event with time <= t,
    or NA before its first event. The grid of a recording runs from 0 to the
    first grid point at or after its last event. All nodes are kept, constant
    ones included; nodes absent from a recording are all-NA there.
    """
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError(f"interval_s must be positive, got {interval_s}")
    required = {"recording_id", "time_s", "event_seq", "node", "type", "value_num", "value_cat"}
    missing = required - set(events.columns)
    if missing:
        raise SchemaError(f"Events lack columns {sorted(missing)}; run parse_values first")
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
    # events are in time order, so the last row per (node, step) is the state
    # at the end of that grid step.
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


def transitions_per_bin(events: pd.DataFrame, interval_s: float) -> pd.DataFrame:
    """Number of value changes per (recording, grid step, node).

    Grid step k covers (t_{k-1}, t_k]. Several changes in one step, such as a
    short on/off pulse, are counted here even when the state table hides them.
    """
    marked = events if "is_change" in events.columns else mark_changes(events)
    changes = marked[marked["is_change"]]
    return (
        changes.assign(step=grid_index(changes["time_s"], interval_s))
        .groupby(["recording_id", "step", "node"])
        .size()
        .rename("n_changes")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# verification against raw events
# ---------------------------------------------------------------------------


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

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def verify_states(
    events: pd.DataFrame, states: pd.DataFrame, interval_s: float, max_examples: int = 20
) -> VerificationResult:
    """Re-derive every state cell from the events with a separate method
    (binary search over event times) and compare it with the state table."""
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


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def event_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """Per recording and node: event and change counts, inter-event gaps,
    initial snapshot presence and counter resets."""
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
    """What a regular grid of each candidate interval loses.

    Per interval, source and variable type:

    * ``n_true_changes``: value changes in the raw events.
    * ``n_visible_transitions``: changes still visible between consecutive
      grid points; ``transition_retention`` is their share.
    * ``n_bins_with_masked_change``: grid steps with changes but no net change
      (e.g. a pulse that starts and ends inside one step).
    * ``multi_change_step_fraction`` (rows with type "ALL"): share of grid steps
      with a visible transition in which more than one node changed, i.e. where
      the grid cannot order the changes.
    """
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


def transition_bins(events: pd.DataFrame, interval_s: float) -> pd.DataFrame:
    """per (recording, node, grid step with events): value changes inside the step,
    whether the grid shows a change (``visible``) and whether changes are hidden (``masked``).

    same logic as ``sampling_sensitivity``, kept per node instead of per type.
    """
    marked = events if "is_change" in events.columns else mark_changes(events)
    ordered = marked.sort_values(["recording_id", "time_s", "event_seq"], kind="mergesort")
    ordered = ordered.assign(_key=_value_key(ordered), step=grid_index(ordered["time_s"], interval_s))
    bins = (
        ordered.groupby(["recording_id", "node", "step"], sort=True)
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
    return bins.drop(columns="key")


def build_quality_report(
    infos: Sequence[RecordingInfo],
    events: pd.DataFrame,
    duplicate_log: pd.DataFrame,
    stats: pd.DataFrame,
    verification: VerificationResult,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    recordings = pd.DataFrame([asdict(info) for info in infos])
    in_file_order = events.sort_values(["recording_id", "event_seq"])
    out_of_order = int((in_file_order.groupby("recording_id")["time_s"].diff() < 0).sum())
    after_start = events[events["time_s"] > 0]
    sharing = after_start.duplicated(["recording_id", "time_s"], keep=False).mean() if len(after_start) else 0.0
    durations = events.groupby("recording_id").agg(source=("source", "first"), t_end=("time_s", "max"))

    boolean = events[events["type"].isin(BOOLEAN_TYPES)]
    categorical = events[events["type"] == "Categorical"]
    numeric_looking = sorted(
        node
        for node, values in categorical.groupby("node")["raw_value"]
        if pd.to_numeric(values, errors="coerce").notna().all()
    )

    nodes_by_source: dict[str, Any] = {}
    for source, group in stats.groupby("source"):
        n_recordings = group["recording_id"].nunique()
        coverage = group.groupby("node")["recording_id"].nunique()
        changes = group.groupby("node")["n_changes"].sum()
        nodes_by_source[source] = {
            "n_recordings": int(n_recordings),
            "n_nodes_observed": int(len(coverage)),
            "nodes_missing_in_some_recordings": sorted(coverage[coverage < n_recordings].index),
            "nodes_never_changing": sorted(changes[changes == 0].index),
        }

    resets = stats[stats["n_counter_resets"] > 0].groupby("node")["n_counter_resets"].sum()
    dropped = int((~duplicate_log["kept"]).sum()) if len(duplicate_log) else 0
    return {
        "config": config.to_dict(),
        "policies": {
            "timestamps": "OPC UA publish times; ties broken by file row order, not interpreted causally",
            "duplicates": f"same (recording, node, time_s): keep {config.duplicate_policy}",
            "state": "zero-order hold (forward fill) within a recording; NA before first event; no backfill",
            "grid": "t_k = k * interval_s from 0 to the first grid point at or after the last event",
            "categorical": "kept as strings, never converted to numbers",
        },
        "recordings": {
            "total": len(recordings),
            "by_source": recordings["source"].value_counts().sort_index().to_dict(),
            "by_subsystem": recordings["subsystem"].dropna().value_counts().sort_index().to_dict(),
            "naming_notes": recordings.dropna(subset=["naming_note"])[["recording_id", "naming_note"]].to_dict("records"),
        },
        "duration_s": {
            source: {
                "min": float(group["t_end"].min()),
                "median": float(group["t_end"].median()),
                "max": float(group["t_end"].max()),
            }
            for source, group in durations.groupby("source")
        },
        "events": {
            "total_after_duplicate_policy": len(events),
            "duplicate_events": len(duplicate_log),
            "duplicates_dropped": dropped,
            "out_of_order_rows": out_of_order,
            "share_sharing_timestamp_with_other_node": float(sharing),
            "repeat_events_without_change": int(events["is_repeat"].sum()),
        },
        "variable_types": events.drop_duplicates("node")["type"].value_counts().sort_index().to_dict(),
        "boolean_raw_encodings": boolean["raw_value"].value_counts().sort_index().to_dict(),
        "numeric_looking_categoricals": numeric_looking,
        "nodes": {
            "n_total": int(events["node"].nunique()),
            "without_t0_snapshot": sorted(stats.loc[~stats["has_t0_snapshot"], "node"].unique()),
            "by_source": nodes_by_source,
        },
        "counter_resets": {node: int(n) for node, n in resets.items()},
        "verification": verification.to_dict(),
    }


def render_quality_markdown(report: dict[str, Any]) -> str:
    rec, ev, nodes = report["recordings"], report["events"], report["nodes"]
    lines = [
        "# causRCA preprocessing: data-quality report",
        "",
        "## Policies",
        "",
        *[f"- **{key}**: {value}" for key, value in report["policies"].items()],
        "",
        "## Recordings",
        "",
        f"- Total: {rec['total']} ({', '.join(f'{k}: {v}' for k, v in rec['by_source'].items())})",
        f"- Fault runs per subsystem: {', '.join(f'{k}: {v}' for k, v in rec['by_subsystem'].items())}",
        *[
            f"- Duration ({source}): min {d['min']:.1f} s, median {d['median']:.1f} s, max {d['max']:.1f} s"
            for source, d in report["duration_s"].items()
        ],
    ]
    if rec["naming_notes"]:
        lines.append(f"- Naming notes ({len(rec['naming_notes'])}): {rec['naming_notes'][0]['naming_note']}")
    lines += [
        "",
        "## Events",
        "",
        f"- Events after duplicate policy: {ev['total_after_duplicate_policy']}",
        f"- Duplicate events: {ev['duplicate_events']} (dropped {ev['duplicates_dropped']})",
        f"- Out-of-order rows in files: {ev['out_of_order_rows']}",
        f"- Events (t > 0) sharing their timestamp with another node: "
        f"{ev['share_sharing_timestamp_with_other_node']:.1%}",
        f"- Re-published values without change: {ev['repeat_events_without_change']}",
        f"- Binary/Alarm raw encodings: {report['boolean_raw_encodings']}",
        f"- Numeric-looking categoricals kept as strings: {', '.join(report['numeric_looking_categoricals']) or 'none'}",
        f"- Counter resets: {report['counter_resets'] or 'none'}",
        "",
        "## Nodes",
        "",
        f"- Distinct nodes: {nodes['n_total']}; types: {report['variable_types']}",
        f"- Nodes without a t = 0 snapshot in some recording: {', '.join(nodes['without_t0_snapshot']) or 'none'}",
    ]
    for source, info in nodes["by_source"].items():
        lines += [
            f"- **{source}**: {info['n_nodes_observed']} nodes observed in {info['n_recordings']} recordings",
            f"  - missing in some recordings: {', '.join(info['nodes_missing_in_some_recordings']) or 'none'}",
            f"  - never change ({len(info['nodes_never_changing'])}): "
            f"{', '.join(info['nodes_never_changing']) or 'none'}",
        ]
    ver = report["verification"]
    lines += [
        "",
        "## State-table verification",
        "",
        f"- Interval {ver['interval_s']} s, {ver['n_recordings']} recordings, {ver['n_cells_checked']} cells checked",
        f"- Mismatches: {ver['n_mismatches']}, grid errors: {ver['n_grid_errors']} -> "
        f"**{'PASSED' if ver['passed'] else 'FAILED'}**",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# model-ready views
# ---------------------------------------------------------------------------


@dataclass
class ViewData:
    """Per-recording arrays for an estimator; lags never cross recordings."""

    var_names: list[str]
    arrays: dict[str, np.ndarray]  # recording_id -> (T, N); unknown cells hold 0
    masks: dict[str, np.ndarray]  # recording_id -> (T, N); True where the state is unknown
    data_type: np.ndarray  # (N,): 0 continuous, 1 discrete (tigramite convention)


@dataclass
class ModelView:
    """Column selection and encodings for one family of estimators.

    * ``continuous``: Continuous variables, standardized with training mean/std.
    * ``discrete``: Binary/Alarm as 0/1 and Categorical as integer codes from a
      training vocabulary; unseen categories get the code ``len(vocabulary)``.
    * ``mixed``: the union, with ``data_type`` marking discrete columns.

    Cumulative counters are excluded unless ``counter_first_differences`` is set,
    in which case they enter as within-recording first differences. Columns that
    are all-NA or constant in the training recordings are excluded. Every
    exclusion is recorded with its reason.
    """

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
    def fit(
        cls,
        states: pd.DataFrame,
        types: dict[str, str],
        train_ids: Sequence[str],
        kind: ViewKind,
        nodes: Sequence[str] | None = None,
        counter_first_differences: bool = False,
    ) -> ModelView:
        if kind not in VIEW_KINDS:
            raise ValueError(f"Unknown view kind {kind!r}; expected one of {VIEW_KINDS}")
        train_ids = sorted(set(train_ids))
        absent = set(train_ids) - set(states["recording_id"])
        if not train_ids or absent:
            raise KeyError(f"Training recordings missing from states: {sorted(absent) or 'none given'}")

        exclusions: list[dict[str, str]] = []
        sources: dict[str, str] = {}
        column_types: dict[str, str] = {}
        for node in state_node_columns(states) if nodes is None else nodes:
            if node not in states.columns or node not in types:
                exclusions.append({"column": node, "reason": "not_observed"})
                continue
            reason = _type_exclusion(types[node], kind, counter_first_differences)
            if reason:
                exclusions.append({"column": node, "reason": reason})
                continue
            name = f"{node}__diff" if types[node] == "Counter" else node
            sources[name] = node
            column_types[name] = types[node]

        view = cls(
            kind=kind,
            train_ids=train_ids,
            columns=[],
            column_types={},
            sources={},
            means={},
            stds={},
            vocabularies={},
            exclusions=exclusions,
            counter_first_differences=counter_first_differences,
        )
        train = view._source_frame(states[states["recording_id"].isin(train_ids)], sources, column_types)
        for name, node in sources.items():
            values = train[name].dropna()
            if values.empty:
                exclusions.append({"column": name, "reason": "all_missing_in_training"})
                continue
            if values.nunique() <= 1:
                exclusions.append({"column": name, "reason": "constant_in_training"})
                continue
            view.columns.append(name)
            view.sources[name] = node
            view.column_types[name] = column_types[name]
            if column_types[name] in NUMERIC_TYPES:
                view.means[name] = float(values.astype(float).mean())
                view.stds[name] = float(values.astype(float).std(ddof=0))
            elif column_types[name] == "Categorical":
                view.vocabularies[name] = sorted(values.astype(str).unique())
        return view

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

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "data_type": self.data_type.tolist()}


def save_view_data(
    view: ModelView, view_data: ViewData, directory: Path, name: str, interval_s: float
) -> tuple[Path, Path]:
    """write arrays and masks to <directory>/<name>.npz and the view spec to <name>.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids = list(view_data.arrays)
    arrays = {f"data__{rid}": view_data.arrays[rid] for rid in ids}
    arrays.update({f"mask__{rid}": view_data.masks[rid] for rid in ids})
    npz_path = directory / f"{name}.npz"
    np.savez_compressed(
        npz_path,
        var_names=np.array(view_data.var_names),
        data_type=view_data.data_type,
        recording_ids=np.array(ids),
        **arrays,
    )
    spec_path = directory / f"{name}.json"
    _write_json(spec_path, {"name": name, "interval_s": interval_s, "recording_ids": ids, **view.to_dict()})
    return npz_path, spec_path


def load_view_data(directory: Path, name: str) -> tuple[ViewData, dict[str, Any]]:
    """read what ``save_view_data`` wrote."""
    directory = Path(directory)
    with np.load(directory / f"{name}.npz", allow_pickle=False) as npz:
        ids = [str(rid) for rid in npz["recording_ids"]]
        view_data = ViewData(
            var_names=[str(v) for v in npz["var_names"]],
            arrays={rid: npz[f"data__{rid}"] for rid in ids},
            masks={rid: npz[f"mask__{rid}"] for rid in ids},
            data_type=npz["data_type"],
        )
    spec = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    return view_data, spec


def _type_exclusion(variable_type: str, kind: str, counter_first_differences: bool) -> str | None:
    if variable_type == "Counter" and kind != "discrete" and not counter_first_differences:
        return "cumulative_counter_nonstationary"
    if kind == "continuous" and variable_type in DISCRETE_TYPES:
        return "discrete_type_not_in_continuous_view"
    if kind == "discrete" and variable_type in NUMERIC_TYPES:
        return "numeric_type_not_in_discrete_view"
    return None


# ---------------------------------------------------------------------------
# pipeline and CLI
# ---------------------------------------------------------------------------


def run_preprocessing(config: PreprocessingConfig, max_recordings_per_source: int | None = None) -> dict[str, Any]:
    """Run the full preprocessing stage and write all outputs. Returns the quality report."""
    infos = discover_recordings(config.raw_dir)
    if max_recordings_per_source is not None:
        kept: dict[str, int] = {}
        selected = []
        for info in infos:
            if kept.get(info.source, 0) < max_recordings_per_source:
                selected.append(info)
                kept[info.source] = kept.get(info.source, 0) + 1
        infos = selected
    logger.info("Processing %d recordings", len(infos))

    events = parse_values(load_all_events(infos, config.raw_dir), strict_binary=config.strict_binary)
    events, duplicate_log = resolve_duplicates(events, config.duplicate_policy)
    events = mark_changes(events)
    types = node_types(events)

    stats = event_statistics(events)
    sensitivity = sampling_sensitivity(events, config.sensitivity_intervals_s)
    states = reconstruct_states(events, config.sampling_interval_s)
    verification = verify_states(events, states, config.sampling_interval_s)
    report = build_quality_report(infos, events, duplicate_log, stats, verification, config)

    # diagnostic only: which variables each view would drop if it were fitted on
    # all normal-operation recordings. The experiment splits are defined later.
    normal_ids = [info.recording_id for info in infos if info.source == "real_op"]
    exclusions = pd.DataFrame(
        [
            {"view": kind, **item}
            for kind in VIEW_KINDS
            if normal_ids
            for item in ModelView.fit(
                states, types, normal_ids, kind, counter_first_differences=config.counter_first_differences
            ).exclusions
        ],
        columns=["view", "column", "reason"],
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(info) for info in infos]).to_csv(config.output_dir / "recordings.csv", index=False)
    events.to_parquet(config.output_dir / "events.parquet", index=False)
    states.to_parquet(config.output_dir / f"states_dt{config.sampling_interval_s:g}.parquet", index=False)

    stats.to_csv(config.report_dir / "variable_event_stats.csv", index=False)
    sensitivity.to_csv(config.report_dir / "sampling_sensitivity.csv", index=False)
    duplicate_log.to_csv(config.report_dir / "duplicates_log.csv", index=False)
    exclusions.to_csv(config.report_dir / "view_exclusions_real_op.csv", index=False)
    _write_json(config.report_dir / "verification.json", verification.to_dict())
    _write_json(config.report_dir / "data_quality.json", report)
    (config.report_dir / "data_quality.md").write_text(render_quality_markdown(report), encoding="utf-8")
    logger.info("Wrote processed data to %s and reports to %s", config.output_dir, config.report_dir)

    if not verification.passed:
        raise VerificationError(
            f"State table failed verification: {verification.n_mismatches} mismatches, "
            f"{verification.n_grid_errors} grid errors (see {config.report_dir / 'verification.json'})"
        )
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess causRCA event recordings into state tables.")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    parser.add_argument(
        "--max-recordings-per-source",
        type=int,
        default=None,
        help="Process only the first N recordings of each source (quick smoke runs).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    run_preprocessing(PreprocessingConfig.from_yaml(args.config), args.max_recordings_per_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
