import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from core.plots import write_pdf
from core.preprocessing import ViewData, read_states, states_file
from utils.config import ROOT, load_config, setting
from utils.errors import ConfigError, MissingInputError
from utils.logger import log

# a missing or empty value in a written table
DASH = "–"
# column widths of the metric blocks printed by the evaluation
LABEL_WIDTH = 40
CELL_WIDTH = 14

# file passed between steps -> (folder in config paths, stage of `python main.py run <stage>` that writes it)
STEP_FILES = {
    "splits": ("processed", "preprocessing"),
    "discovered_graphs": ("graphs", "discovery"),
    "discovery_settings": ("graphs", "discovery"),
    "scm_graphs": ("graphs", "scm"),
    "dag_removals": ("graphs", "scm"),
    "scm_settings": ("models", "scm"),
    "scm_selection_inner": ("models", "scm"),
    "rca_rankings": ("results", "rca"),
    "rca_runs": ("results", "rca"),
    "replay": ("results", "interventions"),
    "recommendations": ("results", "counterfactuals"),
    "tried_action_sets": ("results", "counterfactuals"),
    "recommendation_refits": ("results", "counterfactuals"),
}


# paths


def path(key: str) -> Path:
    """absolute path of an entry of `paths` in config/config.yaml."""
    return ROOT / setting(load_config(), "paths", key)


def raw_dir() -> Path:
    return path("raw")


def processed_dir() -> Path:
    return path("processed")


def cache_dir() -> Path:
    return path("cache")


def models_dir(name: str | None = None) -> Path:
    """data/models, or one named folder in it."""
    return path("models") if name is None else path("models") / name


def results_dir(name: str | None = None) -> Path:
    """data/results, or one named folder in it."""
    return path("results") if name is None else path("results") / name


def tables_dir() -> Path:
    return path("tables")


def figures_dir() -> Path:
    return path("figures")


def related_recordings_path() -> Path:
    return path("related_recordings")


def log_file() -> Path:
    return path("log_file")


def relative(target: Path) -> str:
    """path relative to the repository root, for messages."""
    try:
        return str(Path(target).relative_to(ROOT))
    except ValueError:
        return str(target)


# state tables


def view_name(subsystem: str, interval_s: float) -> str:
    """name of the state data in the pcmci+ run keys, e.g. coolant_dt0.5."""
    return f"{subsystem}_dt{interval_s:g}"


def states_path(subsystem: str) -> Path:
    return states_file(processed_dir(), subsystem)


def load_states(subsystem: str) -> ViewData:
    """the state table of one subsystem written by the preprocessing stage."""
    file = states_path(subsystem)
    if not file.exists():
        raise MissingInputError(f"{relative(file)} not found: run \"python main.py run preprocessing\" first")
    return read_states(file)


# files passed between steps


def step_file(name: str) -> Path:
    if name not in STEP_FILES:
        raise ConfigError(f"no stage writes the file {name!r} (utils/io.py STEP_FILES)")
    return path(STEP_FILES[name][0]) / f"{name}.csv"


def read_input(name: str, **read_csv_args: Any) -> pd.DataFrame:
    """a file written by an earlier stage. if it is missing, MissingInputError names that stage."""
    file = step_file(name)
    if not file.exists():
        raise MissingInputError(f"{relative(file)} not found: run \"python main.py run {STEP_FILES[name][1]}\" first")
    return pd.read_csv(file, **read_csv_args)


def write_output(table: pd.DataFrame, name: str) -> Path:
    """write a file for the later stages."""
    return write_csv(table, step_file(name))


def save_result(table: pd.DataFrame, name: str) -> Path:
    """write a detailed evaluation result to data/results/evaluation/<name>.csv."""
    return write_csv(table, results_dir("evaluation") / f"{name}.csv")


def write_csv(table: pd.DataFrame, file: Path) -> Path:
    """csv without index. list, set and dict cells become json text."""
    file.parent.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    for column in out.columns:
        if out[column].map(lambda v: isinstance(v, (list, tuple, set, dict))).any():
            out[column] = out[column].map(_json_cell)
    out.to_csv(file, index=False)
    log.debug(f"wrote {relative(file)} ({len(table)} rows)")
    return file


def _json_cell(value: Any) -> Any:
    if isinstance(value, set):
        value = sorted(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    return value


# numbered tables and figures


def thesis_name(kind: str, number: str, title: str) -> str:
    """file stem of a numbered table or figure, e.g. table_6_4_root_cause_ranking_performance."""
    if kind not in ("table", "figure"):
        raise ConfigError(f"kind must be 'table' or 'figure', got {kind!r}")
    parts = [kind, _snake(str(number)), _snake(title)]
    return "_".join(p for p in parts if p)


def save_table(table: pd.DataFrame, number: str, title: str) -> Path:
    """write a numbered table as csv under its number and title and log the saved line."""
    file = write_csv(table, tables_dir() / f"{thesis_name('table', number, title)}.csv")
    log.info(f"saved table {number} -> {relative(file)}")
    return file


def save_figure(fig: plt.Figure, number: str, title: str) -> Path:
    """write a numbered figure as pdf under its number and title, close it and log the saved line."""
    file = figures_dir() / f"{thesis_name('figure', number, title)}.pdf"
    write_pdf(fig, file)
    log.info(f"saved figure {number} -> {relative(file)}")
    return file


def _snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# printed numbers


def half_up(value: float, decimals: int) -> float:
    """value rounded as printed (halves away from zero, unlike python's round)."""
    return float(_quantized(value, decimals))


def printed(value: float, decimals: int = 2) -> str:
    """value as printed in a table (0.945 -> '0.95'), not rounded to the nearest binary float."""
    return str(_quantized(value, decimals))


def printed_range(values: pd.Series, decimals: int = 2) -> str:
    """'0.94–0.96', or one number when both ends print the same."""
    low, high = printed(values.min(), decimals), printed(values.max(), decimals)
    return low if low == high else f"{low}{DASH}{high}"


def _quantized(value: float, decimals: int) -> Decimal:
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


# metric blocks of the evaluation


def metric_row(label: str, *cells: str, label_width: int = LABEL_WIDTH, cell_width: int = CELL_WIDTH) -> str:
    """one aligned line of a metric block: a label and its cells."""
    return (f"  {label:<{label_width}}" + "".join(f"{cell:<{cell_width}}" for cell in cells)).rstrip()


def report(lines: list[str]) -> None:
    """a metric block and its closing sentence on the terminal, after an empty line."""
    for line in ["", *lines]:
        log.info(line)
