import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from core.plots import write_pdf
from core.preprocessing import ViewData, load_view_data
from utils.config import ROOT, load_config, setting
from utils.errors import ConfigError, MissingInputError
from utils.logger import log

PARTS = ("pipeline", "evaluation")
# a missing or empty value in a written table
DASH = "–"
# column widths of the metric blocks printed by the evaluation
LABEL_WIDTH = 40
CELL_WIDTH = 14

# contract file in outputs/pipeline/tables -> stage of `python main.py run <stage>` that writes it
PRODUCERS = {
    "splits": "preprocessing",
    "discovered_graphs": "discovery",
    "discovery_settings": "discovery",
    "scm_graphs": "scm",
    "dag_removals": "scm",
    "scm_settings": "scm",
    "scm_selection_inner": "scm",
    "rca_rankings": "rca",
    "rca_runs": "rca",
    "replay": "interventions",
    "recommendations": "counterfactuals",
    "tried_action_sets": "counterfactuals",
    "recommendation_refits": "counterfactuals",
}


# paths


def path(key: str) -> Path:
    """absolute path of an entry of `paths` in config/config.yaml."""
    return ROOT / setting(load_config(), "paths", key)


def raw_dir() -> Path:
    return path("raw")


def processed_dir() -> Path:
    return path("processed")


def cache_dir(name: str | None = None) -> Path:
    """data/cache, or one named sub-directory of it."""
    base = path("cache")
    return base if name is None else base / name


def related_recordings_path() -> Path:
    return path("related_recordings")


def log_file() -> Path:
    return path("log_file")


def tables_dir(part: str = "pipeline") -> Path:
    return _part_dir(part) / "tables"


def figures_dir(part: str = "pipeline") -> Path:
    return _part_dir(part) / "figures"


def _part_dir(part: str) -> Path:
    if part not in PARTS:
        raise ConfigError(f"unknown output part {part!r}; expected one of {PARTS}")
    return path("outputs") / part


def relative(target: Path) -> str:
    """path relative to the repository root, for messages."""
    try:
        return str(Path(target).relative_to(ROOT))
    except ValueError:
        return str(target)


# state views


def view_name(subsystem: str, interval_s: float) -> str:
    return f"{subsystem}_dt{interval_s:g}"


def load_view(name: str) -> ViewData:
    """a state view of data/processed written by the preprocessing stage."""
    file = processed_dir() / f"{name}.npz"
    if not file.exists():
        raise MissingInputError(f"{relative(file)} not found: run \"python main.py run preprocessing\" first")
    return load_view_data(processed_dir(), name)[0]


# contract files between stages


def contract_path(name: str) -> Path:
    return tables_dir("pipeline") / f"{name}.csv"


def read_input(name: str, **read_csv_args: Any) -> pd.DataFrame:
    """a contract file of outputs/pipeline/tables; missing -> MissingInputError naming its stage."""
    file = contract_path(name)
    if not file.exists():
        stage = PRODUCERS.get(name)
        if stage is None:
            raise ConfigError(f"no stage is registered for the input {name!r} (utils/io.py PRODUCERS)")
        raise MissingInputError(f"{relative(file)} not found: run \"python main.py run {stage}\" first")
    return pd.read_csv(file, **read_csv_args)


def write_output(table: pd.DataFrame, name: str) -> Path:
    """write a contract file to outputs/pipeline/tables/<name>.csv."""
    return write_csv(table, contract_path(name))


def save_result(table: pd.DataFrame, name: str) -> Path:
    """write an unnumbered result table to outputs/evaluation/tables/<name>.csv."""
    return write_csv(table, tables_dir("evaluation") / f"{name}.csv")


def write_csv(table: pd.DataFrame, file: Path) -> Path:
    """csv without index; list, set and dict cells become json text."""
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


def save_table(table: pd.DataFrame, number: str, title: str, part: str) -> Path:
    """write a numbered table as csv under its number and title and log the saved line."""
    file = write_csv(table, tables_dir(part) / f"{thesis_name('table', number, title)}.csv")
    log.info(f"saved table {number} -> {relative(file)}")
    return file


def save_figure(fig: plt.Figure, number: str, title: str, part: str) -> Path:
    """write a numbered figure as pdf under its number and title, close it and log the saved line."""
    file = figures_dir(part) / f"{thesis_name('figure', number, title)}.pdf"
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
