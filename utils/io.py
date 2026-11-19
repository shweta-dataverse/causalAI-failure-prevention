import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from core.plots import write_pdf
from utils.config import ROOT, load_config, setting
from utils.errors import ConfigError, MissingInputError
from utils.logger import log

PARTS = ("pipeline", "evaluation")

# contract file in outputs/pipeline/tables -> stage of `python main.py run <stage>` that writes it
PRODUCERS = {
    "splits": "preprocessing",
    "discovered_graphs": "discovery",
    "discovery_settings": "discovery",
    "scm_graphs": "scm",
    "dag_removals": "scm",
    "scm_settings": "scm",
    "rca_rankings": "rca",
    "rca_runs": "rca",
    "replay": "interventions",
    "recommendations": "counterfactuals",
}


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    return ROOT


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


# ---------------------------------------------------------------------------
# contract files between stages
# ---------------------------------------------------------------------------


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
    file = contract_path(name)
    _write_csv(table, file)
    log.debug(f"wrote {relative(file)} ({len(table)} rows)")
    return file


# ---------------------------------------------------------------------------
# numbered tables and figures
# ---------------------------------------------------------------------------


def thesis_name(kind: str, number: str, title: str) -> str:
    """file stem of a numbered table or figure, e.g. ('table', '6.4', 'Root-cause ranking performance')
    -> 'table_6_4_root_cause_ranking_performance'."""
    if kind not in ("table", "figure"):
        raise ConfigError(f"kind must be 'table' or 'figure', got {kind!r}")
    parts = [kind, _snake(str(number)), _snake(title)]
    return "_".join(p for p in parts if p)


def save_table(table: pd.DataFrame, number: str, title: str, part: str) -> Path:
    """write a numbered table as csv under its number and title and log the saved line."""
    file = tables_dir(part) / f"{thesis_name('table', number, title)}.csv"
    _write_csv(table, file)
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


def _write_csv(table: pd.DataFrame, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    # list, set and dict cells become json text
    for column in out.columns:
        if out[column].map(lambda v: isinstance(v, (list, tuple, set, dict))).any():
            out[column] = out[column].map(_json_cell)
    out.to_csv(file, index=False)


def _json_cell(value: Any) -> Any:
    if isinstance(value, set):
        value = sorted(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    return value
