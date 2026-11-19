import logging
from pathlib import Path

log = logging.getLogger("causal_rca")


def setup_logging(log_file: Path) -> logging.Logger:
    """debug lines with timestamps to the log file, plain info lines to the terminal."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()

    to_file = logging.FileHandler(log_file, encoding="utf-8")
    to_file.setLevel(logging.DEBUG)
    to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(to_file)

    to_terminal = logging.StreamHandler()
    to_terminal.setLevel(logging.INFO)
    to_terminal.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(to_terminal)
    return log
