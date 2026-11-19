"""command line of the pipeline and the evaluation.

  python main.py run [stage]          all six stages in order, or one stage
  python main.py evaluate <part>      one evaluation part
  --force                             rerun cached heavy steps (pcmci+ runs, lstm training)
"""

import argparse
import importlib
import sys

import matplotlib

# figures are only written to files
matplotlib.use("Agg")

from utils.config import load_config  # noqa: E402
from utils.errors import PipelineError  # noqa: E402
from utils.io import log_file  # noqa: E402
from utils.logger import log, setup_logging  # noqa: E402

# command -> name -> module with run(cfg, force=False); stages are listed in pipeline order
COMMANDS = {
    "run": {
        "preprocessing": "pipeline.preprocessing",
        "discovery": "pipeline.causal_discovery",
        "scm": "pipeline.scm_fitting",
        "rca": "pipeline.root_cause_analysis",
        "interventions": "pipeline.interventions",
        "counterfactuals": "pipeline.counterfactuals",
    },
    "evaluate": {
        "discovery": "evaluation.discovery",
        "scm": "evaluation.scm",
        "rca": "evaluation.rca",
        "counterfactuals": "evaluation.counterfactuals",
        "comparisons": "evaluation.comparisons",
    },
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="main.py", description="causal root cause analysis on causRCA")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run all pipeline stages, or one")
    run.add_argument("name", nargs="?", choices=list(COMMANDS["run"]), metavar="stage",
                     help=", ".join(COMMANDS["run"]))
    evaluate = commands.add_parser("evaluate", help="run one evaluation part")
    evaluate.add_argument("name", choices=list(COMMANDS["evaluate"]), metavar="part",
                          help=", ".join(COMMANDS["evaluate"]))
    for sub in (run, evaluate):
        sub.add_argument("--force", action="store_true", help="rerun cached heavy steps")
    return parser.parse_args(argv)


def dispatch(command: str, names: list[str], force: bool) -> None:
    cfg = load_config()
    for name in names:
        module = importlib.import_module(COMMANDS[command][name])
        log.debug(f"{command} {name} (force={force})")
        try:
            module.run(cfg, force=force)
        except NotImplementedError as exc:
            raise PipelineError(f"{command} {name}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        setup_logging(log_file())
    except PipelineError as exc:
        print(exc, file=sys.stderr)
        return 1
    names = [args.name] if args.name else list(COMMANDS[args.command])
    try:
        dispatch(args.command, names, args.force)
    except PipelineError as exc:
        log.debug("stopped", exc_info=True)
        log.error(str(exc))
        return 1
    except Exception:
        log.debug("unexpected error", exc_info=True)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
