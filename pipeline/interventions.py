from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import data, graphs, scm
from core import interventions as itv
from core import preprocessing as pp
from core.preprocessing import ViewData
from utils.config import setting
from utils.errors import MissingInputError
from utils.io import processed_dir, raw_dir, read_input, relative, write_output
from utils.logger import log


@dataclass
class Setting:
    subsystem: str
    interval_s: float
    view: ViewData
    splits: pd.DataFrame
    manipulated: dict[str, set[str]]
    alarms: list[str]
    folds: list[str]
    # fitted scms, data/cache/scm when not set
    scm_dir: Path | None = None

    def runs_of(self, fold: str) -> list[str]:
        return self.splits.loc[self.splits["fold"] == fold, "recording_id"].tolist()

    def fold_model(self, graph: str, fold: str) -> scm.TemporalSCM:
        return scm.load_fitted(self.subsystem, graph, data.scope_name(fold), self.scm_dir)


def load_setting(cfg: dict, subsystem: str) -> Setting:
    """model view, splits, ground truth and alarms of one subsystem."""
    interval = setting(cfg, "model_interval_s")
    name = f"{subsystem}_dt{interval:g}"
    if not (processed_dir() / f"{name}.npz").exists():
        raise MissingInputError(f"{relative(processed_dir() / name)}.npz not found: "
                                "run \"python main.py run preprocessing\" first")
    view, _ = pp.load_view_data(processed_dir(), name)
    splits = read_input("splits")
    # keep the row order: bootstrap draws depend on it
    splits = splits[splits["subsystem"] == subsystem].reset_index(drop=True)
    expert = graphs.simple_digraph(data.load_expert_graph(subsystem=subsystem))
    alarms = sorted(n for n in expert.nodes if expert.nodes[n]["kind"] == "Alarm")
    manipulated = data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments())
    folds = sorted(splits.loc[splits["split"] == "fault", "fold"].unique())
    return Setting(subsystem, interval, view, splits, manipulated, alarms, folds)


def fault_onsets(subsystem: str, interval_s: float) -> dict[tuple[str, str], int]:
    """grid step of the first recorded change of every manipulated variable after the fault start."""
    infos = [r for r in pp.discover_recordings(raw_dir()) if r.subsystem == subsystem]
    events = pp.mark_changes(pp.parse_values(pp.load_all_events(infos, raw_dir())))
    events = events[events["source"] == "dig_twin"]
    runs = data.load_fault_runs()
    changes = data.fault_variable_changes(events, runs[runs["subsystem"] == subsystem], data.load_experiments())
    return itv.onset_steps(changes, interval_s)


def replay_runs(cfg: dict, s: Setting) -> pd.DataFrame:
    """sustained replay of the recorded fault injections of every held-out run with its fold scm."""
    worlds = setting(cfg, "interventions", "replay_worlds")
    seed = setting(cfg, "interventions", "seed")
    onsets = fault_onsets(s.subsystem, s.interval_s)
    frames = []
    for graph in setting(cfg, "graphs"):
        for fold in s.folds:
            model = s.fold_model(graph, fold)
            for rid in s.runs_of(fold):
                series = scm.recording_series(s.view, rid)
                starts = {v: onsets[(rid, v)] for v in sorted(s.manipulated[rid]) if (rid, v) in onsets}
                response = itv.replay_manipulation(model, series, starts, s.alarms, worlds, np.random.default_rng(seed))
                frames.append(response.assign(graph=graph, fold=fold, recording_id=rid,
                                              model_id=f"{s.subsystem}-{graph}-{data.scope_name(fold)}",
                                              worlds=worlds))
    replay = pd.concat(frames, ignore_index=True)
    replay["observed_delay_s"] = replay["observed_delay_steps"] * s.interval_s
    replay["predicted_delay_s"] = replay["predicted_median_delay_steps"] * s.interval_s
    replay["abs_onset_error_s"] = (replay["predicted_delay_s"] - replay["observed_delay_s"]).abs()
    replay["never_activated_in_simulation"] = replay["p_fires"] == 0
    return replay


def run(cfg: dict, force: bool = False) -> None:
    s = load_setting(cfg, setting(cfg, "interventions", "subsystem"))
    replay = replay_runs(cfg, s)
    file = write_output(replay, "replay")
    n_runs = replay["recording_id"].nunique()
    log.info(f"intervention analysis done. {len(replay)} alarm responses of {n_runs} {s.subsystem} runs "
             f"({replay['graph'].nunique()} graphs) saved to {relative(file)}")
