import numpy as np
import pandas as pd

from core import data, scm
from core import interventions as itv
from core import preprocessing as pp
from core.data import Subsystem, load_subsystem
from utils.config import setting
from utils.io import raw_dir, relative, write_output
from utils.logger import log


def fault_onsets(subsystem: str, interval_s: float) -> dict[tuple[str, str], int]:
    """grid step of the first recorded change of every manipulated variable after the fault start."""
    infos = [r for r in pp.discover_recordings(raw_dir()) if r.subsystem == subsystem]
    events = pp.mark_changes(pp.parse_values(pp.load_all_events(infos, raw_dir())))
    events = events[events["source"] == "dig_twin"]
    runs = data.load_fault_runs()
    changes = data.fault_variable_changes(events, runs[runs["subsystem"] == subsystem], data.load_experiments())
    return itv.onset_steps(changes, interval_s)


def replay_runs(cfg: dict, s: Subsystem) -> pd.DataFrame:
    """sustained replay of the recorded fault injections of every held-out run with its fold scm."""
    worlds = setting(cfg, "interventions", "replay_worlds")
    seed = setting(cfg, "interventions", "seed")
    onsets = fault_onsets(s.name, s.interval_s)
    frames = []
    for graph in setting(cfg, "graphs"):
        for fold in s.folds:
            model = s.fold_model(graph, fold)
            for rid in s.runs_of(fold):
                series = scm.recording_series(s.view, rid)
                starts = {v: onsets[(rid, v)] for v in sorted(s.manipulated[rid]) if (rid, v) in onsets}
                response = itv.replay_manipulation(model, series, starts, s.alarms, worlds, np.random.default_rng(seed))
                frames.append(response.assign(graph=graph, fold=fold, recording_id=rid,
                                              model_id=f"{s.name}-{graph}-{data.scope_name(fold)}",
                                              worlds=worlds))
    replay = pd.concat(frames, ignore_index=True)
    replay["observed_delay_s"] = replay["observed_delay_steps"] * s.interval_s
    replay["predicted_delay_s"] = replay["predicted_median_delay_steps"] * s.interval_s
    replay["abs_onset_error_s"] = (replay["predicted_delay_s"] - replay["observed_delay_s"]).abs()
    replay["never_activated_in_simulation"] = replay["p_fires"] == 0
    return replay


def run(cfg: dict, force: bool = False) -> None:
    s = load_subsystem(cfg, setting(cfg, "interventions", "subsystem"))
    replay = replay_runs(cfg, s)
    file = write_output(replay, "replay")
    n_runs = replay["recording_id"].nunique()
    log.info(f"intervention analysis done. {len(replay)} alarm responses of {n_runs} {s.name} runs "
             f"({replay['graph'].nunique()} graphs) saved to {relative(file)}")
