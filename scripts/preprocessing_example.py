"""thesis tables 5.4 and 5.5: one raw event-log excerpt and its reconstructed state rows.

raw: the first five events of coolant variables in run 1 of scenario exp_22 (original strings).
states: the first five 0.5 s steps of the stored model input (stage-15 arrays), restricted to the
variables usable in the exp_22 discovery regime, in the array's column order; masked cells as NA.
also writes the full 15-variable rows and a generation record (source file, recording, columns,
commit), so the thesis tables can be checked against the current pipeline output.
usage: PYTHONPATH=src .venv/bin/python scripts/preprocessing_example.py
"""

import json
import subprocess

import pandas as pd

from causal_rca import causal_discovery as cd
from causal_rca import data
from causal_rca import graph_evaluation as ge
from causal_rca import preprocessing as pp

T = data.RESULTS_DIR / "tables"
SOURCE = "dig_twin/exp_coolant/exp_22/run_1/faultDataset_coolant_exp22_run_1.csv"
RECORDING = "coolant_exp_22_run_1"
STEPS = 5


def main():
    raw = pd.read_csv(data.RAW_DIR / SOURCE, dtype=str)
    nodes = sorted(ge.simple_digraph(data.load_expert_graph(subsystem="coolant")).nodes)
    coolant = raw[raw.node.isin(nodes)]
    coolant.head(STEPS).to_csv(T / "15_preprocessing_example_raw.csv", index=False)

    view, _ = pp.load_view_data(data.PROCESSED_DIR, "coolant_dt0.5")
    splits = pd.read_csv(T / "02_coolant_splits.csv")
    usable = cd.regime_variables(view, cd.define_regimes(splits)["exp_22"])
    a, m = view.arrays[RECORDING], view.masks[RECORDING]
    full = pd.DataFrame(a[:STEPS], columns=view.var_names).astype("Int64").mask(pd.DataFrame(m[:STEPS], columns=view.var_names))
    full.insert(0, "t_s", [k * 0.5 for k in range(STEPS)])
    full.insert(0, "step", range(STEPS))
    full.to_csv(T / "15_preprocessing_example_states_full.csv", index=False, na_rep="NA")
    full[["step", "t_s", *usable]].to_csv(T / "15_preprocessing_example_states.csv", index=False, na_rep="NA")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=data.PROJECT_ROOT).stdout.strip()
    record = {"source_file": SOURCE, "recording_id": RECORDING, "raw_rows": len(raw), "coolant_rows": len(coolant),
              "grid_s": 0.5, "array_columns": list(view.var_names), "regime": "exp_22", "usable_columns": usable,
              "first_change_s": float(coolant.time_s.astype(float).loc[lambda x: x > 0].min()),
              "code_commit": commit}
    (T / "15_preprocessing_example.json").write_text(json.dumps(record, indent=2))
    print(coolant.head(STEPS).to_string(index=False))
    print(full[["step", "t_s", *usable]].to_string(index=False))
    print(record)


if __name__ == "__main__":
    main()
