"""stage 15f: which scm setting makes the exp_22 counterfactual explanations invalid?

the action set is held fixed at do(HP_Pump_Ok = 1) from its rca detection step (the set that was
valid in stage 14), and only the pseudo-count a and the extra lag of the exp_22 fold model are
varied: (0.5, none) = original, (0.01, none), (0.5, 2), (0.01, 2) = selected. validity uses the
same 2,000 independent simulations and random streams as stages 14 and 15. nothing is selected
here; the table only attributes the drop from 25/25 to 19/25 valid runs.
also reports the fitted probability that the pump alarm stays active with a healthy pump, and the
share of failed simulations whose first alarm activation lies after the recorded pump recovery.
output: results/tables/15_coolant_exp22_settings.csv
usage: PYTHONPATH=src .venv/bin/python scripts/cf_exp22_settings.py
"""

import numpy as np
import pandas as pd

from causal_rca import causal_discovery as cd
from causal_rca import counterfactuals as cfx
from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import graph_evaluation as ge
from causal_rca import interventions as itv
from causal_rca import preprocessing as pp
from causal_rca import root_cause_analysis as rca
from causal_rca import scm as sc

T = data.RESULTS_DIR / "tables"
INTERVAL = 0.5
CF_ENTROPY = 20261109
SETTINGS = [(0.5, None, "original"), (0.01, None, "pseudo-count only"), (0.5, 2, "extra lag only"),
            (0.01, 2, "selected")]
ALARM, PUMP = "HP_A_700304", "HP_Pump_Ok"


def main():
    view, _ = pp.load_view_data(data.PROCESSED_DIR, "coolant_dt0.5")
    splits = pd.read_csv(T / "02_coolant_splits.csv")
    manipulated = data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments())
    expert = ge.simple_digraph(data.load_expert_graph(subsystem="coolant"))
    nodes = sorted(expert.nodes)
    alarms = sorted(n for n in nodes if expert.nodes[n]["kind"] == "Alarm")
    edges = pd.read_csv(T / "15_coolant_discovery_edges.csv")
    graphs = {"expert": expert,
              "discovered": ge.to_dag(cd.edges_to_graph(edges[edges.graph_id == "discovered_without_exp_22"], nodes))[0]}
    # the same independent streams as stages 14 and 15 (spawned per graph and run, in this order)
    run_ids = sorted(splits.loc[splits.split == "fault", "recording_id"])
    streams = dict(zip([(g, r) for g in ("expert", "discovered") for r in run_ids],
                       np.random.SeedSequence(CF_ENTROPY).spawn(2 * len(run_ids))))
    rankings = pd.read_csv(T / "15_rca_rankings.csv")
    train = data.fold_training_ids(splits, "exp_22")
    rows = []
    for g, graph in graphs.items():
        for a, lag, label in SETTINGS:
            lags = None if lag is None else {(u, v): [lag] for u, v in graph.edges if 3 * graph.in_degree(v) <= 20}
            model = sc.fit_scm(view, graph, train, manipulated, INTERVAL, smoothing=a, lags=lags)
            mech = model.mechanisms[ALARM]
            # p(alarm stays active | pump healthy now and at every lag of the mechanism)
            healthy = (1 << mech.width) - 1
            p_stay = float(mech.tables[1][healthy])
            for rid in splits.loc[splits.fold == "exp_22", "recording_id"]:
                series = sc.recording_series(view, rid)
                rk = rankings[(rankings.subsystem == "coolant") & (rankings.graph == g) & (rankings.recording_id == rid)]
                det = dict(zip(rk.variable, rk.detection_step))
                acts = [itv.Action(PUMP, 1, max(1, int(det[PUMP])))]
                window = rca.analysis_window(series, alarms, INTERVAL)
                recorded = {v for v in series if not np.isnan(series[v]).all()}
                outcome = [x for x in alarms if x in recorded and model.mechanisms[x].has_data]
                res = cfx.reevaluate_validity(model, series, acts, outcome, window[0], recorded, streams[(g, rid)])
                # where do the failed simulations activate the alarm? (one fresh batch, same stream family)
                cf = cfx.counterfactual(model, series, acts, outcome, window[0], 2000,
                                        np.random.default_rng(streams[(g, rid)].spawn(1)[0]), recorded)
                recovery = next(i for i in range(acts[0].start_step + 1, len(series[PUMP])) if series[PUMP][i] == 1)
                noise = model.abduct_noise(series, 2000, np.random.default_rng(7))
                world = model.simulate(cfx.initial_state(model, series), noise,
                                       interventions={PUMP: (1, acts[0].start_step)},
                                       fixed=cfx.unrecorded_context(model, series))
                hp = world[ALARM][:, window[0]:]
                fired = hp.max(axis=1) > 0
                first = np.argmax(hp > 0, axis=1)[fired] + window[0]
                rows.append({"graph": g, "setting": label, "pseudo_count": a, "extra_lag": lag or 0,
                             "recording_id": rid, "p_alarm_stays_with_healthy_pump": p_stay,
                             "validity_2000": res["validity_independent"], "valid": res["validity_independent"] >= 0.9,
                             "pump_recovery_s": recovery * INTERVAL,
                             "share_failures_after_recovery": float((first >= recovery).mean()) if fired.any() else np.nan,
                             "validity_check_batch": cf.validity_probability})
            d = pd.DataFrame([r for r in rows if r["graph"] == g and r["setting"] == label])
            print(f"{g:10s} {label:18s} a={a} lag={lag}: p_stay {p_stay:.4f}; validity {d.validity_2000.min():.3f}-"
                  f"{d.validity_2000.max():.3f}; valid {int(d.valid.sum())}/6; failures after recovery "
                  f"{d.share_failures_after_recovery.mean():.2f}", flush=True)
    ev.save_table(pd.DataFrame(rows), "15_coolant_exp22_settings")


if __name__ == "__main__":
    main()
