"""stage 16: lstm anomaly detection + shap variable ranking, a non-causal diagnosis baseline.

matched to the causal pipeline (stage 15):
  * same 0.5 s state series; unrecorded or unknown states filled with the training mode of the fold
    (as the scm fills them) and left out of the loss;
  * same data per held-out scenario k: normal training recordings + all other fault scenarios;
    in a fault run, the manipulated variables are left out of the training loss (the scm leaves
    them out of fitting their own mechanism);
  * lstm size chosen once per subsystem on the normal validation recordings, from a model trained
    on the normal training recordings only (no fault outcome enters the choice);
  * same held-out runs, candidates (recorded non-alarm variables), alarm-anchored window and metrics.
model: one lstm predicts the next state of every variable from the last 20 s (40 steps). anomaly
score of a step: mean binary cross-entropy of the predicted next states over the recorded variables
of the run. shap (expected gradients) explains the anomaly score of the 20 most anomalous steps in
the analysis window; a candidate's score is its summed absolute attribution over these steps and
all time positions. 3 seeds; metrics are averaged over seeds.
outputs: results/tables/16_lstm_shap_{selection,runs,per_scenario,summary}.csv, 16_lstm_shap_config.json
usage: PYTHONPATH=src .venv/bin/python scripts/lstm_shap_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import graph_evaluation as ge
from causal_rca import preprocessing as pp
from causal_rca import root_cause_analysis as rca
from causal_rca import scm as sc

T = data.RESULTS_DIR / "tables"
INTERVAL = 0.5
LOOKBACK = 40          # 20 s, the longest discovery lag
HIDDEN = (32, 64)      # chosen on normal validation
SEEDS = (0, 1, 2)
MAX_EPOCHS = 12
BATCH = 512
TOP_STEPS = 20         # most anomalous steps explained per run
BACKGROUND = 100       # shap background windows from normal training data
SHAP_SAMPLES = 50
SPLITS = {"coolant": "02_coolant_splits.csv", "probe": "10_probe_splits.csv"}
torch.set_num_threads(8)


class Forecaster(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(n_vars, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class AnomalyScore(nn.Module):
    """input: lookback rows + the observed next row; output: mean bce over the given variables."""

    def __init__(self, model: Forecaster, weights: torch.Tensor):
        super().__init__()
        self.model, self.weights = model, weights

    def forward(self, seq):
        logits = self.model(seq[:, :-1])
        bce = nn.functional.binary_cross_entropy_with_logits(logits, seq[:, -1], reduction="none")
        return ((bce * self.weights).sum(1) / self.weights.sum()).unsqueeze(1)


def run_arrays(view, rid, fills, manipulated=frozenset()):
    """filled states (t, v) and loss weights (observed, not manipulated)."""
    a = view.arrays[rid].astype(np.float32).copy()
    m = view.masks[rid]
    w = (~m).astype(np.float32)
    for j, v in enumerate(view.var_names):
        a[m[:, j], j] = fills[v]
        if v in manipulated:
            w[:, j] = 0.0
    return a, w


class Windows:
    """all (lookback, next row) pairs of a set of recordings; the start is padded with the first row."""

    def __init__(self, arrays):
        blocks, wblocks, idx, offset = [], [], [], 0
        for a, w in arrays:
            pad = np.repeat(a[:1], LOOKBACK, axis=0)
            blocks.append(np.vstack([pad, a]))
            wblocks.append(np.vstack([np.zeros((LOOKBACK, a.shape[1]), np.float32), w]))
            idx.append(offset + np.arange(1, len(a)))
            offset += len(a) + LOOKBACK
        self.x = torch.from_numpy(np.vstack(blocks))
        self.w = torch.from_numpy(np.vstack(wblocks))
        self.idx = np.concatenate(idx)
        self.steps = torch.arange(LOOKBACK)

    def batch(self, ids):
        ids = torch.from_numpy(ids)
        return self.x[ids[:, None] + self.steps], self.x[ids + LOOKBACK], self.w[ids + LOOKBACK]


def loss_of(model, win, batches):
    total, n = 0.0, 0.0
    with torch.no_grad():
        for ids in batches:
            x, y, w = win.batch(ids)
            bce = nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none")
            total += float((bce * w).sum())
            n += float(w.sum())
    return total / n


def train(train_win, val_win, n_vars, hidden, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Forecaster(n_vars, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    val_batches = np.array_split(val_win.idx, max(1, len(val_win.idx) // 4096))
    best, best_state, history = np.inf, None, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(train_win.idx)
        for ids in np.array_split(order, max(1, len(order) // BATCH)):
            x, y, w = train_win.batch(ids)
            bce = nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none")
            loss = (bce * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val = loss_of(model, val_win, val_batches)
        history.append(val)
        if val < best - 1e-6:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            break  # early stop on the normal validation loss
    model.load_state_dict(best_state)
    model.eval()
    return model, best, history


def explain_run(model, view, rid, fills, alarms, nodes, background):
    series = sc.recording_series(view, rid)
    window = rca.analysis_window(series, alarms, INTERVAL)
    cands = rca.candidates(series, nodes, alarms)
    a, _ = run_arrays(view, rid, fills)
    recorded = torch.tensor([0.0 if np.isnan(series[v]).all() else 1.0 for v in view.var_names])
    pad = np.vstack([np.repeat(a[:1], LOOKBACK, axis=0), a])
    steps = np.arange(window[0], window[1] + 1)
    seqs = torch.from_numpy(np.stack([pad[t: t + LOOKBACK + 1] for t in steps]))
    score = AnomalyScore(model, recorded)
    with torch.no_grad():
        anomaly = score(seqs).squeeze(1).numpy()
    top = np.argsort(-anomaly, kind="stable")[:TOP_STEPS]
    explainer = shap.GradientExplainer(score, background)
    phi = explainer.shap_values(seqs[top], nsamples=SHAP_SAMPLES, rseed=0)
    phi = np.asarray(phi).reshape(len(top), LOOKBACK + 1, len(view.var_names))
    importance = np.abs(phi).sum(axis=(0, 1))
    imp = {v: float(importance[view.var_names.index(v)]) for v in cands}
    ranking = sorted(cands, key=lambda v: (-imp[v], v))
    return ranking, cands, imp, float(anomaly.max())


def main():
    t0 = time.perf_counter()
    manipulated = data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments())
    sel_rows, run_rows = [], []
    for sub, split_file in SPLITS.items():
        view, _ = pp.load_view_data(data.PROCESSED_DIR, f"{sub}_dt{INTERVAL:g}")
        splits = pd.read_csv(T / split_file)
        expert = ge.simple_digraph(data.load_expert_graph(subsystem=sub))
        nodes = sorted(expert.nodes)
        alarms = sorted(n for n in nodes if expert.nodes[n]["kind"] == "Alarm")
        n_vars = len(view.var_names)
        normal_train = splits.loc[splits.split == "train", "recording_id"].tolist()
        normal_val = splits.loc[splits.split == "validation", "recording_id"].tolist()
        folds = sorted(splits.loc[splits.split == "fault", "fold"].unique())

        # size selection on normal validation, trained on normal training recordings only
        fills_n = sc.fill_values(view, normal_train)
        win_tr = Windows([run_arrays(view, r, fills_n) for r in normal_train])
        win_va = Windows([run_arrays(view, r, fills_n) for r in normal_val])
        scores = {}
        for h in HIDDEN:
            _, val, hist = train(win_tr, win_va, n_vars, h, seed=0)
            scores[h] = val
            sel_rows.append({"subsystem": sub, "hidden": h, "normal_validation_loss": val, "epochs": len(hist)})
            print(f"{sub}: hidden {h} validation loss {val:.5f} ({len(hist)} epochs)", flush=True)
        hidden = min(HIDDEN, key=lambda h: (round(scores[h], 9), h))
        print(f"{sub}: selected hidden {hidden}", flush=True)

        for k in folds:
            train_ids = data.fold_training_ids(splits, k)
            fills = sc.fill_values(view, train_ids)
            win_k = Windows([run_arrays(view, r, fills, manipulated.get(r, set())) for r in train_ids])
            win_v = Windows([run_arrays(view, r, fills) for r in normal_val])
            rng = np.random.default_rng(0)
            bg_ids = rng.choice(Windows([run_arrays(view, r, fills) for r in normal_train]).idx, BACKGROUND, replace=False)
            bg_win = Windows([run_arrays(view, r, fills) for r in normal_train])
            bx, by, _ = bg_win.batch(bg_ids)
            background = torch.cat([bx, by[:, None]], dim=1)
            test_ids = splits.loc[splits.fold == k, "recording_id"].tolist()
            for seed in SEEDS:
                model, val, hist = train(win_k, win_v, n_vars, hidden, seed)
                for rid in test_ids:
                    ranking, cands, imp, peak = explain_run(model, view, rid, fills, alarms, nodes, background)
                    truth = set(manipulated[rid])
                    run_rows.append({"subsystem": sub, "fold": k, "recording_id": rid, "seed": seed, "hidden": hidden,
                                     "n_candidates": len(cands), "n_causes": len(truth), "ranking": " > ".join(ranking),
                                     "peak_anomaly": peak, **rca.ranking_metrics(ranking, truth),
                                     **{f"chance_{m}": v for m, v in rca.chance_metrics(cands, truth, seed=0).items()}})
                print(f"{sub} {k} seed {seed}: epochs {len(hist)}, "
                      f"mean ap@3 {np.mean([r['ap@3'] for r in run_rows if r['fold'] == k and r['seed'] == seed and r['subsystem'] == sub]):.2f}",
                      flush=True)

    runs = pd.DataFrame(run_rows)
    ev.save_table(pd.DataFrame(sel_rows), "16_lstm_shap_selection")
    ev.save_table(runs, "16_lstm_shap_runs")
    # average over seeds per run, then runs -> scenarios, as for the causal rca
    metrics = ["top1", "hit@3", "recall@3", "ap@3", "rr"]
    per_run = runs.groupby(["subsystem", "fold", "recording_id"])[metrics + ["chance_top1", "chance_ap@3"]].mean().reset_index()
    per_scen = per_run.groupby(["subsystem", "fold"])[metrics + ["chance_ap@3"]].mean().reset_index()
    ev.save_table(per_scen, "16_lstm_shap_per_scenario")
    seed_level = runs.groupby(["subsystem", "seed"])[["top1", "ap@3"]].mean().groupby("subsystem").agg(["min", "max"])
    summ = []
    for sub, d in per_run.groupby("subsystem"):
        ps = per_scen[per_scen.subsystem == sub]
        summ.append({"subsystem": sub, "runs": len(d), "top1_run_mean": d.top1.mean(), "hit3_run_mean": d["hit@3"].mean(),
                     "recall3_run_mean": d["recall@3"].mean(), "map3_run_mean": d["ap@3"].mean(), "mrr_run_mean": d.rr.mean(),
                     "ap3_scenario_macro": ps["ap@3"].mean(), "top1_scenario_macro": ps.top1.mean(),
                     "chance_map3_run_mean": d["chance_ap@3"].mean(),
                     "map3_seed_min": seed_level.loc[sub, ("ap@3", "min")], "map3_seed_max": seed_level.loc[sub, ("ap@3", "max")],
                     "top1_seed_min": seed_level.loc[sub, ("top1", "min")], "top1_seed_max": seed_level.loc[sub, ("top1", "max")]})
    summary = pd.DataFrame(summ)
    ev.save_table(summary, "16_lstm_shap_summary")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=data.PROJECT_ROOT).stdout.strip()
    config = {"lookback_steps": LOOKBACK, "hidden_candidates": HIDDEN, "seeds": SEEDS, "max_epochs": MAX_EPOCHS, "batch": BATCH,
              "optimizer": "adam 1e-3", "early_stopping": "normal validation loss, patience 1",
              "explained_steps": TOP_STEPS, "background": BACKGROUND, "shap": f"GradientExplainer, nsamples {SHAP_SAMPLES}",
              "torch": torch.__version__, "shap_version": shap.__version__, "code_commit": commit,
              "runtime_s": round(time.perf_counter() - t0)}
    (T / "16_lstm_shap_config.json").write_text(json.dumps(config, indent=2))
    print(summary.round(3).to_string())
    print(per_scen.round(2).to_string())
    print(f"finished in {config['runtime_s']} s")


if __name__ == "__main__":
    main()
