"""stage 16: lstm anomaly detection + shap variable ranking, a non-causal diagnosis baseline.

matched to the causal pipeline (stage 15):
  * same 0.5 s state series; unrecorded or unknown states filled with the training mode of the fold
    (as the scm fills them) and left out of the loss;
  * same data per held-out scenario k: normal training recordings + all other fault scenarios;
    in a fault run, the manipulated variables are left out of the training loss (the scm leaves
    them out of fitting their own mechanism);
  * same held-out runs, candidates (recorded non-alarm variables), alarm-anchored window and metrics.
tuning (same objective as the scm settings: held-out predictive loss, never a fault outcome):
  hidden units {32, 64, 128} x look-back {20, 40} steps, trained on the normal training recordings
  only, chosen by the loss on the normal validation recordings; up to 30 epochs with early stopping
  (patience 2) on that loss. the chosen setting is then retrained in every fold with 3 seeds.
model: one lstm predicts the next state of every variable from the look-back window. anomaly score
of a step: mean binary cross-entropy of the predicted next states over the recorded variables of
the run. shap (expected gradients) explains the anomaly score of the steps of the analysis window.
explanation settings cannot be tuned without fault labels, so they are fixed in advance (main: the
20 most anomalous steps, summed absolute attribution) and reported with a sensitivity check
(10 / 20 / all steps x absolute / positive attribution).
outputs: results/tables/16_lstm_shap_*.csv, 16_lstm_shap_config.json, results/figures/16_*.pdf
usage: PYTHONPATH=src .venv/bin/python scripts/lstm_shap_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from causal_rca import data
from causal_rca import evaluation as ev

T = data.RESULTS_DIR / "tables"
INTERVAL = 0.5
GRID = [(h, lb) for h in (32, 64, 128) for lb in (20, 40)]
SEEDS = (0, 1, 2)
MAX_EPOCHS = 30
PATIENCE = 2
LR = 1e-3
BATCH = 512
BACKGROUND = 100
SHAP_SAMPLES = 50
STEP_VARIANTS = {"top10": 10, "top20": 20, "all": None}
MAIN = ("top20", "abs")
WORKERS = 4
THREADS = 3
SPLITS = {"coolant": "02_coolant_splits.csv", "probe": "10_probe_splits.csv"}

_CACHE: dict = {}


def setting(sub):
    """data of one subsystem, loaded once per worker process."""
    if sub not in _CACHE:
        from causal_rca import graph_evaluation as ge
        from causal_rca import preprocessing as pp
        view, _ = pp.load_view_data(data.PROCESSED_DIR, f"{sub}_dt{INTERVAL:g}")
        splits = pd.read_csv(T / SPLITS[sub])
        expert = ge.simple_digraph(data.load_expert_graph(subsystem=sub))
        nodes = sorted(expert.nodes)
        _CACHE[sub] = dict(view=view, splits=splits, nodes=nodes,
                           alarms=sorted(n for n in nodes if expert.nodes[n]["kind"] == "Alarm"),
                           manipulated=data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments()))
    return _CACHE[sub]


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


def make_windows(arrays, lookback):
    """all (look-back window, next row) pairs; the start of a recording is padded with its first row."""
    import torch
    blocks, wblocks, idx, offset = [], [], [], 0
    for a, w in arrays:
        blocks.append(np.vstack([np.repeat(a[:1], lookback, axis=0), a]))
        wblocks.append(np.vstack([np.zeros((lookback, a.shape[1]), np.float32), w]))
        idx.append(offset + np.arange(1, len(a)))
        offset += len(a) + lookback
    return dict(x=torch.from_numpy(np.vstack(blocks)), w=torch.from_numpy(np.vstack(wblocks)),
                idx=np.concatenate(idx), steps=torch.arange(lookback), lookback=lookback)


def batch(win, ids):
    import torch
    ids = torch.from_numpy(ids)
    lb = win["lookback"]
    return win["x"][ids[:, None] + win["steps"]], win["x"][ids + lb], win["w"][ids + lb]


def build_model(n_vars, hidden):
    from torch import nn

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_vars, hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_vars)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    return Forecaster()


def val_loss(model, win):
    import torch
    from torch import nn
    total, n = 0.0, 0.0
    with torch.no_grad():
        for ids in np.array_split(win["idx"], max(1, len(win["idx"]) // 4096)):
            x, y, w = batch(win, ids)
            bce = nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none")
            total += float((bce * w).sum())
            n += float(w.sum())
    return total / n


def train(train_win, val_win, n_vars, hidden, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(n_vars, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, history, waited = np.inf, None, [], 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(train_win["idx"])
        for ids in np.array_split(order, max(1, len(order) // BATCH)):
            x, y, w = batch(train_win, ids)
            bce = nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none")
            loss = (bce * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val = val_loss(model, val_win)
        history.append(val)
        if val < best - 1e-6:
            best, best_state, waited = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, history


def job_select(args):
    """one grid point, trained on normal training recordings only, scored on normal validation."""
    import torch
    from causal_rca import scm as sc
    torch.set_num_threads(THREADS)
    sub, hidden, lookback = args
    s = setting(sub)
    view, splits = s["view"], s["splits"]
    tr = splits.loc[splits.split == "train", "recording_id"].tolist()
    va = splits.loc[splits.split == "validation", "recording_id"].tolist()
    fills = sc.fill_values(view, tr)
    _, val, hist = train(make_windows([run_arrays(view, r, fills) for r in tr], lookback),
                         make_windows([run_arrays(view, r, fills) for r in va], lookback),
                         len(view.var_names), hidden, seed=0)
    return {"subsystem": sub, "hidden": hidden, "lookback_steps": lookback, "lookback_s": lookback * INTERVAL,
            "normal_validation_loss": val, "epochs": len(hist)}


def explain(model, view, rid, fills, alarms, nodes, background, lookback):
    """anomaly scores and shap attributions of every step of the analysis window."""
    import shap
    import torch
    from torch import nn
    from causal_rca import root_cause_analysis as rca
    from causal_rca import scm as sc

    class AnomalyScore(nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.model, self.weights = model, weights

        def forward(self, seq):
            logits = self.model(seq[:, :-1])
            bce = nn.functional.binary_cross_entropy_with_logits(logits, seq[:, -1], reduction="none")
            return ((bce * self.weights).sum(1) / self.weights.sum()).unsqueeze(1)

    series = sc.recording_series(view, rid)
    window = rca.analysis_window(series, alarms, INTERVAL)
    cands = rca.candidates(series, nodes, alarms)
    a, _ = run_arrays(view, rid, fills)
    recorded = torch.tensor([0.0 if np.isnan(series[v]).all() else 1.0 for v in view.var_names])
    pad = np.vstack([np.repeat(a[:1], lookback, axis=0), a])
    steps = np.arange(window[0], window[1] + 1)
    seqs = torch.from_numpy(np.stack([pad[t: t + lookback + 1] for t in steps]))
    score = AnomalyScore(recorded)
    with torch.no_grad():
        anomaly = score(seqs).squeeze(1).numpy()
    phi = shap.GradientExplainer(score, background).shap_values(seqs, nsamples=SHAP_SAMPLES, rseed=0)
    phi = np.asarray(phi).reshape(len(steps), lookback + 1, len(view.var_names)).sum(axis=1)  # step x variable
    order = np.argsort(-anomaly, kind="stable")
    importance = {}
    for name, n in STEP_VARIANTS.items():
        rows = order if n is None else order[:n]
        importance[(name, "abs")] = np.abs(phi[rows]).sum(0)
        importance[(name, "pos")] = np.clip(phi[rows], 0, None).sum(0)
    return cands, importance, anomaly


def job_fold(args):
    """retrain the chosen setting without scenario k and rank the candidates of its runs."""
    import torch
    from causal_rca import root_cause_analysis as rca
    from causal_rca import scm as sc
    torch.set_num_threads(THREADS)
    sub, k, seed, hidden, lookback = args
    s = setting(sub)
    view, splits, man = s["view"], s["splits"], s["manipulated"]
    train_ids = data.fold_training_ids(splits, k)
    fills = sc.fill_values(view, train_ids)
    win_k = make_windows([run_arrays(view, r, fills, man.get(r, set())) for r in train_ids], lookback)
    va = splits.loc[splits.split == "validation", "recording_id"].tolist()
    win_v = make_windows([run_arrays(view, r, fills) for r in va], lookback)
    model, val, hist = train(win_k, win_v, len(view.var_names), hidden, seed)
    normal_train = splits.loc[splits.split == "train", "recording_id"].tolist()
    bg_win = make_windows([run_arrays(view, r, fills) for r in normal_train], lookback)
    bx, by, _ = batch(bg_win, np.random.default_rng(0).choice(bg_win["idx"], BACKGROUND, replace=False))
    background = torch.cat([bx, by[:, None]], dim=1)
    rows, imp_rows = [], []
    for rid in splits.loc[splits.fold == k, "recording_id"]:
        cands, importance, anomaly = explain(model, view, rid, fills, s["alarms"], s["nodes"], background, lookback)
        truth = set(man[rid])
        chance = rca.chance_metrics(cands, truth, seed=0)
        for (steps_v, agg), vec in importance.items():
            imp = {v: float(vec[view.var_names.index(v)]) for v in cands}
            ranking = sorted(cands, key=lambda v: (-imp[v], v))
            rows.append({"subsystem": sub, "fold": k, "recording_id": rid, "seed": seed, "steps": steps_v,
                         "aggregation": agg, "n_candidates": len(cands), "n_causes": len(truth),
                         "epochs": len(hist), "fold_validation_loss": val, "ranking": " > ".join(ranking),
                         **rca.ranking_metrics(ranking, truth),
                         **{f"chance_{m}": x for m, x in chance.items()}})
            if (steps_v, agg) == MAIN:
                imp_rows += [{"subsystem": sub, "fold": k, "recording_id": rid, "seed": seed, "variable": v,
                              "importance": imp[v], "is_cause": v in truth} for v in cands]
    return rows, imp_rows


def figures(runs, imp):
    import matplotlib.pyplot as plt
    ev.set_plot_style()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                         "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5})
    blue, orange, grey = ev.COLORS[0], ev.COLORS[1], ev.INK["muted"]

    # (1) attribution in one probe run, chosen by a fixed rule: first run of the first probe scenario, seed 0
    rid = sorted(imp[(imp.subsystem == "probe") & (imp.fold == "exp_1")].recording_id.unique())[0]
    d = imp[(imp.recording_id == rid) & (imp.seed == 0)].sort_values("importance")
    fig, ax = plt.subplots(figsize=(5.5, 3.4), layout="constrained")
    ax.barh(d.variable, d.importance, color=[orange if c else blue for c in d.is_cause])
    ax.set_xlabel("Summed absolute SHAP value (20 most anomalous steps)")
    ax.set_title("SHAP variable attribution in a probe fault run", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=orange, label="Injected cause"), Patch(color=blue, label="Other candidate")],
              loc="lower right", frameon=False)
    ev.save_figure(fig, "16_lstm_shap_attribution_example")
    plt.close(fig)

    # (2) per-scenario ap@3: causal rca (expert, discovered) versus lstm + shap
    causal = pd.read_csv(T / "15_rca_per_scenario.csv")
    main = runs[(runs.steps == MAIN[0]) & (runs.aggregation == MAIN[1])]
    lstm = main.groupby(["subsystem", "fold", "recording_id"])[["ap@3", "chance_ap@3"]].mean().groupby(
        ["subsystem", "fold"]).mean().reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.6), layout="constrained")
    for ax, sub in zip(axes, ("coolant", "probe")):
        folds = sorted(lstm[lstm.subsystem == sub].fold)
        x = np.arange(len(folds))
        series = [("Causal RCA, expert graph", causal[(causal.subsystem == sub) & (causal.graph == "expert")].set_index("fold").ap3, blue),
                  ("Causal RCA, discovered graph", causal[(causal.subsystem == sub) & (causal.graph == "discovered")].set_index("fold").ap3, orange),
                  ("LSTM + SHAP", lstm[lstm.subsystem == sub].set_index("fold")["ap@3"], grey)]
        for i, (lab, val, col) in enumerate(series):
            ax.bar(x + (i - 1) * 0.27, val.reindex(folds), width=0.27, color=col, label=lab)
        ch = lstm[lstm.subsystem == sub].set_index("fold")["chance_ap@3"].reindex(folds)
        ax.scatter(x, ch, marker="_", s=500, color="black", zorder=3, label="Chance level")
        ax.set_xticks(x, folds)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AP@3")
        ax.set_title(f"({'ab'[('coolant', 'probe').index(sub)]}) {sub.capitalize()}", loc="left")
    axes[1].set_xlabel("Held-out fault scenario")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Root-cause ranking by scenario: causal RCA versus LSTM + SHAP", fontsize=10.5)
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    ev.save_figure(fig, "16_rca_vs_lstm_shap")
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    ctx = get_context("spawn")
    with ctx.Pool(WORKERS) as pool:
        sel = pd.DataFrame(pool.map(job_select, [(sub, h, lb) for sub in SPLITS for h, lb in GRID]))
    ev.save_table(sel, "16_lstm_shap_selection")
    chosen = {}
    for sub, d in sel.groupby("subsystem"):
        best = d.sort_values(["normal_validation_loss", "hidden", "lookback_steps"]).iloc[0]
        chosen[sub] = (int(best.hidden), int(best.lookback_steps))
    print(sel.round(5).to_string(), "\nchosen:", chosen, flush=True)

    jobs = []
    for sub in SPLITS:
        splits = pd.read_csv(T / SPLITS[sub])
        for k in sorted(splits.loc[splits.split == "fault", "fold"].unique()):
            jobs += [(sub, k, seed, *chosen[sub]) for seed in SEEDS]
    with ctx.Pool(WORKERS) as pool:
        out = pool.map(job_fold, jobs)
    runs = pd.DataFrame([r for rows, _ in out for r in rows])
    imp = pd.DataFrame([r for _, rows in out for r in rows])
    ev.save_table(runs, "16_lstm_shap_runs")
    ev.save_table(imp, "16_lstm_shap_importance")

    metrics = ["top1", "hit@3", "recall@3", "ap@3", "rr"]
    # every variant: average over seeds per run, then runs -> scenarios (as for the causal rca)
    sens, per_scen_main, summ = [], None, []
    for (steps_v, agg), d in runs.groupby(["steps", "aggregation"]):
        per_run = d.groupby(["subsystem", "fold", "recording_id"])[metrics + ["chance_ap@3"]].mean().reset_index()
        per_scen = per_run.groupby(["subsystem", "fold"])[metrics + ["chance_ap@3"]].mean().reset_index()
        for sub, pr in per_run.groupby("subsystem"):
            ps = per_scen[per_scen.subsystem == sub]
            sens.append({"subsystem": sub, "steps": steps_v, "aggregation": agg, "main": (steps_v, agg) == MAIN,
                         "top1_run_mean": pr.top1.mean(), "map3_run_mean": pr["ap@3"].mean(),
                         "ap3_scenario_macro": ps["ap@3"].mean()})
        if (steps_v, agg) == MAIN:
            per_scen_main = per_scen
            seed_level = d.groupby(["subsystem", "seed"])[["top1", "ap@3"]].mean()
            for sub, pr in per_run.groupby("subsystem"):
                ps = per_scen[per_scen.subsystem == sub]
                sl = seed_level.loc[sub]
                summ.append({"subsystem": sub, "hidden": chosen[sub][0], "lookback_s": chosen[sub][1] * INTERVAL,
                             "runs": len(pr), "top1_run_mean": pr.top1.mean(), "hit3_run_mean": pr["hit@3"].mean(),
                             "recall3_run_mean": pr["recall@3"].mean(), "map3_run_mean": pr["ap@3"].mean(),
                             "mrr_run_mean": pr.rr.mean(), "ap3_scenario_macro": ps["ap@3"].mean(),
                             "chance_map3_run_mean": pr["chance_ap@3"].mean(),
                             "map3_seed_min": sl["ap@3"].min(), "map3_seed_max": sl["ap@3"].max(),
                             "top1_seed_min": sl.top1.min(), "top1_seed_max": sl.top1.max()})
    ev.save_table(pd.DataFrame(sens), "16_lstm_shap_sensitivity")
    ev.save_table(per_scen_main, "16_lstm_shap_per_scenario")
    summary = pd.DataFrame(summ)
    ev.save_table(summary, "16_lstm_shap_summary")
    figures(runs, imp)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=data.PROJECT_ROOT).stdout.strip()
    import shap
    import torch
    config = {"grid": {"hidden": [32, 64, 128], "lookback_steps": [20, 40]}, "chosen": chosen,
              "selection": "lowest normal-validation loss of a model trained on normal training recordings only",
              "seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "learning_rate": LR, "batch": BATCH,
              "shap": f"GradientExplainer, background {BACKGROUND} normal-training windows, nsamples {SHAP_SAMPLES}",
              "main_explanation": "20 most anomalous steps, summed absolute attribution",
              "sensitivity": "10 / 20 / all window steps x absolute / positive attribution",
              "torch": torch.__version__, "shap_version": shap.__version__, "code_commit": commit,
              "runtime_s": round(time.perf_counter() - t0)}
    (T / "16_lstm_shap_config.json").write_text(json.dumps(config, indent=2))
    print(summary.round(3).to_string())
    print(per_scen_main.round(2).to_string())
    print(pd.DataFrame(sens).round(3).to_string())
    print(f"finished in {config['runtime_s']} s")


if __name__ == "__main__":
    main()
