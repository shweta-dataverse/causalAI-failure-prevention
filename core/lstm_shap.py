"""lstm next-state forecaster with a shap explanation of its anomaly score (non-causal baseline).

the lstm predicts the next state of every variable from a look-back window. the anomaly score of a
step is the mean binary cross-entropy of the predicted next states over the recorded variables of
the run, and expected-gradient shap values of that score rank the candidate variables.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import shap
import torch
from torch import nn

from core.preprocessing import ViewData
from core.rca import analysis_window, candidates
from core.scm import recording_series

# windows per forward pass when scoring; changes only the summation order of the loss
EVAL_CHUNK = 4096


def run_arrays(view: ViewData, rid: str, fills: dict[str, float],
               manipulated: frozenset[str] | set[str] = frozenset()) -> tuple[np.ndarray, np.ndarray]:
    """filled states (steps, variables) and loss weights (1 = observed and not manipulated)."""
    a = view.arrays[rid].astype(np.float32).copy()
    m = view.masks[rid]
    w = (~m).astype(np.float32)
    for j, v in enumerate(view.var_names):
        a[m[:, j], j] = fills[v]
        if v in manipulated:
            w[:, j] = 0.0
    return a, w


def make_windows(arrays: Sequence[tuple[np.ndarray, np.ndarray]], lookback: int) -> dict:
    """every (look-back window, next row) pair; a recording start is padded with its first row."""
    blocks, wblocks, idx, offset = [], [], [], 0
    for a, w in arrays:
        blocks.append(np.vstack([np.repeat(a[:1], lookback, axis=0), a]))
        wblocks.append(np.vstack([np.zeros((lookback, a.shape[1]), np.float32), w]))
        idx.append(offset + np.arange(1, len(a)))
        offset += len(a) + lookback
    return dict(x=torch.from_numpy(np.vstack(blocks)), w=torch.from_numpy(np.vstack(wblocks)),
                idx=np.concatenate(idx), steps=torch.arange(lookback), lookback=lookback)


def batch(win: dict, ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """inputs, next states and loss weights of the windows ending before the given rows."""
    ids = torch.from_numpy(ids)
    lb = win["lookback"]
    return win["x"][ids[:, None] + win["steps"]], win["x"][ids + lb], win["w"][ids + lb]


class Forecaster(nn.Module):
    def __init__(self, n_vars: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(n_vars, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def val_loss(model: nn.Module, win: dict) -> float:
    """weighted mean binary cross-entropy of the next-state predictions."""
    total, n = 0.0, 0.0
    with torch.no_grad():
        for ids in np.array_split(win["idx"], max(1, len(win["idx"]) // EVAL_CHUNK)):
            x, y, w = batch(win, ids)
            bce = nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none")
            total += float((bce * w).sum())
            n += float(w.sum())
    return total / n


def train(train_win: dict, val_win: dict, n_vars: int, hidden: int, seed: int, learning_rate: float,
          batch_size: int, max_epochs: int, patience: int, min_improvement: float) -> tuple[nn.Module, float, list]:
    """adam training with early stopping on the validation loss; returns the best model, its loss and the history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Forecaster(n_vars, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best, best_state, history, waited = np.inf, None, [], 0
    for _ in range(max_epochs):
        model.train()
        order = rng.permutation(train_win["idx"])
        for ids in np.array_split(order, max(1, len(order) // batch_size)):
            x, y, w = batch(train_win, ids)
            bce = nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none")
            loss = (bce * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val = val_loss(model, val_win)
        history.append(val)
        if val < best - min_improvement:
            best, best_state, waited = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, history


def background_windows(win: dict, n: int, seed: int) -> torch.Tensor:
    """n random (look-back + next row) sequences used as the shap background."""
    bx, by, _ = batch(win, np.random.default_rng(seed).choice(win["idx"], n, replace=False))
    return torch.cat([bx, by[:, None]], dim=1)


class AnomalyScore(nn.Module):
    """mean binary cross-entropy of the next-state prediction over the weighted variables."""

    def __init__(self, model: nn.Module, weights: torch.Tensor):
        super().__init__()
        self.model, self.weights = model, weights

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        logits = self.model(seq[:, :-1])
        bce = nn.functional.binary_cross_entropy_with_logits(logits, seq[:, -1], reduction="none")
        return ((bce * self.weights).sum(1) / self.weights.sum()).unsqueeze(1)


def explain(model: nn.Module, view: ViewData, rid: str, fills: dict[str, float], alarms: Sequence[str],
            nodes: Sequence[str], background: torch.Tensor, lookback: int, interval_s: float, shap_samples: int,
            step_variants: dict[str, int | None]) -> tuple[list[str], dict, np.ndarray] | None:
    """candidates, attributions per (steps variant, abs/pos) and anomaly scores; None without alarm."""
    series = recording_series(view, rid)
    window = analysis_window(series, alarms, interval_s)
    if window is None:
        return None
    cands = candidates(series, nodes, alarms)
    a, _ = run_arrays(view, rid, fills)
    recorded = torch.tensor([0.0 if np.isnan(series[v]).all() else 1.0 for v in view.var_names])
    pad = np.vstack([np.repeat(a[:1], lookback, axis=0), a])
    steps = np.arange(window[0], window[1] + 1)
    seqs = torch.from_numpy(np.stack([pad[t: t + lookback + 1] for t in steps]))
    score = AnomalyScore(model, recorded)
    with torch.no_grad():
        anomaly = score(seqs).squeeze(1).numpy()
    phi = shap.GradientExplainer(score, background).shap_values(seqs, nsamples=shap_samples, rseed=0)
    # step x variable, summed over the window positions
    phi = np.asarray(phi).reshape(len(steps), lookback + 1, len(view.var_names)).sum(axis=1)
    order = np.argsort(-anomaly, kind="stable")
    importance = {}
    for name, n in step_variants.items():
        rows = order if n is None else order[:n]
        importance[(name, "abs")] = np.abs(phi[rows]).sum(0)
        importance[(name, "pos")] = np.clip(phi[rows], 0, None).sum(0)
    return cands, importance, anomaly


def importance_ranking(vector: np.ndarray, var_names: Sequence[str],
                       cands: Sequence[str]) -> tuple[list[str], dict[str, float]]:
    """candidates by decreasing attribution (ties by name) and the attribution of each."""
    importance = {v: float(vector[list(var_names).index(v)]) for v in cands}
    return sorted(cands, key=lambda v: (-importance[v], v)), importance
