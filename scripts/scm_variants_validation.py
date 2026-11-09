"""validation-only comparison of binary transition SCM variants (graphs fixed: expert, empty).

nested leave-one-scenario-out: for every outer test scenario k, variants are scored only on
(a) the normal *validation* recordings, fitted on normal train + fault runs except k, and
(b) every inner scenario j != k, fitted on normal train + fault runs except k and j.
the normal test recordings and the outer scenario k are never scored. the research code is
imported unchanged; variants are built by replacing the probability tables or the
missing-parent rule outside the package.

usage: python scripts/scm_variants_validation.py <research_repo> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "src"))
from causal_rca import data  # noqa: E402
from causal_rca import graph_evaluation as ge  # noqa: E402
from causal_rca import preprocessing as pp  # noqa: E402
from causal_rca import root_cause_analysis as rca  # noqa: E402
from causal_rca import scm as sc  # noqa: E402

INTERVAL = 0.5
SPLIT_TABLE = {"coolant": "02_coolant_splits", "probe": "10_probe_splits"}


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------

def logistic_tables(model: sc.TemporalSCM, view_data, train_ids, manipulated, C: float) -> sc.TemporalSCM:
    """replace every cpt by an l2 logistic model with main effects, one per previous state.

    fitted on the same rows as the cpt (aggregated per configuration, weighted), then
    evaluated for every configuration so that the scm code is reused unchanged.
    unseen configurations get the logistic extrapolation instead of the cpt back-off.
    """
    series = {r: sc.recording_series(view_data, r) for r in train_ids}
    for v, mech in model.mechanisms.items():
        width = mech.width
        if width == 0 or not mech.has_data:
            continue
        codes_all = np.arange(1 << width)
        X_all = ((codes_all[:, None] >> np.arange(width)) & 1).astype(float)
        for state in (0, 1):
            n = mech.counts[state]
            seen = n > 0
            ones = np.round(mech.tables[state] * (n + 2 * sc.SMOOTHING) - sc.SMOOTHING) * seen
            if ones[seen].sum() in (0, n[seen].sum()):
                continue  # one class only: keep the smoothed (fallback-equivalent) table
            X = np.vstack([X_all[seen], X_all[seen]])
            y = np.r_[np.ones(seen.sum()), np.zeros(seen.sum())]
            w = np.r_[ones[seen], n[seen] - ones[seen]]
            keep = w > 0
            clf = LogisticRegression(C=C, max_iter=2000).fit(X[keep], y[keep], sample_weight=w[keep])
            mech.tables[state] = clf.predict_proba(X_all)[:, 1]
    return model


def half_fallback(model: sc.TemporalSCM) -> sc.TemporalSCM:
    """unseen parent configurations get p = 0.5 instead of the state-specific share."""
    for mech in model.mechanisms.values():
        for state in (0, 1):
            if state in mech.counts:
                mech.tables[state] = np.where(mech.counts[state] > 0, mech.tables[state], 0.5)
    return model


class MixtureSCM(sc.TemporalSCM):
    """a parent that is unrecorded at t and t-1 is averaged over its two constant states,
    weighted by its training share of state 1, instead of being set to its training mode."""
    means: dict

    def transition_probs(self, series, variable):
        mech = self.mechanisms[variable]
        x = series[variable]
        prev = x[:-1]
        probs = np.full(len(prev), np.nan)
        ok = ~np.isnan(prev)
        if not ok.any():
            return probs
        n = len(x)
        missing = {}
        for p in mech.parents:
            s = series.get(p, np.full(n, np.nan))
            missing[p] = np.isnan(s[1:]) & np.isnan(s[:-1])
        miss_parents = [p for p in mech.parents if missing[p].any()]
        if not miss_parents:
            return super().transition_probs(series, variable)
        total = np.zeros(len(prev))
        for combo in range(1 << len(miss_parents)):
            weight = np.ones(len(prev))
            s2 = dict(series)
            for i, p in enumerate(miss_parents):
                b = (combo >> i) & 1
                m = self.means[p]
                weight = np.where(missing[p], weight * (m if b else 1 - m), weight)
                base = series.get(p, np.full(n, np.nan))
                filled = base.copy()
                both = np.r_[missing[p], False] | np.r_[False, missing[p]]
                filled[both] = b
                s2[p] = filled
            f = sc.parent_features(s2, mech.parents, self.fills, mech.lags)
            total += weight * np.where(ok, mech.prob_one(np.where(ok, prev, 0), f), 0)
        probs[ok] = total[ok]
        return np.clip(probs, sc.PROB_CLIP, 1 - sc.PROB_CLIP)


def fit_variant(variant, view_data, graph, train_ids, manipulated):
    kind, value = variant
    if kind == "smoothing":
        return sc.fit_scm(view_data, graph, train_ids, manipulated, INTERVAL, smoothing=value)
    base = sc.fit_scm(view_data, graph, train_ids, manipulated, INTERVAL)
    if kind == "baseline":
        return base
    if kind == "logistic":
        return logistic_tables(base, view_data, train_ids, manipulated, C=value)
    if kind == "fallback_half":
        return half_fallback(base)
    if kind == "include_manipulated":
        return sc.fit_scm(view_data, graph, train_ids, {}, INTERVAL)
    if kind == "extra_lag":
        lags = {}
        for u, v in graph.edges:
            n_par = graph.in_degree(v)
            if 3 * n_par <= 20:  # stays a valid table: width = 2 * parents + one extra lag each
                lags[(u, v)] = [value]
        return sc.fit_scm(view_data, graph, train_ids, manipulated, INTERVAL, lags=lags)
    if kind == "parent_mixture":
        means = {}
        for j, name in enumerate(view_data.var_names):
            obs = np.concatenate([view_data.arrays[r][~view_data.masks[r][:, j], j] for r in train_ids])
            means[name] = float(obs.mean()) if obs.size else 0.0
        m = MixtureSCM(graph=base.graph, mechanisms=base.mechanisms, fills=base.fills, interval_s=INTERVAL)
        m.means = means
        return m
    raise ValueError(kind)


VARIANTS = [("baseline", None)] + \
    [("smoothing", a) for a in (0.01, 0.1, 1.0)] + \
    [("logistic", c) for c in (0.1, 1.0, 10.0)] + \
    [("fallback_half", None), ("parent_mixture", None), ("include_manipulated", None)] + \
    [("extra_lag", l) for l in (2, 4, 10)]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def predictions(model, view_data, ids, manipulated):
    """like scm.held_out_predictions, but through model.transition_probs (so subclasses apply)."""
    frames = []
    for r in ids:
        s = sc.recording_series(view_data, r)
        for v in model.mechanisms:
            if v in manipulated.get(r, set()) or v not in s:
                continue
            p = model.transition_probs(s, v)
            y, prev = s[v][1:], s[v][:-1]
            ok = ~np.isnan(p) & ~np.isnan(y)
            if ok.any():
                frames.append(pd.DataFrame({"variable": v, "p": p[ok], "y": y[ok], "prev": prev[ok]}))
    return pd.concat(frames, ignore_index=True)


def fit_scores(pred):
    lik = np.where(pred.y == 1, pred.p, 1 - pred.p)
    change = (pred.y != pred.prev).to_numpy()
    return {"rows": len(pred), "changes": int(change.sum()),
            "log_loss": float(-np.log(lik).mean()),
            "change_log_loss": float(-np.log(lik[change]).mean()) if change.any() else np.nan,
            "stay_log_loss": float(-np.log(lik[~change]).mean())}


def rca_scores(model, view_data, run_ids, manipulated, alarms, nodes):
    rows = []
    for rid in run_ids:
        series = sc.recording_series(view_data, rid)
        window = rca.analysis_window(series, alarms, INTERVAL)
        if window is None:
            continue
        cands = rca.candidates(series, nodes, alarms)
        ranking = rca.score_candidates(model, series, cands, window)
        rows.append(rca.ranking_metrics(ranking["variable"], set(manipulated[rid])))
    return pd.DataFrame(rows).mean().to_dict() if rows else {}


def main():
    runs = data.load_fault_runs()
    experiments = data.load_experiments()
    manipulated = data.manipulated_by_recording(runs, experiments)
    results = []
    for subsystem in ("coolant", "probe"):
        view_data, _ = pp.load_view_data(data.PROCESSED_DIR, f"{subsystem}_dt{INTERVAL:g}")
        splits = pd.read_csv(data.RESULTS_DIR / "tables" / f"{SPLIT_TABLE[subsystem]}.csv")
        folds = sorted(splits.loc[splits.split == "fault", "fold"].unique())
        validation = splits.loc[splits.split == "validation", "recording_id"].tolist()
        normal_train = splits.loc[splits.split == "train", "recording_id"].tolist()
        expert = ge.simple_digraph(data.load_expert_graph(subsystem=subsystem))
        nodes = sorted(expert.nodes)
        alarms = sorted(n for n in nodes if expert.nodes[n]["kind"] == "Alarm")
        empty = expert.copy()
        empty.remove_edges_from(list(empty.edges))
        for gname, graph in (("expert", expert), ("empty", empty)):
            for variant in VARIANTS:
                if gname == "empty" and variant[0] in ("logistic", "extra_lag", "parent_mixture", "fallback_half"):
                    continue  # no parents: these variants equal the baseline
                for k in folds:
                    def train_without(*held):
                        return normal_train + splits.loc[(splits.split == "fault") & (~splits.fold.isin(held)), "recording_id"].tolist()
                    model = fit_variant(variant, view_data, graph, train_without(k), manipulated)
                    results.append({"subsystem": subsystem, "graph": gname, "variant": variant[0], "value": variant[1],
                                    "outer_fold": k, "data": "normal validation",
                                    **fit_scores(predictions(model, view_data, validation, manipulated))})
                    for j in folds:
                        if j == k:
                            continue
                        model = fit_variant(variant, view_data, graph, train_without(k, j), manipulated)
                        ids = splits.loc[splits.fold == j, "recording_id"].tolist()
                        results.append({"subsystem": subsystem, "graph": gname, "variant": variant[0], "value": variant[1],
                                        "outer_fold": k, "inner_fold": j, "data": "inner held-out scenario",
                                        **fit_scores(predictions(model, view_data, ids, manipulated)),
                                        **rca_scores(model, view_data, ids, manipulated, alarms, nodes)})
                print(subsystem, gname, variant, flush=True)
    pd.DataFrame(results).to_csv(out / "13_scm_variants_validation.csv", index=False)


if __name__ == "__main__":
    main()
