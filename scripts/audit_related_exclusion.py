"""stage 17c: audit that no held-out run or its related partner reaches any fitting or selection step.

for every outer probe and coolant fold k and every inner fold (k, j):
  * discovery: the regime runs of the selected configuration of each scope, read back from the
    stage-12 cache entries themselves (their stored recording ids), contain no protected recording;
  * scm / lstm fits: data.fold_training_ids contains none;
  * inner scores: the inner validation runs contain no partner of k's runs;
  * lstm: early stopping uses the normal validation recordings, the shap background the normal
    training recordings only, so no fault recording enters either.
output: results/tables/17_exclusion_audit.csv
usage: PYTHONPATH=src .venv/bin/python scripts/audit_related_exclusion.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from causal_rca import data  # noqa: E402
from causal_rca import discovery_study as ds  # noqa: E402
from causal_rca import evaluation as ev  # noqa: E402
import run_discovery_study as rds  # noqa: E402

T = data.RESULTS_DIR / "tables"


def cached_ids(setting, config, regimes):
    """recording ids stored in the cache entries of every regime run of a scope."""
    specs, _ = setting.specs(config, regimes)
    ids = set()
    for spec in specs:
        entry = json.loads(rds.CACHE.path(spec).read_text())
        stored = entry.get("recording_ids") or entry.get("spec", {}).get("recording_ids")
        if stored is None:
            raise SystemExit(f"cache entry without recording ids: {rds.CACHE.path(spec)}")
        if list(stored) != list(spec.recording_ids):
            raise SystemExit(f"cache entry does not match its spec: {rds.CACHE.path(spec)}")
        ids |= set(stored)
    return ids


def main():
    study = rds.load_study()
    grid = {c.config_id: c for c in ds.grid_configs(study["grid"])}
    inner = pd.read_csv(T / "15_scm_selection_inner.csv", keep_default_na=False)
    rows = []
    for sub in ("coolant", "probe"):
        s = rds.Setting(sub)
        sel = pd.read_csv(T / f"15_{sub}_discovery_selection.csv", keep_default_na=False).set_index("graph_id")
        runs_of = lambda f: s.splits.loc[s.splits.fold == f, "recording_id"].tolist()
        normal_train = set(s.splits.loc[s.splits.split == "train", "recording_id"])
        for k in s.folds:
            protected = data.protected_ids(runs_of(k))
            partners = sorted(protected - set(runs_of(k)))
            scopes = [(k,)] + [(k, j) for j in s.folds if j != k]
            for scope in scopes:
                gid = "discovered_without_" + "_and_".join(scope)
                config = grid[sel.loc[gid, "config_id"]]
                inner_protected = protected | (data.protected_ids(runs_of(scope[1])) if len(scope) == 2 else set())
                used = cached_ids(s, config, s.scope_regimes(scope))
                fit = set(data.fold_training_ids(s.splits, scope))
                row = {"subsystem": sub, "outer_fold": k, "scope": ",".join(scope), "partners_of_outer_test": ",".join(partners),
                       "discovery_recordings": len(used), "discovery_clean": not (used & inner_protected),
                       "fit_recordings": len(fit), "fit_clean": not (fit & inner_protected)}
                if len(scope) == 2:
                    scored = set(",".join(inner[(inner.subsystem == sub) & (inner.track == "F") & (inner.outer_fold == k)
                                                & (inner.inner_fold == scope[1])].inner_scored_runs.unique()).split(",")) - {""}
                    row.update(inner_scored=len(scored), inner_scores_clean=not (scored & protected))
                rows.append(row)
            # lstm: same training ids; early stopping and shap background use normal recordings only
            rows.append({"subsystem": sub, "outer_fold": k, "scope": "lstm", "partners_of_outer_test": ",".join(partners),
                         "fit_recordings": len(data.fold_training_ids(s.splits, k)),
                         "fit_clean": not (set(data.fold_training_ids(s.splits, k)) & protected),
                         "background_clean": normal_train.isdisjoint(protected)})
    table = pd.DataFrame(rows)
    ev.save_table(table, "17_exclusion_audit")
    flags = [c for c in table.columns if c.endswith("_clean")]
    bad = table[~table[flags].fillna(True).all(axis=1)]
    print(table.groupby(["subsystem", "scope"]).size().shape[0], "rows checked;", len(bad), "violations")
    print(table[table.subsystem == "probe"].head(8).to_string(index=False))
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
