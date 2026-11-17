"""stage 17a: related-recording check and the frozen group manifest.

rule (fixed before any model was refitted):
  two recordings are related when their raw event logs share at least 90% of their rows after
  t = 0 (row = exact time_s, node and value; jaccard of the row sets). all 270 raw recordings of
  the dataset are compared (coolant, probe and hydraulics fault runs and the real-operation
  recordings), so normal recordings are checked as well.
  the initial rows at t = 0 are left out, since every recording starts with a full state snapshot.
supporting evidence, not used for the decision: the change events of the 0.5 s state grid
(step, variable, new value), compared at the same time and at the best common time shift,
for pairs in which both recordings change at least 5 times.
constant recordings are never related by this rule (they have no rows after t = 0 in common).
nothing here claims how the related recordings were produced; only their content is compared.

outputs:
  configs/related_recordings.csv          frozen manifest (group_id, subsystem, recording_id, ...);
                                          written once, later runs only verify it
  results/tables/17_related_pairs.csv     evidence per related pair
  results/tables/17_related_raw_differences.csv   raw rows present in only one recording of a pair
  results/tables/17_raw_similarity_top.csv        most similar raw pairs per kind (the gap to the rule)
  results/tables/17_grid_similarity_top.csv       most similar state-grid pairs per subsystem and kind
usage: PYTHONPATH=src .venv/bin/python scripts/related_recordings.py
"""

from __future__ import annotations

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd

from causal_rca import data
from causal_rca import evaluation as ev
from causal_rca import preprocessing as pp

THRESHOLD = 0.9
MIN_EVENTS = 5  # grid evidence only: pairs with fewer changes say nothing about similarity
HEADER = "time_s,node,value,type"
T = data.RESULTS_DIR / "tables"
MANIFEST = data.RELATED_PATH
SPLITS = {"coolant": "02_coolant_splits.csv", "probe": "10_probe_splits.csv"}


def recording_id(path) -> str:
    """dataset file name -> pipeline recording id."""
    name = path.name
    if name.startswith("dataset_real_op_"):
        return "real_op_" + name.removeprefix("dataset_real_op_").removesuffix(".csv")
    sub, exp = path.parts[-4].removeprefix("exp_"), path.parts[-3]
    return f"{sub}_{exp}_{path.parts[-2]}"


def raw_rows():
    files = [f for f in sorted(data.RAW_DIR.rglob("*.csv")) if f.open().readline().strip() == HEADER]
    rows = {}
    for f in files:
        d = pd.read_csv(f, dtype=str)
        d = d[d.time_s.astype(float) > 0]
        rows[f] = set(zip(d.time_s, d.node, d.value))
    return rows


def grid_events(view, rid):
    """(step, column, new value) of every observed state change on the 0.5 s grid."""
    a, m = view.arrays[rid], view.masks[rid]
    out = set()
    for j in range(a.shape[1]):
        obs = np.flatnonzero(~m[:, j])
        x = a[obs, j]
        out |= {(int(obs[i]), j, int(x[i])) for i in np.flatnonzero(np.diff(x) != 0) + 1}
    return out


def grid_similarity(A, B):
    """jaccard at the same time and at the best common shift (b minus a, in steps)."""
    j0 = len(A & B) / len(A | B) if A | B else np.nan
    by_key = {}
    for t, j, x in B:
        by_key.setdefault((j, x), []).append(t)
    shifts = Counter(tb - t for t, j, x in A for tb in by_key.get((j, x), []))
    d = shifts.most_common(1)[0][0] if shifts else 0
    n = len({(t + d, j, x) for t, j, x in A} & B)
    return j0, d, n / (len(A) + len(B) - n) if A or B else np.nan


def main():
    rows = raw_rows()
    files = {recording_id(f): f for f in rows}
    pairs = []
    for a, b in itertools.combinations(rows, 2):
        A, B = rows[a], rows[b]
        shared = len(A & B)
        pairs.append({"a": recording_id(a), "b": recording_id(b), "file_a": str(a.relative_to(data.RAW_DIR)),
                      "file_b": str(b.relative_to(data.RAW_DIR)), "rows_a": len(A), "rows_b": len(B),
                      "shared_rows": shared, "raw_jaccard": shared / len(A | B) if A | B else np.nan})
    raw = pd.DataFrame(pairs)
    raw["kind"] = ["-".join(sorted("normal" if x.startswith("real_op") else "fault" for x in (a, b)))
                   for a, b in zip(raw.a, raw.b)]
    related = raw[raw.raw_jaccard >= THRESHOLD].copy()
    top = raw.sort_values("raw_jaccard", ascending=False).groupby("kind").head(10)
    ev.save_table(top, "17_raw_similarity_top")
    print(f"{len(rows)} raw recordings, {len(raw)} pairs; related (raw jaccard >= {THRESHOLD}): {len(related)}")
    print(top.groupby("kind").head(3)[["a", "b", "rows_a", "rows_b", "shared_rows", "raw_jaccard"]].to_string(index=False))

    # groups: connected components of the related pairs
    groups, seen = [], {}
    for a, b in zip(related.a, related.b):
        g = seen.get(a) or seen.get(b)
        if g is None:
            g = set()
            groups.append(g)
        g |= {a, b}
        seen[a] = seen[b] = g

    # where each recording is used in the pipeline
    use = {}
    for sub, file in SPLITS.items():
        s = pd.read_csv(T / file)
        for r in s.itertuples(index=False):
            use.setdefault(r.recording_id, []).append((sub, r.split, r.fold if isinstance(r.fold, str) else ""))
    runs = data.load_fault_runs().set_index("recording_id")
    manifest = []
    for k, g in enumerate(sorted(groups, key=lambda g: sorted(g)), start=1):
        for rid in sorted(g):
            for sub, split, fold in use.get(rid, [("not used", "", "")]):
                manifest.append({"group_id": f"G{k}", "subsystem": sub, "recording_id": rid, "split": split,
                                 "scenario": fold, "raw_file": str(files[rid].relative_to(data.RAW_DIR)),
                                 "rule": f"raw-row jaccard >= {THRESHOLD} (rows after t = 0)"})
    manifest = pd.DataFrame(manifest)
    if MANIFEST.exists():
        frozen = pd.read_csv(MANIFEST, keep_default_na=False)
        same = frozen[["group_id", "recording_id"]].equals(manifest[["group_id", "recording_id"]])
        print(f"frozen manifest {MANIFEST.relative_to(data.PROJECT_ROOT)} reproduced: {same}")
        if not same:
            raise SystemExit("the frozen related-recording manifest differs from the recomputed groups")
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(MANIFEST, index=False)
        print(f"wrote {MANIFEST.relative_to(data.PROJECT_ROOT)} ({len(manifest)} rows)")

    # evidence per related pair: raw differences, annotations and grid similarity
    views = {sub: pp.load_view_data(data.PROCESSED_DIR, f"{sub}_dt0.5")[0] for sub in SPLITS}
    manipulated = data.manipulated_by_recording(data.load_fault_runs(), data.load_experiments())
    evidence, diffs = [], []
    for r in related.itertuples(index=False):
        fa, fb = data.RAW_DIR / r.file_a, data.RAW_DIR / r.file_b
        A, B = rows[fa], rows[fb]
        for who, only in ((r.a, A - B), (r.b, B - A)):
            diffs += [{"pair": f"{r.a} ~ {r.b}", "only_in": who, "time_s": t, "node": n, "value": v}
                      for t, n, v in sorted(only, key=lambda x: float(x[0]))]
        sub = next(u[0] for u in use[r.a])
        ga, gb = grid_events(views[sub], r.a), grid_events(views[sub], r.b)
        j0, shift, js = grid_similarity(ga, gb)
        la, lb = len(pd.read_csv(fa)), len(pd.read_csv(fb))
        evidence.append({
            "subsystem": sub, "a": r.a, "b": r.b, "file_a": r.file_a, "file_b": r.file_b,
            "raw_rows_a": la, "raw_rows_b": lb, "rows_after_t0_a": r.rows_a, "rows_after_t0_b": r.rows_b,
            "shared_rows_after_t0": r.shared_rows, "raw_jaccard": r.raw_jaccard,
            "differing_rows": len(A ^ B), "differing_nodes": ",".join(sorted({n for _, n, _ in A ^ B})),
            "cause_start_a": runs.cause_start_at.get(r.a), "cause_start_b": runs.cause_start_at.get(r.b),
            "cause_end_a": runs.cause_end_at.get(r.a), "cause_end_b": runs.cause_end_at.get(r.b),
            "injected_a": ",".join(sorted(manipulated.get(r.a, ()))), "injected_b": ",".join(sorted(manipulated.get(r.b, ()))),
            "grid_states_identical": bool(np.array_equal(views[sub].arrays[r.a], views[sub].arrays[r.b])
                                          and np.array_equal(views[sub].masks[r.a], views[sub].masks[r.b])),
            "grid_change_events_a": len(ga), "grid_change_events_b": len(gb), "grid_jaccard": j0,
            "scenario_a": next(u[2] for u in use[r.a]), "scenario_b": next(u[2] for u in use[r.b]),
        })
    ev.save_table(pd.DataFrame(evidence), "17_related_pairs")
    ev.save_table(pd.DataFrame(diffs), "17_related_raw_differences")
    print(pd.DataFrame(evidence).T.to_string())
    print(pd.DataFrame(diffs).to_string(index=False))

    # state-grid similarity of every pair within a subsystem (supporting evidence)
    grid = []
    for sub, file in SPLITS.items():
        s = pd.read_csv(T / file).set_index("recording_id")
        evs = {r: grid_events(views[sub], r) for r in s.index}
        for a, b in itertools.combinations(s.index, 2):
            if min(len(evs[a]), len(evs[b])) < MIN_EVENTS:
                continue
            j0, shift, js = grid_similarity(evs[a], evs[b])
            kind = "-".join(sorted("normal" if s.loc[x, "split"] != "fault" else "fault" for x in (a, b)))
            grid.append({"subsystem": sub, "kind": kind, "a": a, "b": b, "events_a": len(evs[a]), "events_b": len(evs[b]),
                         "grid_jaccard": j0, "best_shift_steps": shift, "grid_jaccard_best_shift": js,
                         "related": bool(((related.a == a) & (related.b == b)).any() or ((related.a == b) & (related.b == a)).any())})
    grid = pd.DataFrame(grid)
    gtop = grid.sort_values("grid_jaccard_best_shift", ascending=False).groupby(["subsystem", "kind"]).head(5)
    ev.save_table(gtop, "17_grid_similarity_top")
    print(gtop.to_string(index=False))


if __name__ == "__main__":
    main()
