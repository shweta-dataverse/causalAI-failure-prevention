# Related recordings: retrospective evaluation correction (stage 17, 17 Nov 2026)

## What was found

All 270 raw recordings of the dataset were compared in pairs: coolant, probe and hydraulics fault runs, and the real-operation recordings.

**Rule (fixed before any refit).** Two recordings are related when at least 90% of their raw rows after t = 0 are identical. A row is the exact `time_s`, `node` and `value`, and similarity is the Jaccard index of the row sets.

| Pair | Shared rows | Only difference | Annotated fault |
|---|---|---|---|
| `probe_exp_1_run_3` ~ `probe_exp_6_run_2` | 634 of 635 | `MPA_InitPos` change at 100.31 s vs 150.536 s | identical (86.559 – 150.562 s) |
| `probe_exp_2_3_run_1` ~ `probe_exp_7_run_1` | 989 of 990 | `MPA_WorkPos` change at 123.59 s vs 183.59 s | identical (109.088 – 183.522 s) |

- **The gap to the rule is large.** The next most similar raw pair shares 2.1% of its rows.
- **Normal recordings.** No two share more than one raw row. Their 0.5 s grid similarity of up to 0.8 comes from the periodic machining cycle, not from shared rows.
- **Coolant and the normal split.** No coolant recording is related. The normal split (119 / 26 / 25) is unchanged.
- **Provenance.** Not documented. Only the content was compared.

**Files.**
- Frozen manifest: `configs/related_recordings.csv`.
- Evidence: `results/tables/17_related_pairs.csv`, `17_related_raw_differences.csv`, `17_raw_similarity_top.csv`, `17_grid_similarity_top.csv`.

## The corrected protocol

Scenario holdout is kept. When scenario k is held out:
- **Discovery.** The related partners of k's runs are removed from every discovery regime of k's scopes, both outer (without k) and inner (without k and j). A reduced regime is run anew through PCMCI+ with the same 9 settings.
- **Fitting.** The partners are removed from the SCM fit, the fill values and the LSTM training data. All of these go through `data.fold_training_ids`.
- **Inner folds.** The partners of j's runs are removed from inner training, and the partners of k's runs from the inner validation scores.
- **Checks.** Every fold is checked in `results/tables/15_checks.csv`: no partner in the fit, and the outer test and its partners absent from the discovery scopes, inner fits and inner scores.
- **Fold-level record.**
  - Roles `fit`, `outer test` and `excluded (related to an outer-test run)`: `results/tables/15_probe_fold_manifest.csv`.
  - Inner folds (`inner_train_recordings`, `inner_scored_runs`, `related_left_out`): `15_scm_selection_inner.csv`.
  - Discovery scopes (`related_removed`): `15_probe_discovery_selection.csv`.

**What is unchanged:** methods, candidate settings, metrics, aggregation, simulation budgets and seeds.

## Before / after

Source: `results/tables/17_protocol_comparison.csv` and `17_protocol_affected_runs.csv`. The scenario-only values are read from git commit `00c0767`.

| Probe | Scenario only | Corrected |
|---|---|---|
| Change-row log loss, held-out scenarios (E; D) | 2.05; 2.93 | 2.09; 2.62 |
| Top-1 (E; D; LSTM) | 0.62; 0.65; 0.56 | 0.62; 0.53; 0.56 |
| MAP@3 (E; D; LSTM) | 0.67; 0.65; 0.59 | 0.67; 0.63; 0.58 |
| Scenario AP@3 (E; D; LSTM) | 0.64; 0.63; 0.52 | 0.64; 0.57; 0.51 |
| AP@3 of the 4 related runs (E; D; LSTM) | 0.21; 0.25; 0.49 | 0.21; 0.15; 0.46 |
| Discovered fold settings changed | – | exp_2_3, exp_6, exp_7 |

**Coolant:** every result is identical (`17_protocol_coolant_identity.csv`), including the 19/25 accepted recommendations and the 37/56 coverage.

## Why the results changed

- **Expert graph.** The settings and every probe ranking are unchanged. Only the held-out log loss rises, most in exp_6 (1.80 → 2.02 nats), where one run's near-copy had been in training.
- **Discovered graph.**
  - Removing one run from a regime of 3–8 runs changed the PCMCI+ output and the selected settings in three folds. The SCM fold graphs went from 5, 7 and 16 edges to 13, 18 and 9.
  - AP@3 changed: exp_2_3 0.63 → 0.79, exp_6 0.50 → 0.19, exp_7 0.35 → 0.10.
- **Uncertain.** Whether the discovered graph profited from the near-copies or only reacts to removing any recording is not separated. Its low bootstrap stability (34% Jaccard) fits the second explanation.

## Claim audit

| Claim | Evidence | Status | Change |
|---|---|---|---|
| Scenario separation correct | `15_checks.csv`, fold manifests | verified | now also free of related pairs |
| Normal split has no related recordings | `17_raw_similarity_top.csv` (max 1 shared row) | verified | none |
| Probe windows: first alarm onset − 60 s to last onset + 5 s; all 34 runs scored; no fallback | `root_cause_analysis.analysis_window`; LSTM log "unscored runs: 0" | verified | LSTM now skips a run without an alarm onset, as the causal RCA does |
| 19/25 accepted, 37/56 covered, 31 in accepted runs | `15_coolant_cf_runs.csv` (identical) | verified | none |
| Discovered probe graph at the diagnostic level of the expert graph | `15_rca_summary.csv` | **no longer holds** | thesis text revised |
| LSTM MAP@3 0.59 | `16_lstm_shap_summary.csv` | changed to 0.58 | thesis text revised |
| Empty-graph comparison | stage 13, scenario-only protocol | historical | labelled as historical in Appendix A |

## Reproduce (research repo, branch `thesis-pipeline`)

Seeds and streams are unchanged: RCA chance 0, replay 0, counterfactual selection 0, independent stream 20261109, LSTM seeds 0, 1 and 2.

Commands, in order:

```bash
PYTHONPATH=src .venv/bin/python scripts/related_recordings.py
PYTHONPATH=src .venv/bin/python scripts/run_discovery_study.py grid select report --workers 9
PYTHONPATH=src .venv/bin/python scripts/selected_settings.py
PYTHONPATH=src .venv/bin/python scripts/final_evaluation.py main
PYTHONPATH=src .venv/bin/python scripts/final_evaluation.py main selected
PYTHONPATH=src .venv/bin/python scripts/probe_rca_decomposition.py
PYTHONPATH=src .venv/bin/python scripts/compare_original_selected.py
PYTHONPATH=src .venv/bin/python scripts/lstm_shap_baseline.py
PYTHONPATH=src .venv/bin/python scripts/thesis_figures.py
PYTHONPATH=src .venv/bin/python scripts/protocol_comparison.py
PYTHONPATH=src .venv/bin/python scripts/audit_related_exclusion.py   # 17_exclusion_audit.csv, 0 violations
PYTHONPATH=src .venv/bin/python scripts/probe_window_table.py        # 17_probe_window_runs.csv
```

| Stage | Runtime |
|---|---|
| Related-recording check | ~1 min |
| Discovery grid, select and report | ~2 min, 36 new probe regime runs |
| Selection of settings | ~3 min |
| Final evaluation | ~70 s per mode |
| LSTM | ~41 min |

## The two failing stage-14 checks

The checks `coolant expert/discovered: counterfactual selection reproduces notebook 08` (`14_checks.csv`) compare the original-settings search with the superseded notebook-08 outputs.

- **What matches:** the recommended actions are identical in all 50 run-graph cases.
- **What differs:** the 200-trajectory search-stage validities differ by at most 0.04.
- **Why:** since commit `8147d60`, the search excludes actions without a path to a modelled alarm. Fewer candidate sets are simulated, so the shared random stream of the search is consumed differently.
- **Effect on the thesis:** none. The accepted recommendations use the independent 2,000-simulation stream (entropy 20261109), and the stage-15 checks all pass. The checks are kept and documented, not suppressed.

## Further verification (17 Nov 2026)

- **Fold audit:** `17_exclusion_audit.csv` covers every outer fold and every inner fold of both subsystems. It reads the recording ids stored in the PCMCI+ cache entries themselves. No held-out run or related partner appears in:
  - the discovery regimes,
  - the scm, fill-value or lstm fits,
  - the inner validation scores.

  Lstm early stopping uses normal validation recordings only, and the shap background uses normal training recordings only.
- **Testable pairs:** the grouped regimes can change which pairs are testable, and so the denominator of the "testable pairs" graph scores. In the evaluated data, every probe fold keeps 54 testable pairs covering all 15 reference edges, so the denominator is unchanged.
- **Before/after comparisons:** both protocols use the same 34 probe runs, the same candidate sets per run, the same windows, and unchanged metric code (only `data.py` and `causal_discovery.py` changed).
- **Probe windows (`17_probe_window_runs.csv`):**
  - Every probe run has exactly one alarm onset, of its expected alarm, 11.5–40.5 s after the annotated start and before the annotated end. No fallback is used.
  - In 13 runs, the first observed change of the injected sensor is its release, 0.1–1.0 s after the annotated end. This explains the Table 5.2 counts.
- **Tests:** 91 tests pass.
