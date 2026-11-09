# Evaluation audit against the final evaluation protocol (9 Nov 2026)

**Scope.** This is a read-only audit of the current repository (HEAD `e70eaa1`, branch `thesis-pipeline`, plus the uncommitted stage-12 files) against the protocol "Evaluation protocol for the causRCA thesis". No code or result was changed.

**Method.** Notebooks 02–10, `src/causal_rca/*`, and scripts `scm_variants_validation.py`, `scm_nested_selection.py`, `run_robustness.py` and `run_discovery_study.py` were inspected. Recording-ID sets were checked for overlap with small read-only scripts. The key claims below were re-checked by hand: the view training IDs, `fit_scm` keying, the notebook 05 refit cell, and the probe scenario-macro AP@3.

**Thesis state.** The thesis in Overleaf (HEAD `8a1bf45`) still contains the empty-graph and perturbation material: Ch. 1 contribution list, Ch. 4 §Graph Perturbation Study, Ch. 5 RQ4, Ch. 6 §RQ4.

---

## 1. Boundaries that hold (verified)

| Check | Evidence | Verdict |
|---|---|---|
| Normal manifest 119 / 26 / 25, same for both subsystems, whole recordings | `data.make_splits` (`data.py:453`); `02_coolant_splits` and `10_probe_splits` normal labels are identical; ID ranges are disjoint (train 28610–58890, validation 59058–66496, test 66631–72825) | OK |
| Track N full fit = 119 normal train + all subsystem fault runs; normal validation/test never fitted or used for selection | notebook 05 `fold_training_ids(held_out=None)`; notebook 10 likewise | OK |
| Fold graphs exclude regime k | `aggregate_relations(exclude=...)`; every downstream fold uses `discovered_without_{k}` | OK |
| Fold SCM training IDs exclude k; fills and fallbacks come from training IDs only | `data.py:516`; `scm.py:62, 190, 347`; overlap check = 0 in every fold | OK |
| Lag sets for the lag-aware SCM exclude k | `05_coolant_lag_selection`: 0 rows cite the held-out regime | OK |
| Stage-13 nested selection: inner fits exclude both k and j; outer fit excludes k | `scm_nested_selection.py:134–170`; 0 overlaps; no j == k | OK |
| Stage-13 results are not propagated downstream (default SCM stays primary) | no notebook or script reads `13_*` | OK |
| Stage-12 discovery study: training regimes only; nothing downstream reads it | `run_discovery_study.py`; grep | OK |
| RCA ranking does not read cause labels; window = first alarm − 60 s to last alarm + 5 s, untuned; deterministic ties | `root_cause_analysis.py:44–86, 164–178` | OK |
| Replay inputs = targets, values, onsets, pre-onset history; nothing tuned | `interventions.py:183–192` | OK. The X1 time-limited replay also reads the observed release time; this is disclosed. |
| Counterfactual action search uses only the label-free ranking and actionability rules; labels are used afterwards | `counterfactuals.py:140–150, 213–222` | OK |
| Fresh-world re-evaluation of fixed action sets | `run_robustness.py` X2: 2,000 worlds, `SeedSequence(20261023).spawn`, no reselection | OK |
| Threshold 0.90 applied as `>=` everywhere | `counterfactuals.py:47`, `run_robustness.py:253` | OK |
| Normal test never used for selection (stage 13) | `scm_variants_validation.py` scores normal validation and inner held-out scenarios only | OK |

## 2. Violations and inconsistencies (verified)

| # | Problem | Affected results | Severity | Proposed handling |
|---|---|---|---|---|
| V1 | **Global column view.** The ModelView is fitted on normal train + *all* fault runs, including the held-out scenario (`coolant_dt0.5.json` train_ids = 119 + 25; probe 119 + 34). A per-fold view would drop 2 coolant variables in each of exp_22, exp_27 and exp_28. | Column sets in leave-one-scenario-out folds | Low: all variables are binary pass-through, with no fitted values, vocabularies or exclusions. The output equals the predeclared expert node set. | Declare the schema = expert node set explicitly and disclose. No numbers change. |
| V2 | **Bootstrap duplicates collapse.** `fit_scm` keys training series by recording ID (`scm.py:348`), so a recording drawn twice counts once. `fill_values` keeps the multiplicity, which is inconsistent. A 138-recording draw has only ~81 unique recordings. | notebook 05 refit stability; notebook 08 "validity under refitted SCMs" (thesis Ch. 6 RQ3b robustness figure; Ch. 5 configuration table "refitted SCMs 5") | Medium: these are subsample refits, not bootstrap refits | Pass draws with multiplicity (small change to `fit_scm`; identical output without duplicates); rerun the notebook 08 refit check (5 refits) and the notebook 05 refits |
| V3 | **Refit training/evaluation overlap.** The notebook 05 refits train on `full_train` (incl. all 25 fault runs) and score on normal validation/test **and all fault runs**. | `05_coolant_refit_stability` (listed in the Ch. 5 configuration table, "refits on bootstrapped training recordings 10") | Medium (partly in-sample) | Score refits on normal validation/test only, or per fold |
| V4 | **Two discovered-graph versions are mixed.** `04_*`/`10_probe_graph_metrics` use the raw graph; `09_*`/`10_perturbation_summary` use the SCM DAG (undirected edges removed). Coolant skeleton F1 0.80 (raw) vs 0.75 (DAG); probe 0.522 vs 0.125. | Graph tables | Reporting | Separate raw and DAG rows (protocol T3) |
| V5 | **Counterfactual validity from three budgets.** Notebook 08 (200 worlds) gives 25/25 valid; notebook 09 (100 worlds) gives 23/25; X2 re-checks the *notebook 09* sets with 2,000 worlds. Notebook 08's own sets were never re-checked independently. | RQ3b numbers | Reporting | Final numbers = notebook 09 selection + X2 2,000-world re-evaluation; state the selection budget |
| V6 | **Redundant replay metric.** Coolant `p_fires_all` (0.661) equals alarm coverage (37/56), because supported alarms fire with p = 1 and unsupported ones with p = 0. | RQ3a | Reporting | Report activation probability on supported alarms with coverage beside it, not the pooled value |
| V7 | **Averaging scheme.** Every SCM loss is a row-weighted pooled mean; the protocol's recording → scenario macro is not implemented. For RCA, the probe scenario-macro AP@3 is expert 0.431 / discovered 0.611 / empty 0.486, vs run-weighted 0.475 / 0.559 / 0.444, so **expert vs empty flips**. | RQ2 probe, SCM fit | Reporting | Compute and label both |
| V8 | **Normal validation and test are pooled** in notebooks 05 and 10. | F2 (train / validation / test bars) | Reporting | Score the same fitted model per split (no refit) |
| V9 | **Outputs not reproducible from current notebooks.** `08_alarm_coverage`, `08_validity_2000`, `08_validity_thresholds`, `10_probe_rca_scenario_level`, `10_rca_ranking_variability`, `07_replay_delay_per_alarm` and `05_coolant_lag_selection` were produced by cells removed in commits 06fb7f8 / c717ed5. | Provenance | Reporting | Regenerate in one evaluation script, or cite the producing commit |
| V10 | **Stage-13 declaration not verifiable.** The objective is declared in the docstring, but the script and results arrived in one commit (`e70eaa1`). | §6 SCM sensitivity | Disclosure | Label it a separate sensitivity experiment ("rule stated in the script; prior specification not verifiable") |
| V11 | **Probe testable-only graph scope missing.** | Supplementary graph table | Low | Compute from `10_probe_links` |
| V12 | **Identical model inputs across splits.** Normal recordings in which no subsystem variable changes are bit-identical: coolant 39, probe 45, a few in validation/test. They are not acquisition duplicates (raw files all differ). They add easy "stay" rows to held-out log loss. | Track N overall log loss | Disclosure | Keep the declared split. Report change-row loss beside overall loss, and the number of zero-change recordings per split |
| V13 | **Chronology not verified.** The split is identifier-ordered; the raw files have no absolute times. | Track N | Disclosure (already in METHODOLOGY) | Call it "identifier-ordered" |
| V14 | **Grid interval chosen on all events.** The 0.5 s vs 1 s choice in notebook 02 used statistics over all recordings, including held-out ones. | Preprocessing | Low (not outcome-tuned) | Disclose |
| V15 | **"Supported" has two meanings.** `has_data` (replay, coverage) vs `Mechanism.supported` (changes seen in training). | Terminology | Low | Define both in the thesis |

**Intervention specificity.** Coolant has **no negative examples**: all 56 recorded alarm–run pairs activated, so false-alarm specificity cannot be assessed. Probe has 13 negatives (MPA_A_701124 ×8, MPA_A_701125 ×5), with mean p_fires 0.263 (expert) and 0.128 (discovered). None of these is reported yet.

## 3. Verdicts on the protocol itself

- **Accepted as written:**
  - two tracks (N / F);
  - LOSO boundaries and inner LOSO for any fault-outcome selection;
  - permitted inference on held-out runs;
  - metric definitions A, B2, C, D, which match the implementation (zero-denominator rule: F1 = 0, P/R = NaN; SHD: reversal = 1);
  - figure and table plan F0–F5 and T1–T6;
  - the uncertainty and sufficiency rules;
  - the final thesis claim and its exclusions.
- **Modified:**
  - **B1 primary averaging.** Switching the primary SCM summary to recording/scenario-macro *after* the results are known is itself a post-hoc choice. Proposal: keep the existing row-weighted change-row/overall loss as the primary (declared earlier) and report the macro summary beside it, labelled. If the macro is made primary, the thesis must say when and why.
  - **Duplicate groups (§3).** "Correct the grouping" would change the declared split after seeing results. The identical inputs are zero-change recordings, not shared acquisitions. Proposal: keep the split, disclose, and report change-row loss, which these recordings do not affect.
  - **Graphviz (§7).** Graphviz is not installed, and installing it adds a system dependency. The existing figures already use a layered layout with shared node positions (`04_coolant_expert_vs_discovered`). Proposal: keep that layout and add correct / extra / reversed / missed colouring.
  - **F2 metric.** Overall log loss is dominated by persistent states and the zero-change recordings. F2 follows the protocol (log loss); change-row loss goes in the T4 table, as specified.
- **Kept despite removing the empty graph:** the protocol asks not to attribute perfect coolant RCA to causal structure. The only evidence for that is the empty-graph RCA result (1.00). Proposal: one sentence plus one appendix row, not a main-evaluation arm.
