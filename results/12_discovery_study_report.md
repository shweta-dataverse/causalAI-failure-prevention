# Stage 12: causal-discovery hyperparameter study

**Scope.** This study covers causal discovery only (PCMCI+ settings and the aggregation over regimes). SCM fitting, RCA, interventions and counterfactuals are unchanged. Notebooks 03 and 10 still produce the primary graphs with the original configuration. The original result tables are left untouched, and every output of this study is written to new `12_*` files.

**Provenance**
- git commit `06fb7f85b40bec094a7a5f57d87152d5bead112d` (branch `thesis-pipeline`), with uncommitted study files on top. These are listed per invocation in `results/tables/12_study_manifest.jsonl`.
- Software: tigramite 5.2.10.1, Python in `.venv`, numba 0.68.1 (installed for CMIsymb only).
- Configuration: `configs/discovery_study.yaml`, written before any result existed.
- Code: `src/causal_rca/discovery_study.py`, `scripts/run_discovery_study.py`. Tests are in `tests/test_discovery_study.py`; the full suite gives 91 passed.
- Run cache: `data/cache/discovery_study/<subsystem>/<key>.json`, one file per regime run. The key is a SHA-1 over the subsystem, regime, recording IDs, variables, PCMCI+ settings, a data fingerprint and the tigramite version.
- Log: `results/logs/12_discovery_study.log`. Run events: `results/tables/12_run_events.csv`.

**Label used for all scores in this document:** *reference-guided tuning performance on the causRCA subsystem*.
- The expert graph was used to choose hyperparameters. The scores of the selected configuration are therefore optimistically biased (Cawley & Talbot 2010; Varma & Simon 2006). They are not independent test accuracy.
- PCMCI+ itself never receives expert edges. Expert information enters only through the selection objective and through the node set, which was already used in notebooks 03/10.

---

## 1. What was already in the pipeline: four separate things

| Category | Where | What it is | What it is not |
|---|---|---|---|
| **Primary manual settings** | notebook 03 / 10, `DiscoveryConfig` | PCMCI+ with G², Δ = 0.5 s, τmin = 0, τmax = 10 s (20 steps), pc_alpha = 0.01, caps 3/3/3, BH correction plus lagged-link filter, support ≥ 0.5. Fixed a priori from domain reasoning and library defaults. | Not tuned. Not prescribed by the PCMCI+ paper: Runge (2020) gives no default values and suggests choosing αPC by cross-validation or an information criterion. |
| **Sensitivity experiments** | notebook 03 (one setting at a time: Δ, τmax, α, FDR filter, support) | Each variant is compared with the *primary graph* (Jaccard), not with the expert graph. | Not tuning. There was no selection objective and no selection procedure, so they must not be described as hyperparameter tuning. |
| **Bootstrap stability** | notebook 03 (20 recording resamples); stage 12d here | Edge frequencies under resampling of recordings within regimes. | Not selection. It measures how stable a graph is, not which configuration is best. |
| **Actual selection** | **stage 12 (this study)** | Explicit objective, grid, tie-breaks and eligibility, written to `configs/discovery_study.yaml` before running. | Not a global optimum, and not independent accuracy. It is reference-guided tuning. |

---

## 2. Selection objective (specified before executing the search; not formally preregistered)

The candidates are ranked lexicographically (`selection.order` in the yaml):

1. **directed F1** against the expert reference (maximise). This is the primary objective.
2. **skeleton F1** (maximise). This is the first tie-break.
3. **τmax** (minimise): prefer the shorter maximum lag.
4. **pc_alpha** (minimise): deterministic rule, prefer the more conservative significance level.
5. **support threshold** (maximise): deterministic rule, prefer the stricter aggregation.
6. **config_id** (string order): final deterministic tie-break.

Rules around the ranking:
- **Eligibility:** status `ok` in every regime of the scope.
- **Ties:** the number of configurations tied on the objective, and on the objective plus the first tie-break, is reported (`n_tied_objective`, `n_tied_objective_and_first_tiebreak`).
- **Also reported:** directed and skeleton precision/recall, SHD, number of adjacencies, unresolved orientations, testable-pair share and orientation conflicts. These are not part of the ranking.
- **Numerical stability:** floats are rounded to 12 decimals before ranking, so float noise cannot break a tie. This is covered by a unit test.

The objective was not changed after results were inspected. The yaml was amended once (CMIsymb restricted to fault regimes, see §7). That amendment was made at 13:21:22, before any grid table existed, and only because of runtime.

**Wording for the thesis:** "best-performing configuration among the evaluated candidates under the stated selection objective". Do not call it optimal.

---

## 3. Design

**Stage A: grid (27 configurations per subsystem)**
- Settings: pc_alpha ∈ {0.001, 0.01, 0.05} × τmax ∈ {5, 10, 20} s (= 10, 20, 40 steps) × support ∈ {0.25, 0.5, 0.75}.
- The support threshold acts only at aggregation, so each subsystem needs 9 PCMCI+ settings × its regimes:
  - coolant: 9 × 5 regimes;
  - probe: 9 × 7 regimes;
  - 108 regime runs in total, all `ok`.
- Fixed: G², Δ = 0.5 s, caps 3/3/3, BH with filter.

**Stage B: leave-one-scenario-out folds**
- For each held-out fault scenario, the same grid is re-aggregated without that scenario's regime, and the same selection rule is applied.
- The held-out regime is never used for discovery, aggregation or selection in its fold. Because regime runs are independent, this is exact: discovery in one regime does not see the others (unit test `held-out regime never affects its fold`).
- Settings and training IDs are recorded per fold: `12_{sub}_selection.csv`, `12_{sub}_scope_training_ids.csv`.

**Stage C: staged comparisons on a shortlist** (not an exhaustive search)
- Shortlist: the top 3 configurations of stage A plus the original configuration, if it is not already among them. For coolant the original is rank 3, so the shortlist has 3 entries; for probe it has 4.
- Conditioning caps k ∈ {2, 4}, for the selected and the original configuration. `max_conds_dim = max_conds_px = max_conds_py = k` are changed together.
- τmax = 60 s, for the selected and the original configuration.
- FDR filter off, for all 27 configurations (re-aggregation only, no rerun).
- CI test G² vs CMIsymb, for the selected configuration, on fault regimes only (§7).
- Per-run timeout: 3 h.

**Stage D: bootstrap stability of the shortlist**
- 20 draws, seed 0. Recordings are resampled with replacement within every regime, the same scheme as notebook 03.
- Check: the coolant original-configuration bootstrap reproduces `03_coolant_bootstrap_stability` exactly (`12_coolant_bootstrap_reproduction_check.csv`: identical = True).

**Reproduction check of the pipeline**
- The original configuration recomputed in this study gives edge sets identical to `03_coolant_edges` and `10_probe_edges` in every scope (5 coolant and 7 probe scopes): `12_{sub}_reproduction_check.csv`.

**Split hygiene**
- Discovery uses the training recordings only: normal training split plus fault runs.
- Normal validation and test recordings never enter discovery or selection.
- Recording boundaries are kept: recordings are passed as separate datasets, so lags never cross recordings.
- The node set is the expert node set, which is independent of held-out data.
- Downstream outcomes (RCA, interventions) are **not** used in the objective.

---

## 4. Results: stage A and B

### 4.1 Original vs selected (`results/tables/12_original_vs_selected.csv`)

| Subsystem | Scope | Configuration | τmax | α | support | dir. P | dir. R | **dir. F1** | skel. F1 | SHD | adj. | unresolved | ties (obj.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coolant | all scenarios | original | 10 s | 0.01 | 0.5 | 1.00 | 0.60 | **0.750** | 0.800 | 6 | 10 | 4 | – |
| coolant | all scenarios | selected | 5 s | 0.01 | 0.5 | 1.00 | 0.60 | **0.750** | 0.800 | 6 | 10 | 4 | 4 |
| probe | all scenarios | original | 10 s | 0.01 | 0.5 | 0.00 | 0.00 | **0.000** | 0.522 | 17 | 8 | 7 | – |
| probe | all scenarios | selected | 20 s | 0.01 | 0.25 | 0.39 | 0.47 | **0.424** | 0.588 | 18 | 19 | 3 | 1 |

**Coolant**
- Four configurations tie on directed F1 (0.75) *and* skeleton F1 (0.80): τmax 5 s or 10 s × support 0.25 or 0.5, all at α = 0.01. They produce the identical graph.
- The tie is broken by the deterministic "shorter τmax" rule and then "stricter support", which picks τ 5 s / s 0.5.
- The selected graph is **identical to the original graph** (Jaccard 1.0). The study therefore gives no reason to change the coolant graph. It shows that the original choice is among the best evaluated candidates, and that τmax 5 s vs 10 s makes no difference.
- α = 0.01 is best at every τmax. α = 0.001 loses one correct edge (directed F1 0.667), and α = 0.05 adds adjacencies (up to 13–14 at support ≤ 0.5; directed F1 0.46–0.67).

**Probe**
- There is a unique winner: τ 20 s / α 0.01 / s 0.25.
- The original configuration orients no expert edge correctly (directed F1 0). The selected configuration recovers 7 of 15 expert edges in the correct direction (directed precision 0.39), mainly arm-position → alarm links.
- The cost is many extra edges: 19 adjacencies, 9 of them not in the reference, 2 reversed, and SHD 18 vs 17.
- The two graphs share almost nothing (Jaccard 0.037).
- **Support 0.75 collapses every probe configuration to 2 adjacencies** (directed F1 0). Most probe pairs are testable in only a few regimes, and the threshold acts in coarse steps.

**Single configuration for both subsystems?**
- No configuration is best on both subsystems. The four tied coolant winners score 0.000–0.286 on probe, and the probe winner scores 0.706 on coolant.
- The best worst-case configuration is τ 20 s / α 0.01 / s 0.25 (coolant 0.706, probe 0.424). This is a descriptive observation only; it was not a pre-specified objective.

### 4.2 Heatmaps

`results/figures/12_coolant_heatmap_directed_f1.{pdf,png}` and `12_probe_heatmap_directed_f1.{pdf,png}`:
- rows: α; columns: τmax (s, with steps); one panel per support threshold (0.25 / 0.5 / 0.75);
- cells: directed F1, with skeleton F1 in brackets;
- `*` marks the selected configuration;
- scope: all scenarios; label: reference-guided tuning performance.

### 4.3 Leave-one-scenario-out folds (selection repeated without the held-out scenario)

| Subsystem | Held out | Fold winner | dir. F1 | ties (obj.) | Original in that fold |
|---|---|---|---|---|---|
| coolant | exp_22 | τ5 / 0.01 / 0.5 | 0.750 | 3 | 0.750 |
| coolant | exp_27 | τ10 / 0.01 / 0.75 | 0.667 | 3 | 0.667 |
| coolant | exp_28 | τ10 / 0.01 / 0.5 (= original) | 0.667 | 1 | 0.667 |
| coolant | exp_35 | τ5 / 0.01 / 0.5 | 0.750 | 4 | 0.750 |
| probe | exp_1 | τ20 / 0.01 / 0.25 | 0.452 | 1 | 0.000 |
| probe | exp_2_3 | τ20 / 0.01 / 0.5 | 0.400 | 1 | 0.118 |
| probe | exp_4 | τ20 / 0.05 / 0.25 | 0.364 | 1 | 0.000 |
| probe | exp_5 | τ20 / 0.01 / 0.25 | 0.375 | 1 | 0.118 |
| probe | exp_6 | τ10 / 0.001 / 0.25 | 0.435 | 1 | 0.118 |
| probe | exp_7 | τ20 / 0.01 / 0.25 | 0.438 | 1 | 0.118 |

Fold scores are still reference-guided: they are computed on the same expert graph that drives the fold's selection. The folds show **selection stability**, not out-of-sample accuracy.
- **Coolant:** in every fold the winner's directed F1 equals the original's. The selection never improves on the original.
- **Probe:** τ 20 s is chosen in 5 of 6 folds and support 0.25 in 5 of 6 folds, but α varies (0.001 / 0.01 / 0.05).

---

## 5. Results: stage C (staged comparisons)

### 5.1 Conditioning caps (`12_{sub}_staged.csv`)

| Subsystem | Variant of | caps 2/2/2 | **caps 3/3/3** | caps 4/4/4 |
|---|---|---|---|---|
| coolant | selected (τ5) | 0.706 / 0.727 / 12 adj. | **0.750 / 0.800 / 10 adj.** | 0.750 / 0.762 / 11 adj. |
| coolant | original (τ10) | 0.667 / 0.727 / 12 adj. | **0.750 / 0.800 / 10 adj.** | 0.750 / 0.762 / 11 adj. |
| probe | selected (τ20, s0.25) | 0.389 / 0.541 / 22 adj. | **0.424 / 0.588 / 19 adj.** | 0.414 / 0.516 / 16 adj. |
| probe | original (τ10, s0.5) | 0.111 / 0.435 / 8 adj. | **0.000 / 0.522 / 8 adj.** | 0.125 / 0.381 / 6 adj. |

(Each cell: directed F1 / skeleton F1 / adjacencies.)

Observations:
- Caps of 2 produced more adjacencies (coolant 12 vs 10; probe selected 22 vs 19) and lower directed F1 on three of the four configurations. On the probe original, directed F1 rose from 0.000 to 0.111.
- Caps of 4 did not improve directed F1 on the selected configurations, and lowered skeleton F1 everywhere.

Plausible explanations, **not tested separately**:
- caps of 2 may allow too few conditions to block indirect paths;
- with 3 + 3 + 3 binary conditions there are already up to 2⁹ = 512 strata against 108–294 state changes per fault regime, so larger caps may mostly add sparsely populated strata. Stratum occupancy was not measured.

Conclusion: retaining caps 3 is supported **within this comparison**. It is a staged comparison on two configurations per subsystem, not a search over caps.

### 5.2 Longer maximum lag (60 s)

**Why 60 s was examined.** In the fault runs, the first change of the manipulated variable follows the injection annotation (`annotation_to_change_s` in `12_{sub}_annotation_delay.csv`) by:
- coolant: 20.4–50.2 s over all 25 fault runs (scenario medians 33–43 s);
- probe: 8.5–80.7 s over all 34 fault runs (scenario medians 13–68 s; 13 runs above 50 s, spread over all six scenarios: exp_5 4 of 6, exp_6 3 of 3, exp_2_3 2 of 8, exp_7 2 of 5, exp_1 1 of 6, exp_4 1 of 6).

These are full ranges, not a subset. The values of about 65–144 s in the same table are a different column, `cause_start_at_s`: the time of the annotation *within* the recording, not a delay.

That delay runs from **the injection annotation to a signal**. It reflects the test script and the digital twin, not a mechanism between two recorded variables, and τmax has to cover only the latter.

**Expert-free co-change timing between recorded signals** (`12_{sub}_timing_evidence.csv`). The unit is an **ordered pair × regime**, using the variables recorded in that regime:
- coolant: 176 observations (normal 6, exp_22 42, exp_27 30, exp_28 56, exp_35 42);
- probe: 612 observations (normal 72, six fault regimes × 90).

This is not the number of distinct variable pairs: probe has 11 nodes, i.e. 55 unordered pairs. For each observation, the lag in 0–60 s at which co-changes most exceed chance (lift) was computed on training data. Observations with no co-change at any lag have no peak. Lag-0 peaks are symmetric, so each such unordered pair appears twice.

| | observations | with a peak | at lag 0 | 0.5–10 s | 10.5–20 s | > 20 s |
|---|---|---|---|---|---|---|
| coolant | 176 | 55 | 27 | 22 | 4 | 2 |
| probe | 612 | 541 | 225 | 289 | 4 | 23 |

How to read this:
- Most co-change peaks lie within 10 s. This supports the **search range** of 5–20 s.
- **Co-change peaks are not causal delays.** A peak can also come from shared operating cycles, simultaneous injections of a multi-cause scenario, or a common driver.
- The probe peaks beyond 20 s (at 20–25 s and 43–47 s, in every probe fault regime except exp_4, most often in exp_7, and involving arm positions and alarms) fit the period of the scripted arm cycle and the alarm timeouts. They are not evidence of direct mechanisms.
- In coolant normal data, `LP_Pump_On → LT_Pump_Ok` peaks at 45.5 s with a weak lift.

**Sample cost of a 60 s lag.** Tigramite's tests use `cut_off='2xtau_max'`, so the first 2·τmax steps of every recording never serve as test time t (verified in the source and by a unit test). With 60 s this removes 120 s from recordings that are only 158–365 s long. Share of state changes left after the cut (`12_{sub}_sample_availability.csv`):

| | τ 5 s | τ 10 s | τ 20 s | τ 60 s |
|---|---|---|---|---|
| coolant fault regimes | 100% | 100% | 100% | 52–72% |
| probe fault regimes | 100% | 100% | 93–100% | 21–31% |
| normal regimes (coolant / probe) | ≥ 98% | ≥ 97% | ≥ 92% | 79% / 92% |

**60 s results**

| Subsystem | Variant | dir. F1 | skel. F1 | SHD | adj. | runtime (all regimes) |
|---|---|---|---|---|---|---|
| coolant | selected = original at 60 s (τ60 / 0.01 / 0.5) | 0.571 | 0.632 | 9 | 9 | 158 s (vs 20 s at τ5) |
| probe | selected at 60 s (τ60 / 0.01 / 0.25) | 0.190 | 0.519 | 18 | 12 | 1,844 s (normal regime ≈ 1,750 s) |
| probe | original at 60 s (τ60 / 0.01 / 0.5) | 0.125 | 0.125 | 14 | 1 | 1,844 s (same regime runs) |

Interpretation:
- The 60 s graphs are worse on the reference-guided objective.
- This result is **confounded**: 60 s removes 28–79% of the fault-regime state changes, and it multiplies the BH family size (N²·τmax lagged hypotheses per regime) by 3–6.
- Conclusion: under Tigramite's current sample-cutting rule (`cut_off='2xtau_max'`), a 60 s maximum lag substantially reduces the usable fault data, and it was not retained.
- This is **not** evidence that 10 s (or 20 s) is scientifically optimal, nor that no delay longer than 20 s exists.
- Runge (2020) notes that τmax should be at least the largest true lag, and that choosing it too large does not degrade performance much for long series. The series here are short, which is consistent with the sample cut dominating. That was not isolated, e.g. with a different cut-off rule.

### 5.3 FDR filter on / off

**Verified against tigramite 5.2.10.1.**
- `run_pcmciplus(fdr_method='fdr_bh')` calls `get_corrected_pvalues(exclude_contemporaneous=True)` at the end. This rewrites only the reported `p_matrix`. The returned graph is unchanged (unit test).
- Corrected: the lagged (τ ≥ 1) entries of one regime, autodependencies included, i.e. N²·τmax hypotheses per regime. The family grows with τmax.
- Not corrected: contemporaneous (lag-0) p-values.
- The correction is applied separately in each regime run.
- The reported p of a link is the maximum over all its tests.
- Our post-hoc filter removes lagged links whose corrected p exceeds α.
- **Orientation:** collider and Meek orientation of lag-0 links ran on the *uncorrected* graph and is not recomputed after the filter. The diagnostic counts lag-0 orientations where a removed lagged link enters either endpoint. This is an upper bound on orientations that could have depended on a removed link.

**Diagnostics for the selected configuration** (`12_{sub}_fdr_diagnostics.csv`, summed over regimes):

| | lagged links | removed by filter | lag-0 orientations | possibly dependent on a removed link |
|---|---|---|---|---|
| coolant τ5 / 0.01 | 15 | 8 | 5 | ≤ 1 |
| probe τ20 / 0.01 | 141 | 83 | 27 | ≤ 21 |

**Effect on the graphs** (`12_{sub}_fdr_off.csv`; filter off is reported, never selected):
- **Coolant:** mean adjacencies over the 27 configurations go from 9.9 (filtered) to 12.3 (unfiltered). The best unfiltered directed F1 is 0.737, below the filtered best of 0.75. Unfiltered is higher in 10 of 27 configurations and lower in 8.
- **Probe:** adjacencies go from 9.4 to 13.4. The best unfiltered directed F1 is 0.483 (τ5 / 0.01 / 0.25), *above* the filtered best of 0.424. Unfiltered is higher in 13 of 27 configurations and lower in 4.
- On probe, the filter therefore costs reference-guided F1. It is kept because it was fixed a priori and has a reason independent of the score: the number of lagged hypotheses grows with τmax. Selecting "no correction" because it scores better would be outcome-driven selection.

**What must not be claimed:** that the final aggregated graph has FDR ≤ 1%. The reasons:
- the PC-type tests are sequential and data-adaptive (Li & Wang 2009; Strobl et al. 2019: post-hoc edge-wise control holds only under strong conditions);
- lag-0 links are not corrected;
- orientations are not recomputed;
- the graph is a vote over regimes.

The filter is a pragmatic multiplicity adjustment of lagged links, not formal FDR control.

---

## 6. Results: stage D (bootstrap stability of the shortlist)

`12_{sub}_bootstrap_summary.csv`, `_winners.csv`, `_edges.csv`, `_draws.csv`: 20 draws, all runs `ok` (732 regime runs).

| Subsystem | Configuration | dir. F1 mean ± sd [min, max] | skel. F1 mean | Jaccard with full-data graph | wins under the selection rule |
|---|---|---|---|---|---|
| coolant | **τ5 / 0.01 / 0.5 (selected)** | 0.647 ± 0.056 [0.56, 0.75] | 0.712 | 0.68 | 13 / 20 |
| coolant | τ10 / 0.01 / 0.5 (original) | 0.642 ± 0.058 [0.55, 0.74] | 0.703 | 0.64 | 6 / 20 |
| coolant | τ5 / 0.01 / 0.25 | 0.620 ± 0.054 [0.53, 0.71] | 0.677 | 0.63 | 1 / 20 |
| probe | **τ20 / 0.01 / 0.25 (selected)** | 0.386 ± 0.079 [0.22, 0.52] | 0.547 | 0.34 | 6 / 20 |
| probe | τ20 / 0.05 / 0.25 | 0.400 ± 0.081 [0.26, 0.56] | 0.553 | 0.37 | 11 / 20 |
| probe | τ10 / 0.001 / 0.25 | 0.294 ± 0.129 [0.08, 0.50] | 0.541 | 0.23 | 3 / 20 |
| probe | τ10 / 0.01 / 0.5 (original) | 0.137 ± 0.094 [0.00, 0.24] | 0.468 | 0.44 | 0 / 20 |

**Coolant**
- The selected and original configurations cannot be told apart: means differ by 0.005 with sd ≈ 0.06. The data do not prefer τ 5 s over 10 s.
- Seven edges appear in ≥ 90% of draws, all in the correct expert direction except the alarm–alarm pair `HP_A_700304 — LT_A_700317`. That pair is extra, explained in §12 of METHODOLOGY by multi-cause scenarios.

**Probe**
- The bootstrap confirms that **τ 20 s with support 0.25 beats the original** (the original never wins; the τ20 / s0.25 configurations win 17 of 20 draws).
- It does **not** resolve α. α = 0.05 wins more draws (11) than the selected α = 0.01 (6), and the mean directed F1 is within one sd.
- The selected probe graph is unstable: mean Jaccard with the full-data graph is only 0.34. Only `MPC_close → MPC_isOpen` (extra) and `MPA_InitPos → MPA_A_701124` (correct) appear in ≥ 85% of draws.
- The selected probe graph should be read as the best candidate under the objective, with low structural stability.

**Caveat.** Duplicated recordings make tests over-confident, so the frequencies of weak edges are biased upward. With 3–8 recordings per fault regime, the bootstrap is coarse.

---

## 7. Alternative CI test: CMIsymb (incomplete)

**Why test it at all.**
- G² uses an asymptotic χ² null. The tigramite docstring calls it valid only asymptotically and recommends above 1,000–2,000 samples. In fault regimes, the effective sample per stratum is small and many cells are empty (dof are reduced for empty margins, following Bishop, Fienberg & Holland 1975).
- CMIsymb with `significance='shuffle_test'` builds the null by permuting X within strata of Z. This is a check on whether G²'s calibration changes the graph.
- **It is not a fix for autocorrelation.** The local permutation shuffles X across time inside a Z stratum. That preserves auto-dependence only through the conditioning set (Runge 2018). G² is not replaced merely because another test exists.

**Compatibility, verified with tigramite 5.2.10.1:**
- multi-dataset dataframes and `mask_type='xyz'` work;
- the shuffle loop needs numba (installed, 0.68.1);
- settings: `sig_samples` 200, seed 0.

**Cost**
- G² takes about 3 s per coolant fault regime. CMIsymb took more than 7 CPU-minutes on the same regime in a pilot (the reason for the 13:21 amendment, which restricted the comparison to fault regimes).
- **Probe:** all 6 fault-regime runs were stopped by hand after 30 min, logged as `stopped (manual)`. The 6 runs at the 3 h timeout on 2 workers would have needed about 9 h. The probe comparison is **not completed**.
- **Coolant:** 4 fault-regime runs at τ 5 s on 2 workers. All four reached the 3 h timeout (exp_22 and exp_28 at 17:13, exp_27 and exp_35 at 20:13) and are logged as `timeout`. The coolant comparison is therefore incomplete as well.
- For comparison, the G² baseline on the same scope gives: coolant directed F1 0.750 / skeleton 0.800 / 10 adj.; probe 0.462 / 0.645 / 16 adj.

**Status:** the CMIsymb comparison does not support any conclusion about graph differences.
- What it shows: the tested configuration (CMIsymb local permutation test, 200 shuffles, single-threaded runs, 3 h budget per regime run) is computationally impractical under the available resources.
- What it does not show: that every permutation-based approach is unsuitable. Fewer shuffles, a faster implementation or more compute could change this.

No CMIsymb result exists for either subsystem.

---

## 8. Parameter justification

Full table with references: `results/tables/12_parameter_justification.csv`. Summary, with the outcome of this study:

| Setting | Original | Controls | Status | Outcome / evidence |
|---|---|---|---|---|
| Algorithm | PCMCI+ | lagged + lag-0 links in autocorrelated series | fixed | not studied (out of scope) |
| CI test | G² (analytic) | conditional independence for binary data | fixed; CMIsymb sensitivity | CMIsymb incomplete (cost); no evidence to replace G² |
| Δ | 0.5 s | grid; 1 lag step | fixed (nb 03 sensitivity: 1 s → Jaccard 0.53) | not studied here |
| τmin | 0 | whether lag-0 links are tested | fixed | most co-changes are at lag 0 (§5.2); τmin = 1 would hide them |
| τmax | 10 s | longest lag; BH family size; 2·τmax sample cut | **tuned** 5/10/20 s; 60 s sensitivity | coolant: 5 = 10 s (identical graph); probe: 20 s chosen in 5/6 folds and 17/20 bootstrap draws; 60 s not retained (sample cut reduces usable fault data) |
| pc_alpha | 0.01 | threshold of every CI test (a regularisation parameter) | **tuned** 0.001/0.01/0.05 | coolant: 0.01 best at every τ; probe: 0.01 selected, but 0.05 and 0.001 win some folds/draws (not resolved) |
| caps dim/px/py | 3/3/3 | conditioning-set size | staged 2/4 (changed together) | 3 is never worse on directed F1 than 2 or 4 for the selected configurations |
| max_combinations, collider rule, conflict resolution | 1, majority, True | PC1 subsets; lag-0 orientation | library defaults | not studied |
| FDR | BH on lagged p per regime | multiplicity of lagged links | fixed; filter on/off sensitivity | filter costs probe F1; kept a priori; no FDR claim for the final graph |
| Support threshold | 0.5 | share of testable regimes needed | **tuned** 0.25/0.5/0.75 | coolant 0.25 = 0.5; probe 0.25 needed, 0.75 collapses the graph |

**Evidence that would justify changing a setting:**
- for τmax: a delay between two *recorded* signals beyond τmax, shown on training data, plus enough recording length left after the 2·τmax cut;
- for α: stability under resampling (Liu et al. 2010 StARS; Meinshausen & Bühlmann 2010) or an out-of-sample criterion (Biza et al. 2020, 2024), rather than reference F1 alone;
- for the CI test: graphs that differ under a permutation-based or otherwise better-calibrated test at feasible cost.

---

## 9. Exact commands

Run from the repository root with `.venv`.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_discovery_study.py timing grid select
```
```bash
PYTHONPATH=src .venv/bin/python scripts/run_discovery_study.py staged bootstrap --staged-part nonci --workers 7
```
```bash
PYTHONPATH=src .venv/bin/python scripts/run_discovery_study.py staged --staged-part ci --subsystems coolant --workers 2
```
```bash
PYTHONPATH=src .venv/bin/python scripts/run_discovery_study.py staged report
```
```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Notes:
- The first invocation started at 13:20:41.
- The probe CI part (`staged --staged-part ci --workers 2`) was started at 13:41:52 and stopped by hand.
- Every regime run is cached by key, so re-running any stage only aggregates.
- Run times: grid 1,175 s; staged non-CI 1,750 s; bootstrap 8,747 s (probe normal-regime runs take 800–1,250 s each).
- **Manifest note:** the original JSON manifest of the first three invocations was corrupted by two processes writing it at the same second, and then deleted by mistake during recovery. The first line of `12_study_manifest.jsonl` is a reconstruction (commit, data timestamps, yaml amendment time, timeline). The manifest is append-only since then.

---

## 10. Run status (completed, timed out, stopped)

Unique regime runs in the cache (`data/cache/discovery_study`). Bootstrap draws reuse identical runs.

| Subsystem | CI test | Stages | ok | timeout (3 h) | stopped by hand | failed | still running |
|---|---|---|---|---|---|---|---|
| coolant | G² | grid, caps, 60 s, bootstrap | 270 | 0 | 0 | 0 | 0 |
| probe | G² | grid, caps, 60 s, bootstrap | 630 | 0 | 0 | 0 | 0 |
| coolant | CMIsymb | CI comparison (4 fault regimes) | 0 | 4 (exp_22, exp_28 at 17:13; exp_27, exp_35 at 20:13) | 0 | 0 | 0 |
| probe | CMIsymb | CI comparison (6 fault regimes) | 0 | 0 | 6 (after 30 min, 14:12) | 0 | 0 |

Every G² configuration in the grid, the staged comparisons and the bootstrap completed. No configuration was omitted. The per-run event log is `results/tables/12_run_events.csv`. The 60 s scores are in §5.2.

## 11. What to adopt

- **Coolant:** keep the original configuration as the primary setting, and report the three tied alternatives. They give the identical graph.
- **Probe:** keep the original configuration as the **baseline**. Present τ 20 s / α 0.01 / support 0.25 as the **reference-guided tuned comparison**, with its instability (α unresolved, mean bootstrap Jaccard 0.34).
- If the tuned probe graph ever becomes the main pipeline graph:
  - regenerate the fold-specific graphs, SCMs and RCA results from it;
  - do not combine new graph scores with downstream results computed on the old graph;
  - do not justify the switch by downstream (RCA) performance.
- No further algorithms or grid extensions are planned. The study is closed; all CMIsymb runs ended by timeout or manual stop.

**Main conclusion:** reference-guided tuning supports the original coolant configuration and improves the probe graph's agreement with the expert reference, while revealing substantial structural instability in the probe graph. It does not establish an optimal or a definitively correct causal graph.

## 12. Limitations

- **Reference-guided bias.** Selection and scoring use the same expert graph, so the selected scores are optimistic. The LOSO folds show selection stability, not independent accuracy. No held-out reference exists.
- **Probe instability.** The selection of τ 20 s / support 0.25 is stable, but α is not, and the selected graph has a mean bootstrap Jaccard of 0.34. Its extra edges (9 of 19 adjacencies) would enter any downstream use.
- **Coarse support threshold.** Most pairs are testable in 1–2 regimes, so the support threshold changes the graph in discrete jumps.
- **Long lags are poorly supported** with recordings of 158–365 s under the 2·τmax cut. The 60 s result is confounded by sample loss.
- **The selection rule was written before the search, not formally preregistered.** The yaml was written before any result, but it is not under version control or externally timestamped, and it was amended once during the run (CMIsymb scope, for runtime).
- **CMIsymb** is incomplete: probe runs stopped by hand, all coolant runs timed out after 3 h.
- **No adjustment for model-selection uncertainty** in the reported F1. Ties (coolant: 4) and per-draw winners are reported instead.
- **No downstream re-run.** The study does not re-run SCM/RCA with the selected probe graph. Choosing that graph because of better RCA would be selection on downstream outcomes. Whether to adopt it as the primary probe graph is a separate decision, and it must be disclosed as reference-guided.

---

## 13. Thesis paragraph (draft)

> The PCMCI+ settings of the primary analysis (G² test, Δ = 0.5 s, τmax = 10 s, αPC = 0.01, conditioning caps of three, Benjamini–Hochberg filtering of lagged links, majority support across regimes) were fixed a priori; PCMCI+ itself prescribes no default values \citep{runge2020}. To examine whether these choices are defensible, we ran a reference-guided search over αPC ∈ {0.001, 0.01, 0.05}, τmax ∈ {5, 10, 20} s and support ∈ {0.25, 0.5, 0.75}, with a selection rule specified before executing the search: directed F1 against the expert graph, then skeleton F1, then the shorter τmax. Because the expert graph is used for selection, the resulting scores are optimistically biased and are reported as reference-guided tuning performance, not as test accuracy \citep{cawley2010,varma2006}. For the coolant subsystem, four configurations tied for the best directed F1 (0.75) and all yield the original graph. For the probe subsystem, the best-performing configuration among the evaluated candidates (τmax = 20 s, αPC = 0.01, support 0.25) raised directed F1 from 0.00 to 0.42, at the cost of more extra edges. Leave-one-scenario-out repetition of the selection and a 20-draw bootstrap consistently favoured τmax = 20 s with support 0.25 but did not resolve αPC, and the probe graph remained structurally unstable (mean bootstrap Jaccard 0.34). Within a staged comparison, conditioning caps of two or four did not improve on three. Under Tigramite's sample-cutting rule, a 60 s maximum lag removed 28–79 % of the fault-regime state changes and was not retained; this does not establish an optimal lag. A permutation-based CMIsymb test was computationally impractical for the tested configuration under the available resources. The Benjamini–Hochberg filter corrects only lagged p-values within each regime after PCMCI+ has oriented the graph, so it is a multiplicity adjustment rather than formal FDR control of the final graph \citep{li2009,strobl2019}.

### New BibTeX entries (not yet in `thesis.bib`)

```bibtex
@article{li2009,
  author  = {Li, Junning and Wang, Z. Jane},
  title   = {Controlling the False Discovery Rate of the Association/Causality Structure Learned with the {PC} Algorithm},
  journal = {Journal of Machine Learning Research},
  volume  = {10},
  pages   = {475--514},
  year    = {2009}
}
@article{strobl2019,
  author  = {Strobl, Eric V. and Spirtes, Peter L. and Visweswaran, Shyam},
  title   = {Estimating and Controlling the False Discovery Rate of the {PC} Algorithm Using Edge-specific P-Values},
  journal = {ACM Transactions on Intelligent Systems and Technology},
  volume  = {10},
  number  = {5},
  pages   = {46},
  year    = {2019}
}
@article{cawley2010,
  author  = {Cawley, Gavin C. and Talbot, Nicola L. C.},
  title   = {On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation},
  journal = {Journal of Machine Learning Research},
  volume  = {11},
  pages   = {2079--2107},
  year    = {2010}
}
@article{varma2006,
  author  = {Varma, Sudhir and Simon, Richard},
  title   = {Bias in Error Estimation when Using Cross-Validation for Model Selection},
  journal = {BMC Bioinformatics},
  volume  = {7},
  pages   = {91},
  year    = {2006}
}
@inproceedings{biza2020,
  author    = {Biza, Konstantina and Tsamardinos, Ioannis and Triantafillou, Sofia},
  title     = {Tuning Causal Discovery Algorithms},
  booktitle = {Proceedings of the 10th International Conference on Probabilistic Graphical Models},
  series    = {PMLR},
  volume    = {138},
  pages     = {17--28},
  year      = {2020}
}
@article{biza2024,
  author  = {Biza, Konstantina and Tsamardinos, Ioannis and Triantafillou, Sofia},
  title   = {Out-of-Sample Tuning for Causal Discovery},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume  = {35},
  number  = {4},
  pages   = {4963--4973},
  year    = {2024}
}
@inproceedings{liu2010,
  author    = {Liu, Han and Roeder, Kathryn and Wasserman, Larry},
  title     = {Stability Approach to Regularization Selection ({StARS}) for High Dimensional Graphical Models},
  booktitle = {Advances in Neural Information Processing Systems 23},
  year      = {2010}
}
@article{meinshausen2010,
  author  = {Meinshausen, Nicolai and B{\"u}hlmann, Peter},
  title   = {Stability Selection},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {72},
  number  = {4},
  pages   = {417--473},
  year    = {2010}
}
@article{kalisch2007,
  author  = {Kalisch, Markus and B{\"u}hlmann, Peter},
  title   = {Estimating High-Dimensional Directed Acyclic Graphs with the {PC}-Algorithm},
  journal = {Journal of Machine Learning Research},
  volume  = {8},
  pages   = {613--636},
  year    = {2007}
}
@article{runge2018,
  author  = {Runge, Jakob},
  title   = {Causal Network Reconstruction from Time Series: From Theoretical Assumptions to Practical Estimation},
  journal = {Chaos},
  volume  = {28},
  number  = {7},
  pages   = {075310},
  year    = {2018}
}
@book{bishop1975,
  author    = {Bishop, Yvonne M. M. and Fienberg, Stephen E. and Holland, Paul W.},
  title     = {Discrete Multivariate Analysis: Theory and Practice},
  publisher = {MIT Press},
  year      = {1975}
}
```

Existing keys used: `runge2019`, `runge2020`, `agresti2013`, `benjamini1995`, `benjamini2001`. Check the BibTeX fields above against the publisher pages before submission.
