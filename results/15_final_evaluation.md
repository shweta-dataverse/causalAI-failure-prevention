# Stage 15: final evaluation with the selected settings (main pipeline of the thesis)

**Commands** (repository root, `.venv`):

```bash
PYTHONPATH=src .venv/bin/python scripts/selected_settings.py
```
```bash
PYTHONPATH=src .venv/bin/python scripts/final_evaluation.py main selected
```
```bash
PYTHONPATH=src .venv/bin/python scripts/compare_original_selected.py
```
```bash
PYTHONPATH=src .venv/bin/python scripts/edge_diagnostics.py
```
```bash
PYTHONPATH=src .venv/bin/python scripts/probe_rca_decomposition.py
```
```bash
PYTHONPATH=src .venv/bin/python scripts/normal_diagnostics.py
```
```bash
PYTHONPATH=src .venv/bin/python scripts/thesis_figures.py
```

- Runtime: selection about 4 min (discovery reuses the stage-12 run cache); final evaluation 87 s.
- Checks: 0 failed, in `15_checks.csv`. Every fold's graph and SCM settings were selected without its outer scenario (check per fold).
- Tests: 91 passed.
- Stage 14 (original configuration) is kept unchanged as the comparison.

## Reconciliation: what the main pipeline uses

| Component | Selected configuration or selection rule | Used in main code? | Main thesis table | Dependent results regenerated? |
|---|---|---|---|---|
| Coolant discovery | Stage-12 rule per scope. All data: τ5 s / α0.01 / κ0.5 (tie-break winner of 4 ties). Its graph is identical to the original τ10 graph. Folds: exp_22 τ5/0.5, exp_27 τ10/κ0.75 (graph differs), exp_28 τ10/0.5, exp_35 τ5/0.5 | yes (`15_coolant_discovery_edges.csv`) | Tab. 5.1, A.2, 6.1 | yes |
| Probe discovery | All data: τ20 / α0.01 / κ0.25. Folds: same, except exp_2_3 κ0.5, exp_4 α0.05, exp_6 τ10/α0.001 | yes | Tab. 5.1, A.2, 6.1 | yes |
| Inner-fold graphs (SCM selection) | Same rule without k and j (54 scopes) | yes | Sec. 5.4 | yes |
| SCM settings | Lowest held-out one-step log loss of non-manipulated variables, same rule for both graphs. Track N: normal validation. Fold k: inner LOSO without k. Ties go to a=0.5 with no lag | yes (`15_scm_selection.csv`) | Tab. 5.1, A.2 | yes: prediction, RCA, intervention analysis, counterfactuals, refits |
| Stage-13 AP@3 rule | Sensitivity only (original graphs) | no | App. A.4 | n/a |

Why log loss and not AP@3:
- The SCM is a probabilistic transition model, and every downstream task uses it.
- Log loss uses no injected-cause labels, so RCA is not tuned to its own metric.
- The rule was fixed after stage 13 had been inspected. The thesis discloses this.

## Headline results (selected vs original, `15_original_vs_selected.csv`)

**Coolant**
- **Graph:** unchanged, dir. F1 0.75.
- **Prediction:**
  - normal test change-LL 4.45 → 4.22;
  - fault LL 0.138 → 0.174 (exp_22 pump alarm), fault change-LL 3.04 → 2.98 / 2.88.
- **RCA:** 1.00.
- **Replay:** onset max error 3.0 → 2.0 s.
- **Counterfactuals:**
  - valid **25/25 → 19/25** for both graphs; all 6 failures are in exp_22;
  - refit minimum 0.88 → 0.78.

**Probe**
- **Graph:** dir. F1 0.00 → 0.42; the SCM DAG has 1 → 17 edges.
- **Prediction (fault change-LL, expert / discovered):** 2.11 / 5.21 → 2.05 / 2.93.
- **RCA:**
  - MAP@3 expert 0.48 → 0.67, discovered 0.56 → 0.65;
  - Top-1 expert 0.44 → 0.62, discovered 0.24 → 0.65.

## Explanations established in this stage

1. **Missed coolant edges** (`15_coolant_missing_edges.csv`).
   - **LP_Pump_Ok → LP_Pump_On.** LP_Pump_On is 0 in all 12 fault intervals, so there is no co-change. The edge is dropped by the unconditional test (p ≥ 0.14). No evaluated setting and no bootstrap draw recovers it.
   - **LT_Level_Ok → LT_A_700317.** The edge is separated by LT_Pump_Ok(t), which drops in the same step in 5 of 6 runs. Only α = 0.05 finds it, and the BH filter removes it. With the filter off (α 0.05, τ10, κ0.5), 9 of 10 adjacencies are found, but with 18 adjacencies instead of 10.
2. **Extra coolant edges** (`15_coolant_extra_alarm_edges.csv`).
   - The two alarms reset in the same step in 6/6 and 12/12 runs.
   - Their onsets are never within one step of each other.
3. **Probe RCA decomposition** (`15_probe_rca_decomposition.csv`).
   - With the selected graph and default mechanisms, the discovered model ranks like the expert model: MAP 0.48 vs 0.48, 28 vs 30 orders.
   - The selected SCM settings add +0.17 to +0.19 MAP@3 to both graphs.
   - The original discovered graph's MAP 0.56 came from 3 distinct rankings.
4. **Counterfactual failure in exp_22.**
   - In exp_22, HP_A stays on for about 40 s after the pump recovers, until the joint reset. In exp_27 (training) it clears within 1 s.
   - With a = 0.01 and lag 2, the probability that the alarm stays on with a healthy pump is 0.002, against 0.46 under the original settings.
   - Abduction therefore confines the noise, and the counterfactual alarm activates after the recovery.
   - HP_A loss in exp_22 rises from 407 to 3,263 nats.
   - The partial sets add do(LT_Level_Ok=1) through Monte Carlo noise (no path to HP_A). This inflates the action count and recall.

The selection rules were not changed after these outcomes were seen.

## Why 25/25 became 19/25 (`scripts/cf_exp22_settings.py`, `15_coolant_exp22_settings.csv`)

**Same procedure.** The counterfactual configuration of stages 14 and 15 is identical (`14_config.json` = `15_config.json` for counterfactuals, replay and rca):
- threshold 0.90;
- 200 selection worlds with seed 0;
- top-k 3 and at most 3 actions;
- 2,000 independent worlds from entropy 20261109;
- coverage = activated alarms with a fitted mechanism.

`counterfactuals.py`, `interventions.py` and `scm.py` did not change between b6bb0cf and 7845b78.

**What changed in the exp_22 fold**
- **Discovery:** τ 10 → 5 s, but the graph is identical, so discovery has no effect here.
- **SCM:** a 0.5, no lag → a 0.01, extra lag 2 steps, for both graphs. Selected by the inner log loss on exp_27, exp_28 and exp_35 only.

**Attribution** (fixed action do(HP_Pump_Ok=1), same streams, expert graph; discovered within 0.01):

| SCM setting, exp_22 fold | p(alarm stays on, pump healthy) | validity | valid | failures after pump recovery |
|---|---|---|---|---|
| a 0.5, no lag (original) | 0.46 | 0.94–0.96 | 6/6 | 48% |
| a 0.01, no lag | 0.46 | 1.00 | 6/6 | 17% |
| a 0.5, lag 2 | 0.07 | 0.79–0.83 | 0/6 | 86% |
| a 0.01, lag 2 (selected) | 0.002 | 0.83–0.87 | 0/6 | 99% |

The HP_A log loss in exp_22 follows the same pattern: 407 / 406 / 1,354 / 3,263 nats.

**Traced cause: the extra lag.**
- The lag creates a "pump healthy for ≥2 steps" configuration. In training (exp_27 only for HP_A), the alarm is active in it in 0 of 6 steps.
- In exp_22 the alarm stays on for about 40 s after the pump recovers.
- Abduction therefore confines the noise of these steps, and the counterfactual alarm fires after the recovery.

**Not isolated (interpretation):**
- why the alarm latches until the joint reset in exp_22;
- whether other lags would avoid the effect.

Settings were not changed after this result.
