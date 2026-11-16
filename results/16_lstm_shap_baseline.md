# LSTM + SHAP diagnosis baseline (stage 16)

## Purpose

A common data-driven approach in predictive maintenance is anomaly detection followed by a feature-attribution explanation. The approach is:
- A recurrent network (LSTM) learns the normal behaviour of the machine.
- It flags time steps that deviate from its predictions.
- SHAP then ranks the input variables that contribute most to a flagged anomaly.

This baseline answers one question:

> On the same data, how well does the "detect and explain" ranking (LSTM + SHAP) find the injected fault causes, compared with the mechanism-break root cause analysis (RCA) of the causal framework?

It is a baseline for **diagnosis only (RQ2)**. LSTM + SHAP has no causal graph and no intervention semantics. It cannot predict what an action would change, so it is **not** compared on intervention analysis or counterfactual explanations.

## Where the code and results are

| What | Path (research repository `causalAI-failure-prevention`) |
|---|---|
| Code | `scripts/lstm_shap_baseline.py` |
| Run command | `PYTHONPATH=src .venv/bin/python scripts/lstm_shap_baseline.py` |
| Size selection | `results/tables/16_lstm_shap_selection.csv` |
| Ranking of every run and seed | `results/tables/16_lstm_shap_runs.csv` |
| Per-scenario results | `results/tables/16_lstm_shap_per_scenario.csv` |
| Summary | `results/tables/16_lstm_shap_summary.csv` |
| Settings and versions | `results/tables/16_lstm_shap_config.json` |
| Log | `results/logs/16_lstm_shap.log` |

New packages in `.venv`: PyTorch 2.15.0 and SHAP 0.54.1.

## What was implemented, step by step

### 1. Input data: identical to the causal framework

- **Series:** the same 0.5 s binary state series (`data/processed/{coolant,probe}_dt0.5`) that the SCM uses.
- **Missing states:** a variable that is not recorded in a run, or whose state is still unknown, is filled with its most frequent training state, exactly as the SCM fills it. These cells are left out of the training loss.

### 2. Training data: identical to the causal framework in every fold

The evaluation is leave-one-scenario-out, as for the SCM: coolant has 4 folds and probe has 6. For a held-out scenario k:
- **Training recordings:** the 119 normal training recordings plus all runs of the other fault scenarios. This is the same set as `data.fold_training_ids(splits, k)`, which the SCM uses.
- **Manipulated variables:** in a fault run, the manipulated variables are left out of the training loss. The SCM likewise leaves them out when fitting their own mechanism, because an injection replaces the normal behaviour of that variable.
- **The held-out scenario k** is never used for training, size selection or early stopping.

### 3. The model

- **Network:** one LSTM layer followed by a linear output layer. It reads the last 40 steps (20 s, the same as the longest discovery lag) of all variables and predicts the probability that each variable is 1 in the next step.
- **Loss:** binary cross-entropy over the non-masked variables.
- **Training:** Adam with learning rate 0.001 and batch size 512, for at most 12 epochs. Training stops early when the loss on the 26 normal validation recordings no longer decreases, and the best epoch is kept.
- **Seeds:** 3 (0, 1, 2). All metrics are averaged over the seeds.

### 4. Choosing the model size, without any fault outcome

- **Candidates:** 32 or 64 hidden units.
- **How chosen:** each candidate is trained on the 119 normal training recordings only and compared by its loss on the 26 normal validation recordings.
- **Result:** 64 for both subsystems.

| Subsystem | Hidden units | Validation loss |
|---|---|---|
| coolant | 32 | 0.01051 |
| coolant | 64 | **0.01013** |
| probe | 32 | 0.01079 |
| probe | 64 | **0.01007** |

For comparison, the coolant SCM reaches a normal validation log loss of 0.0101, so the LSTM is an equally good predictor of normal operation.

### 5. Anomaly score

For each time step t of a held-out run, the anomaly score is the mean binary cross-entropy of the LSTM's prediction for step t. It is averaged over the variables recorded in that run. A high score means the observed states are unexpected given the last 20 s.

### 6. SHAP variable ranking

- **Window and candidates:** the same alarm-anchored analysis window as the causal RCA (first alarm onset minus 60 s to last alarm onset plus 5 s), and the same candidates (recorded non-alarm variables).
- **Explained steps:** the **20 most anomalous steps** in the window, i.e. the detected anomalies.
- **Explainer:** SHAP `GradientExplainer` (expected gradients), with 50 samples. The background is 100 random windows from the normal training recordings.
- **What SHAP explains:** the anomaly score. Its input is the 40 history rows plus the observed next row, so a variable's own unexpected value can also receive attribution.
- **Variable score:** the sum of absolute SHAP values over the explained steps and all time positions. Candidates are ranked by this score, with ties broken by name.

### 7. Metrics: identical to the causal RCA

- **Per run:** Top-1, Hit@3, Recall@3, AP@3 and MRR, plus the chance level of a random ranking of the same candidates.
- **Aggregation:** averaged over seeds per run, then reported as the run mean (MAP@3) and the scenario mean (scenario-macro AP@3). The minimum and maximum over seeds are also reported.

## Honest differences from the causal framework

| Point | Causal framework | LSTM + SHAP | Effect |
|---|---|---|---|
| Size or setting selection | SCM settings chosen per fold by inner leave-one-scenario-out log loss | LSTM size chosen once by normal validation loss | The baseline's choice uses no fault data at all; this does not favour the causal method |
| Randomness | Deterministic ranking | Depends on the training seed | Three seeds are reported, with their spread |
| What is explained | Each variable's own mechanism break | The detector's anomaly score | This is the conceptual difference the comparison tests |

## Results

Run time: 64 minutes in total, covering 10 folds × 3 seeds plus the size selection. No error occurred.

### Overall (same 25 coolant and 34 probe held-out runs as the causal RCA)

| Subsystem | Method | Top-1 | MAP@3 (run mean) | AP@3 (scenario mean) | Chance MAP@3 |
|---|---|---|---|---|---|
| Coolant | Causal RCA, expert graph | 1.00 | 1.00 | 1.00 | 0.47 |
| | Causal RCA, discovered graph | 1.00 | 1.00 | 1.00 | |
| | **LSTM + SHAP** | **1.00** | **0.91** (seeds 0.84–1.00) | **0.91** | |
| Probe | Causal RCA, expert graph | 0.62 | 0.67 | 0.64 | 0.21 |
| | Causal RCA, discovered graph | 0.65 | 0.65 | 0.63 | |
| | **LSTM + SHAP** | **0.21** | **0.24** (seeds 0.12–0.30) | **0.21** | |

### Per held-out scenario (AP@3)

| Scenario | Causal, expert | Causal, discovered | LSTM + SHAP | Chance |
|---|---|---|---|---|
| coolant exp_22 | 1.00 | 1.00 | 0.78 | 0.42 |
| coolant exp_27 | 1.00 | 1.00 | 0.95 | 0.55 |
| coolant exp_28 | 1.00 | 1.00 | 1.00 | 0.49 |
| coolant exp_35 | 1.00 | 1.00 | 0.91 | 0.43 |
| probe exp_1 | 0.89 | 0.83 | 0.00 | 0.21 |
| probe exp_2_3 | 0.42 | 0.62 | 0.38 | 0.21 |
| probe exp_4 | 1.00 | 0.72 | 0.75 | 0.21 |
| probe exp_5 | 0.92 | 0.75 | 0.00 | 0.21 |
| probe exp_6 | 0.33 | 0.50 | 0.00 | 0.22 |
| probe exp_7 | 0.30 | 0.35 | 0.12 | 0.22 |

The causal RCA is at least as good as LSTM + SHAP in every scenario, with both graphs, and better in 9 of 10 scenarios with the expert graph.

### Why: what LSTM + SHAP ranks first

**Probe.** The top-ranked variable is the cover close command `MPC_close` in 71 of the 102 seed-runs. It is never an injected cause: every probe scenario manipulates only the arm position sensors `MPA_InitPos` and/or `MPA_WorkPos`. The true causes end up on average at rank 6–8 of 9 in exp_1, exp_5 and exp_6, which is worse than a random ranking (average rank 5).
- **Observed:** the SHAP ranking points to the control commands that react to the faulty sensor, the symptoms, and not to the sensor whose behaviour was changed.
- **Interpretation (not isolated by an experiment):** when a sensor is manipulated, the control logic changes its command sequence. These command deviations dominate the anomaly score, so SHAP attributes most of the anomaly to them. SHAP explains *where the anomaly shows*, not *where it starts*.

**Coolant.** The first-ranked variable is always an injected cause, because these causes never change in normal operation. Any detector flags their first change. The MAP@3 is below 1.00 because in the scenarios with two or three causes, the *second* cause is ranked behind a downstream symptom. In exp_22, the level fault `LT_Level_Ok` is on average at rank 3, behind the pump-off state `HP_Pump_isOff`. In exp_27, `HP_Pump_Ok` is at rank 2.3 on average.

### What this means for the thesis

- **Same data, same runs, same metrics.** The causal mechanism-break RCA locates the injected causes far better on probe (MAP@3 0.67 or 0.65, against 0.24 at chance level). On coolant it is slightly better (1.00 against 0.91).
- **Why this supports the causal approach (observed):** SHAP explains the anomaly detector, so it ranks the variables in which the fault becomes visible. The mechanism-break score asks whether a variable behaves unusually *given its causes*, which separates a changed mechanism from its downstream effects.
- **Limits:**
  - one LSTM architecture, one explanation setup and 3 seeds;
  - the variation over seeds is large for probe (MAP@3 0.12–0.30);
  - like the causal RCA, both methods remain weak in the two-sensor scenarios exp_6 and exp_7.
- **Unchanged by this baseline:** LSTM + SHAP cannot predict intervention effects or evaluate preventive actions; only the causal framework can.
