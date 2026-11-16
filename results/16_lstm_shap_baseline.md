# LSTM + SHAP diagnosis baseline (stage 16, tuned version)

## Purpose

A common data-driven approach in predictive maintenance is anomaly detection followed by a feature-attribution explanation:
- A recurrent network (LSTM) learns how the machine normally behaves and flags time steps that deviate from its predictions.
- SHAP then ranks the input variables that contribute most to the flagged anomaly.

This baseline answers one question:

> On the same data, how well does the "detect and explain" ranking (LSTM + SHAP) find the injected fault causes, compared with the mechanism-break root cause analysis (RCA) of the causal framework?

It is a baseline for **diagnosis only (RQ2)**. LSTM + SHAP has no causal graph and no intervention semantics: it cannot predict what an action would change. It is therefore **not** compared on intervention analysis or counterfactual explanations.

An earlier, untuned version (commit `9e15f9e`, 12-epoch cap, only the hidden size tuned) gave a probe MAP@3 of 0.24. Those LSTMs were undertrained, so that result is **superseded** by this tuned version and must not be reported.

## Where the code and results are

| What | Path (research repository `causalAI-failure-prevention`) |
|---|---|
| Code | `scripts/lstm_shap_baseline.py` |
| Run command | `PYTHONPATH=src .venv/bin/python scripts/lstm_shap_baseline.py` (41 minutes, 4 parallel workers) |
| Tuning results | `results/tables/16_lstm_shap_selection.csv` |
| Ranking of every run, seed and SHAP variant | `results/tables/16_lstm_shap_runs.csv` |
| SHAP importance per variable (main variant) | `results/tables/16_lstm_shap_importance.csv` |
| Per-scenario results | `results/tables/16_lstm_shap_per_scenario.csv` |
| Summary | `results/tables/16_lstm_shap_summary.csv` |
| SHAP sensitivity check | `results/tables/16_lstm_shap_sensitivity.csv` |
| Settings and versions | `results/tables/16_lstm_shap_config.json` |
| Figures | `results/figures/16_lstm_shap_attribution_example.{pdf,png}`, `results/figures/16_rca_vs_lstm_shap.{pdf,png}` |
| Log | `results/logs/16_lstm_shap.log` |

New packages in `.venv`: PyTorch 2.15.0 and SHAP 0.54.1.

## Shared protocol (the methods share splits, candidates and metrics, but not every input or selection step)

| | Causal framework | LSTM + SHAP |
|---|---|---|
| Input series | 0.5 s binary state series | the same |
| Training data for held-out scenario k | 119 normal training recordings plus all other fault scenarios | the same |
| Manipulated variables in training fault runs | left out of fitting that variable's mechanism | left out of the training loss (the same rule) |
| Unrecorded or unknown states | filled with the most frequent training state | the same, and left out of the loss |
| How settings are chosen | lowest held-out predictive log loss, no fault outcome | lowest predictive loss on the normal validation recordings, no fault outcome |
| Test runs | the runs of scenario k (25 coolant, 34 probe) | the same |
| Candidates and window | recorded non-alarm variables; first alarm onset −60 s to last alarm onset +5 s | the same |
| Metrics | Top-1, Hit@3, Recall@3, MAP@3, MRR, scenario AP@3, chance | the same, averaged over 3 seeds |

## Hyperparameters

| Symbol | Parameter | Meaning | Candidates | Selected (coolant / probe) |
|---|---|---|---|---|
| H | Hidden units | Size of the LSTM memory | 32, 64, 128 | 128 / 32 |
| W | Look-back | History the LSTM reads to predict the next step | 10 s, 20 s (20, 40 steps) | 20 s / 10 s |
| – | Learning rate | Adam step size | fixed | 0.001 |
| – | Batch size | Samples per update | fixed | 512 |
| – | Epochs | Passes over the training data | early stopping on normal validation loss, patience 2 | at most 30 |
| – | Seeds | Independent trainings per fold | fixed | 3 (0, 1, 2) |
| – | Explained steps | Steps of the analysis window whose anomaly score SHAP explains | 10, 20, all (sensitivity only) | 20 most anomalous |
| – | Attribution | How SHAP values are summed per variable | absolute, positive (sensitivity only) | absolute |
| – | SHAP background | Reference windows for expected gradients | fixed | 100 normal-training windows |
| – | SHAP samples | Samples per explanation | fixed | 50 |

**How the LSTM settings were chosen.** Each of the 6 combinations of H and W was trained on the 119 normal training recordings only and compared by its loss on the 26 normal validation recordings. This is the same kind of rule as for the SCM settings: predictive loss, never a fault outcome.

| Subsystem | H | W | Validation loss | Epochs |
|---|---|---|---|---|
| coolant | 32 | 10 s | 0.00921 | 27 |
| coolant | 32 | 20 s | 0.00885 | 30 |
| coolant | 64 | 10 s | 0.00905 | 27 |
| coolant | 64 | 20 s | 0.00883 | 27 |
| coolant | **128** | **20 s** | **0.00872** | 27 |
| coolant | 128 | 10 s | 0.00957 | 16 |
| probe | **32** | **10 s** | **0.00955** | 25 |
| probe | 32 | 20 s | 0.00984 | 20 |
| probe | 64 | 10 s | 0.00970 | 20 |
| probe | 64 | 20 s | 0.00957 | 20 |
| probe | 128 | 10 s | 0.00998 | 14 |
| probe | 128 | 20 s | 0.00982 | 13 |

These losses are not directly comparable with the SCM log losses. They are averaged over all steps and variables rather than per recording, and the SCM also conditions on the current states of the parents. No claim about which model predicts better is made.

**Why the SHAP settings were not tuned.** The validation recordings contain no faults, so there is nothing to tune explanation settings against. Tuning them on the test runs would be selection on the evaluation outcome. They were therefore fixed in advance, and a sensitivity check reports how much the result depends on them.

## How the baseline works

1. **Model.** One LSTM layer and one linear output layer read the last W of all variables and predict the probability that each variable is 1 in the next step. They are trained with binary cross-entropy (Adam, learning rate 0.001, batch size 512).
2. **Anomaly score of a step.** The mean binary cross-entropy of the predicted next states over the variables recorded in the run. A high score means the observed states are unexpected given their history.
3. **SHAP ranking.**
   - SHAP `GradientExplainer` (expected gradients) attributes the anomaly score of the 20 most anomalous steps in the analysis window to the input variables, over all time positions and the observed next row.
   - The background is 100 windows of normal training data.
   - Each candidate's score is its summed absolute attribution, and candidates are ranked by this score.

## Results

### Overall (same 25 coolant and 34 probe held-out runs)

| Subsystem | Method | Top-1 | MAP@3 (run mean) | AP@3 (scenario mean) |
|---|---|---|---|---|
| Coolant | Causal RCA, expert graph | 1.00 | 1.00 | 1.00 |
| | Causal RCA, discovered graph | 1.00 | 1.00 | 1.00 |
| | **LSTM + SHAP** | **1.00** | **1.00** | **1.00** |
| Probe | Causal RCA, expert graph | 0.62 | **0.67** | **0.64** |
| | Causal RCA, discovered graph | **0.65** | 0.65 | 0.63 |
| | **LSTM + SHAP** | 0.56 | 0.59 | 0.52 |
| | Chance | 0.14 | 0.21 | 0.21 |

Seed spread of LSTM + SHAP on probe: MAP@3 0.53 to 0.62. Top-1 is 0.56 in every seed.

### Per held-out scenario (AP@3), see figure `16_rca_vs_lstm_shap`

| Scenario | Causes | Causal, expert | Causal, discovered | LSTM + SHAP | Chance |
|---|---|---|---|---|---|
| coolant (all four) | 2–3 components each | 1.00 | 1.00 | 1.00 | 0.42–0.55 |
| probe exp_1 | MPA_WorkPos | **0.89** | 0.83 | 0.07 | 0.21 |
| probe exp_2_3 | MPA_InitPos | 0.42 | 0.62 | **1.00** | 0.21 |
| probe exp_4 | MPA_WorkPos | **1.00** | 0.72 | **1.00** | 0.21 |
| probe exp_5 | MPA_InitPos | **0.92** | 0.75 | 0.27 | 0.21 |
| probe exp_6 | both sensors | 0.33 | **0.50** | 0.00 | 0.22 |
| probe exp_7 | both sensors | 0.30 | 0.35 | **0.81** | 0.22 |

### SHAP sensitivity check (probe MAP@3; coolant is 0.99–1.00 in every variant)

| Explained steps | Absolute attribution | Positive attribution |
|---|---|---|
| 10 most anomalous | 0.52 | 0.55 |
| 20 most anomalous (main) | **0.59** | 0.65 |
| all window steps | 0.55 | 0.47 |

### Why: what LSTM + SHAP ranks first (main variant, all seeds)

- **Where LSTM + SHAP is better** (exp_2_3, exp_7): it ranks the injected sensor first. In exp_7, where both sensors are manipulated, it finds them at mean ranks 1.7 and 2.3. The causal RCA is weak there, because each injected sensor state is not unusual given its parents.
- **Where LSTM + SHAP fails** (exp_1, exp_5, exp_6): it ranks reacting variables first. These are the inactive state `MP_Inactive`, the cover states and commands, and the arm commands. The injected sensor lands at mean rank 5.7 (exp_1), 3.5 (exp_5), and 4.6 and 7.7 (exp_6).
- **The example run** (figure `16_lstm_shap_attribution_example`, exp_1 run 1, seed 0, chosen by a fixed rule): `MP_Inactive` and `MPC_Closed` receive more attribution than the injected `MPA_WorkPos`, which comes third.
- **Observed:** SHAP ranks the variables whose values make the detector's prediction error large. These can be the cause or its downstream reaction, depending on the scenario.
- **Interpretation (not isolated by an experiment):** the two methods fail in different scenarios.
  - The mechanism-break score fails when the injected state is plausible given its parents.
  - SHAP fails when the reaction of the control logic dominates the prediction error.

## Can the result be trusted?

- **The data comparison is fair.** Training data, test runs, candidates, window and metrics are identical. Every setting was chosen without fault outcomes.
- **The tuning mattered.** Without it, the baseline looked much worse (0.24). The tuned result is the one to report.
- **Remaining uncertainty:**
  - The learning rate was not tuned.
  - The comparison is not a controlled experiment on causal structure. The LSTM predicts from the preceding steps only, its settings are selected differently, and the discovered graph used the expert reference during selection.
  - On probe, the result depends moderately on the fixed SHAP settings (MAP@3 0.47 to 0.65) and on the seed (0.53 to 0.62). The best SHAP variant (0.65) is level with the causal RCA.
- **With 6 probe scenarios and mixed per-scenario results, there is no consistent winner.**
  - The causal RCA has a slightly higher mean and wins 3 scenarios.
  - LSTM + SHAP wins 2.
  - One scenario is tied with the expert graph.

## What this means for the thesis

- **Diagnosis (RQ2).** A tuned LSTM + SHAP baseline on the same data matches the causal RCA on coolant and comes close on probe. The causal RCA has a slightly higher mean (MAP@3 0.67 / 0.65 against 0.59) but no consistent advantage. The thesis should **not** claim that the causal framework diagnoses better than a data-driven baseline.
- **The contribution lies beyond diagnosis.** LSTM + SHAP stops at "these variables contributed to the anomaly". It cannot say what an intervention would change or which allowed action would have prevented the alarms. The causal framework answers these questions with the intervention analysis and the counterfactual explanations. That is where its added value lies.
- **Complementary failure modes.** The two rankings fail in different scenarios. That is an honest finding for the Discussion and a direction for future work, for example combining both scores.
