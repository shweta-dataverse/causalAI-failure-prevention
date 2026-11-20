# Actionable Root Cause Analysis with Causal Discovery and Counterfactual Explanations

Code of the master's thesis *Toward Actionable Root Cause Analysis: Integrating Causal Discovery and
Counterfactual XAI for Predictive Maintenance* (Otto von Guericke University Magdeburg, 2026).

When a machine raises an alarm, maintenance staff need two answers: what caused it, and what could
have prevented it. This framework answers both from the alarm and state logs of an industrial CNC
lathe. It learns a causal graph from the logs and fits a structural causal model (SCM) on that graph.
It then ranks the likely root causes of a fault, simulates the recorded fault injections, and
searches for maintenance actions that would have kept the alarms off.

Each step runs twice: once with a graph learned from the data (the **discovered graph**) and once with
the graph drawn by the dataset's experts (the **expert graph**). The two are compared at every step.
The data is the [causRCA](https://doi.org/10.5281/zenodo.15876410) dataset. The analysis covers its
**coolant** and **probe** subsystems.

```
event logs ─► 01 preprocessing ─► 02 causal discovery ─► 03 SCM fitting
           ─► 04 root cause analysis ─► 05 intervention analysis ─► 06 counterfactual explanations
```

---

## Contents

1. [Tech stack](#1-tech-stack)
2. [Setup](#2-setup)
3. [Quick start](#3-quick-start)
4. [Folder structure](#4-folder-structure)
5. [Pipeline](#5-pipeline)
6. [Evaluation](#6-evaluation)
7. [Configuration](#7-configuration)
8. [Future scope](#8-future-scope)
9. [Data, licence and citation](#9-data-licence-and-citation)

---

## 1. Tech stack

| Purpose | Library | Version |
|---|---|---|
| Language | Python | 3.12 |
| Causal discovery (PCMCI+, G² independence test) | tigramite | 5.2.10.1 |
| Tables and arrays | pandas, NumPy | 3.0.6, 2.5.4 |
| Statistics | SciPy | 1.18.1 |
| Graphs | NetworkX | 3.7 |
| LSTM baseline | PyTorch | 2.15.0 |
| Feature attribution of the baseline | SHAP | 0.54.1 |
| Figures | Matplotlib | 3.11.2 |
| Configuration file | PyYAML | 6.0.3 |

`requirements.txt` lists the exact versions of every library that produced the thesis results.

## 2. Setup

From the repository root, create an environment and install the libraries:

```bash
python3.12 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

The raw dataset (causRCA 1.0.0) is already included in `data/raw/causRCA/`. The results of every step
are stored too, including the PCMCI+ test results. Every command therefore runs in minutes, with no
need to recompute everything.

## 3. Quick start

**Run the whole pipeline** (all six steps in order, about 7 minutes):

```bash
python main.py run
```

**Run one step** of the pipeline. Each step reads what the steps before it saved:

```bash
python main.py run preprocessing
```
```bash
python main.py run discovery
```
```bash
python main.py run scm
```
```bash
python main.py run rca
```
```bash
python main.py run interventions
```
```bash
python main.py run counterfactuals
```

**Run one evaluation.** Each part is run on its own and prints its metrics with one sentence on
what they mean:

```bash
python main.py evaluate discovery
```
```bash
python main.py evaluate scm
```
```bash
python main.py evaluate rca
```
```bash
python main.py evaluate counterfactuals
```
```bash
python main.py evaluate comparisons
```

Every step and part saves its thesis tables to `outputs/tables/` and its thesis figures to
`outputs/figures/`, under the number and title they have in the thesis. Everything printed is also
written to `logs/run.log`.

Adding `--force` to a command recomputes the stored heavy results instead of reusing them:

| What `--force` recomputes | Time |
|---|---|
| PCMCI+ runs of the hyperparameter grid and the discovery bootstrap | about 20 CPU hours (a few hours with 9 parallel workers) |
| LSTM + SHAP training (`evaluate rca`, `evaluate comparisons`) | about 45 minutes |

Approximate run times with the stored results:

| Command | Time |
|---|---|
| `run` (whole pipeline) | 7 min |
| `evaluate discovery` | under 1 min |
| `evaluate scm` | 3 min |
| `evaluate rca` | under 1 min |
| `evaluate counterfactuals` | 1 min |
| `evaluate comparisons` | 4 min |

## 4. Folder structure

```
causalAI-failure-prevention/
├── main.py                      command line: runs a pipeline step or an evaluation part
├── requirements.txt             exact library versions
│
├── config/
│   ├── config.yaml              every path, hyperparameter, seed and threshold
│   └── related_recordings.csv   near-duplicate recordings, kept out of training with their partner
│
├── pipeline/                    the six steps of the framework, run in this order
│   ├── 01_preprocessing.py
│   ├── 02_causal_discovery.py
│   ├── 03_scm_fitting.py
│   ├── 04_root_cause_analysis.py
│   ├── 05_interventions.py
│   └── 06_counterfactuals.py
│
├── evaluation/                  the five evaluation parts, in this order
│   ├── 01_causal_discovery.py
│   ├── 02_scm.py
│   ├── 03_root_cause_analysis.py
│   ├── 04_counterfactuals.py
│   └── 05_comparisons.py
│
├── core/                        the methods that the steps and evaluation parts use
│   ├── data.py                  dataset metadata, expert graphs, data splits and folds
│   ├── preprocessing.py         event logs to state tables
│   ├── discovery.py             regime-wise PCMCI+, aggregation, hyperparameter selection
│   ├── graphs.py                graph comparison, conversion to a DAG, graph drawings
│   ├── scm.py                   binary transition SCM: fitting, simulation, log loss
│   ├── rca.py                   mechanism-break scores and ranking metrics
│   ├── interventions.py         simulated fault injections, allowed actions
│   ├── counterfactuals.py       counterfactual search and its validity
│   ├── lstm_shap.py             LSTM forecaster with SHAP attributions (baseline)
│   └── plots.py                 colours and figure style
│
├── utils/
│   ├── config.py                reads config/config.yaml
│   ├── io.py                    where every file is read from and saved to
│   ├── logger.py                messages on the terminal and in logs/run.log
│   └── errors.py                error messages that say what to run first
│
├── data/
│   ├── raw/causRCA/             the causRCA dataset (event logs, expert graphs, fault annotations)
│   ├── processed/               coolant_states.csv, probe_states.csv, splits.csv
│   ├── graphs/                  discovered graphs, the DAGs used by the SCMs, discovery hyperparameters
│   ├── models/                  fitted SCMs (scm/), their hyperparameters, LSTM models (lstm/)
│   ├── results/                 rankings, simulated fault injections, recommendations,
│   │                            evaluation/ details, comparisons/ for the reruns
│   └── cache/                   PCMCI+ test results, one file per regime run
│
├── outputs/
│   ├── tables/                  every thesis table that comes from code, e.g. table_6_4_...csv
│   └── figures/                 every thesis figure that comes from code, e.g. figure_6_1_...pdf
│
└── notebooks/01_EDA.ipynb       exploration of the dataset
```

**What `data/cache/` contains:** PCMCI+ is the slowest part of the framework. Selecting the discovery
hyperparameters and the bootstrap needs about 950 PCMCI+ runs, one per regime and hyperparameter combination **(about 20 CPU
hours in total)**. Every run is saved as one small file, named after its data, variables and hyperparameters. When the same run is needed
again, it is read from this folder instead of being recomputed.

**What the state tables look like.** `data/processed/coolant_states.csv` has one row per recording
and time step, like Table 5.5 of the thesis:

| recording_id | step | t_s | CLF_A_700307 | CLF_Filter_Ok | CLT_A_700310 | CLT_Level_lt_Min | ... |
|---|---|---|---|---|---|---|---|
| real_op_28610 | 0 | 0.0 | | 1 | | 0 | ... |
| real_op_28610 | 1 | 0.5 | | 1 | | 0 | ... |
| real_op_28610 | 2 | 1.0 | | 1 | | 0 | ... |

Each variable column holds the state (0 or 1) at that step. A cell is empty while the state is still
unknown, before the variable's first event in the recording. Above, the two alarms have not been
recorded yet.

---

## 5. Pipeline

The six steps run in order. Each step reads the files saved by the steps before it.

### 5.1 Preprocessing

Converts the raw event logs into state tables with one row every 0.5 s and one column per variable.

- **Zero-order hold:** each cell keeps the last value the variable reported. Before its first event,
  the cell stays empty and is never filled backwards.
- **Data split:** 119 training, 26 validation and 25 test recordings of normal operation. Fault runs
  are grouped by scenario for leave-one-scenario-out evaluation.
- **Near-duplicates excluded:** two pairs of probe recordings from different scenarios share 634 of
  635 and 989 of 990 rows. Neither recording is used for training while its partner is tested.
- **Verified:** an independent reconstruction from the raw events matches all 11.7 million cells.

```bash
python main.py run preprocessing
```

**Output:** `data/processed/coolant_states.csv`, `probe_states.csv`, `splits.csv`

### 5.2 Causal discovery

Learns a causal graph from the state tables with PCMCI+ and the G² test.

- **One run per regime:** normal operation and every fault scenario are analysed separately, because
  in pooled data, a variable missing from some recordings would look independent of the others.
- **Fault runs are essential:** only 3 of the 15 coolant variables are usable in normal operation,
  compared with 7 in exp_22 and 8 in exp_28.
- **Aggregation:** a false discovery rate filter removes weak lagged links, and a link is kept when
  enough regimes support it.
- **Hyperparameters:** maximum lag, significance level α and support threshold are selected on a grid
  by agreement with the expert graph, separately for every evaluation fold.

```bash
python main.py run discovery
```
```bash
python main.py evaluate discovery
```

**Output:** `data/graphs/discovered_graphs.csv`, `discovery_settings.csv`

### 5.3 SCM fitting

Fits a binary transition SCM on the expert graph and on the discovered graph.

- **DAG conversion:** undirected edges are removed, and every cycle is broken at its weakest edge.
- **Mechanisms:** the next state of each variable depends on its previous state and on its parents,
  for example `LP_Pump_On` on `LP_Pump_Ok` in the expert graph.
- **Hyperparameters:** a pseudo-count *a* keeps unseen transitions above probability zero, and an
  extra lag captures how long a variable has stayed in its state. Both are selected on inner folds,
  so the held-out scenario is never used.
- **Coolant:** both graphs select *a* = 0.01 and an extra lag of four steps (2 s).

```bash
python main.py run scm
```
```bash
python main.py evaluate scm
```

**Output:** `data/graphs/scm_graphs.csv`, `dag_removals.csv`, `data/models/scm_settings.csv`,
`data/models/scm/*.pkl`

### 5.4 Root cause analysis

Ranks the candidate variables of each fault run by how strongly they break their learned mechanism.

- **Surprise score:** the negative log-likelihood of the observed state under the variable's
  mechanism, taken at its maximum in the analysis window.
- **Analysis window:** from 60 s before the first alarm onset to 5 s after the last one, for example
  40.5–105.5 s in run 1 of exp_6, whose alarm activates at 100.5 s.
- **Coolant causes stand out:** they never change in the training data, so their first change scores
  up to 16.43 nats, a probability of about 1 in 14 million.

```bash
python main.py run rca
```
```bash
python main.py evaluate rca
```

**Output:** `data/results/rca_rankings.csv`, `rca_runs.csv`

### 5.5 Intervention analysis

Simulates every recorded fault injection in the SCM and compares the simulated alarms with the
recorded ones.

- **do-operator:** the injected variable is set to its recorded fault value from its observed onset,
  and each run is simulated 200 times.
- **37 of 56 alarm activations can be simulated:** the other 19 belong to three alarms that occur in
  only one coolant scenario, so no training data exists for them when that scenario is held out.

```bash
python main.py run interventions
```
```bash
python main.py evaluate counterfactuals
```

**Output:** `data/results/replay.csv`

### 5.6 Counterfactual explanations

Searches for the smallest set of maintenance actions that would have prevented the alarms in a
recorded coolant fault.

1. **Abduction:** infer the noise of the SCM from the observed run.
2. **Action:** keep selected components healthy from the step at which their fault was detected, for
   example `do(HP_Pump_Ok = 1)`. The candidates come from the root cause ranking.
3. **Prediction:** simulate the run again and check whether the alarms stay off.

- **Validity:** a set is valid when the alarms stay off in at least 90% of the simulations, both in
  the search (200 simulations) and in 2,000 independent simulations.
- **Refitting:** every valid set is checked again on five SCMs refitted on bootstrap samples of the
  training data.

```bash
python main.py run counterfactuals
```
```bash
python main.py evaluate counterfactuals
```

**Output:** `data/results/recommendations.csv`, `tried_action_sets.csv`, `recommendation_refits.csv`

---

## 6. Evaluation

Each part reproduces the thesis results exactly.

### 6.1 Causal discovery

Compares the discovered graph with the expert graph. Random graphs with the same number of edges provide
a chance reference (random baseline).

```bash
python main.py evaluate discovery
```

| Metric | Coolant | Coolant, random graphs | Probe | Probe, random graphs |
|---|---|---|---|---|
| Skeleton F1 | 0.80 | 0.09 | 0.59 | 0.30 |
| Directed F1 | 0.75 | 0.05 | 0.42 | 0.15 |
| SHD | 6 | 18.6 | 18 | 26.2 |

- **Coolant: 8 of the 10 expert connections are recovered, and every directed edge is correctly
  oriented.**
- **Every connection with lagged evidence is correct.** All errors occur among connections found only
  within the same time step.
- **One missed edge cannot be learned from the data:** `LP_Pump_Ok` → `LP_Pump_On` is absent because
  the pump command is already off in all 12 pump-fault intervals.
- **Probe: 10 of the 15 expert connections are recovered, but only 7 of the 18 directed edges are
  correctly oriented.**
- **Under bootstrap resampling (20 draws), the coolant graph changes little (mean Jaccard 0.68),
  the probe graph considerably (0.34).**

Agreement is measured against the expert graph that also selected the hyperparameters, so it is not
an independent test of accuracy.

### 6.2 SCM

Measures how well each SCM predicts the next state, as log loss in nats (lower is better). *At changes*
counts only the steps in which a variable changes its state.

```bash
python main.py evaluate scm
```

| Subsystem | Data | Expert log loss | Discovered log loss | Expert, at changes | Discovered, at changes |
|---|---|---|---|---|---|
| Coolant | Test | 0.013 | 0.013 | 4.22 | 4.21 |
| Coolant | Fault scenarios | 0.174 | 0.174 | 2.98 | 2.88 |
| Probe | Test | 0.009 | 0.009 | 1.63 | 1.79 |
| Probe | Fault scenarios | 0.022 | 0.026 | 2.09 | 2.62 |

- **On normal test data, both graphs reach the same log loss** (0.013 for coolant, 0.009 for probe).
  Only three coolant variables change during normal operation.
- **On held-out fault scenarios, the expert graph predicts probe state changes clearly better** (2.09
  against 2.62 nats). For coolant, the discovered graph is slightly better (2.88 against 2.98).
- **The predicted probabilities are well calibrated:** the quantile residuals on normal data have a
  standard deviation of 1.00, the value expected from a correct model.
- **Refitting the coolant SCM on bootstrap samples changes the log loss at changes by only
  0.05–0.06 nats.**

### 6.3 Root cause analysis

Measures how high the injected causes are ranked. Top-1 counts the runs with an injected cause in
first place, and MAP@3 rewards all injected causes among the first three. LSTM + SHAP is a baseline
without a causal graph. Chance (random baseline) is the expected score from randomly ranking the same candidates.

```bash
python main.py evaluate rca
```

| Subsystem | Method | Top-1 | MAP@3 | Scenario avg. AP@3 |
|---|---|---|---|---|
| Coolant | Causal, both graphs | 1.00 | 1.00 | 1.00 |
| Coolant | LSTM + SHAP | 1.00 | 1.00 | 1.00 |
| Coolant | Chance (random baseline) | 0.47 | 0.47 | 0.47 |
| Probe | Causal, expert graph | 0.62 | 0.67 | 0.64 |
| Probe | Causal, discovered graph | 0.53 | 0.63 | 0.57 |
| Probe | LSTM + SHAP | 0.56 | 0.58 | 0.51 |
| Probe | Chance (random baseline) | 0.14 | 0.21 | 0.21 |

- **Coolant: every method ranks all injected causes at the top**, because these causes never change
  during normal operation. The median score gap to the next candidate is 7.42 nats.
- **Probe: the expert graph ranks best (MAP@3 0.67), followed by the discovered graph (0.63) and
  LSTM + SHAP (0.58).** The probe sensors also change during normal machining, which makes their
  faults harder to detect.
- **No significant difference between the two graphs:** their 95% bootstrap intervals of MAP@3
  overlap, [0.54, 0.79] and [0.49, 0.76].
- **The selected SCM hyperparameters bring most of the improvement:** +0.19 MAP@3 for the expert
  graph and +0.16 for the discovered graph.
- **LSTM + SHAP shows complementary strengths, not consistent superiority:** lower MAP@3 than both
  causal models, but a higher Top-1 than the discovered graph (0.56 against 0.53). It leads in exp_2_3
  and exp_7, and its MAP@3 varies between 0.54 and 0.63 across training seeds.
- **SHAP often ranks the machine's reaction above the injected cause:** in run 1 of exp_1, the
  faulty arm sensor `MPA_WorkPos` ranks only sixth, behind reacting
  variables such as `MP_Inactive` and `MPC_Closed`.
- **Only the causal framework continues beyond diagnosis:** LSTM + SHAP gives rankings, but no
  intervention analysis or counterfactual recommendations.

### 6.4 Interventions and counterfactual explanations

Tests whether the SCM reproduces the recorded alarms and how many coolant runs receive a valid
recommendation.

```bash
python main.py evaluate counterfactuals
```

| Coolant, 25 held-out runs | Expert graph | Discovered graph |
|---|---|---|
| Alarm activations reproduced (with training data) | 37 of 37 | 37 of 37 |
| Alarm onset within one step (0.5 s) | 25 of 37 | 25 of 37 |
| Valid recommendations (2,000 independent simulations) | 19 / 25 | 19 / 25 |
| Mean validity | 0.933 | 0.936 |
| Alarm coverage (activations with training data) | 37 / 56 | 37 / 56 |
| Valid on all five refitted SCMs | 19 / 19 | 19 / 19 |

- **The SCM reproduces all 37 evaluated alarm activations**, each in at least 199 of 200 simulations,
  with a median onset error of 0.5 s and a maximum of 2.0 s.
- **19 of 25 runs receive a valid recommendation, identical for both graphs:** keep the high-pressure
  pump healthy (exp_27), and keep the coolant filter and the low-pressure pump healthy (exp_28,
  exp_35).
- **Every recommended action targets an injected cause** (action precision 1.00).
- **The discovered graph gives the same recommendations as the expert graph**, because none of its
  errors lie on the paths from components to the evaluated alarms.
- **exp_22 stays below the 0.90 threshold (0.84–0.87):** the selected extra lag makes its long alarm
  activity unlikely. With the original SCM hyperparameters, all six runs are valid.
- **All 19 recommendations remain valid on five SCMs refitted on bootstrap samples.**

Validity is predicted by the SCM. It is not observed prevention on the machine.

### 6.5 Comparisons

Reruns the pipeline with the original hyperparameters and, for probe, without the near-duplicate
exclusion.

```bash
python main.py evaluate comparisons
```

- **Probe: hyperparameter selection raises the discovered graph's directed F1 from 0.00 to 0.42** and
  the expert graph's MAP@3 from 0.48 to 0.67.
- **Coolant: graph agreement and diagnosis are unchanged, but 19 instead of 25 runs receive a valid
  recommendation.**
- **Excluding the near-duplicates leaves the expert graph's MAP@3 unchanged** and changes the
  discovered graph and LSTM + SHAP by at most 0.02.

---

## 7. Configuration

All paths and hyperparameters are set in `config/config.yaml`:
- the file locations
- the subsystems
- the 0.5 s time grid and the data split fractions
- the discovery hyperparameter grid and its selection rule
- the SCM pseudo-counts and extra lags
- the root cause analysis window
- the numbers of simulations
- the 0.90 validity threshold
- the LSTM hyperparameter grid
- the random seeds

Changing a value there changes the run. No value is hidden in the code.

## 8. Future scope

- **Compare discovery methods.** Run other methods, including event-sequence approaches, on the same
  data splits. Judge them by how they change diagnosis and recommendations, not only by graph
  accuracy.
- **Improve temporal modelling.** Model recovery and alarm resets explicitly. Compare hyperparameters
  selected for prediction with hyperparameters selected for intervention behaviour and for
  recommendations that remain valid under refitting. Try other noise assumptions and bounds on the
  counterfactual outcomes.
- **Extend and validate actions.** Add maintenance cost, timing and operating constraints. Make
  recommendations from only the information available at the first alarm. Test the predicted
  prevention in controlled digital-twin experiments, then against recorded maintenance outcomes on
  real machines.

## 9. Data, licence and citation

The causRCA dataset is by Fraunhofer IWU and is published under the Apache 2.0 licence. Its own README
and licence are in `data/raw/causRCA/`.

> C. W. Mehling et al. causRCA: Real-World Dataset for Causal Discovery and Root Cause Analysis in
> Machinery. Zenodo, 2025. https://doi.org/10.5281/zenodo.15876410

The code is released under the MIT licence (`LICENSE`).

---

**Author:** Shweta Bambal, Otto von Guericke University Magdeburg
