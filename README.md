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
5. [The pipeline, step by step](#5-the-pipeline-step-by-step)
6. [The evaluation, part by part](#6-the-evaluation-part-by-part)
7. [Thesis tables and figures](#7-thesis-tables-and-figures)
8. [Settings](#8-settings)
9. [Future scope](#9-future-scope)
10. [Data, licence and citation](#10-data-licence-and-citation)

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
| Settings file | PyYAML | 6.0.3 |

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

**Run one evaluation part.** Each part is run on its own and prints its metrics with one sentence on
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
| PCMCI+ runs of the settings grid and the discovery bootstrap | about 20 CPU hours (a few hours with 9 parallel workers) |
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
│   ├── config.yaml              every path, setting, seed and threshold
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
│   ├── discovery.py             regime-wise PCMCI+, aggregation, selection of the settings
│   ├── graphs.py                graph comparison, conversion to a DAG, graph drawings
│   ├── scm.py                   binary transition SCM: fitting, simulation, log loss
│   ├── rca.py                   mechanism-break scores and ranking metrics
│   ├── interventions.py         replay of fault injections, allowed actions
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
│   ├── graphs/                  discovered graphs, the DAGs used by the SCMs, discovery settings
│   ├── models/                  fitted SCMs (scm/), their settings, LSTM models (lstm/)
│   ├── results/                 rankings, replays, recommendations; evaluation/ details;
│   │                            comparisons/ for the reruns with other settings
│   └── cache/                   PCMCI+ test results, one file per regime run
│
├── outputs/
│   ├── tables/                  every thesis table that comes from code, e.g. table_6_4_...csv
│   └── figures/                 every thesis figure that comes from code, e.g. figure_6_1_...pdf
│
└── notebooks/01_EDA.ipynb       exploration of the dataset
```

**Why `pipeline/` and `core/` are separate.** A file in `pipeline/` is one step: it reads the step's
input, calls the methods, and saves the step's results, tables and figures. A file in `core/`
holds the method itself, for example how the SCM is fitted or how a root cause is scored. The same
method is used by several steps and evaluation parts. For example, the root cause scores are needed by
the root cause step, the counterfactual step and three evaluation parts. Keeping each method in one
place means all of them use exactly the same code.

**What `data/cache/` is for.** PCMCI+ is the slowest part of the framework. Selecting the discovery
settings and the bootstrap needs about 950 PCMCI+ runs, one per regime and setting (about 20 CPU
hours in total). Every run is saved as one small file, named after its data, variables and settings. When the same run is needed
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

## 5. The pipeline, step by step

Every step reads the files the earlier steps saved, so the steps run in this order. If an input is
missing, the step says which command to run first.

### Step 1: Preprocessing

```bash
python main.py run preprocessing
```

The step turns the raw event logs into state tables. The state of every variable is read every 0.5 s,
keeping the last recorded value of each variable (zero-order hold). Every cell is checked against the
raw events. Normal recordings are split by number into training, validation and test (70/15/15). Fault
runs are grouped by scenario, so that each scenario can be held out in turn (leave-one-scenario-out).
The step also checks for near-duplicate recordings, which must not be in training while their partner
is being tested.

- **Input:** `data/raw/causRCA/`
- **Output:** `data/processed/coolant_states.csv`, `probe_states.csv`, `splits.csv`
- **Thesis items:** Tables 5.1–5.6, Table A.4, Figure 5.1

### Step 2: Causal discovery

```bash
python main.py run discovery
```

The step learns a causal graph from the state tables. PCMCI+ with the G² test runs separately in
each regime: normal operation forms one regime, and each fault scenario forms another. A
false-discovery-rate filter removes weak lagged links. The links of all regimes are then combined by
how often they appear (support).

The discovery settings (maximum lag, significance level α, support threshold) are chosen from a grid
by agreement with the expert graph. One graph is learned for every evaluation fold, without the
scenario held out in that fold.

- **Input:** state tables, splits
- **Output:** `data/graphs/discovered_graphs.csv`, `discovery_settings.csv`

### Step 3: SCM fitting

```bash
python main.py run scm
```

The step converts each graph to an acyclic graph (DAG), removing undirected edges and breaking cycles
at the weakest edge. On that graph it fits a binary transition SCM. Each variable switches between 0
and 1 with a probability that depends on its previous state and on its parents. The probabilities are
counted from the training data, with a pseudo-count *a* so that unseen cases never get probability
zero. An optional extra lag lets a variable's switching depend on how long it has stayed in its
state.

*a* and the extra lag are chosen per fold by log loss on inner folds, so that the held-out scenario
is never used. Models are fitted for both the expert and the discovered graph.

- **Input:** discovered graphs, expert graphs, state tables, splits
- **Output:** `data/graphs/scm_graphs.csv`, `dag_removals.csv`; `data/models/scm_settings.csv`,
  `data/models/scm/*.pkl`
- **Thesis items:** Table 5.7, Table A.2

### Step 4: Root cause analysis

```bash
python main.py run rca
```

For every held-out fault run, the step scores each candidate variable by how strongly it breaks its
learned mechanism. The score is the surprise (negative log-likelihood) of its observed state, taken at
its highest point in the window around the first alarms. The candidates are then ranked from most to
least surprising, so the injected cause should come first.

- **Input:** fitted SCMs, state tables
- **Output:** `data/results/rca_rankings.csv`, `rca_runs.csv`
- **Thesis items:** Table A.6, Figure 6.4

### Step 5: Intervention analysis

```bash
python main.py run interventions
```

The step replays every recorded fault injection as an intervention in the SCM, with 200 simulations
per run. It records which alarms the model activates and when, and compares this with the alarms
recorded in the run.

- **Input:** fitted SCMs, state tables, fault annotations
- **Output:** `data/results/replay.csv`

### Step 6: Counterfactual explanations

```bash
python main.py run counterfactuals
```

For every coolant fault run, the step asks: which small set of allowed actions would have kept the
alarms off? An example action is keeping a pump healthy from a given time. The noise of the SCM is
first inferred from the observed run (abduction). Candidate actions are then taken from the root
cause ranking and simulated.

A set is valid when the alarms stay off in at least 90% of the simulations. It must reach this in the
search (200 simulations) and again in 2,000 new, independent simulations. Valid sets are also checked
on five SCMs refitted on resampled training data.

- **Input:** fitted SCMs, root cause rankings, state tables
- **Output:** `data/results/recommendations.csv`, `tried_action_sets.csv`, `recommendation_refits.csv`
- **Thesis items:** Tables 6.7 and A.7, Figure 6.9

---

## 6. The evaluation, part by part

Each part reads the pipeline results and is run on its own. The numbers below are the thesis results,
which the commands reproduce exactly.

### Part 1: Causal discovery

```bash
python main.py evaluate discovery
```

This part asks how close the discovered graph is to the expert graph. It compares their connections
(skeleton F1), their directions (directed F1) and the number of edits between them (structural Hamming
distance, SHD). Random graphs with the same number of edges serve as a reference.

| Metric | Coolant | Coolant, random | Probe | Probe, random |
|---|---|---|---|---|
| Skeleton F1 | 0.80 | 0.09 | 0.59 | 0.30 |
| Directed F1 | 0.75 | 0.05 | 0.42 | 0.15 |
| SHD | 6 | 18.6 | 18 | 26.2 |

**What it means.** For coolant, discovery recovers 8 of the 10 expert connections and orients every
directed edge correctly. For probe, it recovers 10 of 15 connections but orients only 7 of its 18
directed edges correctly.

**Stability.** In 20 bootstrap draws, the coolant graph stays close to the selected graph (mean
Jaccard similarity 0.68, directed F1 0.65). The probe graph changes much more (Jaccard 0.34).

These scores measure agreement with the expert graph that was also used to select the discovery
settings. They are not an independent test of accuracy.

**Thesis items:** Tables 6.1, 6.2 and A.1, Figures 6.1 and 6.2

### Part 2: SCM

```bash
python main.py evaluate scm
```

This part asks how well each SCM predicts the next state, measured by log loss in nats (lower is
better). Log loss at changes is the loss on only the steps where a variable changes. Those steps are
the hard part to predict.

| Subsystem | Data | Expert log loss | Discovered log loss | Expert, at changes | Discovered, at changes |
|---|---|---|---|---|---|
| Coolant | Test | 0.013 | 0.013 | 4.22 | 4.21 |
| Coolant | Fault scenarios | 0.174 | 0.174 | 2.98 | 2.88 |
| Probe | Test | 0.009 | 0.009 | 1.63 | 1.79 |
| Probe | Fault scenarios | 0.022 | 0.026 | 2.09 | 2.62 |

**What it means.** Both graphs predict normal test data equally well. At state changes in unseen
faults, the discovered graph is slightly better for coolant and the expert graph is better for probe.

**Stability.** The predicted probabilities are well calibrated on normal data: the quantile
residuals have standard deviation 1.00. Refitting the coolant SCM on resampled training data changes
the log loss at changes by only 0.05–0.06 nats.

**Thesis items:** Table 6.3, Tables A.8 and A.9, Figure 6.3

### Part 3: Root cause analysis

```bash
python main.py evaluate rca
```

This part asks how often the injected cause is ranked first (Top-1), and how high the injected causes
are ranked on average (MAP@3). The LSTM + SHAP baseline uses no graph. It is an anomaly detector whose
SHAP attributions rank the variables. Chance is the average of random rankings.

| Subsystem | Method | Top-1 | MAP@3 | Scenario avg. AP@3 |
|---|---|---|---|---|
| Coolant | Causal, both graphs | 1.00 | 1.00 | 1.00 |
| Coolant | LSTM + SHAP | 1.00 | 1.00 | 1.00 |
| Coolant | Chance | 0.47 | 0.47 | 0.47 |
| Probe | Causal, expert graph | 0.62 | 0.67 | 0.64 |
| Probe | Causal, discovered graph | 0.53 | 0.63 | 0.57 |
| Probe | LSTM + SHAP | 0.56 | 0.58 | 0.51 |
| Probe | Chance | 0.14 | 0.21 | 0.21 |

**What it means.** Every method finds all injected coolant causes. For probe, the causal model with
the expert graph ranks best, followed by the discovered graph and then LSTM + SHAP. Selecting the SCM
settings, not the new graph, gives most of the gain over the original configuration.

**Stability.** The 95% run intervals of probe MAP@3 overlap: [0.54, 0.79] for the expert graph and
[0.49, 0.76] for the discovered graph. No graph is therefore significantly better. LSTM + SHAP varies
between 0.54 and 0.63 MAP@3 across training seeds.

This is a retrospective diagnosis of injected faults.

**Thesis items:** Table 5.8, Tables 6.4–6.6, Figures 6.5 and 6.6

### Part 4: Interventions and counterfactual explanations

```bash
python main.py evaluate counterfactuals
```

This part asks two questions. Does the SCM reproduce the alarms when the recorded faults are replayed?
And how many runs receive a valid preventive recommendation?

| Coolant, 25 held-out runs | Expert graph | Discovered graph |
|---|---|---|
| Alarm activations reproduced (with training data) | 37 of 37 | 37 of 37 |
| Alarm onset within one step (0.5 s) | 25 of 37 | 25 of 37 |
| Valid recommendations (2,000 independent simulations) | 19 / 25 | 19 / 25 |
| Mean validity | 0.933 | 0.936 |
| Alarm coverage (activations with training data) | 37 / 56 | 37 / 56 |
| Valid on all five refitted SCMs | 19 / 19 | 19 / 19 |

**What it means.** Both graphs give the same valid recommendations for 19 of 25 runs. Each
recommendation keeps one or two components healthy, for example the high-pressure pump in scenario
exp_27. The six runs of scenario exp_22 stay below the 0.90 threshold. The selected extra lag makes
their long alarm activity unlikely, and with the original SCM settings all six are valid.

**Stability.** All 19 valid recommendations stay valid on all five refitted SCMs.

Validity is predicted by the SCM, not observed prevention on the machine.

**Thesis items:** Tables 6.8–6.10, Figures 6.7 and 6.8

### Part 5: Comparisons (appendix)

```bash
python main.py evaluate comparisons
```

This part reruns the pipeline twice: with the original discovery and SCM settings, and for probe
without the near-duplicate exclusion. It compares both reruns with the main results.

**What it means.** For coolant, the selected settings keep graph agreement and diagnosis but give
fewer valid recommendations (19 instead of 25). For probe, they raise the discovered graph's directed
F1 from 0.00 to 0.42 and the expert graph's MAP@3 from 0.48 to 0.67. Excluding the near-duplicates leaves the expert graph's
MAP@3 unchanged. It moves the discovered graph and LSTM + SHAP by at most 0.02.

**Thesis items:** Tables A.3 and A.5

---

## 7. Thesis tables and figures

Every table and figure of the thesis that comes from code is saved under its thesis number and title.

| Thesis item | File in `outputs/` | Command |
|---|---|---|
| Table 5.1 Analysed subsystems of the causRCA dataset | `tables/table_5_1_...` | `run preprocessing` |
| Table 5.2 Fault scenarios and injected causes | `tables/table_5_2_...` | `run preprocessing` |
| Table 5.3 Coolant variables and their roles | `tables/table_5_3_...` | `run preprocessing` |
| Table 5.4 Raw coolant event log | `tables/table_5_4_...` | `run preprocessing` |
| Table 5.5 Reconstructed coolant states | `tables/table_5_5_...` | `run preprocessing` |
| Table 5.6 Coolant discovery regimes | `tables/table_5_6_...` | `run preprocessing` |
| Table 5.7 Model parameters and selected settings | `tables/table_5_7_...` | `run scm` |
| Table 5.8 Settings of the LSTM + SHAP baseline | `tables/table_5_8_...` | `evaluate rca` |
| Table 6.1 Agreement with the expert graphs | `tables/table_6_1_...` | `evaluate discovery` |
| Table 6.2 Analysis of the missed coolant edges | `tables/table_6_2_...` | `evaluate discovery` |
| Table 6.3 State-prediction performance | `tables/table_6_3_...` | `evaluate scm` |
| Table 6.4 Root-cause ranking performance | `tables/table_6_4_...` | `evaluate rca` |
| Table 6.5 Probe ranking by graph and SCM settings | `tables/table_6_5_...` | `evaluate rca` |
| Table 6.6 Sensitivity of the LSTM + SHAP baseline | `tables/table_6_6_...` | `evaluate rca` |
| Table 6.7 Recommended coolant maintenance actions | `tables/table_6_7_...` | `run counterfactuals` |
| Table 6.8 Counterfactual results for coolant | `tables/table_6_8_...` | `evaluate counterfactuals` |
| Table 6.9 Validity under different SCM settings | `tables/table_6_9_...` | `evaluate counterfactuals` |
| Table 6.10 Examples of counterfactual explanations | `tables/table_6_10_...` | `evaluate counterfactuals` |
| Table A.1 Original and selected discovery settings | `tables/table_a_1_...` | `evaluate discovery` |
| Table A.2 Selected settings by evaluation fold | `tables/table_a_2_...` | `run scm` |
| Table A.3 Results with original and selected settings | `tables/table_a_3_...` | `evaluate comparisons` |
| Table A.4 Near-duplicate probe recordings | `tables/table_a_4_...` | `run preprocessing` |
| Table A.5 Results before and after near-duplicate exclusion | `tables/table_a_5_...` | `evaluate comparisons` |
| Table A.6 Probe analysis windows | `tables/table_a_6_...` | `run rca` |
| Table A.7 Coolant counterfactual results by run | `tables/table_a_7_...` | `run counterfactuals` |
| Table A.8 SCM settings by selection objective | `tables/table_a_8_...` | `evaluate scm` |
| Table A.9 Probe diagnosis by selection objective | `tables/table_a_9_...` | `evaluate scm` |
| Figure 5.1 Machine states during a coolant fault | `figures/figure_5_1_...` | `run preprocessing` |
| Figure 6.1 Expert and discovered coolant graphs | `figures/figure_6_1_...` | `evaluate discovery` |
| Figure 6.2 Expert and discovered probe graphs | `figures/figure_6_2_...` | `evaluate discovery` |
| Figure 6.3 State-prediction loss across data splits | `figures/figure_6_3_...` | `evaluate scm` |
| Figure 6.4 Root-cause scores during a coolant fault | `figures/figure_6_4_...` | `run rca` |
| Figure 6.5 Root-cause ranking by fault scenario | `figures/figure_6_5_...` | `evaluate rca` |
| Figure 6.6 SHAP attribution in a probe fault run | `figures/figure_6_6_...` | `evaluate rca` |
| Figure 6.7 Alarm responses to fault interventions | `figures/figure_6_7_...` | `evaluate counterfactuals` |
| Figure 6.8 Counterfactual validity and alarm coverage | `figures/figure_6_8_...` | `evaluate counterfactuals` |
| Figure 6.9 Counterfactual proposal below the validity threshold | `figures/figure_6_9_...` | `run counterfactuals` |

Figures 2.1, 4.1 and 5.2 and Tables 2.1, 3.1, 5.9 and 7.1 are concept diagrams and literature tables
drawn in LaTeX. No code produces them.

## 8. Settings

All settings are in `config/config.yaml`:
- the file locations;
- the subsystems;
- the 0.5 s time grid and the data split fractions;
- the discovery settings grid and its selection rule;
- the SCM pseudo-counts and extra lags;
- the root cause analysis window;
- the numbers of simulations;
- the 0.90 validity threshold;
- the LSTM grid;
- the random seeds.

Changing a value there changes the run. No setting is hidden in the code.

## 9. Future scope

- **Compare discovery methods.** Run other methods, including event-sequence approaches, on the same
  data splits. Judge them by how they change diagnosis and recommendations, not only by graph
  accuracy.
- **Improve temporal modelling.** Model recovery and alarm resets explicitly. Compare settings chosen
  for prediction with settings chosen for intervention behaviour and stable recommendations. Try other
  noise assumptions and bounds on the counterfactual outcomes.
- **Extend and validate actions.** Add maintenance cost, timing and operating constraints. Make
  recommendations from only the information available at the first alarm. Test the predicted
  prevention in controlled digital-twin experiments, then against recorded maintenance outcomes on
  real machines.

## 10. Data, licence and citation

The causRCA dataset is by Fraunhofer IWU and is published under the Apache 2.0 licence. Its own README
and licence are in `data/raw/causRCA/`.

> C. W. Mehling et al. causRCA: Real-World Dataset for Causal Discovery and Root Cause Analysis in
> Machinery. Zenodo, 2025. https://doi.org/10.5281/zenodo.15876410

The code is released under the MIT licence (`LICENSE`).

---

**Author:** Shweta Bambal, Otto von Guericke University Magdeburg
