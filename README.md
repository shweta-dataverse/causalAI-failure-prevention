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
│   ├── results/                 rankings, replays, recommendations, evaluation/ details,
│   │                            comparisons/ for the reruns with other settings
│   └── cache/                   PCMCI+ test results, one file per regime run
│
├── outputs/
│   ├── tables/                  every thesis table that comes from code, e.g. table_6_4_...csv
│   └── figures/                 every thesis figure that comes from code, e.g. figure_6_1_...pdf
│
└── notebooks/01_EDA.ipynb       exploration of the dataset
```

**What `data/cache/` contains:** PCMCI+ is the slowest part of the framework. Selecting the discovery
settings and the bootstrap needs about 950 PCMCI+ runs, one per regime and setting **(about 20 CPU
hours in total)**. Every run is saved as one small file, named after its data, variables and settings. When the same run is needed
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

The six steps build on each other: each step reads what the steps before it saved. If an input is
missing, the step tells you which command to run first. Every step below shows two commands: **Run**
does the step, and **Check** runs the evaluation part that measures how well it worked.

| Step | Run | Check |
|---|---|---|
| 1 Preprocessing | `python main.py run preprocessing` | checks itself while it runs |
| 2 Causal discovery | `python main.py run discovery` | `python main.py evaluate discovery` |
| 3 SCM fitting | `python main.py run scm` | `python main.py evaluate scm` |
| 4 Root cause analysis | `python main.py run rca` | `python main.py evaluate rca` |
| 5 Intervention analysis | `python main.py run interventions` | `python main.py evaluate counterfactuals` |
| 6 Counterfactual explanations | `python main.py run counterfactuals` | `python main.py evaluate counterfactuals` |

### Step 1: Preprocessing

**Run:**
```bash
python main.py run preprocessing
```
**Check:** the step checks its own output while it runs. Tables 5.4 and 5.5 in `outputs/tables/` show
the same recording before and after.

**What it does.** The machine does not log its state at fixed times. It logs an event only when
something changes, for example `LP_Pump_On` switching to 0. This step turns those events into a
regular table: one row every 0.5 s, one column per variable. Each cell holds the last value the
variable reported (zero-order hold). Before a variable's first event, its cell stays empty, because
its state is unknown. It is never guessed.

The step then splits the data. The 170 normal recordings go into 119 for training, 26 for validation
and 25 for testing. The fault runs are grouped by scenario, so that each scenario can be held out in
turn and tested on a model that has never seen it (leave-one-scenario-out).

**Found along the way.** Two pairs of probe recordings from *different* scenarios are almost
identical. Run 3 of exp_1 and run 2 of exp_6 share 634 of their 635 rows. If one were tested while
its twin was in training, the model would already have seen the answer. These near-duplicates are
therefore never in training while their partner is tested.

**Why you can trust it.** An independent rebuild from the raw events matched all 11.7 million cells
of the state tables.

- **Input:** `data/raw/causRCA/`
- **Output:** `data/processed/coolant_states.csv`, `probe_states.csv`, `splits.csv`
- **Thesis items:** Tables 5.1–5.6, Table A.4, Figure 5.1

### Step 2: Causal discovery

**Run:**
```bash
python main.py run discovery
```
**Check:**
```bash
python main.py evaluate discovery
```

**What it does.** This step learns from the data which variable affects which, for example that a
failing pump raises a pump alarm. It uses PCMCI+ with the G² test, which looks for dependencies that
remain after the other variables are taken into account.

The key idea is to learn **per regime**. Normal operation is one regime, and each fault scenario is
another. Why? A fault scenario records only some of the variables, and many fault causes never change
in normal operation. If all recordings were simply pooled, a variable that is missing or constant
would look unrelated to everything else. So PCMCI+ runs separately in each regime. A false discovery
rate filter removes weak lagged links, and a link enters the final graph when enough regimes support
it (the support threshold).

The settings (maximum lag, significance level α, support threshold) are picked from a grid by how well
the graph agrees with the expert graph. One graph is learned for every evaluation fold, without the
scenario that the fold holds out.

**Found along the way.** Normal operation alone could not reveal the coolant faults. In the normal
training regime, only 3 of the 15 coolant variables are usable. The fault scenarios carry the
information, with 7 usable variables in exp_22 and 8 in exp_28.

- **Input:** state tables, splits
- **Output:** `data/graphs/discovered_graphs.csv`, `discovery_settings.csv`

### Step 3: SCM fitting

**Run:**
```bash
python main.py run scm
```
**Check:**
```bash
python main.py evaluate scm
```

**What it does.** A graph shows which variable affects which, but not how strongly. The SCM adds that.
First, the graph is made acyclic (a DAG): undirected edges are removed, and every cycle is broken at
its weakest edge. Then each variable gets its own small model, called its mechanism. It gives the
probability that the variable is 0 or 1 at the next step, from its previous state and the states of
its parents. In the expert graph, for example, `LP_Pump_On` depends on whether `LP_Pump_Ok` reports a
healthy pump.

The probabilities are counted from the training data. A pseudo-count *a* keeps unseen cases from ever
getting probability zero. An optional extra lag lets a variable's next state depend on how long it
has stayed in its current state. *a* and the extra lag are chosen per fold on inner folds, so the
held-out scenario never influences its own model. Models are fitted for both the expert and the
discovered graph.

**Found along the way.** For coolant, both graphs end up with the same settings: a pseudo-count of
0.01 and an extra lag of four steps (2 s).

- **Input:** discovered graphs, expert graphs, state tables, splits
- **Output:** `data/graphs/scm_graphs.csv`, `dag_removals.csv`, `data/models/scm_settings.csv`,
  `data/models/scm/*.pkl`
- **Thesis items:** Table 5.7, Table A.2

### Step 4: Root cause analysis

**Run:**
```bash
python main.py run rca
```
**Check:**
```bash
python main.py evaluate rca
```

**What it does.** When a fault happens, which variable started it? The idea is simple: the root cause
is the variable that breaks its own mechanism. Every candidate gets a **surprise score**, which says
how unlikely its observed state was under its mechanism (the negative log-likelihood, in nats). A
variable that just follows its parents is not surprising. A variable that suddenly does something its
parents cannot explain is.

The score is taken at its highest point in an analysis window around the alarms. The window starts
60 s before the first alarm onset and ends 5 s after the last one. In run 1 of exp_6, for example,
the alarm comes on at 100.5 s, so the window spans 40.5–105.5 s. The candidates are then ranked from
most to least surprising, so the injected cause should come out on top.

**Found along the way.** The coolant fault causes stay healthy in all the training data. So when one
of them changes, the model is very surprised: up to 16.43 nats, a probability of about 1 in
14 million. That is why every coolant cause is found.

- **Input:** fitted SCMs, state tables
- **Output:** `data/results/rca_rankings.csv`, `rca_runs.csv`
- **Thesis items:** Table A.6, Figure 6.4

### Step 5: Intervention analysis

**Run:**
```bash
python main.py run interventions
```
**Check:**
```bash
python main.py evaluate counterfactuals
```

**What it does.** Before a model is trusted with advice, it should reproduce what actually happened.
This step replays every recorded fault injection inside the SCM. With the do-operator, the injected
variable is set to its recorded fault value from its observed onset time, and the model simulates the
run 200 times. The
step records which alarms come on and when, so that they can be compared with the alarms the machine
actually raised.

**Found along the way.** The 25 coolant runs contain 56 alarm activations, but only 37 can be
simulated. The other 19 belong to three alarms that occur in only one scenario. When that scenario
is held out, the model has never seen the alarm, so it cannot predict it.

- **Input:** fitted SCMs, state tables, fault annotations
- **Output:** `data/results/replay.csv`

### Step 6: Counterfactual explanations

**Run:**
```bash
python main.py run counterfactuals
```
**Check:**
```bash
python main.py evaluate counterfactuals
```

**What it does.** This step asks the maintenance question: *what small action would have kept the
alarms off?* An action keeps one component healthy from a given time, for example
`do(HP_Pump_Ok = 1)`, "keep the high-pressure pump healthy". The step works in three moves:

1. **Abduction:** infer the hidden noise of the SCM from the observed run, so the simulation starts
   from what really happened.
2. **Action:** apply a set of allowed actions. The candidates come from the root cause ranking, and
   each action starts when its fault was detected.
3. **Prediction:** simulate the run again and check whether the alarms stay off.

A set is **valid** when the alarms stay off in at least 90% of the simulations. It must reach this
twice: in the search (200 simulations), and again in 2,000 new, independent simulations. Valid sets
are also tested on five SCMs refitted on resampled training data.

**Found along the way.** In all seven runs of exp_27, keeping the high-pressure pump healthy keeps
its alarm off, with a validity of 1.00.

- **Input:** fitted SCMs, root cause rankings, state tables
- **Output:** `data/results/recommendations.csv`, `tried_action_sets.csv`, `recommendation_refits.csv`
- **Thesis items:** Tables 6.7 and A.7, Figure 6.9

---

## 6. The evaluation, part by part

Each part reads the pipeline results and runs on its own. The numbers below are the thesis results,
and the commands reproduce them exactly. Each part names the pipeline steps it needs.

### Part 1: Causal discovery

**Run:**
```bash
python main.py evaluate discovery
```
**Needs:** steps 1–2

**The question.** How close is the learned graph to the expert graph? Three scores compare them:
skeleton F1 checks the connections, directed F1 also checks their direction, and the structural
Hamming distance (SHD) counts the edits needed to turn one graph into the other. Random graphs with
the same number of edges show what chance would give.

| Metric | Coolant | Coolant, random | Probe | Probe, random |
|---|---|---|---|---|
| Skeleton F1 | 0.80 | 0.09 | 0.59 | 0.30 |
| Directed F1 | 0.75 | 0.05 | 0.42 | 0.15 |
| SHD | 6 | 18.6 | 18 | 26.2 |

**What we found.**
- **Coolant comes out well.** Discovery recovers 8 of the 10 expert connections, and every edge it
  orients points the right way.
- **Time order helps.** Every connection with lagged evidence is correct and correctly oriented. All
  the errors are among connections seen only within the same time step.
- **The data explains the misses.** `LP_Pump_Ok` → `LP_Pump_On` is never found. The pump command is
  already off in all 12 pump-fault intervals, so the data never shows it switching off in response.
- **The two extra edges have a reason too.** Each links two alarms whose causes are injected together.
  The alarms come on at different times but reset in the same grid step.
- **Probe is harder.** Discovery recovers 10 of 15 connections, but orients only 7 of its 18 directed
  edges correctly.

**How stable it is.** In 20 bootstrap draws, the coolant graph stays close to the selected graph (mean
Jaccard similarity 0.68, directed F1 0.65). The probe graph changes much more (Jaccard 0.34).

**Keep in mind.** The expert graph also chose the discovery settings. These scores show agreement with
that graph, not an independent test of accuracy.

**Thesis items:** Tables 6.1, 6.2 and A.1, Figures 6.1 and 6.2

### Part 2: SCM

**Run:**
```bash
python main.py evaluate scm
```
**Needs:** steps 1–3

**The question.** How well does each SCM predict the next state? The score is log loss in nats, where
lower is better. Most steps are easy, because nothing changes. The real test is **log loss at
changes**: only the steps where a variable actually changes.

| Subsystem | Data | Expert log loss | Discovered log loss | Expert, at changes | Discovered, at changes |
|---|---|---|---|---|---|
| Coolant | Test | 0.013 | 0.013 | 4.22 | 4.21 |
| Coolant | Fault scenarios | 0.174 | 0.174 | 2.98 | 2.88 |
| Probe | Test | 0.009 | 0.009 | 1.63 | 1.79 |
| Probe | Fault scenarios | 0.022 | 0.026 | 2.09 | 2.62 |

**What we found.**
- **On normal data, the graphs tie.** Only three coolant variables change during normal operation, so
  there is little room for the graph to make a difference.
- **In unseen faults, the graph starts to matter.** The discovered graph is slightly better for coolant
  (2.88 against 2.98). The expert graph is clearly better for probe (2.09 against 2.62).

**How stable it is.** The predicted probabilities are well calibrated on normal data: the quantile
residuals have a standard deviation of 1.00, as they should. Refitting the coolant SCM on resampled
training data moves the log loss at changes by only 0.05–0.06 nats.

**Thesis items:** Table 6.3, Tables A.8 and A.9, Figure 6.3

### Part 3: Root cause analysis

**Run:**
```bash
python main.py evaluate rca
```
**Needs:** steps 1–4

**The question.** Does the injected cause come out on top? Top-1 is how often it is ranked first.
MAP@3 rewards finding all injected causes near the top of the list. The baseline, LSTM + SHAP, uses no
graph. It is an anomaly detector whose SHAP attributions rank the variables. Chance is the average of
random rankings.

| Subsystem | Method | Top-1 | MAP@3 | Scenario avg. AP@3 |
|---|---|---|---|---|
| Coolant | Causal, both graphs | 1.00 | 1.00 | 1.00 |
| Coolant | LSTM + SHAP | 1.00 | 1.00 | 1.00 |
| Coolant | Chance | 0.47 | 0.47 | 0.47 |
| Probe | Causal, expert graph | 0.62 | 0.67 | 0.64 |
| Probe | Causal, discovered graph | 0.53 | 0.63 | 0.57 |
| Probe | LSTM + SHAP | 0.56 | 0.58 | 0.51 |
| Probe | Chance | 0.14 | 0.21 | 0.21 |

**What we found.**
- **Coolant is solved by every method.** The causes never change in normal operation, so their first
  change stands out. For the expert graph, the median gap between the lowest injected cause and the
  best other candidate is 7.42 nats.
- **Probe is the real test.** Its fault sensors also switch during normal machining, so a change alone
  proves nothing. Here the expert graph ranks best (MAP@3 0.67), then the discovered graph (0.63),
  then LSTM + SHAP (0.58).
- **No method wins everywhere.** LSTM + SHAP has a higher Top-1 than the discovered graph (0.56
  against 0.53). The discovered graph beats the expert graph in one scenario, exp_2_3 (0.79 against
  0.42).
- **The SCM settings bring most of the gain.** Selecting them raises probe MAP@3 by 0.19 for the
  expert graph and 0.16 for the discovered graph.

**How stable it is.** The 95% run intervals of probe MAP@3 overlap: [0.54, 0.79] for the expert graph
and [0.49, 0.76] for the discovered graph. Neither graph is therefore clearly better. LSTM + SHAP
ranges from 0.54 to 0.63 MAP@3 across training seeds.

**Keep in mind.** This is a retrospective diagnosis of injected faults: the window also uses
observations after the alarms.

**Thesis items:** Table 5.8, Tables 6.4–6.6, Figures 6.5 and 6.6

### Part 4: Interventions and counterfactual explanations

**Run:**
```bash
python main.py evaluate counterfactuals
```
**Needs:** steps 1–6

**The question.** Two questions. When the recorded faults are replayed, does the SCM raise the right
alarms at the right time? And how many runs get a valid recommendation that would have kept the
alarms off?

| Coolant, 25 held-out runs | Expert graph | Discovered graph |
|---|---|---|
| Alarm activations reproduced (with training data) | 37 of 37 | 37 of 37 |
| Alarm onset within one step (0.5 s) | 25 of 37 | 25 of 37 |
| Valid recommendations (2,000 independent simulations) | 19 / 25 | 19 / 25 |
| Mean validity | 0.933 | 0.936 |
| Alarm coverage (activations with training data) | 37 / 56 | 37 / 56 |
| Valid on all five refitted SCMs | 19 / 19 | 19 / 19 |

**What we found.**
- **The replay matches the machine.** Every evaluated alarm comes on in at least 199 of 200
  simulations. The median timing error is 0.5 s, and the largest is 2.0 s.
- **19 of 25 runs get a valid recommendation, with the same actions for both graphs.** For exp_27,
  keep the high-pressure pump healthy. For exp_28 and exp_35, keep the coolant filter and the
  low-pressure pump healthy.
- **Every recommended action targets an injected cause** (action precision 1.00).
- **An imperfect graph can give the same advice.** The discovered graph has errors, but none on the
  paths from the components to the alarms that matter here.
- **exp_22 shows the limit.** Its six proposals reach only 0.84–0.87, below the 0.90 threshold. The
  selected extra lag makes the long alarm activity in these runs unlikely. With the original SCM
  settings, all six are valid.

**How stable it is.** All 19 valid recommendations stay valid on all five refitted SCMs.

**Keep in mind.** Validity is predicted by the SCM. It is not prevention observed on the machine.

**Thesis items:** Tables 6.8–6.10, Figures 6.7 and 6.8

### Part 5: Comparisons (appendix)

**Run:**
```bash
python main.py evaluate comparisons
```
**Needs:** steps 1–6

**The question.** Do the choices made along the way matter? This part reruns the pipeline twice: once
with the original discovery and SCM settings, and once, for probe, without the near-duplicate
exclusion. Both reruns are compared with the main results.

**What we found.**
- **For probe, selecting the settings made the biggest difference.** The original discovered graph kept a
  single edge for the SCM and reached a directed F1 of 0.00. The selected settings raise it to 0.42.
  The expert graph's MAP@3 rises from 0.48 to 0.67.
- **For coolant, there is a trade-off.** The selected settings keep graph agreement and diagnosis but
  give fewer valid recommendations (19 instead of 25).
- **The near-duplicates barely matter.** Excluding them leaves the expert graph's MAP@3 unchanged and
  moves the discovered graph and LSTM + SHAP by at most 0.02.

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
- the file locations
- the subsystems
- the 0.5 s time grid and the data split fractions
- the discovery settings grid and its selection rule
- the SCM pseudo-counts and extra lags
- the root cause analysis window
- the numbers of simulations
- the 0.90 validity threshold
- the LSTM grid
- the random seeds

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
