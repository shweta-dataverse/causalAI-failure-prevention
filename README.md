# Actionable root cause analysis with causal discovery and counterfactual explanations

Code of the master's thesis *Toward Actionable Root Cause Analysis: Integrating Causal Discovery and
Counterfactual XAI for Predictive Maintenance* (Shweta Bambal, Otto von Guericke University Magdeburg, 2026).

The framework learns a causal graph from the alarm and state logs of an industrial lathe, fits a
structural causal model (SCM) on it, ranks the likely root causes of a fault, simulates fault
interventions and searches for maintenance actions that would have prevented the alarms.
Every step compares the discovered graph with the expert graph of the
[causRCA](https://doi.org/10.5281/zenodo.15876410) dataset, on the coolant and probe subsystems.

```
event logs → preprocessing → causal discovery → SCM fitting → root cause analysis
           → intervention analysis → counterfactual explanations
```

## Setup

Python 3.12. From the repository root:

```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the thesis. `pyproject.toml` holds the project
details and the test and lint settings.

The raw dataset (causRCA 1.0.0) is included in `data/raw/causRCA/`.

## Running the framework

All six stages in order:

```bash
python main.py run
```

Or one stage at a time. Each stage reads the outputs of the stages before it:

| Command | Stage | Main output |
|---|---|---|
| `python main.py run preprocessing` | event logs to state series, data splits, near-duplicate check | `data/processed/`, `splits.csv` |
| `python main.py run discovery` | regime-wise PCMCI+, selection of the discovery settings, graph aggregation | `discovered_graphs.csv` |
| `python main.py run scm` | conversion to a DAG, nested selection of the SCM settings, fitted SCMs | `scm_settings.csv`, `data/cache/scm/` |
| `python main.py run rca` | mechanism-break root cause ranking of every held-out fault run | `rca_rankings.csv` |
| `python main.py run interventions` | replay of the recorded fault injections in the SCM | `replay.csv` |
| `python main.py run counterfactuals` | search and independent check of preventive maintenance actions | `recommendations.csv` |

Each stage prints one line when it is done, saying what it produced and where.

## Evaluating the results

Each evaluation part is run on its own:

| Command | What it evaluates |
|---|---|
| `python main.py evaluate discovery` | agreement of the discovered graphs with the expert graphs, stability, missed edges |
| `python main.py evaluate scm` | one-step state prediction (log loss), model checks, the study of the selection objective |
| `python main.py evaluate rca` | root cause ranking (Top-1, MAP@3, ...), with the LSTM + SHAP baseline |
| `python main.py evaluate counterfactuals` | simulated alarm responses, validity and coverage of the recommendations |
| `python main.py evaluate comparisons` | original against selected settings, and the effect of the near-duplicate exclusion |

Each part prints its metrics and one sentence on what they mean.

Heavy steps are cached in `data/cache/`: PCMCI+ runs, LSTM training, and the fitted SCMs. Add `--force` to
recompute them. Everything is logged to `logs/run.log`.

Approximate run times on a laptop, with a warm cache:

| Part | Time |
|---|---|
| Whole pipeline | 7 min |
| `evaluate scm` | 3 min |
| `evaluate counterfactuals` | 1.5 min |
| `evaluate comparisons` | 5 min |
| Other evaluation parts | under a minute each |

With an empty cache:
- the PCMCI+ runs of the settings grid and the discovery bootstrap take about 20 CPU hours, or a few hours with the 9 parallel workers set in the config;
- the LSTM + SHAP training takes about 45 minutes.

## Outputs

```
outputs/
  pipeline/
    tables/       data passed between stages, and the thesis tables produced by the pipeline
    figures/
  evaluation/
    tables/       metrics, and the thesis tables produced by the evaluation
    figures/
```

Every table and figure of the thesis that comes from code is saved under its thesis number and title,
for example `table_6_4_root_cause_ranking_performance.csv`:

| Thesis item | File | Command |
|---|---|---|
| Table 5.1 Analysed subsystems of the causRCA dataset | `pipeline/tables/table_5_1_...` | `run preprocessing` |
| Table 5.2 Fault scenarios and injected causes | `pipeline/tables/table_5_2_...` | `run preprocessing` |
| Table 5.3 Coolant variables and their roles | `pipeline/tables/table_5_3_...` | `run preprocessing` |
| Table 5.4 Raw coolant event log | `pipeline/tables/table_5_4_...` | `run preprocessing` |
| Table 5.5 Reconstructed coolant states | `pipeline/tables/table_5_5_...` | `run preprocessing` |
| Table 5.6 Coolant discovery regimes | `pipeline/tables/table_5_6_...` | `run preprocessing` |
| Table 5.7 Model parameters and selected settings | `pipeline/tables/table_5_7_...` | `run scm` |
| Table 5.8 Settings of the LSTM + SHAP baseline | `evaluation/tables/table_5_8_...` | `evaluate rca` |
| Table 6.1 Agreement with the expert graphs | `evaluation/tables/table_6_1_...` | `evaluate discovery` |
| Table 6.2 Analysis of the missed coolant edges | `evaluation/tables/table_6_2_...` | `evaluate discovery` |
| Table 6.3 State-prediction performance | `evaluation/tables/table_6_3_...` | `evaluate scm` |
| Table 6.4 Root-cause ranking performance | `evaluation/tables/table_6_4_...` | `evaluate rca` |
| Table 6.5 Probe ranking by graph and SCM settings | `evaluation/tables/table_6_5_...` | `evaluate rca` |
| Table 6.6 Sensitivity of the LSTM + SHAP baseline | `evaluation/tables/table_6_6_...` | `evaluate rca` |
| Table 6.7 Recommended coolant maintenance actions | `pipeline/tables/table_6_7_...` | `run counterfactuals` |
| Table 6.8 Counterfactual results for coolant | `evaluation/tables/table_6_8_...` | `evaluate counterfactuals` |
| Table 6.9 Validity under different SCM settings | `evaluation/tables/table_6_9_...` | `evaluate counterfactuals` |
| Table 6.10 Examples of counterfactual explanations | `evaluation/tables/table_6_10_...` | `evaluate counterfactuals` |
| Table A.1 Original and selected discovery settings | `evaluation/tables/table_a_1_...` | `evaluate discovery` |
| Table A.2 Selected settings by evaluation fold | `pipeline/tables/table_a_2_...` | `run scm` |
| Table A.3 Results with original and selected settings | `evaluation/tables/table_a_3_...` | `evaluate comparisons` |
| Table A.4 Near-duplicate probe recordings | `pipeline/tables/table_a_4_...` | `run preprocessing` |
| Table A.5 Results before and after near-duplicate exclusion | `evaluation/tables/table_a_5_...` | `evaluate comparisons` |
| Table A.6 Probe analysis windows | `pipeline/tables/table_a_6_...` | `run rca` |
| Table A.7 Coolant counterfactual results by run | `pipeline/tables/table_a_7_...` | `run counterfactuals` |
| Table A.8 SCM settings by selection objective | `evaluation/tables/table_a_8_...` | `evaluate scm` |
| Table A.9 Probe diagnosis by selection objective | `evaluation/tables/table_a_9_...` | `evaluate scm` |
| Figure 5.1 Machine states during a coolant fault | `pipeline/figures/figure_5_1_...` | `run preprocessing` |
| Figure 6.1 Expert and discovered coolant graphs | `evaluation/figures/figure_6_1_...` (PDF and TikZ) | `evaluate discovery` |
| Figure 6.2 Expert and discovered probe graphs | `evaluation/figures/figure_6_2_...` (PDF and TikZ) | `evaluate discovery` |
| Figure 6.3 State-prediction loss across data splits | `evaluation/figures/figure_6_3_...` | `evaluate scm` |
| Figure 6.4 Root-cause scores during a coolant fault | `pipeline/figures/figure_6_4_...` | `run rca` |
| Figure 6.5 Root-cause ranking by fault scenario | `evaluation/figures/figure_6_5_...` | `evaluate rca` |
| Figure 6.6 SHAP attribution in a probe fault run | `evaluation/figures/figure_6_6_...` | `evaluate rca` |
| Figure 6.7 Alarm responses to fault interventions | `evaluation/figures/figure_6_7_...` | `evaluate counterfactuals` |
| Figure 6.8 Counterfactual validity and alarm coverage | `evaluation/figures/figure_6_8_...` | `evaluate counterfactuals` |
| Figure 6.9 Counterfactual proposal below the validity threshold | `pipeline/figures/figure_6_9_...` | `run counterfactuals` |

No code produces Figures 2.1, 4.1 and 5.2 or Tables 2.1, 3.1, 5.9 and 7.1. They are concept
diagrams and literature tables drawn in LaTeX.

## Settings

All settings are in `config/config.yaml`:
- paths and subsystems;
- the grid step;
- the discovery settings grid and its selection rule;
- the SCM pseudo-counts and extra lags;
- the analysis window;
- the simulation counts;
- the validity threshold;
- the LSTM grid;
- the random seeds.

`config/related_recordings.csv` lists the near-duplicate recordings that are kept out of training when
their partner is held out.

## Code

```
main.py            command line
config/            settings
core/              the methods: data, preprocessing, discovery, graphs, scm, rca,
                   interventions, counterfactuals, lstm_shap, plots
pipeline/          one module per stage
evaluation/        one module per evaluation part
utils/             logging, errors, file handling
notebooks/         data exploration
tests/             unit tests, and checks that the outputs match the thesis tables
```

The data exploration notebook needs JupyterLab, which is not part of the pinned requirements:

```bash
pip install jupyterlab
```

Run the tests:

```bash
python -m pytest -q
```

## Data and licence

The causRCA dataset is by Fraunhofer IWU and is published under the Apache 2.0 licence. Its own README
and licence are in `data/raw/causRCA/`.

> C. W. Mehling et al. causRCA: Real-World Dataset for Causal Discovery and Root Cause Analysis in
> Machinery. Zenodo, 2025. https://doi.org/10.5281/zenodo.15876410

The code is released under the MIT licence (`LICENSE`).
