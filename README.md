# causalAI-failure-prevention

Master's thesis: **Toward Actionable Root Cause Analysis: Integrating Causal Discovery and Counterfactual XAI for Predictive Maintenance.**

The repository is a notebook-driven research framework on the [causRCA](https://doi.org/10.5281/zenodo.15876410) dataset (Fraunhofer IWU). It runs one integrated pipeline:

```
event logs → preprocessing → PCMCI+ causal discovery → expert-graph evaluation
          → structural causal model → root cause analysis → interventions
          → counterfactual explanations → framework evaluation (expert vs discovered vs perturbed graphs)
```

The complete experiment uses the **coolant** subsystem; notebook 10 extends discovery, RCA and the graph-perturbation study to the **probe** subsystem. Methods, assumptions and limitations are in [METHODOLOGY.md](METHODOLOGY.md).

## Setup

Python 3.12. From the repository root:

```bash
pip install -r requirements.txt
```

This installs the libraries, JupyterLab and the `causal_rca` package in editable mode, so notebooks can `import causal_rca`.

The raw dataset is expected in `data/raw/causRCA/` (`real_op/`, `dig_twin/`, `expert_graph/`). To download it:

```bash
python scripts/download.py
```

Then extract the archive:

```bash
python scripts/extract_dataset.py
```

## Running

Run the tests:

```bash
python -m pytest -v
```

Run the notebooks in order (each one reads the outputs of the previous ones):

```bash
jupyter lab notebooks/
```

Or execute all of them without opening Jupyter:

```bash
cd notebooks && for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 "$nb"; done
```

Approximate runtimes on a laptop:
- notebook 03: about 15 min (bootstrap and sensitivity runs)
- notebook 09: about 30 min (48 graphs through the full pipeline: expert, discovered, empty and 45 perturbed)
- the others: under 2 min each

The archival preprocessing (all 270 recordings, data-quality report) can also be run as a command:

```bash
python -m causal_rca.preprocessing --config configs/preprocessing.yaml
```

## Notebooks

| Notebook | Stage | Main outputs |
|---|---|---|
| `01_EDA` | dataset composition, ground truth, sampling, constant variables, expert graphs | `results/figures/01_*`, `results/tables/01_*` |
| `02_Preprocessing` | state reconstruction at 0.5 s / 1 s, variable coverage, splits, model-ready matrices | `data/processed/coolant_dt*.npz`, `02_coolant_*` |
| `03_Causal_Discovery` | PCMCI+ (G² test) per regime, combined graph, leave-one-scenario-out graphs, bootstrap, sensitivity | `03_coolant_*` |
| `04_Graph_Evaluation` | discovered vs expert graph: skeleton/directed P/R/F1, SHD, orientation, chance level | `04_coolant_*` |
| `05_SCM_Fitting` | binary transition SCM, held-out fit, residuals, calibration, stability | `05_coolant_*` |
| `06_Root_Cause_Analysis` | SCM mechanism-break attribution, Top-1 / Hit@3 / MAP@3 / MRR vs chance | `06_coolant_*` |
| `07_Intervention_Analysis` | actionability, population-level do() effects, replay of real manipulations | `07_coolant_*` |
| `08_Counterfactual_XAI` | abduction–action–prediction explanations, validity, robustness | `08_coolant_*` |
| `09_Framework_Evaluation` | expert vs discovered vs perturbed graphs through the whole pipeline | `09_coolant_*` |
| `10_Probe_Extension` | the same methods on the probe subsystem, whose root causes also change in normal operation: discovery, graph evaluation, SCM, RCA, perturbations | `10_probe_*` |

Notebooks 02–09 run the coolant subsystem (`SUBSYSTEM` at the top of each notebook), and every output is named per subsystem.

### Robustness checks (stage 11)

After notebooks 07–10, two robustness checks run as a script and write only new `results/tables/11_*` files:

```bash
PYTHONPATH=src python scripts/run_robustness.py all
```

- **x1, time-limited replay** (about 4 min): Every replay of notebooks 07, 09 and 10 is repeated with each manipulation released at its observed reversion step, next to the default sustained replay, with identical SCMs, start states and noise. After release the variable follows its fitted mechanism; a faulty state never seen in training falls back to p = 0.5, which is not a learned recovery.
- **x2, independent counterfactual validity** (about 13 min): The action sets selected in notebook 09 are not reselected but re-evaluated with 2,000 fresh abducted worlds from a separate random stream, with Wilson intervals (Monte Carlo uncertainty given the fitted SCM only).

The script first checks that it reproduces the saved replay and rca results (`11_replay_reproduction_checks.csv`, `11_coolant_cf_reproduction_checks.csv`).

## Code

```
src/causal_rca/
    data.py                 metadata, ground truth, expert graphs, coverage, splits
    preprocessing.py        event parsing, state reconstruction, verification, model views
    causal_discovery.py     PCMCI+ per regime, link tables, aggregation, static projection
    graph_evaluation.py     graph metrics, random reference, DAG conversion, perturbations
    scm.py                  binary transition SCM: fitting, scoring, simulation, abduction
    root_cause_analysis.py  mechanism-break attribution and ranking metrics
    interventions.py        actionability, feasibility, population-level effects
    counterfactuals.py      individual counterfactual explanations and recommendation search
    evaluation.py           plot style, saving helpers, integrated framework evaluation
tests/
    test_preprocessing.py   parsing, reconstruction, verification, views
    test_core.py            metadata helpers, discovery aggregation, graph metrics, scm, rca, counterfactuals
```

Outputs: figures in `results/figures/` (PDF and PNG), tables in `results/tables/` (CSV), configurations and model summaries in `results/models/`.
