# causalAI-failure-prevention

Master's thesis: **Toward Actionable Root Cause Analysis: Integrating Causal Discovery and Counterfactual XAI for Predictive Maintenance.**

The repository implements a framework on the [causRCA](https://doi.org/10.5281/zenodo.15876410) dataset (Fraunhofer IWU, version 1.0.0) that connects

```
event logs → preprocessing → regime-wise PCMCI+ causal discovery → binary transition SCM
          → root cause analysis → intervention analysis → counterfactual recommendations
```

and compares a discovered with an expert causal graph at every stage, on the **coolant** and **probe** subsystems, with an LSTM + SHAP diagnosis baseline. The notebooks (stages 01–10) are the original exploratory pipeline; the results reported in the thesis come from the scripts of stages 12–17 (see [Reproducing the thesis results](#reproducing-the-thesis-results)). Empty and perturbed graphs appear only in the historical notebook results. Methods, assumptions and limitations are in [METHODOLOGY.md](METHODOLOGY.md).

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

## Reproducing the thesis results

The thesis reports the outputs of the scripts below (tables in `results/tables/`, figures in `results/figures/`); all of them are committed, so every number can be checked without rerunning. To recompute them, run the steps in this order from the repository root with Python 3.12 (`pip install -r requirements.txt` into `.venv`; the thesis used Python 3.12.3, tigramite 5.2.10.1, numpy 2.5.4, pandas 3.0.6, networkx 3.7, scipy 1.18.1, PyTorch 2.15.0 and SHAP 0.54.1).

| Step | Command | What it does | Needs |
|---|---|---|---|
| 0 | `python scripts/download.py` and `python scripts/extract_dataset.py` | raw dataset into `data/raw/causRCA/` | internet |
| 1 | `python -m pytest -q` | 91 tests of the package | step 0 for the tests on the raw data (skipped otherwise) |
| 2 | notebooks `01`–`10` in order (see [Running](#running)) | state grids in `data/processed/` (`coolant_dt0.5`, `probe_dt0.5`) and the original-configuration results (stages 01–10) | step 0 |
| 3 | `PYTHONPATH=src .venv/bin/python scripts/related_recordings.py` | near-duplicate check; verifies the frozen manifest `configs/related_recordings.csv` (stage 17a) | step 0 |
| 4 | `PYTHONPATH=src .venv/bin/python scripts/run_discovery_study.py all` | study of the discovery settings (stage 12); PCMCI+ runs are cached in `data/cache/discovery_study/` (not committed); about four hours, mostly the bootstrap | steps 2–3 |
| 5 | `PYTHONPATH=src .venv/bin/python scripts/selected_settings.py` | discovery settings per scope from the stage-12 cache and SCM settings by held-out log loss (stage 15a), about three minutes | step 4 |
| 6 | `PYTHONPATH=src python scripts/final_evaluation.py` | final evaluation with the original configuration (stage 14, `14_*`; reads the stage-12 original-configuration graphs) | step 4 |
| 7 | `PYTHONPATH=src python scripts/final_evaluation.py selected` | final evaluation with the selected settings, the main results of the thesis (stage 15, `15_*`) | steps 5–6 |
| 8 | `PYTHONPATH=src .venv/bin/python scripts/lstm_shap_baseline.py` | LSTM + SHAP baseline (stage 16, `16_*`) | step 2 |
| 9 | `PYTHONPATH=src .venv/bin/python scripts/<name>.py` for `edge_diagnostics`, `normal_diagnostics`, `probe_rca_decomposition`, `cf_exp22_settings`, `compare_original_selected`, `protocol_comparison`, `audit_related_exclusion`, `probe_window_table`, `recommendation_table`, `preprocessing_example` | diagnostics, the exclusion audit and the tables of Chapters 5–6 and Appendix A (stages 15b–15f, 17b–17e) | steps 4–8 |
| 10 | `PYTHONPATH=src python scripts/thesis_figures.py` | thesis figures from the stored stage-15 results | step 7 |

The sensitivity analysis of the SCM selection objective (stage 13, Appendix A.7) is run with `python scripts/scm_nested_selection.py <research_repo> <out_dir>` and summarised with `python scripts/scm_nested_summary.py`. Each script states its inputs, outputs and usage in its header.

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
