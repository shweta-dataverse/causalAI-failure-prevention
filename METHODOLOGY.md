# Methodology, assumptions and limitations

This document describes what the framework does, why, and where its conclusions stop. Numbers refer to the coolant subsystem (the complete experiment) unless they are marked as probe (notebook 10, extension). The notebooks contain every figure and table cited here.

## 1. Research question

How does the quality of a causal graph affect root cause analysis (RCA), intervention predictions and counterfactual explanations for industrial machine faults? We answer it by running one integrated pipeline — graph → structural causal model (SCM) → RCA → interventions → counterfactuals — on the expert graph, a graph discovered from data, and deliberately perturbed graphs.

## 2. Data (notebook 01)

causRCA (Fraunhofer IWU, Zenodo 15876410):
- **Recordings:** 170 normal-operation recordings and 100 digital-twin fault runs, in 19 scenarios.
- **Expert graph:** 92 nodes and 104 edges; the subsystem graphs cover coolant 15/10, hydraulics 17/18 and probe 11/15 (nodes/edges).
- **Metadata checks:** 50 of 51 pass. The one failure is three runs whose alarm is marked resolved 0.03 s after detection.

Facts that shape every later decision:

| Finding | Consequence |
|---|---|
| Ground truth is `manipulatedVars` in the experiment descriptions; `causes.json` holds only timestamps | causes loaded from descriptions; root cause, diagnosis, expected states and alarms kept apart |
| 11 of 19 scenarios have several manipulated variables | multi-cause ranking metrics (Hit@3, MAP@3) |
| Coolant and probe are binary/alarm only | no ParCorr; discrete tests and mechanisms |
| 11 of 12 coolant/hydraulics root-cause variables never change in normal operation; no subsystem alarm ever fires there | discovery must use fault runs; normal data alone cannot identify their effects |
| about 70% of events share their timestamp with another variable; effects can be logged before their causes | lag-0 links cannot be oriented from time order |
| `cause_start_at` lies 17–48 s (median) before the visible change; `alarms_detected_at` is the **last** expected alarm | the RCA window is anchored on observed alarm onsets instead |
| each fault run records only the variables of its scenario | unrecorded = unknown (masked); the recording pattern reveals the scenario (leakage risk) |

## 3. Preprocessing (notebook 02, `preprocessing.py`)

- **Loading:** every recording is loaded separately. Recording, subsystem, experiment and run IDs come from the folder structure.
- **Parsing:** values are parsed by their declared type. Binary values are parsed strictly, and categoricals stay strings even when they look numeric.
- **Duplicates:** for the same (recording, node, timestamp), the last row in the file wins (none occur).
- **State reconstruction:** zero-order hold on a regular grid per recording, with no backfill (unknown before a node's first event) and no filling across recordings. Every cell is re-derived by an independent method; 11.7 M cells were checked with 0 mismatches.
- **Sampling interval:** compared at 0.5 s and 1 s. Both keep 100% of coolant value changes. 0.5 s separates more near-simultaneous changes; 1 s halves the rows. Primary choice: **0.5 s**; 1 s is a sensitivity variant.
- **Model views:** binary 0/1, with masks for unknown states. Columns that are constant in the training recordings are excluded with a recorded reason.
- **Splits:**
  - Normal recordings are ordered by recording number (assumed to grow with time): 119 train, 26 validation, 25 test.
  - Fault runs are grouped into leave-one-scenario-out folds, so a scenario is never divided.

## 4. Causal discovery (notebook 03, `causal_discovery.py`)

- **Algorithm:** PCMCI+ (Tigramite 5.2) with the **G² conditional independence test** for discrete data.
- **Regimes:** one run per regime: normal training recordings, and each fault scenario separately. Pooling all runs fails here for two reasons:
  - Tigramite raises an error when a test has no common samples.
  - It returns p = 1 ("independent") for constant data.

  Pooling would therefore turn "not recorded" into "independent". Inside a regime, only variables recorded in every recording and changing at least once are used.
- **Settings:** Δ = 0.5 s, `tau_max` = 10 s (20 lags), `pc_alpha` = 0.01. Conditioning sets are capped (3 conditions, 3 lagged parents of X and of Y), because larger binary contingency tables are empty. Recordings enter as separate datasets, so lags never cross recordings.
- **FDR** (verified in the tigramite 5.2.10.1 source):
  - `run_pcmciplus` decides and orients every link with uncorrected tests at α.
  - It then Benjamini–Hochberg-corrects only the *reported* p-values, and only those of **lagged** links (`exclude_contemporaneous=True`). The reported value per link is the largest p-value over all its tests.
  - We therefore remove lagged links whose corrected p-value exceeds α. Lag-0 links are unaffected, since their uncorrected maximum p is ≤ α by construction.
  - For coolant the filter removes 9 lagged links. One lag-0 orientation could have relied on a removed link; re-running that regime with the link forbidden gives the same orientation.
  - The unfiltered graph is reported as a sensitivity variant.
- **Combining regimes:** a pair is *testable* in a regime when both variables are used there. A relation enters the static graph when it is found in ≥ 50% of its testable regimes. Each regime counts as one vote, whatever its sample size (normal: 137k steps; fault scenarios: 3–4k). Pairs never testable are reported as **untestable**, not absent (44 of 105 coolant pairs).
- **Conflicting evidence** is kept and reported. Pairs found in opposite directions in different regimes, or directed in some and undirected in others, are flagged (`orientation_conflict`, `mixed_marks`).
  - Coolant: 1 pair (the `LP_Pump_On`–`HP_Pump_isOff` feedback loop).
  - Probe: 14 and 12 pairs.
- **Static projection:**
  - Lagged links (lag ≥ 1) become directed edges, oriented by time.
  - Lag-0 links oriented by PCMCI+ rules become directed edges.
  - `o-o` and `x-x` lag-0 links stay undirected; they are never oriented by guesswork.
  - Autodependencies are dropped.
- **Leakage control:** leave-one-scenario-out graphs are built for RCA by leaving one regime out. Stability is measured by bootstrapping recordings within regimes (20 draws) and by one-at-a-time sensitivity (Δ, `tau_max`, α, FDR filter, support threshold). Sensitivity is compared with the primary graph, never with the expert graph.

## 5. Graph evaluation (notebook 04, `graph_evaluation.py`)

- **Metrics:** skeleton and directed precision, recall and F1, SHD, and orientation accuracy.
- **Edge accounting:** undirected discovered edges count for the skeleton but never as a correct direction.
- **Two scopes:** all pairs, and testable pairs only.
- **Chance level:** random graphs with the same number of edges (2,000 draws).
- **Lagged vs contemporaneous evidence:** reported separately (precision only, since the expert graph has no lags).

Agreement with the expert graph is agreement with expert knowledge, not proof of the true structure. Discovery settings were fixed before this comparison, and the sensitivity variants are reported, not selected.

## 6. Structural causal model (notebook 05, `scm.py`)

**Structural equation (binary transition mechanism)**

  x_j[t] = 1{U_j[t] < p_j(x_j[t-1], pa_j[t], pa_j[t-1])},  U_j[t] ~ Uniform(0, 1)

- **Mechanism family:** p_j is a conditional probability table over the previous state and the parents' values now and one step earlier, smoothed with a Jeffreys prior (add ½). This is the exact maximum-likelihood family for binary data. A first version used logistic regression; it leaked spurious switching probability for rare events and was replaced.
- **Acyclicity:** lag-0 parents enter the equations, so the graph must be acyclic. The discovered graph is made acyclic by dropping undirected edges and breaking cycles at the lowest-support edge; both steps are logged.
- **Modularity:** a variable's mechanism is fitted on all training recordings **except those where it was manipulated** (training-scenario labels only).
- **Unrecorded parents:** set to their usual state.
- **No training data:** variables never recorded in any training scenario (e.g. an alarm that only the held-out scenario records) are held at their usual state and marked **unsupported**.
- **Evaluation:**
  - one-step-ahead predictions on held-out normal recordings and on held-out scenarios (non-manipulated variables)
  - randomized quantile residuals
  - calibration of switching probabilities
  - simulated vs observed behaviour
  - bootstrap refits
- **Reference:** the empty graph (same family, no parents) shows what the structure adds.

## 7. Root cause analysis (notebook 06, `root_cause_analysis.py`)

- **Method:** SCM mechanism-break attribution. An intervention replaces a variable's equation, so a root cause shows values its own mechanism cannot explain given its parents. Score = maximum surprise −log p(x_j[t] | x_j[t-1], pa_j) in the analysis window.
- **Window:** from 60 s before the first observed alarm onset to 5 s after the last one. It is label-free; `causes.json` is never used.
- **Candidates:** non-alarm subsystem variables recorded in the run.
- **Ties:** broken by summed surprise, then by name.
- **Protocol:** leave one scenario out (25 held-out runs).
- **Metrics:** Top-1, Hit@3, Recall@3, MAP@3 and MRR, with 95% bootstrap intervals over runs. Multiple true causes are handled by Hit@k, Recall@k and AP@k.
- **Chance level:** the expected metric of a random ranking of the **same candidate set**, which is high because the recording pattern already narrows the scenario.

## 8. Interventions (notebook 07, `interventions.py`)

- **Actionability:** taken from the node descriptions.
  - *Maintenance* actions restore component conditions (filters, pump states, level switches), allowed only towards the healthy value.
  - *Control* actions set `LP_Pump_On` to 0 or 1.
  - Derived feedback states and alarms are never actionable. Unknown actionability defaults to not actionable.
- **Feasibility:** actionable variables only, allowed and binary values, recorded in the run, at most 3 actions.
- **Population-level effects:** do(X = x) replaces X's equation. Effects are averaged over simulated noise from the usual (healthy) state over 120 s.
- **Validation against real interventions:** the digital-twin manipulations *are* interventions. They are replayed (same variables, values and onset times, same start state) in SCMs fitted without the scenario. Predicted alarm firing and delays are compared with the observed ones.

## 9. Counterfactual explanations (notebook 08, `counterfactuals.py`)

- **Abduction → action → prediction:**
  1. Abduction samples the noise U consistent with the observed run; for x = 1, U < p, and for x = 0, U ≥ p.
  2. Action sets actionable variables to their healthy value from the step where their mechanism break was detected.
  3. Prediction rolls the equations forward.
- **Identification:** individual counterfactuals are **not identified from data**. They rely on the assumed monotone threshold noise. All results are model-predicted.
- **Search:** among the top-3 actionable RCA candidates, by size (1–3 actions) then rank. The first set that keeps every supported outcome alarm off in ≥ 90% of 200 abducted worlds is the explanation; otherwise the best partial set is reported.
- **Metrics:** target improvement (alarm time), validity probability, proximity (seconds of changed values), sparsity, feasibility, and robustness (noise samples, refitted mechanisms, other graphs).

## 10. Framework evaluation (notebook 09)

- **Graphs:** the expert graph, the discovered graph (leave one scenario out), the empty graph, and 45 perturbed expert graphs. The perturbations are deletion, insertion and reversal, each at 10/25/50% of the edges with 5 seeds, all kept acyclic.
- **Pipeline:** for each graph the whole pipeline is rerun. Notebook 10 repeats discovery, graph evaluation, SCM, RCA and the perturbation study (without interventions and counterfactuals) on the probe subsystem. Downstream quality is related to graph quality with Spearman correlations (bootstrap intervals over graphs), plus paired Wilcoxon tests against the expert graph per run.

## 11. Results

**Causal discovery and graph agreement (notebooks 03–04).**
- The combined PCMCI+ graph has 10 edges: 6 oriented, 4 undirected.
- Against the expert graph:

  | Metric | Discovered | Random graph, same size |
  |---|---|---|
  | Skeleton F1 | 0.80 | 0.09 |
  | Directed precision | 1.00 | — |
  | Directed recall | 0.60 | — |
  | Directed F1 | 0.75 | 0.05 |
  | SHD | 6 | — |

- **Lagged vs lag-0 evidence.** Every lagged edge is correct, including the 5 s filter-alarm delays, which were found without using the expert comments. All errors rest on lag-0 evidence: two missed edges, two unorientable edges, and two alarm–alarm extras.
- **Stability.** Correct edges are more stable (mean bootstrap frequency 0.76 vs 0.25 for extra edges). The result is robust to `tau_max` and α, but changes on the 1 s grid (Jaccard 0.53).

**SCM (notebook 05).**
- Structure raises the predicted probability of observed alarm changes in unseen scenarios from about 0.01 (no parents) to 0.40–0.49 (expert or discovered graph).
- Quantile residuals are close to standard normal.
- Alarms recorded in only one scenario cannot be learned when that scenario is held out.

**Root cause analysis (notebook 06).**
- MAP@3, Top-1, Recall@3 and MRR are all 1.00 for the expert, discovered **and empty** graph (chance: MAP@3 0.47, Top-1 0.47).
- Coolant root causes never change in normal operation, so their mechanism break is maximal regardless of structure (margin ≈ 6.3 nats).

**Interventions (notebook 07).** Replaying the real digital-twin manipulations in unseen scenarios:

| Graph | Predicted p(alarm fires) for alarms that fired | Median delay error |
|---|---|---|
| expert | 1.00 | 0.5 s |
| discovered | 1.00 | 0.5 s |
| empty | 0.50 | 41.5 s |

These are the 37 supported alarm activations; 19 unsupported activations cannot be predicted by any graph.

**Counterfactual explanations (notebook 08).**
- Expert graph:
  - every one of the 25 runs has a valid explanation (mean validity 0.955, mean 1.48 actions)
  - precision against the true causes 1.00, recall 0.66 (the remainder are causes of unsupported alarms)
  - abduction reproduces all recorded data exactly
- Robustness: noise resampling gives mean validity 0.954; refitted mechanisms 0.923.
- The discovered graph gives identical recommendations; the empty graph none.

**Framework evaluation (notebook 09; 45 perturbed graphs).**
- **Correlation:** downstream quality correlates positively with directed F1 (Spearman 0.31–0.66, all intervals above 0).
- **RCA:** reacts only to reversed edges (MAP@3 0.91 at 50%).
- **Actionable explanations:** collapse under deleted and reversed edges. At 50% perturbation:
  - deletion: valid share 0.39, recommendation stability 0.52
  - reversal: valid share 0.50, recommendation stability 0.54
- **Inserted false edges** are nearly harmless (0.90 and 1.00 at 50%), because the fitted mechanisms ignore uninformative parents.
- **Not a uniform relationship:** a lower-F1 graph (insert 50%) can be more useful than a higher-F1 graph (delete 25%).

**Answer to the research question (coolant).**
- Graph quality hardly affects *locating* the root cause, because mechanism-break attribution needs little structure when causes are constant in normal operation.
- It strongly affects *acting* on it: interventions and counterfactual recommendations depend on the edges from causes to symptoms.
- Graph F1 is an incomplete proxy for usefulness: which edges are wrong matters more than how many.

**Probe extension (notebook 10; 34 fault runs, 6 scenarios).** Probe root causes are arm-position sensors that also switch during normal machining.
- **Discovery is weak:** skeleton F1 0.52 (random 0.19), directed F1 0.00. Command, position and cover changes fall into the same 0.5 s step and cannot be oriented.
- **The expert graph matters for RCA:**

  | Graph | Top-1 | MAP@3 |
  |---|---|---|
  | expert | 0.44 [0.29, 0.59] | 0.48 |
  | no structure | 0.09 [0.00, 0.21] | 0.44 |
  | chance | 0.14 | 0.21 |

  - It works for work-position sensor faults (Top-1 0.83).
  - It fails when the whole arm is stuck (MAP@3 ≤ 0.10): downstream mechanisms enter unseen states and look more broken than the sensors.
- **No correlation with F1:** across the 45 perturbed probe graphs, RCA quality does not correlate with directed F1 (Spearman 0.04 [−0.27, 0.33]). Alarm replay does (0.35 [0.06, 0.57]).
- **Not applicable:** interventions and counterfactual recommendations, because sensor repair is a soft intervention that is not implemented.

**Overall answer.**
1. Graph quality is decisive for **actionable** outputs: intervention predictions and counterfactual recommendations.
2. Its effect on **locating** root causes depends on the subsystem:
   - none where causes are constant in normal operation (coolant);
   - substantial but scenario-dependent and non-monotonic where they are not (probe).
3. In neither case is graph F1 a sufficient predictor of downstream usefulness.

## 12. Limitations

**Data**
- All fault data come from a hardware-in-the-loop digital twin. Normal data come from the real machine but contain no faults. Conclusions about real failure prevention are out of reach.
- Sample sizes are small: coolant has 25 fault runs in 4 scenarios, so leave-one-scenario-out has 4 folds.
  - Bootstrap intervals over runs ignore the correlation between runs of one scenario and are therefore optimistic.
  - Per-scenario results are reported for this reason.
- Each fault run records only its scenario's variables. Consequences:
  - The recording pattern reveals the scenario, so candidate sets are small and the chance level is high (MAP@3 0.47 on coolant).
  - 44 of 105 coolant pairs are untestable.
  - Alarms recorded in only one scenario have no mechanism when that scenario is held out.
- OPC UA timestamps are publish times: about 70% of events share a timestamp, and effects are sometimes logged before their causes. Lag-0 links cannot be oriented from time, and the 0.5 s grid can place an effect one step before its cause.
- The ordered split of normal recordings assumes that recording numbers grow with time. This is not verified.

**Causal discovery**
- PCMCI+ assumes causal sufficiency, faithfulness and one stationary mechanism per regime. Regime-wise discovery satisfies the last assumption better than pooling, but leaves each relation resting on one or two scenarios.
- The G² test treats samples as independent given the conditioning set and uses an asymptotic χ² approximation.
  - In every coolant fault regime each variable changes only 12–14 times in 3,000–4,000 samples, so many contingency cells are sparse.
  - Conditioning on the variables' own past absorbs most of the persistence of binary states, but p-values remain approximate.
  - They are used as a ranking device for an exploratory graph. Bootstrap stability over recordings is the more reliable evidence.
- Fault regimes are intervention regimes: the manipulated variables are forced externally, with timing set by the experiment design. Discovery is not informed of the intervention targets or times.
  - The graph is therefore an exploratory map of dependencies made visible by the fault excitation, not an identified causal graph.
  - Two of its extra edges (alarm–alarm) are explained by alarms co-occurring because both faults of a multi-cause scenario are triggered by design.
- The bootstrap resamples recordings with replacement. Duplicated recordings make tests more confident than the original sample, so frequencies of weak edges are biased upward, and with 3–7 recordings per fault regime the bootstrap is coarse.
- Prior knowledge used by discovery:
  - the expert graph's **node set** defines each subsystem's variable universe;
  - the **scenario membership** of runs defines the regimes.

  Expert edges, ground-truth causes and `causes.json` timestamps are never used.
- Conditioning sets are capped (3 conditions, 3 lagged parents) to keep contingency tables estimable. This can leave confounding paths open.
- FDR control is applied after PCMCI+ and to lagged links only (tigramite does not correct lag-0 p-values in PCMCI+). This is a pragmatic correction, not a formally FDR-controlled PCMCI+.
- The expert-edge delays were looked at in notebook 02 (marked as context). `tau_max` = 10 s was not tuned on them, and `tau_max` = 5 s gives the identical graph, but the look itself is reported.
- Agreement with the expert graph is agreement with expert knowledge, not with a verified ground truth.

**Structural causal model**
- First-order binary transition mechanisms with parents at lag 0 and 1:
  - delays longer than one step (the 5 s alarm delay) are approximated as constant hazards;
  - the PLC program logic outside the subsystem graph is not modelled.
- Lag-0 parents require an acyclic graph. Undirected discovered edges are therefore *dropped* from the SCM (not oriented), so the discovered SCM loses `HP_Pump_Ok → HP_Pump_isOff` and `LT_Level_Ok → LT_Pump_Ok`.
- Unrecorded parents are assumed to be in their usual state. Variables never recorded in training are held at their usual state and reported as unsupported.
- Only binary variables are supported. Continuous mechanisms (hydraulics `Hyd_Pressure`) are not implemented, so hydraulics was not run.

**Root cause analysis**
- On coolant the method saturates (perfect for every graph), because injected fault variables never change in normal operation. It therefore cannot discriminate between graphs on this subsystem.
- Mechanism-break attribution assumes that only the intervened mechanism breaks. When a fault drives the system into states never seen in training (probe: arm stuck), downstream mechanisms also look broken and can be ranked above the cause.
- The first-order mechanisms also miss slow, uncommanded sensor drops (probe initial-position faults).
- The analysis window needs at least one observed alarm. Runs without alarms could not be scored (none occurred).

**Interventions and counterfactuals**
- Actionability is a hand-made table derived from the node descriptions, with unknown actionability treated as not actionable.
- Only hard interventions (do(X = constant)) are implemented:
  - repairing a *sensor* (probe faults) would need a soft intervention that restores a mechanism;
  - control actions are allowed in interventions but not used in recommendations.
- All intervention effects are model-predicted. The replay of the digital-twin manipulations checks them only for alarm firing and delay, and only for alarms with a mechanism in the fold.
- Individual counterfactuals rely on the assumed monotone threshold noise. They are not identified from data and could not be validated empirically: there are no counterfactual control runs.
- The validity threshold (0.9) is a convention; the continuous validity probability is always reported.

## 13. Implementation status

| Stage | Module(s) | Notebook | Status |
|---|---|---|---|
| EDA and metadata | `data.py` | 01 | implemented, executed; 50/51 metadata checks pass (the failure is a data property) |
| Preprocessing | `preprocessing.py`, `data.py` | 02 | implemented, executed; state verification passed (0 mismatches) |
| Causal discovery | `causal_discovery.py` | 03 | PCMCI+ (G²) per regime, FDR filter, leave-one-out graphs, bootstrap, sensitivity, conflict reporting; executed; independently audited (no change to the discovered edge set) |
| Graph evaluation | `graph_evaluation.py` | 04 | implemented, executed |
| SCM | `scm.py` | 05 | binary transition SCM; executed; continuous mechanisms **not implemented** |
| Root cause analysis | `root_cause_analysis.py` | 06 | mechanism-break attribution; executed |
| Interventions | `interventions.py` | 07 | hard interventions, feasibility, replay validation; executed; soft interventions **not implemented** |
| Counterfactual XAI | `counterfactuals.py` | 08 | abduction–action–prediction with recommendation search; executed |
| Framework evaluation | `evaluation.py`, `graph_evaluation.py` | 09 | expert vs discovered vs 45 perturbed graphs; executed |
| Extension: probe | all modules | 10 | discovery, graph evaluation, SCM, RCA, perturbations; executed; interventions and counterfactuals not applicable (sensor faults) |
| Hydraulics | — | — | **not run**: needs continuous SCM mechanisms for `Hyd_Pressure` |
| Second discovery algorithm | — | — | **not started**, as planned (only after the full framework works) |

Tests: 52 unit and integration tests (`python -m pytest`). They cover parsing, state reconstruction, no backfill, duplicates, recording boundaries, resampling, metadata helpers, splits, graph projection and metrics, perturbations, a synthetic SCM with known structure, RCA ranking metrics with multiple causes, feasibility rules, the counterfactual API contract, orientation-conflict reporting, FDR bookkeeping and the SCM's rejection of undirected edges.
