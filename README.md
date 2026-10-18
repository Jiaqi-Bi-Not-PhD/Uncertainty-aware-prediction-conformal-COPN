# Uncertainty aware prediction from incomplete clinical records: a conformal prediction study of Parkinsonian classification using C-OPN data

Scientific source code for the probability models, conformal prediction policies, information-deficit analyses and figures described in the manuscript and Supplements S1–S8.

**Code only. C-OPN data are confidential, including every result derived from those data.** This repository contains no study inputs, participant identifiers, cohort counts, predictions, fitted parameters, selected configurations, aggregate results, tables, curve coordinates, figures or manuscript files. All scientific inputs and generated outputs must remain in an authorized private location outside this checkout. Access to this code does not grant access to C-OPN data.

Only the reported scientific methods are retained. Historical experimental approaches, execution workers, artifact certificates, manuscript-writing tools and standalone checking utilities are excluded. Algorithm settings such as tuning grids, random seeds, nominal coverage levels, fixed ensemble weights and scenario definitions are method specifications; they are not learned results.

## Code and manuscript map

| Reported component | Implementation | Purpose |
|---|---|---|
| Preparation; S1–S2 | `copn_preparation/diagnosis.py`, `versioned_forms.py`; `scripts/prepare_workbook.py` | Resolve diagnosis fields and join versioned assessments by participant key. |
| Keyed preparation; S2–S3 | `keyed_model1/common.py` and model-specific preprocessors | Align private workbook records with the privately supplied dictionary; learn preprocessing on training records. |
| M1; S3.1 | `keyed_model1/group_enet.py`, `model1_pipeline.py` | Grouped multinomial elastic-net fitting and training-only tuning. |
| M2; S3.1 | `keyed_catboost/` | CatBoost fitting, preprocessing and training-only tuning. |
| M3; S3.1 | `keyed_ebm/` | Custom additive multinomial model with selected pairwise interactions. |
| M4; S3.1 | `keyed_hierarchical/` | Two-stage hierarchical probabilities: disease versus control, then PD versus AP. |
| M5; S3.1–S3.4 | `keyed_model8_adaptive_enet/`, `keyed_model8_mnar_adaptive_enet/` | Multiple imputation, adaptive grouped elastic net, MAR/MNAR sensitivity and outcome-observation weighting. |
| Selected model fits | `copn_analysis/api.py`, `model_fitting.py` | Portable interfaces to the retained probability-model fitting routines. |
| Evaluation splits; S3.2 | `ssc_uact/splits.py` | Separate proper fitting, internal validation, conformal calibration and outer testing. |
| Probability calibration and ensemble; S3 | `ssc_uact/calibration.py`, `probability.py`, `ensemble.py` | Compare uncalibrated, temperature, vector and Dirichlet calibration; combine calibrated M2/M4 probabilities with fixed 0.6/0.4 weights. |
| A0–A3; S3.5 | `ssc_uact/conformal.py`, `robust.py`; `ssc_uact_v3/aggregate_conformal.py`, `scenarios.py` | Class-conditional LAC, scenario union and family-balanced weighted-quantile score aggregation. |
| A4–A5; S3.5 | `ssc_uact_v3/safety.py`, `metrics.py`, `singleton_reliability.py` | Reliability and study-cost enlargement of initial sets, including the reported conservative HC rule. |
| Reliability; S4 | `ssc_uact/reliability.py`, `uncertainty.py`; `ssc_uact_v3/singleton_evaluation.py`, `reliability_statistics.py`, `all_row_reliability.py` | Multicomponent error-risk models, training-selected simple scores, singleton evaluation and all-row error ranking. |
| Earlier M1 deficit; S4.4 | `ssc_uact/information_deficit.py` | Compute the earlier coefficient-norm-weighted missingness measure from private fitted M1 inputs. |
| M5-based Q; S5 | `copn_analysis/q_ridge.py`, `q_fitting.py`, `q_selection.py`, `q_arrays.py`, `q_solver_cache.py`, `q_softmax_solver.py`, `q_group_kernel.py`, `q_gradient.py`; `copn_preparation/q_states.py` | Retain Q optimizer arithmetic, candidate ranking, source weighting, missing-state mappings and development-derived quartiles. |
| Statistical summaries; S7–S8 | `copn_analysis/q_outcome.py`, `q_summaries.py`; `ssc_uact_v3/policy_evaluation.py`, `bootstrap.py`, `stratification.py` | Participant-cluster summaries and paired intervals, exact binomial bounds, subgroup metrics, set size and information gain. |
| Manuscript and supplementary figures | `figures/`, `copn_analysis/q_plotting.py` | Draw the overview, budget comparisons, set metrics, scenario-variation and earlier-deficit plots, Q distributions, selection patterns and risk–coverage curves from explicit private inputs. |
| Main and supplementary tables | `tables/build_highest_variation_table.py`, `supplements/build_table_s13.py` | Format the reported robustness summaries from explicit private aggregate inputs. |

Package and file names such as `model5_hierarchical` and `model8_*` retain original implementation identifiers for compatibility. In manuscript notation these are **M4** and **M5**, respectively. `keyed_ebm` implements the custom additive model; it does not require the `interpret` package.

## Installation

Use Python 3.11 or later in a separate virtual environment:

```bash
python -m pip install -e '.[catboost,figures]'
```

NumPy, pandas, SciPy, scikit-learn, openpyxl and numerical dependencies are declared in `pyproject.toml`. CatBoost and plotting dependencies are optional extras. Editable installation may create ignored package metadata locally; no datasets are installed or downloaded.

## Using the code privately

Supply authorized workbook and dictionary paths through the configuration objects. Their path defaults are intentionally empty. Preserve training, tuning, calibration and test separation when calling the model interfaces. Model selection, imputation, preprocessing and Q coefficient fitting must exclude the target records specified in Supplement S5; `q_arrays` alone does not enforce that study design.

```bash
python scripts/prepare_workbook.py \
  --input /private/authorized/raw.xlsx \
  --output /private/authorized/prepared.xlsx
```

`copn_analysis.api.fit_selected_probability_model` fits M1–M4 from a training-selected parameter row. `fit_selected_m5` fits the MAR or MNAR adaptive model on a training-derived imputation design. Selected parameter rows and fitted objects are private inputs and outputs.

For Q, `q_selection.select_q_candidate` preserves the original ordering: higher balanced accuracy, higher AP average precision, lower log loss, then registry order, among candidates that converge in every required pair fit. `q_fitting.fit_selected_q_model` exposes the final selected-fit numerical kernel. `q_arrays.source_weights` divides each coefficient-block norm by the square root of its encoded dimension and normalizes across sources; `q_values` applies the primary and sensitivity state mappings. `freeze_q_groups` uses development-score quartiles with the higher-quantile convention and keeps equal scores together.

Conformal probability arrays use class order PD, AP, HC. The scenario registry contains the reported five families; aggregation gives each family equal total weight. The retained LAC implementations preserve infinite cutoffs when a finite-sample rank is unattainable. Reliability and set-policy thresholds must be developed on the allowed training outputs and supplied privately.

## Figures and tables

Plotting and table programs require private input paths and an output location outside the checkout. They contain no embedded study results. For example:

```bash
python figures/build_budget_performance.py \
  --data /private/authorized/budget_metrics.csv \
  --output-dir /private/authorized/figures

python figures/build_figure_02.py \
  --data /private/authorized/policy_metrics.csv \
  --output-dir /private/authorized/figures

python figures/build_figure_03.py \
  --ssi /private/authorized/scenario_variation.csv \
  --m1 /private/authorized/earlier_deficit.csv \
  --output-dir /private/authorized/figures

python figures/build_reliability_risk_coverage.py \
  --data /private/authorized/risk_coverage.csv \
  --summary /private/authorized/reliability_summary.json \
  --output-dir /private/authorized/figures

python tables/build_highest_variation_table.py \
  --source /private/authorized/robustness.csv \
  --output /private/authorized/tables/highest_variation.tex

python supplements/build_table_s13.py \
  --ssi /private/authorized/robustness.csv \
  --m1 /private/authorized/earlier_deficit_summary.csv \
  --output /private/authorized/tables/supplement_robustness.tex
```

The overview accepts `--output`. Q plotting functions accept private summary dictionaries and an external output path. Required input column names are defined by each reader and plotting function. No example input files or expected study outputs are deposited.

This source release exposes reusable scientific routines, rather than the confidential execution archive. Reproducing manuscript numerical results requires authorized private data, split assignments, selected parameters and fitted artifacts, none of which are included here.

## Confidentiality

Keep this repository private. Never commit or upload C-OPN raw, processed, row-level, aggregate or derived data, including fitted models, coefficient tables, learned thresholds, result-bearing configurations, generated tables or figures. Keep confidential inputs and outputs outside the checkout. The repository uses a source-file allowlist in `.gitignore`; review explicit staged files before any later commit because ignore rules cannot detect data embedded inside source code.
