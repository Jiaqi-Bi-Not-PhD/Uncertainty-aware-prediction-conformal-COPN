# Uncertainty aware prediction from incomplete clinical records: a conformal prediction study of Parkinsonian classification using C-OPN data

Code accompanying the manuscript and supplements.

| Component | Code location |
|---|---|
| M1: Grouped elastic-net model | `src/keyed_model1/` |
| M2: CatBoost model | `src/keyed_catboost/` |
| M3: Additive model with pairwise interactions | `src/keyed_ebm/` |
| M4: Two-stage hierarchical model | `src/keyed_hierarchical/` |
| M5: Adaptive elastic-net models, MAR and MNAR | `src/keyed_model8_adaptive_enet/`, `src/keyed_model8_mnar_adaptive_enet/` |
| Data preparation and wrangling | `src/copn_preparation/`, `scripts/prepare_workbook.py` |
| Probability calibration, conformal prediction, reliability and evaluation | `src/ssc_uact/`, `src/ssc_uact_v3/` |
| Information-deficit and statistical analyses | `src/copn_analysis/` |
| Figure code | `figures/`, `src/copn_analysis/q_plotting.py` |
| Table code | `tables/`, `supplements/` |

C-OPN data, fitted models and derived results are confidential and are not included.
