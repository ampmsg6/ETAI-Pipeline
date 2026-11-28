# Week 11: requirements and evidence

Source: local `aula 11/12_outliers.ipynb`, cells 23-30, and the professor's public pipeline at [bcd7b53](https://github.com/sofiacper/ETAI-Pipeline/tree/bcd7b5386d3a5b6130432c560862b60085425ede), verified on 28 November 2026. `assignment_reference.json` and `metadata.json` preserve references, notebook/data hashes, seeds and installed versions.

| Required step | Implementation and evidence |
|---|---|
| 1: `AnomalyScore` | `src/preprocessing.py`: Isolation Forest or LOF with novelty enabled; fit inside the Pipeline; append a score without dropping rows. LOF uses training factors in fit_transform and novelty scores in transform. |
| 1: pipeline integration | `src/model.py`: optional anomaly_config, after prep and before selection, retaining the project's full-config API and compatibility with old configs. |
| 2: config and main | `anomaly` in COMPAS/Adult configs, default none; main.py passes it to the builder, supports --anomaly, propagates feature names, and records the method in console, scoreboard and full report. |
| 3: prediction | Real predict.py CLI called on the 20-person example for all three saved methods and the default model. Every row, ID and order preserved; saved files unchanged during inference. Reviewable CSVs in predictions/week11. |
| 4: pre-run guess | README prediction was written before the experiments: little or no consistent gain; rearrest is not equivalent to rare combinations. |
| 4: unusual-person outcomes | anomaly_rates.csv: top 5% vs remaining rows for each detector, separately for held-out fold scores and notebook-style development-fit scores. No reserved data used. |
| 4: three model runs | compare_week11 calls the same run_pipeline entry point for none, isolation_forest and lof, with tuning off and every other statistical setting fixed. Scoreboard, folds, eight metrics and full reports/configs provided. |
| 4: interpretation | README compares the pre-run guess with measured ROC-AUC, rearrest rates and feature selection; none retained. Three corrected paired comparisons with unadjusted exploratory p-values. |

All recipes use 5,771 development records, seed 42 and five stratified folds. XGBoost: 300 trees, learning rate 0.05, depth 4; median imputation, target encoding, robust scaling, two-vote selection. The 1,443 reserved records stay outside training and evaluation. Mean ROC-AUC: none 0.7208, Isolation Forest 0.7163, LOF 0.7150. Both score methods lose to none on all five paired folds. Final refits discard the anomaly score; fold-level selection is documented separately.

Held-out top-5% rearrest rates: Isolation Forest 170/290 (58.62%) vs 2,431/5,481 (44.35%); LOF 166/290 (57.24%) vs 2,435/5,481 (44.43%). Overall: 2,601/5,771 (45.07%). Top groups are selected within each validation fold without labels, with stable input-order tie breaking. Individual scores remain locally in results/week11/anomaly_scores.csv; public tables contain aggregate rates only. LOF k=20 exceeds every training fold's largest duplicate group (maximum 11), and the full-development maximum (14).

Verification: 27 unit/regression tests passed. validate_week11 independently recomputes all 120 metric values from OOF predictions, all three corrected tests and 12 rate groups, independently refits 15 fold models and matches their OOF probabilities, checks feature names and LOF neighbourhoods, and invokes the real prediction CLI. None reproduces the five-fold, eight-metric Week 10 median baseline. Current Adult cleaning/config/prediction tests also pass. No new dependency was installed.

Optional challenges (training-only removal, quantile clipping, tuning k) are not included. Models and individual OOF data stay local, following the previous weeks' delivery format. Predictions and compact reports are committed to GitHub.
