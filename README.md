# ETAI Predictive Pipeline

Antonio Gonçalves — 20260536

## What this project does

This project predicts whether a person will be rearrested within two years (`two_year_recid`) using the COMPAS case dataset. The model uses age, sex, charge degree, and prior offense counts. It does not use `race` or COMPAS's own risk score as predictors. We keep those fields separately to check false-positive rates by race and compare with COMPAS. The [data dictionary](data/README.md) describes the columns.

The current configuration runs XGBoost and jointly tunes its tree settings, regularisation and, when feature selection is enabled, the number of votes required to keep a feature. It reports eight metrics and selects hyperparameters by ROC-AUC. Numeric features use median imputation and robust scaling; categorical features use target encoding. Preprocessing and selection are fitted inside every training fold. The settings are in `config.yaml`.

## Run the pipeline

From the project root, create an environment and install the dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source venv/bin/activate`. Then run:

```powershell
python main.py
```

The command cleans the CSV and removes duplicate labelled rows before making a stratified 80/20 development/test split. It reports default five-fold CV, evaluates tuning with nested CV, and tunes once more on all development rows to choose the final hyperparameters. Classification, confusion matrices and race-group false-positive rates use nested out-of-fold probabilities at a fixed threshold of 0.5 when tuning is enabled. It then refits the selected pipeline on all development rows. The reserved test rows are not scored. A timestamped report, the run configuration, fold tables, OOF probabilities and Optuna trials are saved in `results/`; each execution also adds a row to `results/runs.csv`.

Use short runs while checking a change, or select another model without editing the saved configuration:

```powershell
python main.py --model decision_tree --trials 5
python main.py --model adaboost --no-tuning
python main.py --model oblique_random_forest --trials 5
python main.py --model logistic_regression --features priors_per_adult_year
python main.py --model hist_gradient_boosting --trials 5
python main.py --model naive_bayes --trials 5
python main.py --model dummy
python main.py --model xgboost --no-tuning --selection on
python main.py --model xgboost --trials 5 --selection off
```

`--model` selects the notebook's default parameters for that model. Without it, `main.py` uses the configured model and parameters. `--features` selects an explicit list of engineered features; an empty list selects the baseline. `--selection on|off` overrides learned feature selection. CLI overrides apply only to the current run. The full default budget is 30 trials, with five inner and five outer folds. On macOS, XGBoost may also require `brew install libomp`.

## Pipeline progress

| Week | Focus | Implemented in this project |
|---|---|---|
| 2 | Baseline | Logistic regression and decision tree, train/test evaluation and a simple fairness comparison. |
| 3 | EDA and preprocessing | Data diagnosis, fixed cleaning and recovery rules, imputation, missingness indicators, and preprocessing comparisons. |
| 4 | Cross-validation | Row-preserving cleaning, exact deduplication, reserved test rows, whole-pipeline CV, OOF reports, and development-only comparisons. |
| 5 | Hyperparameter tuning | Seeded Optuna studies, search spaces for logistic regression and decision tree, nested CV, an optimism check, and final development-set tuning/refit. |
| 6 | Ensembles and features | Oblique Random Forest and AdaBoost, ensemble search spaces, a corrected paired test, and separate nested-CV ablations of three fixed engineered features. |
| 7 | Boosting and metrics | Gradient Boosting, HistGradientBoosting and Gaussian Naive Bayes; eight metrics, ROC-AUC selection, OOF probabilities and confusion matrices; a nine-model scoreboard with dummy floor, untuned and nested tuned scores. |
| 8 | XGBoost and feature selection | XGBoost with nine tuned hyperparameters; fold-fitted `VoteSelector` between preprocessing and model; `min_votes` tuned jointly, temporary pipeline cache, selected-feature reports and paired nested-CV comparison with selection on/off. |

## Week 8: XGBoost and feature selection

The implementation follows `08_feature_selection.ipynb` and the [Week 8 course pipeline](https://github.com/sofiacper/ETAI-Pipeline/tree/7dc7713640fe3b97f16be6df6be67ce417f9d950). XGBoost adds regularisation within each tree through `gamma`, `min_child_weight`, `reg_alpha` and `reg_lambda`; its search space also covers tree count, learning rate, depth and row/column subsampling. We tune the tree count with nested CV, without using validation or test labels for early stopping. The existing fold-fitted imputation is retained for a consistent comparison.

`VoteSelector` combines a 200-tree random forest, gradient boosting and L1 logistic regression (`C=0.01`). The trees vote for above-mean importance; logistic regression votes for nonzero coefficients. Selection is fitted after preprocessing and before the final classifier, inside each inner and outer training fold. With selection enabled, Optuna tunes `select__min_votes` from 1 to 3 alongside the nine XGBoost hyperparameters. A temporary `Pipeline(memory=...)` cache reuses fitted steps only for identical training data, labels and parameters; it is removed even if a run fails. The dummy floor has no feature selection.

Both recipes use the same 5,771 development records, five outer folds, five inner folds, seeds 42 and 30 trials per study, followed by another 30-trial study for the final refit on development data. The corrected paired test uses selection-on minus selection-off ROC-AUC across the five outer folds, with validation/training ratio 1/4. The 1,443 reserved test records are not evaluated. This compares two specified tuning procedures; their later selection is not independently assessed on a final test set.

### Results: 7 November 2026

| XGBoost recipe | Untuned ROC-AUC | Tuned nested ROC-AUC | Final inputs | Dummy floor |
|---|---:|---:|---:|---:|
| Selection on | 0.7208 ± 0.0125 | 0.7260 ± 0.0129 | 2 of 10 | 0.5000 |
| Selection off | 0.7245 ± 0.0184 | 0.7285 ± 0.0146 | 10 of 10 | 0.5000 |

**Feature-selection conclusion.** Selection's paired mean change was −0.0025 ROC-AUC, with 2/5 fold wins and corrected t=−1.1116, p=0.3286. This comparison did not detect a difference between the recipes, but does not establish equivalence. The final selector chose `min_votes=2` and kept `numeric__age` and `numeric__priors_count`, both with unanimous votes, reducing the transformed inputs from ten to two. I keep selection enabled for the configured Week 8 model because the observed ranking scores are close and the smaller model is easier to describe, accepting its slightly lower observed mean and the extra training cost of the three voters.

**B2: stability across folds.** The outer studies chose `min_votes` of **1, 1, 3, 1, 2**, keeping **2, 3, 2, 3, 2** features respectively. Age and prior count appeared in every outer model; the second and fourth also kept `categorical__age_cat`. The threshold varies while the core features stay stable, so these runs do not identify a uniquely best number of votes.

Reproduce the submission with:

```powershell
python -m unittest discover -s tests -v
python -m scripts.compare_week8 --trials 30 --resume
```

The comparison writes both executions to `results/runs.csv` and publishes [scoreboard.csv](reports/week8/scoreboard.csv), [folds.csv](reports/week8/folds.csv), [all metrics](reports/week8/metrics.csv), [feature votes](reports/week8/features.csv), [final-study trials](reports/week8/trials.csv), [the corrected comparison](reports/week8/selection_comparison.csv), full text reports and exact configurations in [reports/week8](reports/week8). [metadata.json](reports/week8/metadata.json) records data/software identity, seeds and budget. Individual OOF probabilities stay locally in `results/week8/`, following the previous submissions. `requirements-lock.txt` records the installed dependency versions. `--resume` checks the recipe and complete result files before reusing a run.

To run a short comparison separately from the submitted scoreboard:

```powershell
python -m scripts.compare_week8 --trials 5 --scoreboard results/week8_short/runs.csv --output-dir results/week8_short --publish-dir results/week8_short/tables
```

The optional B1 comparison of other selector methods is not included. The outer-fold tables do include B2's chosen `select__min_votes` and kept features. The Week 6 and Week 7 comparison scripts explicitly disable selection to preserve their original experimental recipes.

## Week 7: boosting, metrics and the scoreboard

The required changes in `06_boosting.ipynb` and `07_metrics.ipynb` follow the [course pipeline](https://github.com/sofiacper/ETAI-Pipeline/tree/a170b26b131659ec9b5aff6d818d21354c3930c2), adapted to this project's existing cleaning and full-config preprocessing interface. Gradient Boosting, HistGradientBoosting and Gaussian Naive Bayes join the six existing model types. The two boosters jointly tune the number of trees, learning rate and tree size; Gaussian Naive Bayes tunes `var_smoothing`. HistGradientBoosting can handle missing values natively, but this comparison retains the same fold-fitted imputers and missingness flags for every model.

**Selection metric.** We choose ROC-AUC because this task produces a risk ranking, and we want to assess how well recorded positive outcomes rank above negative outcomes across thresholds. Choosing hyperparameters by ROC-AUC keeps that ranking question separate from the decision threshold, fixed here at 0.5. Precision, recall, F1, F2, balanced accuracy and the race-group false-positive rates remain necessary to describe the errors at that threshold; a strong ranking alone does not establish acceptable errors or fairness.

Every run reports `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `fbeta_2`, `roc_auc` and `average_precision`. F2 puts four times as much weight on recall as precision in the harmonic-mean formula. Average precision summarises the precision-recall curve; its no-skill reference is the positive-class prevalence, whereas the ROC-AUC floor is 0.5. All threshold reports use out-of-fold probabilities from models that did not train on the reported rows. During nested CV, those rows are also excluded from the corresponding Optuna study.

The scoreboard stores one row per execution, including the untuned score, dummy floor, final-study hyperparameters and each outer-fold score. A hash of the input data, cleaning/splitting settings, fold configuration, relevant source files and dependency versions groups compatible runs automatically. This prevents comparing scores from different data or folds as though they were paired. The standard deviations describe variation over folds, not confidence intervals; corrected paired tests remain exploratory comparisons without a multiple-comparison adjustment.

### Final scoreboard: 31 October 2026

All nine models use the same 5,771 development records and the same five outer folds. Every tuned model uses 30 Optuna trials per five-fold inner study, and a further 30-trial study on all development rows for the final refit. Values below are ROC-AUC means ± sample standard deviations over the outer folds; the dummy has no tuning.

| Model | Untuned CV | Tuned nested CV | Dummy floor |
|---|---:|---:|---:|
| Gradient Boosting | 0.7280 ± 0.0149 | 0.7284 ± 0.0153 | 0.5000 |
| AdaBoost | 0.7233 ± 0.0163 | 0.7281 ± 0.0143 | 0.5000 |
| HistGradientBoosting | 0.7188 ± 0.0187 | 0.7280 ± 0.0155 | 0.5000 |
| Oblique Random Forest | 0.6856 ± 0.0170 | 0.7275 ± 0.0130 | 0.5000 |
| Random Forest | 0.6845 ± 0.0193 | 0.7250 ± 0.0156 | 0.5000 |
| Decision tree | 0.6145 ± 0.0179 | 0.7197 ± 0.0149 | 0.5000 |
| Logistic regression | 0.7185 ± 0.0138 | 0.7181 ± 0.0136 | 0.5000 |
| Gaussian Naive Bayes | 0.6848 ± 0.0111 | 0.7052 ± 0.0117 | 0.5000 |
| Dummy (majority class) | 0.5000 ± 0.0000 | Not tuned | 0.5000 |

Exact scores, settings and paired outer folds are in the [scoreboard CSV](reports/week7/scoreboard.csv), [fold tables](reports/week7/folds.csv) and [all-metric summary](reports/week7/metrics.csv). The [metadata](reports/week7/metadata.json) identifies the data, seeds and software versions. Each model also has a text report and complete run configuration in that folder.

Gradient Boosting has the highest observed mean, but its leads over AdaBoost, HistGradientBoosting and the oblique forest are below 0.001 ROC-AUC. The corrected paired tests give p=0.858, p=0.556 and p=0.738 respectively, so these runs do not establish a winner among those four. Its gain over logistic regression is 0.0103 (5/5 fold wins, corrected p=0.0059), which is evidence for better ranking in this exploratory comparison; the full [paired comparisons](reports/week7/model_comparisons.csv) remain unadjusted for multiple comparisons. We keep Gradient Boosting as the configured Week 7 model, as the assignment requests.

Tuning made the largest change to the single tree (+0.1051 ROC-AUC) and improved both forests by about 0.04; it changed Gradient Boosting's mean by only +0.0004. These are ROC-AUC results, so their numerical differences from Week 6's accuracy figures are not measured performance gains. Nested CV evaluates the specified tuning procedure for each model; choosing the highest row across nine models is another selection step, without a final test-set assessment here.

Reproduce the complete nine-model comparison and its published tables with:

```powershell
python -m unittest discover -s tests -v
python -m scripts.compare_week7 --trials 30 --resume
```

`--resume` reuses completed runs only when the recipe, data, source, dependency versions and trial budget match. Models use the same five outer and five inner folds, seeds 42, and baseline features. The dummy has no tuning search. CV runs in five processes, each restricted to one native thread, and forest fits use one worker to avoid nested CPU parallelism. Full local evidence is in `results/week7/`; the submission includes [results/runs.csv](results/runs.csv) and compact reports in [reports/week7](reports/week7). To test with five trials without mixing short runs into the final board:

```powershell
python -m scripts.compare_week7 --trials 5 --scoreboard results/week7_short/runs.csv --output-dir results/week7_short --publish-dir results/week7_short/tables
```

Part B's `VoteSelector` is optional and is not enabled in this submission. The required comparison keeps `engineered_features: []`, following the Week 6 ablation results. The 1,443 reserved test records remain unevaluated in Week 7; their historical Week 3 use is documented below.

## What changed in Week 3

| Step | What the pipeline does | Why |
|---|---|---|
| Diagnose | Count missing, invalid, repeated, and inconsistent values. | Make cleaning decisions traceable. |
| Clean | Standardize category spelling, mark invalid values as missing, fix supported inconsistencies, and remove exact duplicate rows. | Keep one copy of each recorded case without deleting distinct cases. |
| Split | Use `train_test_split(..., stratify=y, random_state=42)`. | Keep the class proportions similar in train and test. |
| Prepare features | Impute, encode, and scale inside a scikit-learn `Pipeline`. | Learn medians, categories, and scale parameters from training rows only. |
| Compare | Try encoder/scaler combinations and another imputer; compare the cleaned models with the Week 2 baseline. | Check whether the changes help rather than assuming they do. |

Week 2 used complete-case deletion and `get_dummies` before training. Its logistic regression and decision tree are the baselines in the comparison at the end of this README.

### Cleaning decisions

The CSV has 7,286 rows. It contains 72 repeated `id` values, each with two rows that are **identical in every column**. `drop_duplicates()` removes the 72 extra copies, leaving 7,214 distinct rows. It does not discard rows merely because an `id` repeats. The data dictionary calls `id` an internal record identifier; it does not tell us whether different IDs belong to the same person. After exact deduplication, every `id` in this CSV is unique. For that reason, the split does not group by `id`.

The cleaning code converts numeric text and treats `?`, `-`, `n/a`, and similar placeholders as missing. It unifies spellings such as `female`/`Female` and `Felony`/`F`. Configured bounds mark impossible numeric values as missing: the diagnosis found eight invalid ages, six COMPAS scores outside 1–10, five negative juvenile felony counts, and six adult-prior counts outside 0–60 among the distinct rows.

We also checked whether related columns agree. Six valid ages disagreed with `age_cat`, so we corrected those **categories** from the numeric ages; we did not invent an exact age from an age band. Ten valid COMPAS scores disagreed with `score_text`, so we aligned the label with the score's 1–4, 5–7, or 8–10 band. An invalid score leaves its comparison label unavailable.

The pipeline drops three redundant columns: `prior_offenses` repeats `priors_count`; `age_in_months` derives from `age`; and `juvenile_total` is the sum of the juvenile counts. Before dropping the total, we used it to recover 216 missing `juv_fel_count` values when the other two counts made the answer unambiguous. Five juvenile felony counts remain missing or invalid.

The diagnosis checks associations between missingness and other observed columns using chi-square and Cramér's V. These associations can support a missingness indicator, but cannot by themselves prove MCAR or MNAR. The model retains `priors_count_was_missing` and `c_charge_degree_was_missing` before filling the missing values.

### Training and model choice

`src/preprocessing.py` returns model features, the target, and separate fields for auditing. Numeric features use a median imputer and robust scaler. Categorical features use a most-frequent imputer and target encoder. The encoder learns category-to-target patterns from training data; unfamiliar categories can still be transformed at prediction time. These learned steps fit on training data inside the model pipeline. `race`, `decile_score`, `score_text`, and `id` are not model inputs.

`scripts/tune_preprocessing.py` compared four encoders with four scalers using three stratified cross-validation folds **inside the training set**. Target encoding with robust scaling had the highest mean accuracy, **0.6754**, and remains the configured pair. One-hot with standard scaling scored **0.6690**. Choosing a preprocessing recipe from training folds avoids using the held-out test set to pick the encoder and scaler. The full table is in `results/week3_tuning.csv`.

We also compared median imputation with a five-neighbor `KNNImputer`. With the selected encoder and scaler, KNN tied median on the cleaned logistic model's shared-test accuracy; median remains the simpler configured choice. Logistic regression has higher shared-test accuracy than the cleaned decision trees. The test results below also show that the cross-validation winner did not outperform the Week 2 logistic baseline.

## Project files and reproducibility

| Path | Purpose |
|---|---|
| `main.py`, `config.yaml` | Run the configured pipeline. |
| `src/data_diagnostics.py`, `scripts/diagnose_data.py` | Check the data and write `data/diagnosis_log.json`. |
| `src/preprocessing.py` | Clean, split, and build the feature transformer. |
| `src/model.py`, `src/evaluate.py`, `src/results.py` | Build the model, report its results, and save each run. |
| `scripts/tune_preprocessing.py` | Compare encoder and scaler combinations. |
| `scripts/compare_week3.py` | Compare Week 2 and cleaned models on shared test records. |
| `scripts/compare_week4.py` | Compare Week 4 models and preprocessing choices on the same development folds. |
| `src/tuning.py` | Run Optuna tuning, nested cross-validation, and the tuning report. |
| `scripts/compare_week6.py` | Run five baseline models and nine feature ablations, with matching checkpoints for resumable runs. |
| `tests/test_week6.py` | Check fold isolation, feature missingness, ensemble tuning, and paired comparisons. |

To regenerate the diagnosis, experiment tables, and current run report, use these commands from the project root:

```powershell
python -m scripts.diagnose_data
python -m scripts.tune_preprocessing
python -m scripts.compare_week3
python -m scripts.compare_week4
python main.py
```

The scripts write `results/week3_tuning.csv`, `results/week3_comparison.csv`, `results/week4_comparison.csv`, and `results/week4_repeated_cv.csv`. Reports from `main.py` are also in `results/`.

To reproduce the Week 5-6 experiments and compact result tables:

```powershell
python -m unittest discover -s tests -v
python -m scripts.compare_week6 --resume
```

The comparison uses 30 trials per study and 200 trees in each forest. It runs five baseline models and each of three features separately for logistic regression, Random Forest, and AdaBoost (14 runs). Full reports and checkpoints live in `results/week6/`; the compact tables and environment metadata live in `reports/week6/`. `--resume` reuses a completed run only when its statistical settings, input data, source code and dependency versions match; CPU worker counts may differ. Ensemble CV normally uses five processes, with one worker per forest fit and one native thread per process, to avoid parallelizing both levels. A five-trial timing check confirmed identical forest scores with parallel folds and parallel trees. Set `--parallel-folds 1` for serial CV. For a short comparison, use `--trials 5 --output-dir results/week6_short --publish-dir results/week6_short/tables`, so it does not overwrite the final published tables.

`requirements-lock.txt` records the installed versions used for the final experiments (Python 3.13.3). To recreate that environment, install it with `python -m pip install -r requirements-lock.txt`. The ordinary `requirements.txt` retains the course's supported minimum versions.

## Week 2 versus the cleaned pipeline

**What “class 1” means.** The target column is `two_year_recid`: `1` means the person **was rearrested within two years**; `0` means they were **not rearrested within two years**. It describes the recorded outcome, not the number of crimes. The common test set below has 558 class-1 records and 671 class-0 records.

| Measure | How to read it |
|---|---|
| Accuracy | Percentage of all test records predicted correctly. |
| Precision (class 1) | Of the records predicted as rearrested, the percentage actually rearrested. |
| Recall (class 1) | Of the records actually rearrested, the percentage the model found. |
| F1 (class 1) | A combined score for class-1 precision and recall; it falls when either is low. |
| False positive (FP) | Predicted rearrest, but the recorded outcome was no rearrest. |
| False negative (FN) | Predicted no rearrest, but the recorded outcome was rearrest. |

We used one stratified split (`test_size=0.2`, `random_state=42`). All six models were scored on the **same 1,229 test records** with no missing raw values. The Week 2 reconstruction trained on 5,018 complete rows, including duplicate copies in its training portion. The cleaned versions trained on 5,771 distinct rows, including cases that needed imputation. All percentages below refer to this shared test set; precision, recall, and F1 refer to **class 1**.

**Where the error counts come from:** `scripts/compare_week3.py` fits each model on its training rows and calls `model.predict(X_test)` on the shared test rows. It compares each prediction with that row's recorded `two_year_recid` value (`y_test`) using `confusion_matrix(y_test, test_pred, labels=[0, 1])`. False positives and false negatives are counts of those disagreements; they are calculated after prediction, not columns found in the original CSV. For the cleaned logistic regression with median imputation:

| Recorded outcome | Predicted 0: no rearrest | Predicted 1: rearrest |
|---|---:|---:|
| Actual 0: no rearrest | 536 correct (true negatives) | **135 false positives** |
| Actual 1: rearrest | **281 false negatives** | 277 correct (true positives) |

These four cells add up to 1,229 test records. For example, accuracy is `(536 + 277) / 1229 = 66.15%`, and class-1 recall is `277 / (277 + 281) = 49.64%`. The same calculation gives the FP and FN counts for every other model in the table.

| Model | Data and imputer | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | Week 2; drop incomplete rows | 66.97% | 65.14% | 58.60% | 61.70% | 175 | 231 |
| Logistic regression | Cleaned; median | 66.15% | 67.23% | 49.64% | 57.11% | 135 | 281 |
| Logistic regression | Cleaned; KNN | 66.15% | 67.15% | 49.82% | 57.20% | 136 | 280 |
| Decision tree | Week 2; drop incomplete rows | 60.29% | 57.74% | 46.77% | 51.68% | 191 | 297 |
| Decision tree | Cleaned; median | 60.46% | 57.66% | 48.57% | 52.72% | 199 | 287 |
| Decision tree | Cleaned; KNN | 60.13% | 56.77% | 51.08% | 53.77% | 217 | 273 |

The **cleaned logistic regression became worse at finding class-1 cases**. With median imputation, it avoided 40 false positives compared with Week 2 (175 to 135), but produced 50 more false negatives (231 to 281). Its precision rose, but recall fell from 58.60% to 49.64%. That lower recall explains most of the F1 drop from 61.70% to 57.11%. Accuracy also fell slightly, from 66.97% to 66.15%.

For the **decision tree**, median imputation slightly improved both accuracy and F1. KNN improved F1 further by finding more class-1 cases, but also produced more false positives; its accuracy fell just below the Week 2 tree. The logistic models still had higher overall accuracy than the trees on these records.

The comparison covers several changes at once: training rows, duplicate removal, imputation, encoding, and scaling. It cannot tell us which individual change caused a result. Target encoding with robust scaling won the training-fold comparison, but the cleaned logistic model did not beat the Week 2 logistic baseline on these held-out records. The Week 3 logistic model scored **65.8% accuracy on the full 1,443-record cleaned test set**. That set includes records excluded from this common comparison, so its accuracy answers a different question. The exact values, including training accuracy and all error counts, are in `results/week3_comparison.csv`.

## Class 4: preprocessing and cross-validation

`clean_dataset()` now preserves the input row count and order. It can prepare unlabelled data without losing a prediction row. `drop_duplicate_rows()` is called on the labelled dataset before the split. It removes exact copies and raises an error if the same `id` has conflicting case data. The 72 copies in this dataset are still removed, leaving 7,214 records. The Week 3 corrections to juvenile counts, age categories, and COMPAS score categories remain in place.

`split_dev_test()` reserves 1,443 rows (20%) and leaves 5,771 development rows. The preprocessing and model form one scikit-learn `Pipeline`. For a single development holdout, the pipeline is fitted on 75% of development and checked on the other 25%. Five-fold stratified cross-validation then fits a fresh copy of the whole pipeline on each training fold. Imputation, encoding, and scaling are therefore learned again without the validation fold. The report gives each fold's train and validation accuracy, their mean and standard deviation, and the train-validation gap. Classification and race-group false-positive rates use out-of-fold predictions. Finally, the configured pipeline is refitted on all 5,771 development rows. The fitted model is kept in memory; the text report is saved to `results/`.

The table compares the Week 3 logistic recipe with options tried on the same Week 4 development rows and the same five folds. Holdout accuracy is one 75/25 development split. F1 is for class 1 (`two_year_recid=1`).

| Recipe | Holdout accuracy | CV accuracy, mean ± std | CV F1, mean | Train–validation gap, mean |
|---|---:|---:|---:|---:|
| Dummy, majority class | 0.550 | 0.549 ± 0.000 | 0.000 | 0.000 |
| Logistic, median + target + robust | 0.674 | 0.672 ± 0.013 | 0.583 | +0.004 |
| Logistic, KNN + target + robust | 0.681 | 0.674 ± 0.014 | 0.585 | +0.000 |
| Logistic, median + count + robust | 0.662 | 0.674 ± 0.013 | 0.600 | −0.000 |
| Decision tree, median + target + robust | 0.595 | 0.614 ± 0.012 | 0.543 | +0.082 |
| Random forest, default leaves | 0.641 | 0.646 ± 0.016 | 0.591 | +0.086 |
| Random forest, minimum 5 rows per leaf | 0.670 | **0.675 ± 0.013** | **0.614** | +0.026 |

One split favoured KNN imputation for logistic regression. Across five folds its accuracy advantage over median was small. Across three repetitions of five folds, KNN scored 0.672 versus 0.673 for median. The forest with at least five rows per leaf scored 0.676, an average paired accuracy gain of 0.0027 and an F1 gain of 0.0258 over the logistic median recipe. Those differences are small compared with fold-to-fold variation, so they do not establish that the forest will win on new data. We selected it for the observed CV accuracy and class-1 F1; the unpruned forest had a much larger train-validation gap. All nine initial candidates and the repeated comparison are in the Week 4 CSV files.

The logistic model with count encoding raised convergence warnings in three of the five folds. Count-encoded values reach about 4,700 and are not scaled, unlike the numeric features. In a separate check, adding a `StandardScaler` before the model removed the warnings; CV accuracy stayed at 0.674 ± 0.013 and F1 at 0.600, so the table value stands. The configured recipe does not use count encoding.

The Week 3 score of 0.658 came from the old 1,443-row test set. Week 4's holdout and CV scores come from the 5,771 development rows and answer a different question, so their numerical difference is not a measured gain on the same test rows. The same 42-seed test rows were examined during Week 3; from Week 4 onward this script sets them aside, but they are not historically untouched. We will not use them to choose further recipes.

## Model evaluation: Weeks 5–6

The Week 5 tuning and Week 6 ensemble builders follow the [course pipeline at the Week 6 commit](https://github.com/sofiacper/ETAI-Pipeline/tree/c261985d9b32495520eff123ba559d162a087b49). They are adapted to this project's full-config preprocessing interface. The project's cleaning and count recovery rules are retained. Logistic regression also gets the Week 5 challenge search space for `C`, from `0.0001` to `100` on a log scale.

Each final run uses the same 5,771 development records, the same five stratified outer folds (seed 42), and accuracy as its objective. A seeded Optuna TPE study tries 30 candidates, each scored by five inner folds of the whole pipeline. Preprocessing is fitted afresh on the training portion of every inner fold. In nested CV, each outer-validation fold is excluded from the whole study used to predict it. A final 30-trial study on all development rows supplies the parameters for the final refit. The best trial's CV score is a selection score; the nested outer-fold mean ± sample standard deviation is the result reported for a tuned model.

The forests use 200 trees throughout, with tree size tuned rather than tree count. Ordinary Random Forest tunes `max_depth`, `min_samples_leaf`, and `max_features`; the oblique forest tunes its base-tree depth/leaf size and projection `n_components`. AdaBoost tunes base-tree depth, boosting rounds and learning rate together. The mean gap is training accuracy minus validation accuracy. The optimism check is mean `inner_best` minus mean outer validation, using the five nested folds; it is separate from the best trial of the final study.

### Ensembles: final 30-trial results

These results were computed in this project on 14 October 2026. The five rows use the baseline feature set, with the notebook's default model parameters before tuning. All percentages can be read by multiplying the accuracy by 100. The hyperparameters below come from the final study on all development rows; individual outer folds select their own parameters.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Best trial | Optimism | Chosen hyperparameters |
|---|---|---|---:|---:|---|
| Logistic regression | 0.672 ± 0.013 (+0.004) | 0.673 ± 0.015 (+0.004) | 0.6777 | +0.0022 | `C=0.0794` |
| Decision tree | 0.614 ± 0.012 (+0.082) | 0.676 ± 0.019 (+0.009) | 0.6803 | −0.0007 | `max_depth=6`, `min_samples_leaf=110`, `criterion=entropy` |
| Random Forest | 0.647 ± 0.016 (+0.085) | 0.681 ± 0.015 (+0.008) | 0.6838 | +0.0012 | `max_depth=24`, `min_samples_leaf=45`, `max_features=0.5` |
| Oblique Random Forest | 0.648 ± 0.014 (+0.114) | 0.682 ± 0.014 (+0.009) | 0.6831 | +0.0003 | tree `max_depth=4`, `min_samples_leaf=3`; projection `n_components=8` |
| AdaBoost | 0.676 ± 0.013 (+0.006) | 0.684 ± 0.014 (+0.004) | 0.6834 | −0.0005 | base tree `max_depth=4`, `n_estimators=220`, `learning_rate=0.0251` |

Tuning improved the single tree by 0.062 and the forests by about 0.033–0.034, larger than their outer-fold standard deviations, while logistic regression (+0.001) and AdaBoost (+0.008) gained less than their fold-to-fold spread. AdaBoost had the highest mean, but its 0.0029 lead over Random Forest was much smaller than the fold standard deviations and the corrected paired test gave p=0.469; all four comparisons against the other tuned baselines had p>0.05. The parameters limit memorisation: the ordinary forest allows depth up to 24 but requires 45 rows per leaf, the oblique forest caps depth at 4, and AdaBoost uses depth-4 trees with a small learning rate and 220 rounds. For Week 6, I retained the ordinary Random Forest as the configured model: its familiar axis-aligned construction reaches similar measured performance, with no established accuracy winner among these tuned models. I would report its **nested CV accuracy of 0.681 ± 0.015**, while keeping the final study's 0.6838 best-trial score identified as a selection score.

Small negative optimism estimates for the tree and AdaBoost reflect variation between inner and outer folds; they do not turn the final study's maximum into an independent evaluation. Exact values, chosen parameters and run references are in [comparison.csv](reports/week6/comparison.csv), the paired outer scores are in [folds.csv](reports/week6/folds.csv), and the statistical comparisons are in [model_comparisons.csv](reports/week6/model_comparisons.csv).

### Challenge 1: fixed engineered features

| Feature | Formula | Reason to try it |
|---|---|---|
| `priors_per_adult_year` | `priors_count / max(age - 17, 1)` | The notebook's example: expose a ratio that a linear model cannot derive from separate age and prior counts. |
| `juvenile_per_adult_year` | `(juv_fel_count + juv_misd_count + juv_other_count) / max(age - 17, 1)` | A candidate ratio of recorded juvenile history to the same age-based denominator; it is a constructed feature, not an observed exposure rate. |
| `priors_age_interaction` | `priors_count * age` | Let the effect of priors depend on age in a linear model. |

The denominator follows the notebook's convention: age 18 gives one unit. All three formulas use only the current record and learn no dataset statistic, so they run after cleaning/deduplication and before `split_features_target()`. They preserve row order and missingness; the juvenile sum is missing if any of its three components is missing. Engineered missing values are imputed inside the fold-fitted preprocessing pipeline. `race`, `id`, `decile_score`, `score_text` and the target remain excluded from the features.

Each feature is tested **separately**, against the baseline, for logistic regression, Random Forest and AdaBoost. Both versions are retuned with the same budget and folds. The corrected paired test uses the five outer accuracy differences and validation/training ratio `1/4`; an uncorrected t-test would understate uncertainty because the training folds overlap. Comparisons are exploratory pairwise checks without a multiple-comparison adjustment. Feature choice itself is not nested inside these studies, so the table estimates each specified recipe rather than the full process of choosing features.

| Added feature | Model | Nested CV mean ± std | Change vs baseline | Std of paired changes | Wins | Corrected p |
|---|---|---|---:|---:|---:|---:|
| `priors_per_adult_year` | Logistic regression | 0.675 ± 0.012 | +0.0017 | 0.0054 | 3/5 | 0.656 |
| `priors_per_adult_year` | Random Forest | 0.675 ± 0.010 | −0.0054 | 0.0099 | 2/5 | 0.464 |
| `priors_per_adult_year` | AdaBoost | 0.677 ± 0.010 | −0.0068 | 0.0052 | 0/5 | 0.126 |
| `juvenile_per_adult_year` | Logistic regression | 0.675 ± 0.015 | +0.0024 | 0.0058 | 2/5 | 0.568 |
| `juvenile_per_adult_year` | Random Forest | 0.678 ± 0.016 | −0.0029 | 0.0068 | 1/5 | 0.551 |
| `juvenile_per_adult_year` | AdaBoost | 0.683 ± 0.014 | −0.0007 | 0.0034 | 3/5 | 0.775 |
| `priors_age_interaction` | Logistic regression | 0.674 ± 0.016 | +0.0012 | 0.0055 | 3/5 | 0.760 |
| `priors_age_interaction` | Random Forest | 0.682 ± 0.017 | +0.0010 | 0.0050 | 3/5 | 0.771 |
| `priors_age_interaction` | AdaBoost | 0.680 ± 0.018 | −0.0033 | 0.0088 | 1/5 | 0.608 |

Logistic regression gained a little from all three representations, with the largest mean gain from the juvenile ratio (+0.0024), but every gain was smaller than the standard deviation of its paired changes. The forests and boosting did not benefit consistently: the ordinary forest lost accuracy with both ratios, while AdaBoost's means were lower with each added feature. None of the nine corrected paired comparisons had p<0.05, so these runs do not support adding a feature to the selected recipe. The configured pipeline therefore keeps `engineered_features: []`; the formulas and CLI options remain available for reproducible experiments.

The unrounded feature comparisons are in [feature_comparisons.csv](reports/week6/feature_comparisons.csv); [comparison.csv](reports/week6/comparison.csv) also contains the default scores, gaps and final-study hyperparameters for all 14 runs. [metadata.json](reports/week6/metadata.json) records the input-data hash, seeds, fold counts, trial budget and dependency versions. The reserved 1,443 test records were not scored in any Week 5–6 run.
