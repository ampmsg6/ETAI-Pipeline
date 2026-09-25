# ETAI Predictive Pipeline

Antonio Gonçalves — 20260536

## What this project does

This project predicts whether a person will be rearrested within two years (`two_year_recid`) using the COMPAS case dataset. The model uses age, sex, charge degree, and prior offense counts. It does not use `race` or COMPAS's own risk score as predictors. We keep those fields separately to check false-positive rates by race and compare with COMPAS. The [data dictionary](data/README.md) describes the columns.

The current pipeline runs logistic regression with median imputation, target encoding, and robust scaling. These settings are in `config.yaml`.

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

The command cleans the CSV, makes a stratified 80/20 train/test split, fits preprocessing and the model on the training rows, and reports train accuracy, test classification results, and false-positive rates by race. It saves a timestamped report in `results/`.

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

`scripts/tune_preprocessing.py` compared four encoders with four scalers using three stratified cross-validation folds **inside the training set**. Target encoding with robust scaling had the highest mean accuracy, **0.6754**, so it is the current setting. One-hot with standard scaling scored **0.6690**. Choosing a preprocessing recipe from training folds avoids using the held-out test set to pick the encoder and scaler. The full table is in `results/week3_tuning.csv`.

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

To regenerate the diagnosis, experiment tables, and current run report, use these commands from the project root:

```powershell
python -m scripts.diagnose_data
python -m scripts.tune_preprocessing
python -m scripts.compare_week3
python main.py
```

The generated tables are `results/week3_tuning.csv` and `results/week3_comparison.csv`. Reports from `main.py` are also in `results/`.

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

The comparison covers several changes at once: training rows, duplicate removal, imputation, encoding, and scaling. It cannot tell us which individual change caused a result. Target encoding with robust scaling won the training-fold comparison, but the cleaned logistic model did not beat the Week 2 logistic baseline on these held-out records. The current model scored **65.8% accuracy on the full 1,443-record cleaned test set**. That set includes records excluded from this common comparison, so its accuracy answers a different question. The exact values, including training accuracy and all error counts, are in `results/week3_comparison.csv`.
