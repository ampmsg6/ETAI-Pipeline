"""Reusable data checks used to justify the cleaning recipe."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def test_missingness_mechanism(df, target_col, candidate_predictors):
    """Measure observed associations with a column's missingness.

    A weak association does not prove MCAR; an association does not distinguish
    MAR from MNAR. The output is evidence for an imputation decision only.
    """
    missing = df[target_col].isna()
    rows = []
    for predictor in candidate_predictors:
        if predictor == target_col:
            continue
        observed = pd.DataFrame({"missing": missing, "value": df[predictor]}).dropna(subset=["value"])
        if observed["missing"].nunique() < 2 or observed["value"].nunique() < 2:
            continue
        table = pd.crosstab(observed["missing"], observed["value"])
        chi2, p_value, _, _ = chi2_contingency(table)
        n, (r, k) = table.to_numpy().sum(), table.shape
        phi2 = chi2 / n
        corrected = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r_corr = r - (r - 1) ** 2 / (n - 1)
        k_corr = k - (k - 1) ** 2 / (n - 1)
        denom = min(r_corr - 1, k_corr - 1)
        v = np.sqrt(corrected / denom) if denom > 0 else 0.0
        rows.append({"predictor": predictor, "cramers_v": float(v), "p_value": float(p_value), "n": len(observed)})
    return pd.DataFrame(rows, columns=["predictor", "cramers_v", "p_value", "n"]).sort_values(
        "cramers_v", ascending=False
    ).reset_index(drop=True)


def flag_invalid_values(df, rules):
    """Replace values outside configured inclusive numeric bounds with NaN."""
    out = df.copy()
    counts = {}
    for column, bounds in rules.items():
        if column not in out:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        invalid = values.notna() & (
            values.lt(bounds.get("min", -np.inf)) | values.gt(bounds.get("max", np.inf))
        )
        counts[column] = int(invalid.sum())
        out[column] = values.mask(invalid)
    return out, counts


def find_duplicates(df, id_column):
    """Count exact repeated rows and repeated identifiers independently."""
    distinct_records_per_id = df.drop_duplicates().groupby(id_column).size()
    return {
        "exact_rows": int(df.duplicated().sum()),
        "repeated_ids": int(df[id_column].duplicated().sum()),
        "ids_with_conflicting_records": int(distinct_records_per_id.gt(1).sum()),
    }


def inconsistent_age_category(df):
    """Check the data's observed bins: <25, 25-44, >=45."""
    age = pd.to_numeric(df["age"], errors="coerce")
    expected = pd.Series(np.select(
        [age.lt(25), age.lt(45)], ["Less than 25", "25 - 45"], default="Greater than 45"
    ), index=df.index)
    return age.between(18, 100) & df["age_cat"].notna() & df["age_cat"].ne(expected)


def inconsistent_score_category(df):
    """Check COMPAS categories against the observed 1-4, 5-7, 8-10 bins."""
    score = pd.to_numeric(df["decile_score"], errors="coerce")
    expected = pd.Series(np.select(
        [score.le(4), score.le(7)], ["Low", "Medium"], default="High"
    ), index=df.index)
    return score.between(1, 10) & df["score_text"].notna() & df["score_text"].ne(expected)
