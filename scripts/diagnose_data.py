"""Print and save the week 3 data diagnosis. Run from the project root."""

import json
from pathlib import Path

import pandas as pd

from main import load_config
from src.data_diagnostics import (
    find_duplicates, flag_invalid_values, inconsistent_age_category,
    inconsistent_score_category, test_missingness_mechanism,
)
from src.preprocessing import clean_dataset


def main():
    config = load_config()
    raw = pd.read_csv(config["data"]["path"])
    unique = raw.drop_duplicates().copy()
    # Normalize only the comparison label so spelling variants are not
    # mistaken for contradictions between COMPAS score and score category.
    values = unique["score_text"].astype("string").str.strip()
    unique["score_text"] = values.str.lower().map(config["data"]["canonical_maps"]["score_text"]).fillna(values)
    invalid = flag_invalid_values(unique, config["diagnostics"]["validity_rules"])[1]
    clean = clean_dataset(raw, config)
    recovered_juvenile = int((unique["juv_fel_count"].isna() & clean.loc[unique.index, "juv_fel_count"].notna()).sum())
    results = {
        "raw_rows": len(raw),
        "rows_after_cleaning": len(clean),
        "duplicates": find_duplicates(raw, config["diagnostics"]["id_column"]),
        "invalid_numeric_values": invalid,
        "recovered_juvenile_felony_counts": recovered_juvenile,
        "inconsistent_age_cat": int(inconsistent_age_category(unique).sum()),
        "inconsistent_score_text": int(inconsistent_score_category(unique).sum()),
        "missing_after_cleaning": {c: int(clean[c].isna().sum()) for c in clean if clean[c].isna().any()},
        "missingness_associations": {},
    }
    for column in ["age", "juv_fel_count", "priors_count", "c_charge_degree", "race", "sex"]:
        predictors = [p for p in config["diagnostics"]["candidate_predictors"] if p != column]
        associations = test_missingness_mechanism(clean, column, predictors)
        results["missingness_associations"][column] = associations.to_dict(orient="records")
    path = Path("data/diagnosis_log.json")
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "missingness_associations"}, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
