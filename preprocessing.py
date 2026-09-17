import os
import functools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

CSV_PATH = r"C:\Users\venne\PycharmProjects\semproject\creditfair\creditfair.csv"


@functools.lru_cache(maxsize=1)
def load_credit_ml_data(sample_size: int = 50000) -> pd.DataFrame:
    """
    Loads a clean, representative sample of Credit Fair records
    and filters to closed loans ('Fully Paid' and 'Charged Off').
    """
    df = pd.read_csv(CSV_PATH, nrows=sample_size, low_memory=False)
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df["loan_status_binary"] = (df["loan_status"] == "Charged Off").astype(int)
    # Clean whitespace in categorical columns
    for col in ["term", "home_ownership", "verification_status", "grade", "purpose"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def run_preprocessing() -> dict:
    """
    Executes leak-free preprocessing:
    1. Diagnosis of missing values.
    2. Median imputation + missing indicator flags for numeric features.
    3. Categorical encoding (One-Hot, Ordinal Grade, Target Encoding for Purpose).
    Returns a summary dictionary for rendering in the web interface.
    """
    df = load_credit_ml_data()

    # Track missing values before imputation
    raw_missing = df.isnull().sum()
    missing_before = {col: int(cnt) for col, cnt in raw_missing.items() if cnt > 0 and col in [
        "dti", "revol_util", "mths_since_last_delinq", "annual_inc", "tot_cur_bal", "open_acc"
    ]}

    # Train-test split before fitting any transformers to prevent data leakage
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["loan_status_binary"])

    # 1. Numeric features with imputation and indicators
    numeric_missing_cols = ["dti", "revol_util", "mths_since_last_delinq"]
    existing_num_missing = [c for c in numeric_missing_cols if c in df.columns]

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_imputed = imputer.fit_transform(train_df[existing_num_missing])
    test_imputed = imputer.transform(test_df[existing_num_missing])

    indicator_cols = [f"{col}_was_missing" for col in existing_num_missing]
    imputed_col_names = existing_num_missing + indicator_cols

    # 2. Categorical Encodings
    # 2a. One-Hot Encoding
    ohe_cols = ["term", "home_ownership", "verification_status"]
    existing_ohe_cols = [c for c in ohe_cols if c in df.columns]
    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    train_ohe = ohe.fit_transform(train_df[existing_ohe_cols])
    ohe_feature_names = list(ohe.get_feature_names_out(existing_ohe_cols))

    # 2b. Ordinal Encoding for Grade (A < B < C < D < E < F < G)
    grade_categories = [["A", "B", "C", "D", "E", "F", "G"]]
    ord_encoder = OrdinalEncoder(categories=grade_categories, handle_unknown="use_encoded_value", unknown_value=-1)
    train_ord = ord_encoder.fit_transform(train_df[["grade"]])

    # 2c. Target Encoding for Purpose (target: loan_status_binary -> Default Rate)
    purpose_target_means = train_df.groupby("purpose")["loan_status_binary"].mean().to_dict()
    overall_mean = float(train_df["loan_status_binary"].mean())
    train_purpose_encoded = train_df["purpose"].map(purpose_target_means).fillna(overall_mean)

    # 3. Assemble sample preview DataFrame
    preview_df = pd.DataFrame(train_imputed[:5], columns=imputed_col_names)
    for idx, col in enumerate(ohe_feature_names):
        preview_df[col] = train_ohe[:5, idx]
    preview_df["grade_ordinal"] = train_ord[:5, 0]
    preview_df["purpose_default_rate"] = train_purpose_encoded.iloc[:5].values

    return {
        "missing_before": missing_before,
        "imputed_columns": existing_num_missing,
        "indicator_columns": indicator_cols,
        "one_hot_features": ohe_feature_names,
        "ordinal_features": ["grade (A=0, B=1, C=2, D=3, E=4, F=5, G=6)"],
        "purpose_target_encoding": {str(k): round(float(v) * 100, 2) for k, v in purpose_target_means.items()},
        "total_train_samples": len(train_df),
        "total_test_samples": len(test_df),
        "preview_columns": list(preview_df.columns),
        "preview_rows": preview_df.round(3).to_dict("records")
    }


def get_preprocessed_credit_data():
    """
    Returns clean, leak-free:
    (X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names)
    with standardized/encoded features ready for regression and classification models.
    """
    df = load_credit_ml_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["loan_status_binary"])

    # Base continuous features
    base_num_cols = ["loan_amnt", "fico_range_low", "annual_inc", "open_acc", "total_acc"]
    missing_num_cols = ["dti", "revol_util", "mths_since_last_delinq"]

    # Impute missing continuous features
    imputer = SimpleImputer(strategy="median")
    train_missing_imp = imputer.fit_transform(train_df[missing_num_cols])
    test_missing_imp = imputer.transform(test_df[missing_num_cols])

    # Base numeric features with median fillna
    train_base_num = train_df[base_num_cols].fillna(train_df[base_num_cols].median()).values
    test_base_num = test_df[base_num_cols].fillna(train_df[base_num_cols].median()).values

    # One-Hot Encoding
    ohe_cols = ["term", "home_ownership", "verification_status"]
    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    train_ohe = ohe.fit_transform(train_df[ohe_cols])
    test_ohe = ohe.transform(test_df[ohe_cols])
    ohe_feature_names = list(ohe.get_feature_names_out(ohe_cols))

    # Ordinal Encoding for Grade
    grade_categories = [["A", "B", "C", "D", "E", "F", "G"]]
    ord_encoder = OrdinalEncoder(categories=grade_categories, handle_unknown="use_encoded_value", unknown_value=-1)
    train_ord = ord_encoder.fit_transform(train_df[["grade"]])
    test_ord = ord_encoder.transform(test_df[["grade"]])

    # Target Encoding for Purpose
    purpose_target_means = train_df.groupby("purpose")["loan_status_binary"].mean().to_dict()
    overall_mean = float(train_df["loan_status_binary"].mean())
    train_purpose = train_df["purpose"].map(purpose_target_means).fillna(overall_mean).values.reshape(-1, 1)
    test_purpose = test_df["purpose"].map(purpose_target_means).fillna(overall_mean).values.reshape(-1, 1)

    # Combine all features
    X_train = np.hstack([train_base_num, train_missing_imp, train_ohe, train_ord, train_purpose])
    X_test = np.hstack([test_base_num, test_missing_imp, test_ohe, test_ord, test_purpose])

    feature_names = (
        base_num_cols
        + missing_num_cols
        + ohe_feature_names
        + ["grade_ordinal", "purpose_risk_rate"]
    )

    # Targets
    y_train_reg = train_df["int_rate"].values
    y_test_reg = test_df["int_rate"].values

    y_train_clf = train_df["loan_status_binary"].values
    y_test_clf = test_df["loan_status_binary"].values

    return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names


if __name__ == "__main__":
    prep = run_preprocessing()
    print("Preprocessing completed successfully!")
    print(f"Train samples: {prep['total_train_samples']}, Test samples: {prep['total_test_samples']}")
    print(f"Preview features count: {len(prep['preview_columns'])}")
