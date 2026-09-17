import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "creditfair.csv")
GRAPH_FOLDER = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# Standard feature set for Credit Risk Tree & Ensemble models
FEATURES = [
    "fico_range_low",
    "dti",
    "annual_inc",
    "loan_amnt",
    "revol_util",
    "int_rate",
    "installment",
    "open_acc",
    "total_acc",
    "grade_ordinal",
    "term_months"
]

FEATURE_LABELS = {
    "fico_range_low": "FICO Credit Score",
    "dti": "Debt-to-Income (DTI %)",
    "annual_inc": "Annual Income ($)",
    "loan_amnt": "Loan Amount ($)",
    "revol_util": "Revolving Line Util (%)",
    "int_rate": "Interest Rate (%)",
    "installment": "Monthly Installment ($)",
    "open_acc": "Open Credit Lines",
    "total_acc": "Total Credit Accounts",
    "grade_ordinal": "Credit Grade (A-G)",
    "term_months": "Loan Term (Months)"
}

TARGET = "loan_status_binary"

# Cache prepared data in memory to make algorithm switching instant
_CACHED_DATA = None


# =========================================================
# LOAD DATA & PREPARE
# =========================================================

def load_data(sample_size=50000):
    """Loads credit data with low memory footprint."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, nrows=sample_size, low_memory=False)
    return df


def convert_target(value):
    """Converts loan status to binary (0 = Fully Paid / Creditworthy, 1 = Charged Off / Default Risk)."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1 if float(value) == 1 else 0

    val_str = str(value).strip().lower()
    if val_str in ["charged off", "1", "yes", "true", "default", "not placed", "notplaced"]:
        return 1
    if val_str in ["fully paid", "0", "no", "false", "good", "placed"]:
        return 0

    return np.nan


def prepare_data():
    """
    Cleans and prepares data for tree-based algorithms with leak-free processing.
    """
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA

    df = load_data()

    # Filter to closed loans
    if "loan_status" in df.columns:
        df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
        df[TARGET] = df["loan_status"].apply(convert_target)
    elif TARGET in df.columns:
        df[TARGET] = df[TARGET].apply(convert_target)
    else:
        raise ValueError(f"Neither 'loan_status' nor '{TARGET}' found in dataset columns.")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    # Encode grade (A=0, B=1, ..., G=6)
    grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    if "grade" in df.columns:
        df["grade_ordinal"] = df["grade"].astype(str).str.strip().map(grade_map).fillna(2).astype(int)
    else:
        df["grade_ordinal"] = 2

    # Encode term (36 or 60 months)
    if "term" in df.columns:
        df["term_months"] = df["term"].astype(str).apply(
            lambda x: 60 if "60" in x else 36
        )
    else:
        df["term_months"] = 36

    # Clean numeric features
    for col in ["fico_range_low", "dti", "annual_inc", "loan_amnt", "revol_util", "int_rate", "installment", "open_acc", "total_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    _CACHED_DATA = (df, X, y)
    return _CACHED_DATA


# =========================================================
# MODEL FACTORY
# =========================================================

def get_model(model_type):
    """Returns (Model Name, Classifier Instance) for requested algorithm."""
    model_type = str(model_type).lower().strip()

    if model_type == "decision_tree":
        return (
            "Decision Tree Classifier",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        )

    elif model_type == "random_forest":
        return (
            "Random Forest Classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        )

    elif model_type == "adaboost":
        return (
            "AdaBoost Classifier",
            AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                random_state=42
            )
        )

    elif model_type == "gradient_boosting":
        return (
            "Gradient Boosting Classifier",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        )

    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            return (
                "XGBoost Classifier",
                XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1
                )
            )
        except Exception:
            return (
                "XGBoost (GBDT Fallback)",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                )
            )

    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return (
                "LightGBM Classifier",
                LGBMClassifier(
                    boosting_type="gbdt",
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=-1,
                    num_leaves=31,
                    max_bin=255,
                    random_state=42,
                    verbosity=-1,
                    importance_type="gain"
                )
            )
        except Exception:
            return (
                "LightGBM (GBDT Fallback)",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            )

    return get_model("decision_tree")


# =========================================================
# MATHEMATICAL CRITERIA (GINI, ENTROPY, INFO GAIN)
# =========================================================

def calculate_gini(y):
    """Calculates Gini Impurity: Gini = 1 - sum(p_i^2)"""
    probabilities = pd.Series(y).value_counts(normalize=True)
    return 1.0 - sum(probability ** 2 for probability in probabilities)


def calculate_entropy(y):
    """Calculates Shannon Entropy: H(S) = -sum(p_i * log2(p_i))"""
    probabilities = pd.Series(y).value_counts(normalize=True)
    return -sum(
        probability * np.log2(probability)
        for probability in probabilities
        if probability > 0
    )


def calculate_information_gain(model, X_train, y_train):
    """Calculates Information Gain achieved at the root split."""
    if not hasattr(model, "tree_"):
        return 0.0

    tree = model.tree_
    root = 0
    feature_index = tree.feature[root]
    threshold = tree.threshold[root]

    if feature_index < 0:
        return 0.0

    feature_values = X_train.iloc[:, feature_index] if hasattr(X_train, "iloc") else X_train[:, feature_index]
    left_mask = feature_values <= threshold
    right_mask = ~left_mask

    y_left = y_train[left_mask]
    y_right = y_train[right_mask]

    parent_entropy = calculate_entropy(y_train)
    left_entropy = calculate_entropy(y_left) if len(y_left) > 0 else 0
    right_entropy = calculate_entropy(y_right) if len(y_right) > 0 else 0

    total = len(y_train)
    if total == 0:
        return 0.0

    weighted_entropy = (
        (len(y_left) / total) * left_entropy
        + (len(y_right) / total) * right_entropy
    )

    return max(0.0, float(parent_entropy - weighted_entropy))


# =========================================================
# FEATURE IMPORTANCE CHART
# =========================================================

def create_feature_importance(model, model_type):
    """Generates a clean horizontal bar chart of feature importances."""
    if not hasattr(model, "feature_importances_"):
        return None

    importance = model.feature_importances_
    labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]

    importance_df = pd.DataFrame({
        "feature": labels,
        "raw_feature": FEATURES,
        "importance": importance
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 4.8), dpi=120)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    bars = plt.barh(
        importance_df["feature"],
        importance_df["importance"],
        color="#2563eb",
        edgecolor="#1d4ed8",
        height=0.65
    )

    for bar in bars:
        w = bar.get_width()
        if w > 0.01:
            plt.text(
                w + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#334155",
                fontweight="bold"
            )

    plt.xlabel("Gini / Gain Importance Score", fontsize=10, fontweight="bold", color="#1e293b")
    plt.ylabel("Predictor Feature", fontsize=10, fontweight="bold", color="#1e293b")
    plt.title(f"Feature Importance - {model_type.replace('_', ' ').title()}", fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    plt.xlim(0, max(importance) * 1.18 if max(importance) > 0 else 1.0)
    plt.tight_layout()

    filename = f"feature_importance_{model_type}.png"
    filepath = os.path.join(GRAPH_FOLDER, filename)
    plt.savefig(filepath, dpi=120)
    plt.close("all")

    return f"charts/{filename}"


# =========================================================
# ALGORITHM TOPICS & THEORY BREAKDOWN
# =========================================================

def get_topics(model, model_type, X_train, y_train):
    """Returns deep theoretical & mathematical parameter breakdown for each algorithm."""
    topics = {}

    # =====================================================
    # 1. DECISION TREE
    # =====================================================
    if model_type == "decision_tree":
        gini = calculate_gini(y_train)
        entropy = calculate_entropy(y_train)
        information_gain = calculate_information_gain(model, X_train, y_train)

        root_feature_index = model.tree_.feature[0] if hasattr(model, "tree_") else -1
        if root_feature_index >= 0:
            root_feature = FEATURE_LABELS.get(FEATURES[root_feature_index], FEATURES[root_feature_index])
            threshold = model.tree_.threshold[0]
            splitting = f"{root_feature} <= {threshold:.2f}"
        else:
            splitting = "Single Root Leaf"

        topics["Gini Impurity (Root)"] = f"{gini:.4f}"
        topics["Shannon Entropy (Root)"] = f"{entropy:.4f} bits"
        topics["Information Gain (Delta H)"] = f"{information_gain:.4f}"
        topics["Tree Max Depth"] = f"{model.get_depth()} levels (Limit: {model.max_depth})"
        topics["Total Leaf Nodes"] = f"{model.get_n_leaves()} decision leaves"
        topics["Root Splitting Rule"] = f"Gini Minimization | {splitting}"

    # =====================================================
    # 2. RANDOM FOREST
    # =====================================================
    elif model_type == "random_forest":
        num_trees = len(model.estimators_) if hasattr(model, "estimators_") else model.n_estimators
        topics["Ensemble Architecture"] = "Bagging (Bootstrap Aggregation)"
        topics["Bootstrap Sampling"] = "Enabled (with replacement sampling)" if model.bootstrap else "Disabled"
        topics["Forest Size"] = f"{num_trees} Independent Trees"
        topics["Feature Subsampling (m)"] = f"max_features = '{model.max_features}' (~3 features per split)"
        topics["Tree Depth Limit"] = f"max_depth = {model.max_depth}"
        topics["Voting Mechanism"] = "Soft Probability Averaging across all trees"

    # =====================================================
    # 3. ADABOOST
    # =====================================================
    elif model_type == "adaboost":
        learner_name = type(model.estimators_[0]).__name__ if hasattr(model, "estimators_") and len(model.estimators_) > 0 else "DecisionTreeClassifier"
        learner_depth = model.estimators_[0].get_depth() if hasattr(model, "estimators_") and len(model.estimators_) > 0 and hasattr(model.estimators_[0], "get_depth") else 1

        topics["Weak Base Learner"] = f"{learner_name} (Stump Depth = {learner_depth})"
        topics["Boosting Strategy"] = "Adaptive Sequential Error Weighting"

        if hasattr(model, "estimator_weights_") and len(model.estimator_weights_) > 0:
            weights = model.estimator_weights_
            topics["Sample & Learner Weights"] = f"Init = {1/len(X_train):.6f}, Final weight = {weights[-1]:.4f}"
        else:
            topics["Sample & Learner Weights"] = f"Initial Sample Weight = {1/len(X_train):.6f}"

        topics["Number of Estimators"] = f"{model.n_estimators} boosting rounds"
        topics["Learning Rate (eta)"] = f"{model.learning_rate}"
        topics["Loss Function"] = "SAMME.R (Real AdaBoost Probability Loss)"

    # =====================================================
    # 4. GRADIENT BOOSTING
    # =====================================================
    elif model_type == "gradient_boosting":
        topics["Sequential Stages"] = f"{model.n_estimators} gradient-stepped trees"

        try:
            staged_probabilities = list(model.staged_predict_proba(X_train))
            if staged_probabilities:
                first_prob = staged_probabilities[0][:, 1]
                final_prob = staged_probabilities[-1][:, 1]
                first_res = np.abs(y_train.values - first_prob)
                final_res = np.abs(y_train.values - final_prob)
                topics["Residual Optimization"] = f"Init MAE = {np.mean(first_res):.4f} -> Final MAE = {np.mean(final_res):.4f}"
            else:
                topics["Residual Optimization"] = "Sequential Pseudo-Residual Shrinkage"
        except Exception:
            topics["Residual Optimization"] = "Iterative Pseudo-Residual Minimization"

        topics["Optimization Algorithm"] = "Gradient Descent in Function Space"
        topics["Learning Rate (Shrinkage)"] = f"eta = {model.learning_rate}"
        topics["Tree Max Depth"] = f"{model.max_depth} levels"
        topics["Loss Objective"] = "Binomial Deviance (Log-Loss)"

    # =====================================================
    # 5. XGBOOST
    # =====================================================
    elif model_type == "xgboost":
        reg_alpha = getattr(model, "reg_alpha", 0.1)
        reg_lambda = getattr(model, "reg_lambda", 1.0)
        subsample = getattr(model, "subsample", 0.8)

        topics["Algorithm"] = "Extreme Gradient Boosting (Exact Greedy Split)"
        topics["Regularization Penalties"] = f"L1 (alpha) = {reg_alpha}, L2 (lambda) = {reg_lambda}"
        topics["Learning Rate (Shrinkage)"] = f"eta = {model.learning_rate}"
        topics["Tree Max Depth"] = f"{model.max_depth}"
        topics["Subsampling Ratio"] = f"{int(subsample * 100)}% instance subsampling"
        topics["Objective"] = "binary:logistic (2nd-Order Taylor Expansion)"

    # =====================================================
    # 6. LIGHTGBM
    # =====================================================
    elif model_type == "lightgbm":
        boosting_type = getattr(model, "boosting_type", "gbdt")
        num_leaves = getattr(model, "num_leaves", 31)
        max_bin = getattr(model, "max_bin", 255)

        topics["Boosting Architecture"] = f"LightGBM ({boosting_type.upper()})"
        topics["Tree Growth Strategy"] = f"Leaf-Wise (Best-First) | num_leaves = {num_leaves}"
        topics["Histogram Binning"] = f"Histogram-Based (max_bin = {max_bin})"
        topics["Learning Rate"] = f"eta = {model.learning_rate}"
        topics["Number of Estimators"] = f"{model.n_estimators} trees"
        topics["Speed Innovations"] = "GOSS (Gradient-based One-Side Sampling) & EFB"

    return topics


# =========================================================
# RUN ALGORITHM EVALUATION
# =========================================================

def run_decision_tree(model_type="decision_tree"):
    """
    Trains and evaluates the chosen Decision Tree / Ensemble model.
    Returns complete metrics, topics, feature importance, and chart path.
    """
    data, X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model_name, model = get_model(model_type)

    # Train model
    model.fit(X_train, y_train)

    # Test predictions & probabilities
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Classification Metrics
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    roc_auc = 0.0
    if probs is not None:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_test, predictions)

    # Feature Importance
    feature_graph = create_feature_importance(model, model_type)
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        for f, imp in zip(FEATURES, model.feature_importances_):
            feature_importance[FEATURE_LABELS.get(f, f)] = round(float(imp), 4)

    # Algorithm Topics
    topics = get_topics(model, model_type, X_train, y_train)

    return {
        "model_key": model_type,
        "model_name": model_name,
        "total_rows": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "roc_auc": round(roc_auc, 3),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1])
        },
        "topics": topics,
        "feature_importance": feature_importance,
        "feature_importance_graph": feature_graph
    }


# =========================================================
# INTERACTIVE APPLICANT PREDICTION
# =========================================================

def predict_decision_tree(
    fico=695,
    dti=18.0,
    annual_inc=65000,
    loan_amnt=12000,
    revol_util=45.0,
    int_rate=12.5,
    installment=380.0,
    open_acc=10,
    total_acc=22,
    grade="C",
    term="36 months",
    model_type="decision_tree"
):
    """
    Live prediction endpoint that evaluates applicant creditworthiness
    using the trained Decision Tree or Ensemble model.
    """
    data, X, y = prepare_data()
    model_name, model = get_model(model_type)

    # Train on dataset
    model.fit(X, y)

    grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    grade_ordinal = grade_map.get(str(grade).upper().strip(), 2)
    term_months = 60 if "60" in str(term) else 36

    input_df = pd.DataFrame([{
        "fico_range_low": float(fico),
        "dti": float(dti),
        "annual_inc": float(annual_inc),
        "loan_amnt": float(loan_amnt),
        "revol_util": float(revol_util),
        "int_rate": float(int_rate),
        "installment": float(installment),
        "open_acc": float(open_acc),
        "total_acc": float(total_acc),
        "grade_ordinal": int(grade_ordinal),
        "term_months": int(term_months)
    }])[FEATURES]

    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else (0.90 if pred == 1 else 0.10)

    prob_pct = round(prob * 100, 1)

    if pred == 1:
        decision = "High Default Risk (Rejected / Flagged)"
        badge = "high-risk"
        icon = "⚠️"
        risk_level = "High Default Risk"
    else:
        decision = "Creditworthy / Approved"
        badge = "approved"
        icon = "✅"
        risk_level = "Low Default Risk" if prob_pct < 25 else "Moderate Risk"

    # AI decision reasoning
    reasons = []
    if float(fico) >= 720:
        reasons.append(f"FICO score of {int(fico)} provides strong protection against default hazard.")
    elif float(fico) < 660:
        reasons.append(f"Sub-prime FICO score of {int(fico)} increases probability of delinquency.")

    if float(dti) > 24:
        reasons.append(f"Elevated Debt-to-Income ratio ({dti}%) limits financial headroom.")
    else:
        reasons.append(f"Comfortable DTI ratio ({dti}%) ensures reliable debt service capacity.")

    if grade_ordinal <= 1:
        reasons.append(f"Credit Grade {grade} indicates prime historical borrowing behavior.")
    elif grade_ordinal >= 4:
        reasons.append(f"Lower Credit Grade {grade} carries higher baseline default risk.")

    return {
        "model_key": model_type,
        "model_name": model_name,
        "decision": decision,
        "badge": badge,
        "icon": icon,
        "default_probability": prob_pct,
        "approval_probability": round(100.0 - prob_pct, 1),
        "risk_level": risk_level,
        "confidence": round(abs(prob - 0.5) * 200, 1),
        "reasons": reasons[:3]
    }


if __name__ == "__main__":
    print("Testing Decision Tree...")
    res = run_decision_tree("decision_tree")
    print("Model:", res["model_name"])
    print("Accuracy:", res["accuracy"])
    print("Topics:", res["topics"])
    print("Prediction test:", predict_decision_tree(model_type="decision_tree")["decision"])
