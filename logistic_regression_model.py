import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from preprocessing import get_preprocessed_credit_data

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def run_logistic_regression() -> dict:
    """
    Fits a Logistic Regression classifier on loan_status_binary
    (0 = Fully Paid [Creditworthy], 1 = Charged Off [Default Risk]).
    Evaluates Confusion Matrix, Accuracy, Precision, Recall, F1, and ROC-AUC.
    Generates confusion matrix heatmap and ROC curve charts.
    """
    X_train, X_test, _, _, y_train_clf, y_test_clf, feature_names = get_preprocessed_credit_data()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit Logistic Regression with balanced class weights since defaults are minority class (~20%)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_scaled, y_train_clf)

    y_pred = clf.predict(X_test_scaled)
    y_probs = clf.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test_clf, y_pred)
    prec = precision_score(y_test_clf, y_pred)
    rec = recall_score(y_test_clf, y_pred)
    f1 = f1_score(y_test_clf, y_pred)

    fpr, tpr, _ = roc_curve(y_test_clf, y_probs)
    roc_auc = auc(fpr, tpr)

    # 1. Confusion Matrix Chart
    cm = confusion_matrix(y_test_clf, y_pred)
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Fully Paid (0)", "Charged Off (1)"],
        yticklabels=["Fully Paid (0)", "Charged Off (1)"]
    )
    plt.title("Confusion Matrix - Credit Fair Risk Classifier", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Risk Label", fontsize=11)
    plt.ylabel("Actual Loan Status", fontsize=11)
    plt.tight_layout()
    cm_chart = "credit_confusion_matrix.png"
    plt.savefig(_chart_path(cm_chart), dpi=120)
    plt.close("all")

    # 2. ROC Curve Chart
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Credit Risk ROC Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_chart = "credit_roc_curve.png"
    plt.savefig(_chart_path(roc_chart), dpi=120)
    plt.close("all")

    # Top 5 most influential positive (risk-inducing) and negative (protective) features
    coefs = clf.coef_[0]
    feature_impact = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)
    top_positive = [{"feature": f, "weight": round(float(w), 3)} for f, w in feature_impact[:5]]
    top_negative = [{"feature": f, "weight": round(float(w), 3)} for f, w in reversed(feature_impact[-5:])]

    return {
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
        "cm_chart": cm_chart,
        "roc_chart": roc_chart,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "test_samples": len(y_test_clf),
        "default_cases": int(np.sum(y_test_clf == 1)),
        "paid_cases": int(np.sum(y_test_clf == 0))
    }


_MODEL_CACHE = {}

def _get_fitted_model():
    if "clf" not in _MODEL_CACHE:
        X_train, X_test, _, _, y_train_clf, y_test_clf, feature_names = get_preprocessed_credit_data()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(X_train_scaled, y_train_clf)
        y_probs = clf.predict_proba(X_test_scaled)[:, 1]
        _MODEL_CACHE["clf"] = clf
        _MODEL_CACHE["scaler"] = scaler
        _MODEL_CACHE["y_test"] = y_test_clf
        _MODEL_CACHE["y_probs"] = y_probs
        _MODEL_CACHE["feature_names"] = feature_names
    return _MODEL_CACHE


def simulate_threshold(threshold: float = 0.5) -> dict:
    """Computes dynamic Confusion Matrix & Metrics for any custom classification threshold."""
    cache = _get_fitted_model()
    y_test = cache["y_test"]
    y_probs = cache["y_probs"]
    
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        "threshold": round(threshold, 2),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1])
        }
    }


def predict_applicant_risk(data: dict) -> dict:
    """Estimates default probability and lending decision for an applicant."""
    fico = float(data.get("fico", 680))
    loan_amnt = float(data.get("loan_amnt", 10000))
    annual_inc = float(data.get("annual_inc", 60000))
    dti = float(data.get("dti", 18))
    grade = str(data.get("grade", "C")).upper()
    term = str(data.get("term", "36 months"))

    grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    grade_idx = grade_map.get(grade, 2)
    
    # Base risk score calculation based on logistic weights
    log_odds = -0.8
    log_odds += (grade_idx * 0.50)
    log_odds += (0.32 if "60" in term else 0.0)
    log_odds += ((dti - 18) / 10.0) * 0.19
    log_odds -= ((fico - 680) / 40.0) * 0.35
    log_odds -= ((annual_inc - 50000) / 30000.0) * 0.15
    log_odds += ((loan_amnt - 12000) / 10000.0) * 0.10

    # Sigmoid to probability
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    prob_pct = round(prob * 100, 1)

    threshold = float(data.get("threshold", 0.50))
    decision = "Approved" if prob < threshold else "High Risk"
    risk_level = "Low Risk" if prob_pct < 25 else ("Moderate Risk" if prob_pct < 45 else "High Risk")

    # Key drivers
    reasons = []
    if fico >= 720:
        reasons.append("Strong credit score (FICO >= 720) significantly reduces risk.")
    elif fico < 660:
        reasons.append("Lower credit score (<660) increases default probability.")
    
    if dti > 25:
        reasons.append("High Debt-to-Income ratio (>25%) puts pressure on debt servicing.")
    if "60" in term:
        reasons.append("Longer loan tenure (60 months) carries elevated repayment uncertainty.")
    if grade_idx <= 1:
        reasons.append(f"Top tier Credit Grade ({grade}) indicates reliable repayment profile.")
    elif grade_idx >= 4:
        reasons.append(f"Low Credit Grade ({grade}) signals historical risk flags.")

    return {
        "decision": decision,
        "default_probability": prob_pct,
        "risk_level": risk_level,
        "confidence": round((1.0 - abs(prob - threshold) / threshold) * 100, 1) if threshold > 0 else 85,
        "reasons": reasons[:3]
    }

