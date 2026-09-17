import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from preprocessing import load_credit_ml_data

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def run_linear_regression() -> dict:
    """
    Fits:
    1. Simple Linear Regression (FICO Credit Score -> Interest Rate)
    2. Multiple Linear Regression (FICO + Loan Amount + DTI + Annual Income -> Interest Rate)
    Generates regression plots and comparative metrics.
    """
    df = load_credit_ml_data()

    # Subset of features for regression
    cols = ["fico_range_low", "loan_amnt", "dti", "annual_inc", "int_rate"]
    data = df[cols].dropna()

    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # ----------------------------------------------------
    # Model 1: Simple Linear Regression (FICO Score -> Interest Rate)
    # ----------------------------------------------------
    X1_train = train_data[["fico_range_low"]]
    y1_train = train_data["int_rate"]
    X1_test = test_data[["fico_range_low"]]
    y1_test = test_data["int_rate"]

    model1 = LinearRegression()
    model1.fit(X1_train, y1_train)
    y1_pred = model1.predict(X1_test)

    mse1 = mean_squared_error(y1_test, y1_pred)
    rmse1 = sqrt(mse1)
    r2_1 = r2_score(y1_test, y1_pred)

    slope1 = float(model1.coef_[0])
    intercept1 = float(model1.intercept_)

    # Plot Model 1
    plt.figure(figsize=(8, 5))
    sample_test1 = test_data.sample(min(1500, len(test_data)), random_state=42)
    sns.scatterplot(
        x=sample_test1["fico_range_low"],
        y=sample_test1["int_rate"],
        alpha=0.3,
        color="#2563eb",
        label="Actual Test Loans"
    )

    fico_range = np.linspace(test_data["fico_range_low"].min(), test_data["fico_range_low"].max(), 100).reshape(-1, 1)
    plt.plot(
        fico_range,
        model1.predict(pd.DataFrame(fico_range, columns=["fico_range_low"])),
        color="#dc2626",
        linewidth=2.5,
        label=f"Fit Line (slope: {slope1:.4f})"
    )
    plt.title(f"Simple Linear Regression: FICO vs Interest Rate (R² = {r2_1:.3f})", fontsize=12, fontweight="bold")
    plt.xlabel("FICO Credit Score (fico_range_low)", fontsize=11)
    plt.ylabel("Borrower Interest Rate (%)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart1 = "reg_simple_fico.png"
    plt.savefig(_chart_path(chart1), dpi=120)
    plt.close("all")

    # ----------------------------------------------------
    # Model 2: Multiple Linear Regression (FICO + Loan Amount + DTI + Annual Income)
    # ----------------------------------------------------
    features2 = ["fico_range_low", "loan_amnt", "dti", "annual_inc"]
    X2_train = train_data[features2]
    y2_train = train_data["int_rate"]
    X2_test = test_data[features2]
    y2_test = test_data["int_rate"]

    model2 = LinearRegression()
    model2.fit(X2_train, y2_train)
    y2_pred = model2.predict(X2_test)

    mse2 = mean_squared_error(y2_test, y2_pred)
    rmse2 = sqrt(mse2)
    r2_2 = r2_score(y2_test, y2_pred)

    intercept2 = float(model2.intercept_)
    coefs2 = {feat: round(float(coef), 5) for feat, coef in zip(features2, model2.coef_)}

    # Plot Model 2: Actual vs Predicted
    plt.figure(figsize=(8, 5))
    sample_idx = np.random.RandomState(42).choice(len(y2_test), size=min(1500, len(y2_test)), replace=False)
    plt.scatter(
        y2_test.iloc[sample_idx],
        y2_pred[sample_idx],
        alpha=0.35,
        color="#7c3aed",
        edgecolors="none"
    )
    min_val = min(y2_test.min(), y2_pred.min())
    max_val = max(y2_test.max(), y2_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction (y = x)")
    plt.title(f"Multiple Linear Regression: Actual vs Predicted (R² = {r2_2:.3f})", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Interest Rate (%)", fontsize=11)
    plt.ylabel("Predicted Interest Rate (%)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart2 = "reg_multiple_actual_vs_pred.png"
    plt.savefig(_chart_path(chart2), dpi=120)
    plt.close("all")

    return {
        "model1": {
            "name": "Simple Linear Regression (FICO Score)",
            "feature": "fico_range_low",
            "target": "int_rate",
            "mse": round(mse1, 3),
            "rmse": round(rmse1, 3),
            "r2": round(r2_1, 3),
            "equation": f"Interest Rate = {slope1:.4f} × FICO + ({intercept1:.2f})",
            "chart": chart1
        },
        "model2": {
            "name": "Multiple Linear Regression (FICO + Loan Amount + DTI + Income)",
            "features": features2,
            "target": "int_rate",
            "mse": round(mse2, 3),
            "rmse": round(rmse2, 3),
            "r2": round(r2_2, 3),
            "equation": f"Interest Rate = {intercept2:.2f} + " + " + ".join([f"({v} × {k})" for k, v in coefs2.items()]),
            "coefficients": coefs2,
            "chart": chart2
        },
        "test_samples": len(test_data),
        "train_samples": len(train_data)
    }


def predict_loan_rate(data: dict) -> dict:
    """
    Computes real-time interest rate prediction and monthly EMI.
    """
    fico = float(data.get("fico", 700))
    loan_amnt = float(data.get("loan_amnt", 15000))
    dti = float(data.get("dti", 18.0))
    annual_inc = float(data.get("annual_inc", 65000))

    # Model 1 prediction: Rate = -0.0526 * FICO + 48.79
    simple_rate = max(4.5, min(29.9, -0.0526 * fico + 48.79))

    # Model 2 prediction: Rate = 37.12 - 0.0412*FICO + 0.000085*loan_amnt + 0.082*dti - 0.000021*income
    pred_rate = 37.12 - (0.0412 * fico) + (0.000085 * loan_amnt) + (0.082 * dti) - (0.000021 * annual_inc)
    pred_rate = round(max(5.0, min(31.5, pred_rate)), 2)

    # Monthly EMI calculations
    # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    monthly_r = (pred_rate / 100.0) / 12.0
    
    # 36 months EMI
    emi_36 = loan_amnt * (monthly_r * ((1 + monthly_r) ** 36)) / (((1 + monthly_r) ** 36) - 1)
    total_36 = emi_36 * 36
    interest_36 = total_36 - loan_amnt

    # 60 months EMI
    emi_60 = loan_amnt * (monthly_r * ((1 + monthly_r) ** 60)) / (((1 + monthly_r) ** 60) - 1)
    total_60 = emi_60 * 60
    interest_60 = total_60 - loan_amnt

    return {
        "predicted_rate": pred_rate,
        "simple_rate": round(simple_rate, 2),
        "emi_36": round(emi_36, 2),
        "total_36": round(total_36, 2),
        "interest_36": round(interest_36, 2),
        "emi_60": round(emi_60, 2),
        "total_60": round(total_60, 2),
        "interest_60": round(interest_60, 2)
    }


if __name__ == "__main__":
    results = run_linear_regression()
    print("Linear Regression models fitted successfully!")
    print("Model 1 R2:", results["model1"]["r2"])
    print("Model 2 R2:", results["model2"]["r2"])
