import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from preprocessing import get_preprocessed_credit_data

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def run_regularization() -> dict:
    """
    Fits and compares:
    1. Linear Regression (OLS baseline)
    2. Ridge Regression (L2 penalty)
    3. Lasso Regression (L1 penalty with automatic feature selection)
    4. Elastic Net (L1 + L2 blend)
    All models use StandardScaler on preprocessed features.
    """
    X_train, X_test, y_train_reg, y_test_reg, _, _, feature_names = get_preprocessed_credit_data()

    # 1. Feature Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    alphas = np.logspace(-3, 2, 40)

    # Model 1: OLS
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train_reg)
    y_pred_ols = ols.predict(X_test_scaled)

    # Model 2: Ridge (L2) with 5-fold CV
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_train_scaled, y_train_reg)
    y_pred_ridge = ridge.predict(X_test_scaled)

    # Model 3: Lasso (L1) with 5-fold CV
    lasso = LassoCV(alphas=alphas, cv=5, max_iter=3000, random_state=42)
    lasso.fit(X_train_scaled, y_train_reg)
    y_pred_lasso = lasso.predict(X_test_scaled)

    # Model 4: Elastic Net with 5-fold CV
    elastic = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5, max_iter=3000, random_state=42)
    elastic.fit(X_train_scaled, y_train_reg)
    y_pred_elastic = elastic.predict(X_test_scaled)

    def evaluate(y_true, y_pred, model):
        mse = mean_squared_error(y_true, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        non_zeros = int(np.sum(np.abs(model.coef_) > 1e-4))
        return {
            "mse": round(mse, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3),
            "non_zero_features": non_zeros,
            "total_features": len(model.coef_)
        }

    metrics = {
        "OLS": evaluate(y_test_reg, y_pred_ols, ols),
        "Ridge (L2)": {
            **evaluate(y_test_reg, y_pred_ridge, ridge),
            "best_alpha": round(float(ridge.alpha_), 4)
        },
        "Lasso (L1)": {
            **evaluate(y_test_reg, y_pred_lasso, lasso),
            "best_alpha": round(float(lasso.alpha_), 4)
        },
        "Elastic Net": {
            **evaluate(y_test_reg, y_pred_elastic, elastic),
            "best_alpha": round(float(elastic.alpha_), 4),
            "l1_ratio": round(float(elastic.l1_ratio_), 2)
        }
    }

    # Generate visual coefficient shrinkage comparison bar chart
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "OLS": ols.coef_,
        "Ridge": ridge.coef_,
        "Lasso": lasso.coef_,
        "ElasticNet": elastic.coef_
    })

    # Sort by absolute OLS impact and select top 12 features for clear display
    coef_df["abs_ols"] = coef_df["OLS"].abs()
    top_coef = coef_df.sort_values("abs_ols", ascending=False).head(12).drop(columns=["abs_ols"])

    plt.figure(figsize=(12, 6))
    x = np.arange(len(top_coef))
    width = 0.2

    plt.bar(x - 1.5 * width, top_coef["OLS"], width, label="OLS", color="#3b82f6", alpha=0.85)
    plt.bar(x - 0.5 * width, top_coef["Ridge"], width, label=f"Ridge (α={ridge.alpha_:.2f})", color="#10b981", alpha=0.85)
    plt.bar(x + 0.5 * width, top_coef["Lasso"], width, label=f"Lasso (α={lasso.alpha_:.2f})", color="#f59e0b", alpha=0.85)
    plt.bar(x + 1.5 * width, top_coef["ElasticNet"], width, label=f"Elastic Net (α={elastic.alpha_:.2f})", color="#8b5cf6", alpha=0.85)

    plt.xticks(x, top_coef["Feature"], rotation=35, ha="right", fontsize=10)
    plt.title("Regularization Shrinkage Comparison (Top Predictors of Interest Rate)", fontsize=13, fontweight="bold")
    plt.ylabel("Standardized Coefficient Weight", fontsize=11)
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()

    chart_name = "regularization_coefficients.png"
    plt.savefig(_chart_path(chart_name), dpi=120)
    plt.close("all")

    return {
        "metrics": metrics,
        "chart": chart_name,
        "coefficients_table": top_coef.round(4).to_dict("records"),
        "zeroed_by_lasso": [f for f, c in zip(feature_names, lasso.coef_) if abs(c) <= 1e-4]
    }


def simulate_regularization_alpha(data: dict) -> dict:
    """
    Simulates coefficient shrinkage for user-selected alpha and penalty type.
    """
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    alpha = float(data.get("alpha", 1.0))
    model_type = str(data.get("model_type", "lasso")).lower()
    l1_ratio = float(data.get("l1_ratio", 0.5))

    X_train, X_test, y_train_reg, y_test_reg, _, _, feature_names = get_preprocessed_credit_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "elastic":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
    else:
        model = Lasso(alpha=alpha, max_iter=2000, random_state=42)

    model.fit(X_train_scaled, y_train_reg)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test_reg, y_pred)
    rmse = sqrt(mean_squared_error(y_test_reg, y_pred))

    coefs = model.coef_
    zeroed = [f for f, c in zip(feature_names, coefs) if abs(c) <= 1e-4]
    active = [f for f, c in zip(feature_names, coefs) if abs(c) > 1e-4]

    coef_list = [
        {"feature": f, "weight": round(float(c), 4), "is_zero": bool(abs(c) <= 1e-4)}
        for f, c in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    ]

    return {
        "alpha": alpha,
        "model_type": model_type.capitalize(),
        "r2": round(r2, 3),
        "rmse": round(rmse, 3),
        "total_features": len(feature_names),
        "active_features_count": len(active),
        "zeroed_features_count": len(zeroed),
        "zeroed_features": zeroed,
        "coefficients": coef_list[:12]
    }


if __name__ == "__main__":
    reg = run_regularization()
    print("Regularization completed successfully!")
    print("OLS R2:", reg["metrics"]["OLS"]["r2"])
    print("Ridge R2:", reg["metrics"]["Ridge (L2)"]["r2"])
    print("Lasso R2:", reg["metrics"]["Lasso (L1)"]["r2"])
    print("Zeroed by Lasso:", reg["zeroed_by_lasso"])
