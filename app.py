from flask import (
    Flask,
    render_template,
    request,
    jsonify
)
import json
import os

from load_data import get_data_summary
from preprocessing import run_preprocessing
from linear_regression import run_linear_regression
from regularization_models import run_regularization
from logistic_regression_model import run_logistic_regression


app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():
    return render_template("index.html", active="home")


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/load_data")
def load_data_page():
    data = get_data_summary()
    return render_template(
        "data_loading.html",
        data=data,
        active="load_data"
    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda():
    summary_path = os.path.join(
        "static",
        "eda",
        "summary.json"
    )

    if not os.path.exists(summary_path):
        return """
        <div style="
            font-family:Arial;
            padding:40px;
        ">
            <h2>EDA has not been generated yet.</h2>
            <p>
                Run <b>credit_fair_eda.py</b>
                first.
            </p>
        </div>
        """

    with open(summary_path, "r") as file:
        summary = json.load(file)

    return render_template(
        "eda.html",
        summary=summary,
        active="eda"
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing_page():
    error = None
    prep = None
    try:
        prep = run_preprocessing()
    except Exception as e:
        error = f"Preprocessing Error: {e}"

    return render_template(
        "preprocessing.html",
        prep=prep,
        error=error,
        active="preprocessing"
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

@app.route("/linear-regression")
def linear_regression_page():
    error = None
    results = None
    try:
        results = run_linear_regression()
    except Exception as e:
        error = f"Linear Regression Error: {e}"

    return render_template(
        "linear_regression.html",
        results=results,
        error=error,
        active="linear_regression"
    )


# ============================================================
# REGULARIZATION
# ============================================================

@app.route("/regularization")
def regularization_page():
    error = None
    results = None
    try:
        results = run_regularization()
    except Exception as e:
        error = f"Regularization Error: {e}"

    return render_template(
        "regularization.html",
        results=results,
        error=error,
        active="regularization"
    )


# ============================================================
# CREDIT RISK CLASSIFICATION (LOGISTIC REGRESSION)
# ============================================================

@app.route("/logistic-regression")
def logistic_regression_page():
    error = None
    results = None
    try:
        results = run_logistic_regression()
    except Exception as e:
        error = f"Credit Risk Classification Error: {e}"

    return render_template(
        "logistic_regression.html",
        results=results,
        error=error,
        active="logistic_regression"
    )


# ============================================================
# INTERACTIVE SIMULATION API ENDPOINTS
# ============================================================

@app.route("/api/predict-interest-rate", methods=["POST"])
def api_predict_interest_rate():
    from linear_regression import predict_loan_rate
    data = request.get_json() or {}
    res = predict_loan_rate(data)
    return jsonify(res)


@app.route("/api/assess-credit-risk", methods=["POST"])
def api_assess_credit_risk():
    from logistic_regression_model import predict_applicant_risk
    data = request.get_json() or {}
    res = predict_applicant_risk(data)
    return jsonify(res)


@app.route("/api/simulate-threshold", methods=["POST"])
def api_simulate_threshold():
    from logistic_regression_model import simulate_threshold
    data = request.get_json() or {}
    threshold = float(data.get("threshold", 0.5))
    res = simulate_threshold(threshold)
    return jsonify(res)


@app.route("/api/simulate-regularization", methods=["POST"])
def api_simulate_regularization():
    from regularization_models import simulate_regularization_alpha
    data = request.get_json() or {}
    res = simulate_regularization_alpha(data)
    return jsonify(res)


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True
    )