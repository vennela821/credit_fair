import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "creditfair.csv")
GRAPH_FOLDER = os.path.join(BASE_DIR, "static", "charts")
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# Standard clustering feature set
CLUSTER_FEATURES = [
    "fico_range_low",
    "annual_inc",
    "loan_amnt",
    "dti",
    "int_rate",
    "installment",
    "revol_util"
]

FEATURE_LABELS = {
    "fico_range_low": "FICO Credit Score",
    "annual_inc": "Annual Income ($)",
    "loan_amnt": "Loan Amount ($)",
    "dti": "Debt-to-Income (DTI %)",
    "int_rate": "Interest Rate (%)",
    "installment": "Monthly Installment ($)",
    "revol_util": "Revolving Line Util (%)"
}

_CACHED_KMEANS_DATA = None


# =========================================================
# LOAD & PREPARE CLUSTERING DATA
# =========================================================

def load_clustering_data(sample_size=30000):
    """Loads and cleans dataset for unsupervised K-Means segmentation."""
    global _CACHED_KMEANS_DATA
    if _CACHED_KMEANS_DATA is not None:
        return _CACHED_KMEANS_DATA

    df = pd.read_csv(CSV_PATH, nrows=sample_size, low_memory=False).copy()

    # Clean target for cluster default rate analysis
    if "loan_status" in df.columns:
        df["loan_status_binary"] = (df["loan_status"].astype(str).str.strip() == "Charged Off").astype(int)
    else:
        df["loan_status_binary"] = 0

    # Ensure numeric columns are cleanly parsed and imputed
    for col in CLUSTER_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    # Remove extreme outlier noise in income/dti for cleaner clustering
    df = df[
        (df["annual_inc"] > 5000) & (df["annual_inc"] < 350000) &
        (df["dti"] >= 0) & (df["dti"] <= 50) &
        (df["fico_range_low"] >= 600)
    ].copy().reset_index(drop=True)

    _CACHED_KMEANS_DATA = df
    return _CACHED_KMEANS_DATA


# =========================================================
# ELBOW METHOD (WCSS COMPUTATION & PLOT)
# =========================================================

def compute_elbow_curve(max_k=10):
    """
    Computes Within-Cluster Sum of Squares (WCSS / Inertia) for K = 1..max_k
    and generates the Elbow Curve chart.
    """
    df = load_clustering_data()
    X = df[CLUSTER_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []
    k_range = list(range(1, max_k + 1))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(float(km.inertia_))

    # Generate Elbow Plot
    plt.figure(figsize=(7.5, 4.5), dpi=120)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    plt.plot(k_range, wcss, marker="o", markersize=8, color="#2563eb", linewidth=2.4, label="Inertia (WCSS)")
    
    # Highlight recommended K = 3 and K = 4
    if len(wcss) >= 4:
        plt.scatter([3, 4], [wcss[2], wcss[3]], color="#dc2626", s=140, zorder=5, label="Elbow Knee Point (Optimal K)")
        plt.annotate("Optimal K=3", (3, wcss[2]), textcoords="offset points", xytext=(12, 10),
                     arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5), fontweight="bold", color="#1e293b", fontsize=9.5)

    plt.xlabel("Number of Clusters (K)", fontsize=11, fontweight="bold", color="#1e293b")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=11, fontweight="bold", color="#1e293b")
    plt.title("Elbow Method for Optimal K Identification", fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    plt.xticks(k_range)
    plt.legend(frameon=True)
    plt.tight_layout()

    filename = "kmeans_elbow_curve.png"
    filepath = os.path.join(GRAPH_FOLDER, filename)
    plt.savefig(filepath, dpi=120)
    plt.close("all")

    return {
        "k_range": k_range,
        "wcss": [round(w, 2) for w in wcss],
        "elbow_chart": f"charts/{filename}",
        "recommended_k": 3
    }


# =========================================================
# PERSONA GENERATION
# =========================================================

def _assign_persona_label(mean_fico, mean_inc, mean_loan, mean_dti, default_rate):
    """Synthesizes human-interpretable borrower personas based on cluster centroid stats."""
    if mean_fico >= 715 and mean_inc >= 75000:
        return "Prime High-Income Borrowers", "High credit score, solid income buffer, and lowest historical default hazard."
    elif mean_fico < 670 and default_rate >= 22:
        return "Subprime High-Risk Segment", "Lower credit score, elevated default probability, requiring stringent underwriting."
    elif mean_dti >= 22:
        return "High-Debt Leverage Borrowers", "Heavy existing obligations relative to earnings; vulnerable to rate hikes."
    elif mean_loan >= 20000:
        return "Large Capital Consolidators", "Substantial borrowing requirements with significant monthly installments."
    elif mean_inc < 50000:
        return "Moderate-Income Standard Borrowers", "Entry to mid-tier earnings with conservative borrowing patterns."
    else:
        return "Core Mainstream Borrowers", "Balanced credit profile with standard repayment reliability."


# =========================================================
# RUN K-MEANS CLUSTERING
# =========================================================

def run_kmeans_clustering(k=3, feature_x="annual_inc", feature_y="loan_amnt"):
    """
    Executes K-Means with chosen K and generates 2D scatter visualization
    along with cluster centroid metrics and financial personas.
    """
    k = max(2, min(int(k), 8))
    if feature_x not in CLUSTER_FEATURES:
        feature_x = "annual_inc"
    if feature_y not in CLUSTER_FEATURES:
        feature_y = "loan_amnt"

    df = load_clustering_data().copy()
    X = df[CLUSTER_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    # Convert centroids back to unscaled natural units
    centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_unscaled, columns=CLUSTER_FEATURES)

    # 1. Cluster Breakdown Statistics
    cluster_stats = []
    colors = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6"]

    for i in range(k):
        c_df = df[df["Cluster"] == i]
        count = len(c_df)
        pct = round((count / len(df)) * 100, 1)

        m_fico = float(centroids_df.loc[i, "fico_range_low"])
        m_inc = float(centroids_df.loc[i, "annual_inc"])
        m_loan = float(centroids_df.loc[i, "loan_amnt"])
        m_dti = float(centroids_df.loc[i, "dti"])
        m_rate = float(centroids_df.loc[i, "int_rate"])
        m_inst = float(centroids_df.loc[i, "installment"])
        m_util = float(centroids_df.loc[i, "revol_util"])
        def_rate = round(float(c_df["loan_status_binary"].mean()) * 100, 1)

        persona_title, persona_desc = _assign_persona_label(m_fico, m_inc, m_loan, m_dti, def_rate)

        cluster_stats.append({
            "cluster_id": i,
            "color": colors[i % len(colors)],
            "count": count,
            "percentage": pct,
            "persona_title": persona_title,
            "persona_desc": persona_desc,
            "mean_fico": round(m_fico, 1),
            "mean_income": round(m_inc, 0),
            "mean_loan": round(m_loan, 0),
            "mean_dti": round(m_dti, 1),
            "mean_rate": round(m_rate, 2),
            "mean_installment": round(m_inst, 1),
            "mean_util": round(m_util, 1),
            "default_rate": def_rate
        })

    # 2. Cluster Scatter Visualization (using sample for snappy rendering)
    plot_df = df.sample(min(2500, len(df)), random_state=42)

    plt.figure(figsize=(8.5, 5.2), dpi=120)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for i in range(k):
        c_sub = plot_df[plot_df["Cluster"] == i]
        plt.scatter(
            c_sub[feature_x],
            c_sub[feature_y],
            c=colors[i % len(colors)],
            label=f"Cluster {i}: {cluster_stats[i]['persona_title']}",
            alpha=0.60,
            s=32,
            edgecolors="none"
        )

    # Plot Centroids
    plt.scatter(
        centroids_df[feature_x],
        centroids_df[feature_y],
        c="#0f172a",
        marker="X",
        s=220,
        linewidths=2,
        edgecolors="#ffffff",
        label="Cluster Centers (Centroids)",
        zorder=10
    )

    plt.xlabel(FEATURE_LABELS.get(feature_x, feature_x), fontsize=11, fontweight="bold", color="#1e293b")
    plt.ylabel(FEATURE_LABELS.get(feature_y, feature_y), fontsize=11, fontweight="bold", color="#1e293b")
    plt.title(f"K-Means Borrower Segmentation (K = {k})", fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=True, fontsize=9)
    plt.tight_layout()

    filename = f"kmeans_clusters_k{k}_{feature_x}_{feature_y}.png"
    filepath = os.path.join(GRAPH_FOLDER, filename)
    plt.savefig(filepath, dpi=120)
    plt.close("all")

    # Get Elbow results
    elbow_data = compute_elbow_curve()

    return {
        "k": k,
        "feature_x": feature_x,
        "feature_y": feature_y,
        "feature_x_label": FEATURE_LABELS.get(feature_x, feature_x),
        "feature_y_label": FEATURE_LABELS.get(feature_y, feature_y),
        "total_records": len(df),
        "wcss_inertia": round(float(kmeans.inertia_), 2),
        "cluster_stats": cluster_stats,
        "cluster_chart": f"charts/{filename}",
        "elbow_chart": elbow_data["elbow_chart"],
        "available_features": [
            {"key": f, "label": FEATURE_LABELS.get(f, f)} for f in CLUSTER_FEATURES
        ]
    }


# =========================================================
# INTERACTIVE APPLICANT CLUSTER PREDICTION
# =========================================================

def predict_applicant_cluster(
    fico=695,
    annual_inc=65000,
    loan_amnt=12000,
    dti=18.0,
    int_rate=12.5,
    installment=380.0,
    revol_util=45.0,
    k=3
):
    """Assigns an applicant to their closest K-Means financial cluster persona."""
    k = max(2, min(int(k), 8))
    df = load_clustering_data()
    X = df[CLUSTER_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    input_data = np.array([[
        float(fico),
        float(annual_inc),
        float(loan_amnt),
        float(dti),
        float(int_rate),
        float(installment),
        float(revol_util)
    ]])

    input_scaled = scaler.transform(input_data)
    cluster_id = int(kmeans.predict(input_scaled)[0])

    # Unscaled centroids for persona evaluation
    centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    c_fico = centroids_unscaled[cluster_id][0]
    c_inc = centroids_unscaled[cluster_id][1]
    c_loan = centroids_unscaled[cluster_id][2]
    c_dti = centroids_unscaled[cluster_id][3]

    # Calculate distance to centroid
    distances = np.linalg.norm(kmeans.cluster_centers_ - input_scaled, axis=1)
    assigned_distance = float(distances[cluster_id])

    c_df = df[kmeans.labels_ == cluster_id]
    def_rate = round(float(c_df["loan_status_binary"].mean()) * 100, 1)

    persona_title, persona_desc = _assign_persona_label(c_fico, c_inc, c_loan, c_dti, def_rate)

    colors = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6"]

    return {
        "cluster_id": cluster_id,
        "color": colors[cluster_id % len(colors)],
        "persona_title": persona_title,
        "persona_desc": persona_desc,
        "default_rate": def_rate,
        "centroid_distance": round(assigned_distance, 3),
        "cluster_size": len(c_df),
        "cluster_pct": round((len(c_df) / len(df)) * 100, 1)
    }


if __name__ == "__main__":
    print("Testing K-Means Clustering...")
    res = run_kmeans_clustering(k=3)
    print("K:", res["k"])
    print("Inertia:", res["wcss_inertia"])
    print("Clusters:", len(res["cluster_stats"]))
    print("Test prediction:", predict_applicant_cluster(k=3))
