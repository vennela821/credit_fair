import os
import warnings

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "/Users/garimellasrija/PycharmProjects/creditfair /credit fair.csv"

OUTPUT_DIR = "static/eda"

# If dataset is very large, graphs use only 50,000 rows
GRAPH_SAMPLE_SIZE = 50000

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREDIT FAIR - EDA")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    CSV_PATH,
    low_memory=False
)

print("Dataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# GRAPH SAMPLE
# ============================================================

if len(df) > GRAPH_SAMPLE_SIZE:

    graph_df = df.sample(
        GRAPH_SAMPLE_SIZE,
        random_state=42
    )

else:

    graph_df = df.copy()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(name):

    for col in df.columns:

        if col.lower() == name.lower():
            return col

    return None


def save_chart(filename):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=100,
        bbox_inches="tight"
    )

    plt.close()

    print("Created:", filename)


# ============================================================
# TASK 1 - LOAD DATA
# ============================================================

print("\nTASK 1 - LOAD DATA")

rows = len(df)
columns = len(df.columns)

print("Rows:", rows)
print("Columns:", columns)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# TASK 2 - BASIC INFO / STRUCTURE
# ============================================================

print("\nTASK 2 - BASIC INFO / STRUCTURE")

print("\nData Types:")
print(df.dtypes)

print("\nNumeric Columns:")
numeric_columns = (
    df.select_dtypes(include=np.number)
    .columns
    .tolist()
)
print(numeric_columns)

print("\nCategorical Columns:")
categorical_columns = (
    df.select_dtypes(exclude=np.number)
    .columns
    .tolist()
)
print(categorical_columns)

print("\nNumerical Summary:")

if numeric_columns:
    print(
        df[numeric_columns]
        .describe()
        .round(2)
    )


# ============================================================
# TASK 3 - MISSING VALUES
# ============================================================

print("\nTASK 3 - MISSING VALUES")

missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

missing_df = pd.DataFrame({

    "column": df.columns,

    "missing_count":
        missing_count.values,

    "missing_percentage":
        missing_percentage.values

})

missing_df = missing_df[
    missing_df["missing_count"] > 0
]

missing_df = missing_df.sort_values(
    "missing_count",
    ascending=False
)

print(missing_df)


# Missing Values Heatmap

heatmap_df = graph_df.copy()

heatmap_columns = heatmap_df.columns[:40]

heatmap_df = heatmap_df[heatmap_columns]

plt.figure(figsize=(14, 5))

sns.heatmap(
    heatmap_df.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values Heatmap")

save_chart(
    "task03_missing_values_heatmap.png"
)


# ============================================================
# TASK 4 - DUPLICATES
# ============================================================

print("\nTASK 4 - DUPLICATE ROWS")

duplicate_count = int(
    df.duplicated().sum()
)

duplicate_percentage = round(

    (duplicate_count / rows) * 100,

    2

) if rows > 0 else 0

print("Duplicate Rows:", duplicate_count)
print("Duplicate Percentage:", duplicate_percentage, "%")


# ============================================================
# TASK 5 - LOAN AMOUNT DISTRIBUTION
# ============================================================

print("\nTASK 5 - LOAN AMOUNT DISTRIBUTION")

loan_amnt = find_column("loan_amnt")

if loan_amnt:

    plt.figure(figsize=(7, 4))

    sns.histplot(
        graph_df[loan_amnt].dropna(),
        bins=40
    )

    plt.title("Loan Amount Distribution")
    plt.xlabel("Loan Amount")
    plt.ylabel("Frequency")

    save_chart(
        "task05_loan_amount.png"
    )


# ============================================================
# TASK 6 - NUMERIC FEATURE DISTRIBUTIONS
# ============================================================

print("\nTASK 6 - NUMERIC FEATURE DISTRIBUTIONS")

numeric_features = [

    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "annual_inc"

]

for feature in numeric_features:

    column = find_column(feature)

    if column is None:
        continue

    data = graph_df[column].dropna()

    if feature == "annual_inc":

        data = data[
            data <= data.quantile(0.99)
        ]

    filename = f"task06_{feature}.png"

    plt.figure(figsize=(7, 4))

    sns.histplot(
        data,
        bins=40
    )

    plt.title(
        f"{column} Distribution"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    save_chart(filename)


# ============================================================
# TASK 7 - OUTLIER DETECTION
# ============================================================

print("\nTASK 7 - OUTLIER DETECTION")

boxplot_features = [

    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "annual_inc"

]

for feature in boxplot_features:

    column = find_column(feature)

    if column is None:
        continue

    data = graph_df[column].dropna()

    if feature == "annual_inc":

        data = data[
            data <= data.quantile(0.99)
        ]

    filename = (
        f"task07_boxplot_{feature}.png"
    )

    plt.figure(figsize=(7, 3))

    sns.boxplot(
        x=data
    )

    plt.title(
        f"Outlier Detection - {column}"
    )

    save_chart(filename)


# ============================================================
# TASK 8 - CORRELATION ANALYSIS
# ============================================================

print("\nTASK 8 - CORRELATION ANALYSIS")

correlation_features = [

    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "annual_inc"

]

correlation_columns = []

for feature in correlation_features:

    column = find_column(feature)

    if column:
        correlation_columns.append(column)


if len(correlation_columns) >= 2:

    correlation_matrix = (
        graph_df[
            correlation_columns
        ]
        .corr()
    )

    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(2))

    plt.figure(
        figsize=(9, 6)
    )

    sns.heatmap(

        correlation_matrix,

        annot=True,

        fmt=".2f",

        cmap="coolwarm",

        center=0

    )

    plt.title(
        "Credit Fair Correlation Heatmap"
    )

    save_chart(
        "task08_correlation.png"
    )


# ============================================================
# TASK 9 - RELATIONSHIP PLOTS
# ============================================================

print("\nTASK 9 - RELATIONSHIP PLOTS")

annual_inc = find_column("annual_inc")

int_rate = find_column("int_rate")


# Income vs Loan

if annual_inc and loan_amnt:

    relationship = graph_df[
        [
            annual_inc,
            loan_amnt
        ]
    ].dropna()

    relationship = relationship[
        relationship[annual_inc]
        <= relationship[annual_inc].quantile(0.99)
    ]

    relationship = relationship.sample(
        min(10000, len(relationship)),
        random_state=42
    )

    plt.figure(figsize=(7, 4))

    sns.regplot(

        data=relationship,

        x=annual_inc,

        y=loan_amnt,

        scatter_kws={
            "alpha": 0.2,
            "s": 10
        }

    )

    plt.title(
        "Annual Income vs Loan Amount"
    )

    plt.xlabel("Annual Income")
    plt.ylabel("Loan Amount")

    save_chart(
        "task09_income_vs_loan.png"
    )


# Loan vs Interest

if loan_amnt and int_rate:

    relationship = graph_df[
        [
            loan_amnt,
            int_rate
        ]
    ].dropna()

    relationship = relationship.sample(
        min(10000, len(relationship)),
        random_state=42
    )

    plt.figure(figsize=(7, 4))

    sns.regplot(

        data=relationship,

        x=loan_amnt,

        y=int_rate,

        scatter_kws={
            "alpha": 0.2,
            "s": 10
        }

    )

    plt.title(
        "Loan Amount vs Interest Rate"
    )

    plt.xlabel("Loan Amount")
    plt.ylabel("Interest Rate")

    save_chart(
        "task09_loan_vs_interest.png"
    )


# ============================================================
# TASK 10 - CATEGORICAL FEATURE COUNTS
# ============================================================

print("\nTASK 10 - CATEGORICAL FEATURE COUNTS")

categorical_features = [

    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "term",
    "emp_length"

]

for feature in categorical_features:

    column = find_column(feature)

    if column is None:
        continue

    counts = (
        graph_df[column]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(20)
    )

    filename = f"task10_{feature}.png"

    plt.figure(figsize=(7, 4))

    sns.barplot(

        x=counts.values,

        y=counts.index

    )

    plt.title(
        f"{column} Distribution"
    )

    plt.xlabel("Count")
    plt.ylabel(column)

    save_chart(filename)


# ============================================================
# TASK 11 - GRADE VS INTEREST RATE
# ============================================================

print("\nTASK 11 - GRADE VS INTEREST RATE")

grade = find_column("grade")

if grade and int_rate:

    grade_data = graph_df[
        [
            grade,
            int_rate
        ]
    ].dropna()

    plt.figure(figsize=(7, 4))

    sns.boxplot(

        data=grade_data,

        x=grade,

        y=int_rate

    )

    plt.title(
        "Interest Rate by Loan Grade"
    )

    plt.xlabel("Grade")
    plt.ylabel("Interest Rate")

    save_chart(
        "task11_grade_interest.png"
    )


# ============================================================
# TASK 12 - HOME OWNERSHIP VS LOAN AMOUNT
# ============================================================

print("\nTASK 12 - HOME OWNERSHIP VS LOAN AMOUNT")

home = find_column("home_ownership")

if home and loan_amnt:

    home_data = graph_df[
        [
            home,
            loan_amnt
        ]
    ].dropna()

    plt.figure(figsize=(8, 4))

    sns.boxplot(

        data=home_data,

        x=home,

        y=loan_amnt

    )

    plt.title(
        "Loan Amount by Home Ownership"
    )

    plt.xlabel("Home Ownership")
    plt.ylabel("Loan Amount")

    plt.xticks(rotation=20)

    save_chart(
        "task12_home_loan.png"
    )


# ============================================================
# TASK 13 - EMPLOYMENT LENGTH
# ============================================================

print("\nTASK 13 - EMPLOYMENT LENGTH")

emp_length = find_column("emp_length")

if emp_length:

    employment_counts = (
        graph_df[emp_length]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
    )

    plt.figure(figsize=(8, 4))

    sns.barplot(

        x=employment_counts.index,

        y=employment_counts.values

    )

    plt.title(
        "Employment Length Distribution"
    )

    plt.xlabel("Employment Length")
    plt.ylabel("Count")

    plt.xticks(rotation=35)

    save_chart(
        "task13_employment_length.png"
    )


# ============================================================
# TASK 14 - LOAN VS FUNDED AMOUNT
# ============================================================

print("\nTASK 14 - LOAN VS FUNDED AMOUNT")

funded_amnt = find_column("funded_amnt")

if loan_amnt and funded_amnt:

    loan_funded = graph_df[
        [
            loan_amnt,
            funded_amnt
        ]
    ].dropna()

    loan_funded = loan_funded.sample(
        min(10000, len(loan_funded)),
        random_state=42
    )

    plt.figure(figsize=(7, 4))

    sns.scatterplot(

        data=loan_funded,

        x=loan_amnt,

        y=funded_amnt,

        alpha=0.3,

        s=15

    )

    plt.title(
        "Loan Amount vs Funded Amount"
    )

    plt.xlabel("Loan Amount")
    plt.ylabel("Funded Amount")

    save_chart(
        "task14_loan_funded.png"
    )


# ============================================================
# TASK 15 - PAIRPLOT
# ============================================================

print("\nTASK 15 - PAIRPLOT")

pair_features = [

    "loan_amnt",
    "funded_amnt",
    "int_rate",
    "installment",
    "annual_inc"

]

pair_columns = []

for feature in pair_features:

    column = find_column(feature)

    if column:
        pair_columns.append(column)


if len(pair_columns) >= 2:

    pair_data = graph_df[
        pair_columns
    ].dropna()

    pair_data = pair_data.sample(
        min(3000, len(pair_data)),
        random_state=42
    )

    pairplot = sns.pairplot(
        pair_data
    )

    pairplot.fig.suptitle(
        "Credit Fair - Pairwise Relationships",
        y=1.02
    )

    pairplot.fig.savefig(

        os.path.join(
            OUTPUT_DIR,
            "task15_pairplot.png"
        ),

        dpi=90,

        bbox_inches="tight"

    )

    plt.close("all")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("ALL 15 EDA TASKS COMPLETED")
print("=" * 70)