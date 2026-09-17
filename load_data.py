import pandas as pd


# ============================================================
# CSV PATH
# ============================================================

CSV_PATH = r"C:\Users\venne\PycharmProjects\semproject\creditfair\creditfair.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_credit_data(nrows=50000):

    df = pd.read_csv(
        CSV_PATH,
        nrows=nrows,
        low_memory=False
    )

    return df


# ============================================================
# DATA SUMMARY
# ============================================================

def get_data_summary():

    df = load_credit_data()

    # First 5 rows
    first_five = (
        df.head(5)
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    # Structure
    structure = []

    for column in df.columns:

        structure.append({
            "column": column,
            "dtype": str(df[column].dtype),
            "non_null": int(df[column].notna().sum()),
            "missing": int(df[column].isna().sum())
        })

    # Numeric columns
    numeric_columns = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )

    # Categorical columns
    categorical_columns = (
        df.select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    return {

        "rows": len(df),

        "columns": len(df.columns),

        "column_names": df.columns.tolist(),

        "first_five": first_five,

        "structure": structure,

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    df = load_credit_data()

    print("Dataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())