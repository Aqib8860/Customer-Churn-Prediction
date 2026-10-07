import pandas as pd

# ---------------------------------------------------------
# Load the generated customer churn dataset.
# ---------------------------------------------------------
df = pd.read_csv(
    "data/customer_churn.csv"
)

# ---------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------
print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ---------------------------------------------------------
# Data types
# ---------------------------------------------------------
print("\nData Types:")
print(df.dtypes)

# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# ---------------------------------------------------------
# Target distribution
#
# 0 = customer stays
# 1 = customer churns
# ---------------------------------------------------------
print("\nChurn Distribution:")
print(df["churn"].value_counts())

print("\nChurn Percentage:")
print(
    df["churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ---------------------------------------------------------
# Numerical feature summary
# ---------------------------------------------------------
print("\nNumerical Summary:")
print(
    df.describe()
)

# ---------------------------------------------------------
# Categorical feature values
# ---------------------------------------------------------
print("\nContract Types:")
print(
    df["contract_type"].value_counts(dropna=False)
)

print("\nPayment Methods:")
print(
    df["payment_method"].value_counts(dropna=False)
)