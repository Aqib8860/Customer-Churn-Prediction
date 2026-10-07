import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Reproducibility
#
# Same random seed use karne par har run mein
# same dataset generate hoga.
# ---------------------------------------------------------
np.random.seed(42)


# ---------------------------------------------------------
# Number of customers
# ---------------------------------------------------------
n_customers = 2000


# ---------------------------------------------------------
# Generate customer features
# ---------------------------------------------------------
age = np.random.randint(
    18,
    70,
    size=n_customers,
)

tenure_months = np.random.randint(
    1,
    61,
    size=n_customers,
)

monthly_charges = np.round(
    np.random.uniform(
        500,
        5000,
        size=n_customers,
    ),
    2,
)

support_tickets = np.random.poisson(
    lam=2,
    size=n_customers,
)


# ---------------------------------------------------------
# Categorical features
# ---------------------------------------------------------
contract_type = np.random.choice(
    [
        "Monthly",
        "Yearly",
        "Two Year",
    ],
    size=n_customers,
    p=[
        0.55,
        0.30,
        0.15,
    ],
)

payment_method = np.random.choice(
    [
        "UPI",
        "Credit Card",
        "Debit Card",
        "Net Banking",
    ],
    size=n_customers,
)


# ---------------------------------------------------------
# Create churn probability.
#
# This creates realistic relationships:
#
# - Short tenure → higher churn
# - Higher charges → higher churn
# - More support tickets → higher churn
# - Monthly contract → higher churn
# ---------------------------------------------------------
churn_score = (
    0.02 * (60 - tenure_months)
    + 0.00025 * monthly_charges
    + 0.15 * support_tickets
    + (contract_type == "Monthly") * 1.5
    + (contract_type == "Yearly") * 0.3
)


# ---------------------------------------------------------
# Convert score into probability.
# ---------------------------------------------------------
churn_probability = (
    1 / (1 + np.exp(-(
        churn_score - 2.5
    )))
)


# ---------------------------------------------------------
# Generate binary churn target.
#
# 0 = customer stays
# 1 = customer churns
# ---------------------------------------------------------
churn = (
    np.random.random(n_customers)
    < churn_probability
).astype(int)


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------
df = pd.DataFrame(
    {
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "support_tickets": support_tickets,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "churn": churn,
    }
)


# ---------------------------------------------------------
# Introduce realistic missing values.
#
# These are intentionally added so that our
# preprocessing pipeline has to handle them.
# ---------------------------------------------------------
missing_columns = [
    "age",
    "monthly_charges",
    "contract_type",
]

for column in missing_columns:

    # Randomly select approximately 3% of rows.
    missing_indices = np.random.choice(
        df.index,
        size=int(n_customers * 0.03),
        replace=False,
    )

    # Replace selected values with NaN.
    df.loc[
        missing_indices,
        column,
    ] = np.nan


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------
output_path = (
    "data/customer_churn.csv"
)

df.to_csv(
    output_path,
    index=False,
)


print(
    f"Dataset created: {output_path}"
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)