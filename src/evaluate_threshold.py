import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/customer_churn.csv")

X = df.drop(columns=["churn"])

y = df["churn"]

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

model = joblib.load("models/customer_churn_pipeline.joblib")

probabilities = model.predict_proba(X_test)[:, 1]

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y_test, predictions, zero_division=0)

    recall = recall_score(y_test, predictions, zero_division=0)

    f1 = f1_score(y_test, predictions, zero_division=0)

    matrix = confusion_matrix(y_test, predictions)

    # -----------------------------------------------------
    # Display results.
    # -----------------------------------------------------
    print("\n" + "=" * 50)
    print(
        f"Threshold: {threshold:.2f}"
    )
    print("=" * 50)

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print("\nConfusion Matrix:")

    print(matrix)

