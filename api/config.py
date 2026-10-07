import os

from dotenv import load_dotenv


# ---------------------------------------------------------
# Load variables from the project's .env file.
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Read churn threshold from environment.
#
# If CHURN_THRESHOLD is not present,
# use 0.40 as the default.
# ---------------------------------------------------------
CHURN_THRESHOLD = float(
    os.getenv(
        "CHURN_THRESHOLD",
        "0.40",
    )
)