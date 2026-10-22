
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("CREDIT_DATA_DIR", PROJECT_ROOT / "data")).resolve()
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "model_artifacts"
EDA_DIR = DATA_DIR / "eda"

MLFLOW_DB_PATH = DATA_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = DATA_DIR / "mlflow_artifacts"

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
UCI_DATASET_NAME = "Default of Credit Card Clients"
TARGET_COLUMN = "default.payment.next.month"
IDENTIFIER_COLUMN = "ID"

CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLUMNS = [IDENTIFIER_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]

RANDOM_STATE = 42

MODEL_PATH = OUTPUT_DIR / "selected_model.joblib"
RESULTS_PATH = OUTPUT_DIR / "results.json"
EXPERIMENTS_PATH = OUTPUT_DIR / "experiments.json"
