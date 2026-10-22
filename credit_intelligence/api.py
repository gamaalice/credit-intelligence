from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from credit_intelligence.constants import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    RESULTS_PATH,
)

app = FastAPI(
    title="Credit Intelligence API",
    description=(
        "Educational inference and experiment results for the UCI Default of "
        "Credit Card Clients dataset. Not for real credit decisions."
    ),
    version="1.0.0",
)


class CreditPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    LIMIT_BAL: float = Field(ge=0, le=10_000_000)
    SEX: Literal[1, 2]
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=120)
    PAY_0: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="No validated model is available. Run the training pipeline first.",
        )
    try:
        return joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError) as error:
        raise HTTPException(
            status_code=503,
            detail="The trained model artifact could not be loaded.",
        ) from error


def load_results() -> dict[str, object] | None:
    if not RESULTS_PATH.exists():
        return None
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=503,
            detail="The training results file is unavailable or invalid.",
        ) from error


@app.get("/credit/summary")
def credit_summary() -> dict[str, object]:
    results = load_results()
    if results is None:
        return {
            "status": "not_trained",
            "dataset": None,
            "selected_model": None,
            "trained_at": None,
            "experiments": [],
            "message": "Run the training pipeline to generate real experiment results.",
        }

    dataset = results.get("dataset", {})
    return {
        "status": "ready",
        "dataset": {
            "name": dataset.get("name"),
            "source": dataset.get("source"),
            "rows": dataset.get("rows"),
            "features": dataset.get("features"),
            "default_rate": dataset.get("default_rate"),
            "missing_values": dataset.get("missing_values"),
            "duplicate_rows": dataset.get("duplicate_rows"),
        },
        "selected_model": results.get("selected_model"),
        "trained_at": results.get("trained_at"),
        "experiments": [
            {
                "model": experiment["model"],
                "precision": experiment["precision"],
                "recall": experiment["recall"],
                "f1": experiment["f1"],
                "roc_auc": experiment["roc_auc"],
                "accuracy": experiment["accuracy"],
                "brier_score": experiment["brier_score"],
                "confusion_matrix": experiment["confusion_matrix"],
                "selected": experiment["selected"],
            }
            for experiment in results.get("experiments", [])
        ],
        "message": None,
    }


@app.get("/credit/experiments")
def credit_experiments() -> list[dict[str, object]]:
    results = load_results()
    if results is None:
        return []
    return credit_summary()["experiments"]  # type: ignore[return-value]


@app.post("/credit/predict")
def predict_credit_default(payload: CreditPredictionInput) -> dict[str, object]:
    model = get_model()
    features = pd.DataFrame(
        [payload.model_dump()],
        columns=FEATURE_COLUMNS,
    )
    probability = float(model.predict_proba(features)[0, 1])
    results = load_results() or {}
    return {
        "default_probability": probability,
        "model": results.get("selected_model", "Selected trained model"),
        "educational_notice": (
            "Educational estimate only. This probability is not a credit decision "
            "and must not be used to approve or reject a real applicant."
        ),
    }
