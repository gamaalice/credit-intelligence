from __future__ import annotations

from fastapi.testclient import TestClient

import credit_intelligence.api as api

client = TestClient(api.app)


def valid_input() -> dict[str, int | float]:
    return {
        "LIMIT_BAL": 50000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 1000,
        "BILL_AMT2": 1000,
        "BILL_AMT3": 1000,
        "BILL_AMT4": 1000,
        "BILL_AMT5": 1000,
        "BILL_AMT6": 1000,
        "PAY_AMT1": 500,
        "PAY_AMT2": 500,
        "PAY_AMT3": 500,
        "PAY_AMT4": 500,
        "PAY_AMT5": 500,
        "PAY_AMT6": 500,
    }


def test_summary_explains_when_training_has_not_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "RESULTS_PATH", tmp_path / "missing-results.json")

    response = client.get("/credit/summary")

    assert response.status_code == 200
    assert response.json()["status"] == "not_trained"
    assert response.json()["experiments"] == []


def test_experiment_list_is_empty_until_real_results_exist(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "RESULTS_PATH", tmp_path / "missing-results.json")

    response = client.get("/credit/experiments")

    assert response.status_code == 200
    assert response.json() == []


def test_prediction_requires_a_trained_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing-model.joblib")
    api.get_model.cache_clear()

    response = client.post("/credit/predict", json=valid_input())

    assert response.status_code == 503
    assert "training pipeline" in response.json()["detail"]


def test_prediction_input_rejects_an_out_of_range_category() -> None:
    payload = valid_input()
    payload["SEX"] = 9

    response = client.post("/credit/predict", json=payload)

    assert response.status_code == 422
