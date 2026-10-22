# Credit Intelligence

Portfolio project for analyzing the public UCI Default of Credit Card Clients
dataset and comparing reproducible default-probability models. It is for
education only and must never be used to make real credit decisions.

## Run & Operate

- `uv sync --frozen` — install the pinned Python dependencies
- `uv run python -m credit_intelligence.training` — download, analyze, and train
- `uv run pytest -q` — run data and API tests
- `uv run streamlit run credit_intelligence/dashboard.py --server.port 8501` — run Streamlit
- `uv run uvicorn credit_intelligence.api:app --host 0.0.0.0 --port 8000` — run FastAPI independently
- `pnpm --filter @workspace/api-server run dev` — run Express gateway and local FastAPI together
- `pnpm --filter @workspace/credit-intelligence run dev` — run the React dashboard
- `pnpm run typecheck` — check the TypeScript workspace
- `docker compose build` — build the reproducible Python image

The browser dashboard uses the `/api/credit/*` endpoints. The Express API
workflow starts FastAPI on its internal local port and forwards those routes.
Run model training before requesting a prediction; until then, the API reports
an explicit not-trained state.

## Stack

- Python 3.11, Pandas, NumPy, PySpark local mode, SQL, Scikit-learn, MLflow
- FastAPI, Streamlit, pytest, Docker
- React, Vite, TypeScript, OpenAPI-generated API client
- Java 17 is required by local PySpark

## Where things live

- `credit_intelligence/` — dataset loading, Spark analysis, training, API, and Streamlit dashboard
- `tests/` — Python data-contract and API validation tests
- `artifacts/credit-intelligence/` — browser dashboard
- `artifacts/api-server/` — API gateway and local FastAPI process runner
- `lib/api-spec/openapi.yaml` — API contract source of truth
- `data/` — downloaded dataset, model, metrics, Spark reports, and local MLflow runs (ignored by Git)
- `README.md` — setup, execution, model-selection rationale, and limitations

## Architecture decisions

- The UCI source is downloaded as supplied; the spreadsheet's descriptive
  first row is skipped, and its observed target label is mapped explicitly.
- `ID` is used for validation only and excluded from the model features.
- Train, validation, and test data are separated before preprocessing is fit.
  The selected model is chosen on validation F1, then recall and ROC-AUC.
- PySpark performs actual local transformations and SQL aggregations; Pandas is
  used for input validation and Scikit-learn model pipelines.
- No Replit database or paid/external service is needed; generated datasets and
  model artifacts stay local and are excluded from version control.

## Product

Shows provenance and quality checks for the UCI data, real comparative
evaluation metrics, confusion matrices, calibration diagnostics, feature
influence, PySpark SQL summaries, and an educational inference form.

## Gotchas

- The API intentionally returns not-trained or 503 when no real model artifact
  exists; it never substitutes example metrics.
- The downloaded public dataset and generated model artifacts are under `data/`
  and must remain ignored by Git.
- Keep the educational-use disclaimer visible in both dashboard experiences.
