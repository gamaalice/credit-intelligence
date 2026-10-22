# Credit Intelligence

An MVP of a credit risk analysis and default prediction application, developed as part of a project for a company.

## About the Project

**Credit Intelligence** is an application designed for credit data analysis, featuring tools for visualizing key metrics, comparing Machine Learning model experiments, and simulating default predictions.

The project was developed to explore solutions for credit risk analysis, data visualization, and the presentation of predictive model results.

> **Status: MVP (Minimum Viable Product).** This version represents an initial implementation developed during the project and made available for demonstration and portfolio purposes. It does not represent the company's final product and may not include all the features, integrations, or requirements of a production-ready solution.

## MVP Features

- **Analytics Dashboard:** Visualization of key metrics and information related to credit risk.
- **Default Prediction:** A form for simulating predictions based on an individual's credit information.
- **Model Comparison:** Visualization of Machine Learning experiments and predictive model results.
- **Evaluation Metrics:** Presentation of metrics used to assess model performance.
- **Responsive Interface:** An interface designed to organize and present analytical results.

## Screenshots

Screenshots coming soon.

## Technologies

### Frontend

- React
- TypeScript
- Tailwind CSS
- Lucide React
- Vite

### Backend and API

- Python
- FastAPI
- Uvicorn

### Data Science and Machine Learning

- PySpark
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Matplotlib

### Infrastructure and Testing

- Docker
- Docker Compose
- Pytest
- FastAPI TestClient

## Automated Tests

The project includes automated tests for dataset validation and API behavior. The test suite contains **8 tests** covering:

- **Dataset schema:** verifies that the dataset uses the documented 23 predictors.
- **Data quality profile:** checks row count, feature count, default rate, missing values, and exclusion of the identifier from model features.
- **Invalid dataset schemas:** verifies that incomplete schemas are rejected.
- **Target validation:** verifies that target values outside the binary classes are rejected.
- **API summary endpoint:** checks the response when model training has not been run.
- **Experiment endpoint:** checks that no experiments are returned when real results are not available.
- **Prediction without a trained model:** verifies that the API returns an appropriate service-unavailable response.
- **Prediction input validation:** checks that invalid categorical values are rejected.

Run the tests from the repository root with:

```bash
uv run pytest -q
```

The suite is intended to verify core data-validation rules and API behavior. It does not, by itself, establish that the model is suitable for real-world credit decisions or that every frontend workflow has been tested.

## Context and Limitations

This project was developed for a company and made available in its MVP version as part of the solution's development and learning process.

As an initial version, the application should not be considered the company's final product or a system validated for use in real-world credit approval decisions.

The predictions and metrics displayed depend on the data, models, and integrations available in this version. Results presented in a demonstration interface do not guarantee performance in real-world scenarios.

## Project Objective

To explore the development of a credit risk analysis application that integrates data visualization, metric reporting, and prediction simulations into a single interface.

---

**Note:** This repository showcases an MVP version of the project and may not reflect the final implementation delivered to or used by the company.
