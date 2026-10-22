from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from credit_intelligence.constants import EDA_DIR, RESULTS_PATH

st.set_page_config(
    page_title="Credit Intelligence",
    page_icon="CI",
    layout="wide",
)
st.title("Credit Intelligence")
st.caption(
    "Projeto educacional de Machine Learning com o conjunto público da UCI."
)
st.warning(
    "Uso exclusivamente educacional. As probabilidades e métricas não devem "
    "ser usadas para decisões reais de crédito."
)

if not RESULTS_PATH.exists():
    st.info(
        "Ainda não há resultados de treinamento. Execute "
        "`uv run python -m credit_intelligence.training` para baixar o dataset "
        "oficial, treinar os modelos e gerar métricas reais."
    )
    st.stop()

try:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Não foi possível ler os resultados de treinamento: {error}")
    st.stop()

dataset = results["dataset"]
experiments = pd.DataFrame(results["experiments"])
selected = experiments.loc[experiments["selected"]].iloc[0]

st.subheader("Conjunto de dados")
st.markdown(
    f"[{dataset['name']}]({dataset['source']}) · "
    f"{dataset['rows']:,} linhas observadas · "
    f"{dataset['features']} variáveis de entrada · "
    f"taxa de inadimplência {dataset['default_rate']:.1%}"
)
quality_columns = st.columns(3)
quality_columns[0].metric("Valores ausentes", f"{dataset['missing_values']:,}")
quality_columns[1].metric("Linhas duplicadas", f"{dataset['duplicate_rows']:,}")
quality_columns[2].metric("Modelo selecionado", results["selected_model"])

st.subheader("Comparação no conjunto de teste")
st.caption(
    "A seleção foi feita pelo F1 na validação; as métricas abaixo vêm do "
    "conjunto de teste reservado."
)
metric_columns = st.columns(4)
for column, key, label in zip(
    metric_columns,
    ("precision", "recall", "f1", "roc_auc"),
    ("Precisão", "Recall", "F1", "ROC-AUC"),
    strict=True,
):
    column.metric(label, f"{float(selected[key]):.3f}")

st.bar_chart(
    experiments.set_index("model")[["precision", "recall", "f1", "roc_auc"]]
)
st.dataframe(
    experiments[
        [
            "model",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "accuracy",
            "brier_score",
            "selected",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Matriz de confusão do modelo selecionado")
matrix = selected["confusion_matrix"]
st.dataframe(
    pd.DataFrame(
        matrix,
        index=["Real: sem inadimplência", "Real: inadimplência"],
        columns=["Previsto: sem inadimplência", "Previsto: inadimplência"],
    ),
    use_container_width=True,
)

st.subheader("Calibração observada")
st.caption(
    "Bins de probabilidade no teste. A linha ideal coincide com a frequência "
    "observada; isto é um diagnóstico, não uma garantia de calibração."
)
calibration = pd.DataFrame(selected["calibration"])
if not calibration.empty:
    st.line_chart(
        calibration.set_index("mean_predicted_probability")[
            ["fraction_positive"]
        ],
        y="fraction_positive",
    )
    st.dataframe(calibration, use_container_width=True, hide_index=True)
else:
    st.info("Não foi possível formar bins de calibração para este resultado.")

st.subheader("Influência das variáveis")
importance = pd.DataFrame(results["feature_importance"])
st.bar_chart(importance.head(12).set_index("feature"))

age_report = EDA_DIR / "pyspark_age_default_rates.csv"
payment_report = EDA_DIR / "pyspark_repayment_status_rates.csv"
if age_report.exists() and payment_report.exists():
    st.subheader("Análises SQL executadas em PySpark")
    age_tab, payment_tab = st.tabs(["Por faixa etária", "Por atraso recente"])
    with age_tab:
        st.dataframe(pd.read_csv(age_report), use_container_width=True)
    with payment_tab:
        st.dataframe(pd.read_csv(payment_report), use_container_width=True)

st.subheader("Limitações")
for limitation in results["limitations"]:
    st.markdown(f"- {limitation}")
