# Credit Intelligence

Projeto de portfólio para demonstrar análise de dados, PySpark, SQL e Machine
Learning com o conjunto público **Default of Credit Card Clients**, da UCI.
O objetivo é exclusivamente educacional: nenhuma previsão pode ser usada para
aprovar ou recusar crédito real.

## Fonte e limites dos dados

- Fonte oficial: [UCI Machine Learning Repository — dataset 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- O pipeline baixa o arquivo original da UCI quando não encontra uma cópia local.
- A planilha oficial inclui uma linha descritiva antes do cabeçalho; a leitura
  usa o cabeçalho observado e normaliza `default payment next month` para
  `default.payment.next.month`, registrando esse mapeamento no perfil dos dados.
- O alvo é `default.payment.next.month`; `ID` é usado somente para verificações
  de qualidade e nunca entra como variável preditora.
- A amostra representa titulares de cartão em Taiwan em 2005. Não representa
  clientes brasileiros nem condições atuais de crédito.
- O código valida as colunas efetivamente recebidas e falha explicitamente se
  o arquivo não corresponder ao esquema esperado.
- Valores faltantes são tratados dentro dos pipelines ajustados no treino.
  Duplicatas completas são contabilizadas e mantidas; não há remoção silenciosa.
- As probabilidades não são calibradas por pós-processamento. Brier score e
  curva de confiabilidade são apresentados como diagnósticos.

## Requisitos

- Python 3.11
- Java 17 (necessário para PySpark em modo local)
- `uv` para sincronizar as dependências travadas em `uv.lock`

No Replit, o ambiente usa Python 3.11 e Java 17. Em ambiente local:

```bash
uv sync --frozen
```

## Executar o pipeline

```bash
uv run python -m credit_intelligence.training
```

O pipeline baixa o arquivo público original da UCI, valida o esquema, produz
relatórios de qualidade, executa transformações e agregações SQL reais no
PySpark, separa treino/validação/teste com estratificação e semente fixa,
compara Logistic Regression, Random Forest e Gradient Boosting, registra
parâmetros e métricas no MLflow e salva o pipeline selecionado.

A seleção usa o maior F1 da classe positiva na validação; empates são
desempatados por recall e ROC-AUC de validação. O conjunto de teste reservado
é usado apenas para a avaliação final e não para escolher o modelo.

Arquivos gerados em `data/` (ignorados pelo Git):

- `raw/`: cópia local do arquivo UCI baixado
- `eda/`: perfil do dataset e agregações produzidas pelo PySpark/SQL
- `model_artifacts/results.json`: métricas reais e metadados dos experimentos
- `model_artifacts/selected_model.joblib`: pipeline completo para inferência
- `mlruns/`: experimentos locais do MLflow

## API FastAPI

Depois do treinamento:

```bash
uv run uvicorn credit_intelligence.api:app --host 0.0.0.0 --port 8000
```

Documentação interativa: `http://localhost:8000/docs`.

- `GET /credit/summary` — resumo do dataset e dos resultados mais recentes
- `GET /credit/experiments` — métricas comparativas
- `POST /credit/predict` — valida as 23 variáveis e retorna uma probabilidade

Sem modelo treinado, a API responde com um estado explícito `not_trained` ou
HTTP 503; ela não cria previsões de exemplo.

## Dashboard Streamlit

```bash
uv run streamlit run credit_intelligence/dashboard.py --server.port 8501
```

O dashboard lê os resultados e relatórios gerados pelo pipeline: comparação
de métricas no teste, matriz de confusão, calibração, importância por
permutação e agregações SQL do PySpark.

## Aplicação Replit

O projeto também inclui uma interface web em português que consulta os
endpoints `/api/credit/*`. O servidor Express existente encaminha essas
requisições para a API FastAPI local, sem duplicar a lógica de inferência.
Execute o pipeline antes de usar a previsão interativa.

## Docker

```bash
docker compose build
docker compose run --rm api uv run python -m credit_intelligence.training
docker compose up api dashboard
```

- API: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

O volume `credit-data` mantém dados e artefatos locais entre execuções. Não são
necessárias credenciais, serviços pagos ou integrações comerciais.

## Testes

```bash
uv run pytest -q
```

Os testes cobrem o contrato dos dados, validações da API, o estado sem modelo
treinado e as respostas de erro. O treinamento completo pode ser repetido com
o mesmo comando e semente.
