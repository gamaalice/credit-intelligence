import { useState, type FormEvent } from 'react';
import {
  Activity,
  ArrowUpRight,
  CircleHelp,
  Database,
  Gauge,
  RefreshCw,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';
import {
  useGetCreditSummary,
  getGetCreditSummaryQueryKey,
  useGetCreditExperiments,
  getGetCreditExperimentsQueryKey,
  usePredictCreditDefault,
} from '@workspace/api-client-react';
import type { CreditPredictionInput, CreditSummary, ModelExperiment } from '@workspace/api-client-react';

type FieldSpec = {
  key: keyof CreditPredictionInput;
  label: string;
  min?: number;
  max?: number;
  group?: 'profile' | 'history' | 'bills' | 'payments';
};

const fields: FieldSpec[] = [
  { key: 'LIMIT_BAL', label: 'Limite de crédito (NT$)', min: 0, max: 10000000, group: 'profile' },
  { key: 'SEX', label: 'Sexo', group: 'profile' },
  { key: 'EDUCATION', label: 'Escolaridade', min: 0, max: 6, group: 'profile' },
  { key: 'MARRIAGE', label: 'Estado civil', min: 0, max: 3, group: 'profile' },
  { key: 'AGE', label: 'Idade', min: 18, max: 120, group: 'profile' },
  { key: 'PAY_0', label: 'PAY_0 · Set', min: -2, max: 9, group: 'history' },
  { key: 'PAY_2', label: 'PAY_2 · Ago', min: -2, max: 9, group: 'history' },
  { key: 'PAY_3', label: 'PAY_3 · Jul', min: -2, max: 9, group: 'history' },
  { key: 'PAY_4', label: 'PAY_4 · Jun', min: -2, max: 9, group: 'history' },
  { key: 'PAY_5', label: 'PAY_5 · Mai', min: -2, max: 9, group: 'history' },
  { key: 'PAY_6', label: 'PAY_6 · Abr', min: -2, max: 9, group: 'history' },
  { key: 'BILL_AMT1', label: 'BILL_AMT1 · Set', group: 'bills' },
  { key: 'BILL_AMT2', label: 'BILL_AMT2 · Ago', group: 'bills' },
  { key: 'BILL_AMT3', label: 'BILL_AMT3 · Jul', group: 'bills' },
  { key: 'BILL_AMT4', label: 'BILL_AMT4 · Jun', group: 'bills' },
  { key: 'BILL_AMT5', label: 'BILL_AMT5 · Mai', group: 'bills' },
  { key: 'BILL_AMT6', label: 'BILL_AMT6 · Abr', group: 'bills' },
  { key: 'PAY_AMT1', label: 'PAY_AMT1 · Set', min: 0, group: 'payments' },
  { key: 'PAY_AMT2', label: 'PAY_AMT2 · Ago', min: 0, group: 'payments' },
  { key: 'PAY_AMT3', label: 'PAY_AMT3 · Jul', min: 0, group: 'payments' },
  { key: 'PAY_AMT4', label: 'PAY_AMT4 · Jun', min: 0, group: 'payments' },
  { key: 'PAY_AMT5', label: 'PAY_AMT5 · Mai', min: 0, group: 'payments' },
  { key: 'PAY_AMT6', label: 'PAY_AMT6 · Abr', min: 0, group: 'payments' },
];

const groups: Record<NonNullable<FieldSpec['group']>, string> = {
  profile: 'Perfil do titular',
  history: 'Status de pagamento · últimos 6 meses',
  bills: 'Valor da fatura · NT$',
  payments: 'Pagamentos anteriores · NT$',
};

const formatMetric = (value: number | undefined, digits = 3) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';

const formatCount = (value: number | undefined) =>
  typeof value === 'number' ? new Intl.NumberFormat().format(value) : '—';

function getErrorMessage(error: unknown) {
  return error
    ? 'Não foi possível concluir a solicitação. Tente novamente.'
    : 'Não foi possível carregar estas informações.';
}

function LoadingPanel({ rows = 3 }: { rows?: number }) {
  return (
    <div className="panel" aria-label="Carregando dados" data-testid="status-loading">
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} style={{ padding: '16px 19px' }}>
          <div className="skeleton" style={{ height: 12, width: `${55 + (index % 3) * 13}%` }} />
        </div>
      ))}
    </div>
  );
}

function ErrorPanel({ message, onRetry, testId }: { message: string; onRetry: () => void; testId: string }) {
  return (
    <div className="error-box" data-testid={testId} role="alert">
      <span>{message}</span>
      <button className="retry-button" type="button" onClick={onRetry} data-testid={`button-retry-${testId}`}>
        Tentar novamente
      </button>
    </div>
  );
}

function DatasetPanel({ dataset, trainedAt }: { dataset: CreditSummary['dataset']; trainedAt: string | null }) {
  if (!dataset) {
    return (
      <div className="dataset-panel" data-testid="dataset-unavailable">
        <div>
          <div className="eyebrow">Registro do conjunto de dados</div>
          <h2 className="dataset-name">Proveniência indisponível</h2>
          <p className="dataset-source">A API não forneceu detalhes do conjunto de dados para este estado do modelo.</p>
        </div>
      </div>
    );
  }
  const stats = [
    ['Linhas', formatCount(dataset.rows)],
    ['Variáveis', formatCount(dataset.features)],
    ['Taxa de inadimplência', typeof dataset.default_rate === 'number' ? `${(dataset.default_rate * 100).toFixed(2)}%` : '—'],
    ['Ausentes / duplicadas', `${formatCount(dataset.missing_values)} / ${formatCount(dataset.duplicate_rows)}`],
  ];
  return (
    <div className="dataset-panel" data-testid="dataset-summary">
      <div>
        <div className="eyebrow">Proveniência dos dados</div>
        <h2 className="dataset-name" data-testid="text-dataset-name">{dataset.name || 'Nome do conjunto não informado'}</h2>
        <div className="dataset-source">
          Fonte · {dataset.source || 'Não informada'}
          {trainedAt ? <><br />Último treinamento · {new Date(trainedAt).toLocaleString('pt-BR')}</> : null}
        </div>
      </div>
      <div className="dataset-stats">
        {stats.map(([label, value]) => (
          <div className="dataset-stat" key={label} data-testid={`stat-${label.toLowerCase().replace(/[^a-z]+/g, '-')}`}>
            <div className="dataset-value">{value}</div>
            <div className="dataset-label">{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SelectedModelCard({ experiment }: { experiment?: ModelExperiment }) {
  if (!experiment) {
    return (
      <div className="panel" data-testid="selected-model-empty">
        <div className="panel-head"><h3 className="panel-title">Modelo selecionado</h3><span className="panel-meta">Sem seleção</span></div>
        <div style={{ padding: 20, color: '#78817e', fontSize: 11 }}>A API não retornou um experimento selecionado.</div>
      </div>
    );
  }
  const metricRows = [
    ['Precisão', experiment.precision],
    ['Revocação', experiment.recall],
    ['Pontuação F1', experiment.f1],
    ['ROC AUC', experiment.roc_auc],
    ['Acurácia', experiment.accuracy],
    ['Pontuação de Brier', experiment.brier_score],
  ];
  const matrix = experiment.confusion_matrix;
  return (
    <div className="panel" data-testid="selected-model-card">
      <div className="panel-head">
        <div><div className="panel-meta">Experimento ativo</div><h3 className="panel-title">{experiment.model}</h3></div>
        <span className="selected-pill">Selecionado</span>
      </div>
      <div className="metric-grid">
        {metricRows.map(([label, value]) => (
          <div className="metric-cell" key={label}>
            <span className="metric-label">{label}</span>
            <span className="metric-number">{formatMetric(value as number)}</span>
          </div>
        ))}
      </div>
      <div className="cm-wrap">
        <div className="cm-caption">Matriz de confusão · real × previsto</div>
        {Array.isArray(matrix) && matrix.length >= 2 && matrix[0]?.length >= 2 ? (
          <div className="cm-grid" data-testid="confusion-matrix">
            <div className="cm-axis" />
            <div className="cm-axis">Prev. 0</div><div className="cm-axis">Prev. 1</div>
            <div className="cm-axis">Real 0</div>
            <div className="cm-cell">{formatCount(matrix[0][0])}</div><div className="cm-cell false">{formatCount(matrix[0][1])}</div>
            <div className="cm-axis">Real 1</div>
            <div className="cm-cell false">{formatCount(matrix[1][0])}</div><div className="cm-cell">{formatCount(matrix[1][1])}</div>
          </div>
        ) : <div style={{ color: '#858e89', fontSize: 10 }}>Matriz de confusão não fornecida.</div>}
      </div>
    </div>
  );
}

function ExperimentTable({ experiments }: { experiments: ModelExperiment[] }) {
  if (!experiments.length) {
    return <div className="panel" style={{ padding: 20, color: '#76817e', fontSize: 11 }} data-testid="experiments-empty">A API não retornou experimentos.</div>;
  }
  return (
    <div className="table-wrap">
      <table className="experiment-table" data-testid="table-experiments">
        <thead><tr>
          <th>Experimento</th><th>Precisão</th><th>Revocação</th><th>F1</th><th>ROC AUC</th><th>Acurácia</th><th>Brier</th>
        </tr></thead>
        <tbody>
          {experiments.map((experiment, index) => (
            <tr className={experiment.selected ? 'is-selected' : ''} key={`${experiment.model}-${index}`} data-testid={`row-experiment-${index}`}>
              <td className="exp-name">{experiment.model}{experiment.selected ? <span className="matrix-note"> &nbsp;SELECIONADO</span> : null}</td>
              <td>{formatMetric(experiment.precision)}</td><td>{formatMetric(experiment.recall)}</td><td>{formatMetric(experiment.f1)}</td>
              <td>{formatMetric(experiment.roc_auc)}</td><td>{formatMetric(experiment.accuracy)}</td><td>{formatMetric(experiment.brier_score)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function CreditDashboard() {
  const summaryQuery = useGetCreditSummary({ query: { queryKey: getGetCreditSummaryQueryKey() } });
  const experimentQuery = useGetCreditExperiments({ query: { queryKey: getGetCreditExperimentsQueryKey() } });
  const predictMutation = usePredictCreditDefault();
  const [formError, setFormError] = useState('');

  async function refreshDashboard() {
    await Promise.all([summaryQuery.refetch(), experimentQuery.refetch()]);
  }

  function handlePrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError('');
    const form = new FormData(event.currentTarget);
    const input: Record<string, number> = {};
    for (const field of fields) {
      const value = form.get(field.key);
      if (value === null || value === '') {
        setFormError(`Informe um valor para ${field.key}.`);
        return;
      }
      input[field.key] = Number(value);
    }
    if (input.SEX !== 1 && input.SEX !== 2) {
      setFormError('Escolha um código de sexo válido.');
      return;
    }
    predictMutation.mutate({
      data: input as unknown as CreditPredictionInput,
    });
  }

  const summary = summaryQuery.data;
  const experiments = experimentQuery.data;
  const selected = experiments?.find((experiment) => experiment.selected)
    || summary?.experiments?.find((experiment) => experiment.selected);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="side-brand" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="brand-mark"><Activity size={16} strokeWidth={1.8} /></div>
          <div className="brand-lockup"><div className="brand-name">Credit Intelligence</div><div className="brand-sub">Laboratório de modelos / 01</div></div>
        </div>
        <div className="nav-label">Área de trabalho</div>
        <nav aria-label="Navegação principal">
          <div className="nav-item active" aria-current="page" data-testid="nav-dashboard"><Gauge size={15} /><span>Painel de experimentos</span></div>
          <a className="nav-item" href="#dataset" data-testid="link-dataset"><Database size={15} /><span>Registro dos dados</span></a>
          <a className="nav-item" href="#predict" data-testid="link-predict"><Sparkles size={15} /><span>Laboratório de previsões</span></a>
        </nav>
        <div className="side-note"><strong>Contexto de pesquisa</strong>Projeto educacional de inadimplência em cartões de crédito da UCI. Os resultados servem apenas ao estudo do modelo e nunca devem orientar decisões de concessão de crédito.</div>
      </aside>

      <main className="main-area">
        <div className="topbar">
          <div className="topbar-label">UCI / inadimplência · registro de experimentos</div>
          <div className={`api-status ${summaryQuery.isError || experimentQuery.isError ? 'is-error' : summaryQuery.isLoading || experimentQuery.isLoading ? 'is-loading' : 'is-ready'}`} role="status" data-testid="api-status"><span className="live-dot" />{summaryQuery.isError || experimentQuery.isError ? 'API com erro' : summaryQuery.isLoading || experimentQuery.isLoading ? 'Verificando API' : 'Dados recebidos da API'}</div>
        </div>
        <div className="page-content">
          <div className="head-row">
            <div>
              <div className="eyebrow">Avaliação transparente · 01</div>
              <h1 className="page-title">Credit<br />Intelligence</h1>
              <p className="intro">Um caderno de laboratório para um modelo educacional de previsão de inadimplência. <strong>Proveniência, compromissos de avaliação e incerteza sempre à vista.</strong></p>
            </div>
            <button type="button" className="refresh-button" onClick={refreshDashboard} disabled={summaryQuery.isFetching || experimentQuery.isFetching} data-testid="button-refresh">
              <RefreshCw size={13} className={summaryQuery.isFetching || experimentQuery.isFetching ? 'animate-spin' : ''} />
              Atualizar dados
            </button>
          </div>

          <section id="dataset" aria-labelledby="dataset-heading">
            <h2 id="dataset-heading" className="section-kicker">01 / Linhagem dos dados</h2>
            {summaryQuery.isLoading ? <LoadingPanel rows={2} /> : summaryQuery.isError ? (
              <ErrorPanel message={getErrorMessage(summaryQuery.error)} onRetry={() => summaryQuery.refetch()} testId="error-summary" />
            ) : summary ? (
              <>
                <DatasetPanel dataset={summary.dataset} trainedAt={summary.trained_at} />
                {summary.status === 'not_trained' ? (
                  <div className="error-box" data-testid="status-not-trained" role="status">
                    <span><strong>Modelo não treinado.</strong> {summary.message || 'A API informa que não há um modelo treinado no momento.'}</span>
                  </div>
                ) : <div className="model-status" data-testid="status-model-ready">Modelo pronto · {summary.selected_model || 'modelo selecionado sem nome'}</div>}
                {summary.status === 'ready' && summary.message ? <p data-testid="text-summary-message" style={{ color: '#79817d', fontSize: 10, margin: '-22px 0 30px' }}>{summary.message}</p> : null}
              </>
            ) : null}
          </section>

          <section aria-labelledby="evaluation-heading">
            <h2 id="evaluation-heading" className="section-kicker">02 / Avaliação em contexto</h2>
            <div className="metrics-band">
              {experimentQuery.isLoading ? <LoadingPanel rows={5} /> : experimentQuery.isError ? (
                <ErrorPanel message={getErrorMessage(experimentQuery.error)} onRetry={() => experimentQuery.refetch()} testId="error-experiments" />
              ) : (
                <SelectedModelCard experiment={selected} />
              )}
              <div className="panel">
                <div className="panel-head">
                  <div><div className="panel-meta">Comparação de modelos</div><h3 className="panel-title">Registro de experimentos</h3></div>
                  <span className="panel-meta">{experiments?.length ?? '—'} execuções</span>
                </div>
                {experimentQuery.isLoading ? <LoadingPanel rows={4} /> : experimentQuery.isError ? (
                  <div style={{ padding: 17, color: '#945442', fontSize: 10 }}>Registro de experimentos indisponível.</div>
                ) : (
                  <div className="model-list">
                    {(experiments || []).slice(0, 4).map((experiment, index) => (
                      <div className="model-row" key={`${experiment.model}-${index}`} data-testid={`ledger-row-${index}`}>
                        <span className="model-name">{experiment.model}</span>
                        <span><span className="metric-label">F1</span><span className="metric-number">{formatMetric(experiment.f1)}</span></span>
                        <span><span className="metric-label">AUC</span><span className="metric-number">{formatMetric(experiment.roc_auc)}</span></span>
                        <span><span className="metric-label">Brier</span><span className="metric-number">{formatMetric(experiment.brier_score)}</span></span>
                        {experiment.selected ? <span className="selected-pill">●</span> : <span />}
                      </div>
                    ))}
                    {!experiments?.length ? <div style={{ padding: '20px 0', color: '#79817d', fontSize: 10 }} data-testid="ledger-empty">A API não retornou dados para comparação.</div> : null}
                  </div>
                )}
                <div style={{ padding: '0 19px 17px', color: '#8a928d', fontSize: 9, lineHeight: 1.5 }}>
                  A AUC resume a capacidade de ordenação; a pontuação de Brier reflete a calibração das probabilidades. Nenhuma métrica, isoladamente, define um ponto de operação adequado.
                </div>
              </div>
            </div>
          </section>

          <section className="experiment-section" aria-labelledby="experiments-heading">
            <h2 id="experiments-heading" className="section-kicker">03 / Comparação completa</h2>
            {experimentQuery.isLoading ? <LoadingPanel rows={5} /> : experimentQuery.isError ? (
              <ErrorPanel message={getErrorMessage(experimentQuery.error)} onRetry={() => experimentQuery.refetch()} testId="error-experiment-table" />
            ) : <ExperimentTable experiments={experiments || []} />}
          </section>

          <section id="predict" className="prediction-section" aria-labelledby="prediction-heading">
            <div className="prediction-heading">
              <div>
                <div className="eyebrow" style={{ color: '#e2a185' }}>04 / Sonda interativa do modelo</div>
                <h2 id="prediction-heading" className="prediction-title">Faça uma previsão educacional</h2>
                <p className="prediction-copy">Preencha os 23 campos do esquema UCI para consultar o modelo disponível. Nenhum valor vem pré-preenchido; use uma linha de exemplo documentada no conjunto de dados original.</p>
              </div>
              <div className="edu-tag"><ShieldAlert size={12} style={{ verticalAlign: 'middle', marginRight: 6 }} />Não usar para crédito</div>
            </div>
            {summary?.status === 'not_trained' ? (
              <div className="notice-box" data-testid="prediction-disabled">A previsão está indisponível porque a API informa que não há um modelo treinado.</div>
            ) : (
              <form className="form-grid" onSubmit={handlePrediction} data-testid="form-prediction">
                {fields.map((field, index) => {
                  const previous = fields[index - 1];
                  const showGroup = field.group && field.group !== previous?.group;
                  return (
                    <div key={field.key} style={{ display: 'contents' }}>
                      {showGroup ? <div className="group-divider">{groups[field.group as NonNullable<FieldSpec['group']>]}</div> : null}
                      <div className="form-group" data-testid={`field-${field.key.toLowerCase()}`}>
                        <label htmlFor={`input-${field.key}`}>{field.label}</label>
                        {field.key === 'SEX' ? (
                          <select id={`input-${field.key}`} name={field.key} defaultValue="" required data-testid={`input-${field.key.toLowerCase()}`}>
                            <option value="" disabled>Selecione uma opção</option>
                            <option value="1">1 · Masculino</option>
                            <option value="2">2 · Feminino</option>
                          </select>
                        ) : field.key === 'EDUCATION' ? (
                          <select id={`input-${field.key}`} name={field.key} defaultValue="" required data-testid={`input-${field.key.toLowerCase()}`}>
                            <option value="" disabled>Selecione uma opção</option>
                            <option value="0">0 · Não informado</option>
                            <option value="1">1 · Pós-graduação</option>
                            <option value="2">2 · Universidade</option>
                            <option value="3">3 · Ensino médio</option>
                            <option value="4">4 · Outros</option>
                            <option value="5">5 · Desconhecido</option>
                            <option value="6">6 · Desconhecido</option>
                          </select>
                        ) : field.key === 'MARRIAGE' ? (
                          <select id={`input-${field.key}`} name={field.key} defaultValue="" required data-testid={`input-${field.key.toLowerCase()}`}>
                            <option value="" disabled>Selecione uma opção</option>
                            <option value="0">0 · Não informado</option>
                            <option value="1">1 · Casado(a)</option>
                            <option value="2">2 · Solteiro(a)</option>
                            <option value="3">3 · Outros</option>
                          </select>
                        ) : (
                          <input id={`input-${field.key}`} name={field.key} type="number" step="1" min={field.min} max={field.max} required placeholder="—" data-testid={`input-${field.key.toLowerCase()}`} />
                        )}
                      </div>
                    </div>
                  );
                })}
                <div className="form-actions">
                  <button className="predict-button" type="submit" disabled={predictMutation.isPending || summaryQuery.data?.status === 'not_trained'} data-testid="button-submit-prediction">
                    {predictMutation.isPending ? <RefreshCw size={13} className="animate-spin" /> : <ArrowUpRight size={14} />}
                    {predictMutation.isPending ? 'Consultando o modelo…' : 'Solicitar previsão'}
                  </button>
                  <div className="form-footnote"><CircleHelp size={12} style={{ verticalAlign: 'middle', marginRight: 5 }} />Códigos e unidades seguem o conjunto de dados de cartões de crédito de Taiwan da UCI. Envie uma linha conhecida para uma demonstração contextualizada.</div>
                </div>
                {formError ? <div className="notice-box" role="alert" data-testid="error-form">{formError}</div> : null}
                {predictMutation.isError ? <div className="notice-box" role="alert" data-testid="error-prediction">{getErrorMessage(predictMutation.error)}</div> : null}
                {predictMutation.data ? (
                  <>
                    <div className="result-panel" data-testid="prediction-result">
                      <div><div className="result-label">Probabilidade de inadimplência</div><div className="result-number">{(predictMutation.data.default_probability * 100).toFixed(2)}%</div></div>
                      <div className="result-detail">Modelo · <strong>{predictMutation.data.model}</strong><br />Resultado retornado diretamente pela API ativa.</div>
                    </div>
                    <div className="notice-box" data-testid="prediction-notice">{predictMutation.data.educational_notice}</div>
                  </>
                ) : null}
              </form>
            )}
          </section>
          <div className="footer-note">Resultado exclusivamente educacional e de pesquisa · não é uma decisão de crédito</div>
        </div>
      </main>
    </div>
  );
}
