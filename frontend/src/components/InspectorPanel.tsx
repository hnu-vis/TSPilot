import { Activity, AlertCircle, ArrowRight, CheckCircle2, ChevronDown, Clipboard, Code2, FileText, LineChart, ListChecks, PanelRightClose, PanelRightOpen, Table2 } from 'lucide-react';
import { MarkdownContent } from './FinalAnswer';
import { toDisplayStep } from '../lib/traceDisplay';
import type { FinalAnswer, TraceStep } from '../types';
import { useI18n } from '../i18n';

type Props = {
  steps: TraceStep[];
  selectedStepId: string | null;
  answer?: FinalAnswer | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
};

export function InspectorPanel({ steps, selectedStepId, answer, collapsed, onToggleCollapsed }: Props) {
  const { t } = useI18n();
  if (steps.length === 0 && !answer) return null;

  const selectedStep = selectedStepId
    ? steps.find((step) => step.id === selectedStepId) || null
    : null;
  const activeStep = selectedStep || steps[steps.length - 1] || null;
  const displayStep = activeStep ? toDisplayStep(activeStep) : null;
  const activeStepIndex = activeStep ? steps.findIndex((step) => step.id === activeStep.id) : -1;

  if (collapsed) {
    return (
      <aside className="inspector-panel collapsed" aria-label={t('Run details collapsed')}>
        <button type="button" className="inspector-rail-button" onClick={onToggleCollapsed} aria-label={t('Show run details')}>
          <PanelRightOpen size={18} />
          <span>{t('Detail')}</span>
        </button>
      </aside>
    );
  }

  return (
    <aside className="inspector-panel" aria-label={t('Run details')}>
      <header className="inspector-header">
        <div className="inspector-heading">
          <div className="inspector-kicker">
            <span>{t('Inspector')}</span>
            {displayStep && <i aria-hidden="true" />}
            {displayStep && <span>{t('Step')} {activeStepIndex + 1} / {steps.length}</span>}
          </div>
          <div className="inspector-title-row">
            <span className="inspector-title-icon" aria-hidden="true">
              <Activity size={16} />
            </span>
            <div>
              <h2>{t(displayStep?.title || 'Answer')}</h2>
              <p>{t(displayStep?.category || 'Run detail')}</p>
            </div>
          </div>
        </div>
        <div className="inspector-header-actions">
          {displayStep && <span className={`status-line compact ${displayStep.status}`}>{t(statusLabel(displayStep.status))}</span>}
          <button type="button" onClick={onToggleCollapsed} aria-label={t('Collapse run details')}>
            <PanelRightClose size={18} />
          </button>
        </div>
      </header>

      <div className="inspector-body">
        {displayStep && <StepDetail step={displayStep} />}
        {!displayStep && answer?.references && answer.references.length > 0 && (
          <ReferenceList answer={answer} />
        )}
      </div>
    </aside>
  );
}

function ReferenceList({ answer }: { answer: FinalAnswer }) {
  const { t } = useI18n();
  return (
    <section className="inspector-card">
      <div className="inspector-card-title">
        <FileText size={16} />
        <h3>{t('Answer references')}</h3>
      </div>
      <div className="reference-list">
        {answer.references?.map((reference, index) => (
          <div key={`${reference.source_type}-${reference.source_id || index}`} className="reference-row">
            <span>{formatLabel(reference.source_type)}</span>
            <strong>{reference.label}</strong>
            {reference.source_id && <code>{reference.source_id}</code>}
          </div>
        ))}
      </div>
    </section>
  );
}

function StepDetail({ step }: { step: ReturnType<typeof toDisplayStep> }) {
  return (
    <>
      <ReactStepCard step={step} />

      {step.sqlDetail?.query && <QueryPreview detail={step.sqlDetail} />}

      {step.sqlDetail && hasQueryData(step.sqlDetail) && <DataPreview detail={step.sqlDetail} />}

      {step.codeInterpreterDetail && <CodeInterpreterPreview detail={step.codeInterpreterDetail} />}

      {step.forecastDetail && <ForecastPreview detail={step.forecastDetail} />}

      {step.anomalyDetail && <AnomalyPreview detail={step.anomalyDetail} />}

      {step.insightDetail && <InsightPreview detail={step.insightDetail} />}
    </>
  );
}

function ReactStepCard({ step }: { step: ReturnType<typeof toDisplayStep> }) {
  const { t } = useI18n();
  const detail = step.reactDetail;
  const actionInput = compactReactInput(detail.actionInput);
  const observation = detail.observation;
  return (
    <section className="inspector-card react-step-card">
      <div className="inspector-card-title">
        <Activity size={16} />
        <h3>{t('Execution context')}</h3>
      </div>
      {detail.thought && (
        <div className="react-thought">
          <span>{t('Thought')}</span>
          <p>{detail.thought}</p>
        </div>
      )}
      <div className="react-transition" aria-label={t('ReAct action and observation')}>
        <div>
          <span>{t('Action')}</span>
          <strong>{detail.action || step.category}</strong>
        </div>
        <ArrowRight size={15} />
        <div>
          <span>{t('Observation')}</span>
          <strong>{step.summary}</strong>
        </div>
      </div>
      {actionInput && (
        <details className="react-input-details">
          <summary><ChevronDown size={14} className="collapsible-chevron" /> {t('Action input')}</summary>
          <pre>{JSON.stringify(actionInput, null, 2)}</pre>
        </details>
      )}
      {observation && Object.keys(observation).length > 0 && (
        <details className="react-input-details react-observation-details">
          <summary><ChevronDown size={14} className="collapsible-chevron" /> {t('Observation details')}</summary>
          <pre>{JSON.stringify(observation, null, 2)}</pre>
        </details>
      )}
    </section>
  );
}

function QueryPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['sqlDetail']> }) {
  const { t } = useI18n();
  return (
    <details className="inspector-card sql-query-preview collapsible-card" open>
      <summary className="collapsible-summary">
        <span>
          <ChevronDown size={15} className="collapsible-chevron" />
          <Code2 size={16} />
          <strong>{t('Query')}</strong>
        </span>
        <span className="query-summary-actions">
          {detail.queryLanguage && <span className="query-language-badge">{detail.queryLanguage}</span>}
          <button
            type="button"
            className="inspector-icon-button"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              void navigator.clipboard?.writeText(detail.query || '');
            }}
            aria-label={t('Copy generated query')}
            title={t('Copy query')}
          >
            <Clipboard size={13} />
          </button>
        </span>
      </summary>
      <pre className="inspector-source-code"><code>{detail.query}</code></pre>
    </details>
  );
}

function DataPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['sqlDetail']> }) {
  const { t } = useI18n();
  const rows = detail.sampleRows.length > 0 ? detail.sampleRows : detail.samplePoints;
  const tableColumns = detail.columns.length > 0 ? detail.columns : inferColumns(rows);
  return (
    <section className="inspector-card data-preview">
      <div className="inspector-card-title sql-detail-title">
        <Table2 size={16} />
        <h3>{t('Query data')}</h3>
        <span className="query-language-badge">{formatCounts(detail)}</span>
      </div>

      {detail.columns.length > 0 && (
        <div className="column-chip-list" aria-label={t('Result columns')}>
          {detail.columns.map((column) => (
            <span key={column}>{column}</span>
          ))}
        </div>
      )}

      {rows.length > 0 && tableColumns.length > 0 && (
        <div className="sample-table-section">
          <div className="sample-table-wrap">
            <table className="sample-table">
              <thead>
                <tr>
                  {tableColumns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={index}>
                    {tableColumns.map((column) => (
                      <td key={column}>{formatCell(row[column])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {detail.truncated && <p className="sample-note">{t('Preview only')}</p>}
        </div>
      )}

      {rows.length === 0 && (
        <div className="empty-data-preview">
          <Table2 size={16} />
          <span>{t('No sample rows are visible for this result.')}</span>
        </div>
      )}
    </section>
  );
}

function CodeInterpreterPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['codeInterpreterDetail']> }) {
  const { t } = useI18n();
  return (
    <section className="inspector-card tool-detail-card code-detail-card">
      {detail.analysisGoal && (
        <div className="code-analysis-goal">
          <span>{t('Analysis goal')}</span>
          <p>{detail.analysisGoal}</p>
        </div>
      )}

      <div className="query-run-insights" aria-label={t('Code interpreter insights')}>
        {detail.inputRowCount !== null && (
          <div>
            <Table2 size={14} />
            <span>{detail.inputRowCount.toLocaleString()} rows</span>
          </div>
        )}
        {detail.runtimeMs !== null && (
          <div>
            <Activity size={14} />
            <span>{Math.round(detail.runtimeMs).toLocaleString()} ms</span>
          </div>
        )}
      </div>

      {detail.code && (
        <ToolCodeBlock
          title={t('Source')}
          language="python"
          code={detail.code}
          copyLabel="Copy code"
        />
      )}

      {hasStructuredResult(detail) && (
        <StructuredResult
          title={t('Result')}
          summary={detail.summary}
          metrics={{}}
          details={{ computed_insights: detail.computedInsights, derived_evidence: detail.derivedEvidence }}
          fallback={null}
        />
      )}
    </section>
  );
}

function ForecastPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['forecastDetail']> }) {
  const { t } = useI18n();
  const rows = detail.points;
  return (
    <section className="inspector-card tool-detail-card">
      <div className="inspector-card-title sql-detail-title">
        <LineChart size={16} />
        <h3>{t('Forecast result')}</h3>
        {detail.status && <span className="query-language-badge">{detail.status}</span>}
      </div>

      <div className="query-run-insights" aria-label={t('Forecast insights')}>
        {detail.horizon !== null && (
          <div>
            <LineChart size={14} />
            <span>{detail.horizon.toLocaleString()} horizon</span>
          </div>
        )}
        {detail.pointCount !== null && (
          <div>
            <Table2 size={14} />
            <span>{detail.pointCount.toLocaleString()} points</span>
          </div>
        )}
        {detail.modelName && (
          <div>
            <Code2 size={14} />
            <span>{detail.modelName}</span>
          </div>
        )}
      </div>

      {detail.plan && <KeyValuePreview title={t('Forecast plan')} data={detail.plan} />}

      {rows.length > 0 ? (
        <SimpleRecordsTable title={t('Forecast points')} records={rows} />
      ) : (
        <p className="sample-note">{t('No direct forecast points were returned for this step.')}</p>
      )}
    </section>
  );
}

function AnomalyPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['anomalyDetail']> }) {
  const { t } = useI18n();
  return (
    <section className="inspector-card tool-detail-card">
      <div className="inspector-card-title sql-detail-title">
        <AlertCircle size={16} />
        <h3>{t('Anomaly result')}</h3>
        {detail.detectorName && <span className="query-language-badge">{detail.detectorName}</span>}
      </div>

      <div className="query-run-insights" aria-label={t('Anomaly insights')}>
        {detail.pointCount !== null && (
          <div>
            <AlertCircle size={14} />
            <span>{detail.pointCount.toLocaleString()} anomalies</span>
          </div>
        )}
        {detail.spanCount !== null && (
          <div>
            <Activity size={14} />
            <span>{detail.spanCount.toLocaleString()} spans</span>
          </div>
        )}
      </div>

      {detail.points.length > 0 ? (
        <SimpleRecordsTable title={t('Anomaly points')} records={detail.points} />
      ) : (
        <p className="sample-note">{t('No anomaly points are visible for this step.')}</p>
      )}
      {detail.scores.length > 0 && <SimpleRecordsTable title={t('Scores')} records={detail.scores} />}
    </section>
  );
}

function InsightPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['insightDetail']> }) {
  const { t } = useI18n();
  const coverageItems = insightCoverageItems(detail.coverage);
  return (
    <section className="inspector-card key-insight-preview">
      <div className="inspector-card-title sql-detail-title">
        <CheckCircle2 size={16} />
        <h3>{t('Key Insight selection')}</h3>
        {detail.produced.length > 0 && <span className="query-language-badge">{detail.produced.length} {t('key insights')}</span>}
      </div>

      {detail.requested.length > 0 && (
        <div className="tool-result-section">
          <div className="sample-table-caption">
            <span>
              <ListChecks size={14} />
              {t('Requested key insights')}
            </span>
          </div>
          <div className="chip-list compact">
            {detail.requested.slice(0, 16).map((request) => (
              <span key={request.insight_key || `${request.name}-${request.insight_type}`}>
                {request.name} · {request.insight_type}
                {request.derived_from?.length ? ` ← ${request.derived_from.join(', ')}` : ''}
              </span>
            ))}
          </div>
        </div>
      )}

      {coverageItems.length > 0 && (
        <div className="insight-coverage-grid" aria-label={t('Key Insight coverage')}>
          {coverageItems.map((item) => (
            <div key={item.label} className={`insight-coverage-tile ${item.status}`}>
              <span>{item.label}</span>
              <strong>{item.values.length}</strong>
              {item.values.length > 0 && <small>{item.values.slice(0, 4).join(', ')}</small>}
            </div>
          ))}
        </div>
      )}

      {detail.produced.length > 0 ? (
        <div className="key-insight-list">
          {detail.produced.map((insight) => (
            <details key={insight.insight_id} className={`key-insight-card ${insight.status}`} open={insight.status !== 'verified'}>
              <summary>
                <span>
                  <StatusIcon status={insight.status === 'verified' ? 'complete' : insight.status === 'unavailable' ? 'attention' : 'error'} />
                  <strong>{insight.name}</strong>
                  <code>{insight.insight_key || insight.insight_type}</code>
                </span>
                <span className="key-insight-summary-value">
                  {insight.value !== undefined && <strong>{formatCell(insight.value)}</strong>}
                  <span className={`status-line compact ${insightStatusClass(insight.status)}`}>{formatLabel(insight.status)}</span>
                </span>
              </summary>
              <p className="step-summary">{insight.statement}</p>
              <dl className="answer-metric-grid">
                <div>
                  <dt>{t('Method')}</dt>
                  <dd>{insight.method}</dd>
                </div>
                {insight.insight_key && (
                  <div>
                    <dt>{t('Key Insight key')}</dt>
                    <dd>{insight.insight_key}</dd>
                  </div>
                )}
                {insight.value !== undefined && (
                  <div>
                    <dt>{t('Value')}</dt>
                    <dd>{formatCell(insight.value)}</dd>
                  </div>
                )}
                {insight.unavailable_reason && (
                  <div>
                    <dt>{t('Reason')}</dt>
                    <dd>{insight.unavailable_reason}</dd>
                  </div>
                )}
                {insight.derived_from && insight.derived_from.length > 0 && (
                  <div>
                    <dt>{t('Derived from')}</dt>
                    <dd>{insight.derived_from.join(', ')}</dd>
                  </div>
                )}
              </dl>
              {insight.evidence_refs && insight.evidence_refs.length > 0 && (
                <div className="chip-list compact">
                  {insight.evidence_refs.map((reference) => (
                    <span key={`${reference.source_type}-${reference.source_id}`}>
                      {reference.source_type}:{reference.source_id}
                    </span>
                  ))}
                </div>
              )}
              {insight.calculation_trace && Object.keys(insight.calculation_trace).length > 0 && (
                <div className="tool-result-section">
                  <div className="sample-table-caption">
                    <span>
                      <Code2 size={14} />
                      {t('Calculation trace')}
                    </span>
                  </div>
                  <pre className="debug-json">{JSON.stringify(insight.calculation_trace, null, 2)}</pre>
                </div>
              )}
            </details>
          ))}
        </div>
      ) : (
        <p className="sample-note">{t('No structured key insights were produced for this step.')}</p>
      )}
    </section>
  );
}

function StructuredResult({
  title,
  summary,
  metrics,
  details,
  fallback,
}: {
  title: string;
  summary: string | null;
  metrics: Record<string, unknown>;
  details: Record<string, unknown>;
  fallback: Record<string, unknown> | null;
}) {
  const { t } = useI18n();
  const visibleMetrics = deduplicateStructuredEntries(metrics);
  const visibleDetails = deduplicateStructuredEntries(details, visibleMetrics.map(([, value]) => value));
  const hasMetrics = visibleMetrics.length > 0;
  const hasDetails = visibleDetails.length > 0;
  return (
    <div className="tool-result-section">
      <div className="sample-table-caption">
        <span>
          <CheckCircle2 size={14} />
          {title}
        </span>
      </div>
      {summary && <MarkdownContent content={summary} />}
      {hasMetrics && <InspectorMetricGroup entries={visibleMetrics} />}
      {hasDetails && <KeyValuePreview title={t('Details')} data={Object.fromEntries(visibleDetails)} />}
      {!summary && !hasMetrics && !hasDetails && fallback && <pre className="debug-json">{JSON.stringify(fallback, null, 2)}</pre>}
    </div>
  );
}

function InspectorMetricGroup({ entries }: { entries: Array<[string, unknown]> }) {
  return (
    <dl className="answer-metric-grid">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{formatLabel(key)}</dt>
          <dd>{formatCell(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function KeyValuePreview({ title, data }: { title: string; data: Record<string, unknown> }) {
  return (
    <div className="tool-result-section">
      <div className="sample-table-caption">
        <span>
          <FileText size={14} />
          {title}
        </span>
      </div>
      <dl className="answer-metric-grid">
        {Object.entries(data).slice(0, 24).map(([key, value]) => (
          <div key={key}>
            <dt>{formatLabel(key)}</dt>
            <dd>{formatCell(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function SimpleRecordsTable({ title, records }: { title: string; records: Record<string, unknown>[] }) {
  const columns = inferColumns(records).slice(0, 8);
  return (
    <div className="sample-table-section">
      <div className="sample-table-caption">
        <span>
          <Table2 size={14} />
          {title}
        </span>
        <strong>{records.length.toLocaleString()} visible</strong>
      </div>
      {columns.length > 0 && (
        <div className="sample-table-wrap">
          <table className="sample-table">
            <thead>
              <tr>
                {columns.map((column) => <th key={column}>{column}</th>)}
              </tr>
            </thead>
            <tbody>
              {records.map((row, index) => (
                <tr key={index}>
                  {columns.map((column) => <td key={column}>{formatCell(row[column])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ToolCodeBlock({
  title,
  language,
  code,
  copyLabel,
}: {
  title: string;
  language: string;
  code: string;
  copyLabel: string;
}) {
  return (
    <details className="answer-code-details tool-code-details" open>
      <summary>
        <span>
          <ChevronDown size={14} className="collapsible-chevron" />
          <Code2 size={14} />
          <strong className="tool-code-title">{title}</strong>
        </span>
        <span className="tool-code-actions">
          <span className="query-language-badge">{language}</span>
          <button
            type="button"
            className="inspector-icon-button inline-copy-button"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              void navigator.clipboard?.writeText(code);
            }}
            aria-label={copyLabel}
            title={copyLabel}
          >
            <Clipboard size={13} />
          </button>
        </span>
      </summary>
      <pre className="inspector-source-code"><code>{code}</code></pre>
    </details>
  );
}

function StatusIcon({ status }: { status: ReturnType<typeof toDisplayStep>['status'] }) {
  if (status === 'error') return <AlertCircle size={16} />;
  if (status === 'attention') return <AlertCircle size={16} />;
  return <CheckCircle2 size={16} />;
}

function statusLabel(status: string) {
  if (status === 'running') return 'Running';
  if (status === 'complete') return 'Complete';
  if (status === 'attention') return 'Needs more evidence';
  if (status === 'error') return 'Error';
  return 'Not started';
}

function formatLabel(value: string) {
  if (value === 'query') return 'Database evidence';
  if (value === 'sql_query') return 'Database evidence';
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function inferColumns(rows: Record<string, unknown>[]) {
  const columns = new Set<string>();
  rows.slice(0, 5).forEach((row) => {
    Object.keys(row).forEach((key) => columns.add(key));
  });
  return Array.from(columns).slice(0, 12);
}

function formatCounts(detail: NonNullable<ReturnType<typeof toDisplayStep>['sqlDetail']>) {
  const parts = [
    detail.rowCount !== null ? `${detail.rowCount.toLocaleString()} rows` : null,
    shouldShowPointCount(detail) ? `${detail.pointCount?.toLocaleString()} points` : null,
  ].filter(Boolean);
  return parts.length ? parts.join(' / ') : `${detail.sampleRows.length || detail.samplePoints.length} samples`;
}

function shouldShowPointCount(detail: NonNullable<ReturnType<typeof toDisplayStep>['sqlDetail']>) {
  if (detail.pointCount === null) return false;
  if (detail.pointCount === 0 && (detail.rowCount || detail.sampleRows.length > 0)) return false;
  return true;
}

function hasQueryData(detail: NonNullable<ReturnType<typeof toDisplayStep>['sqlDetail']>) {
  return detail.columns.length > 0
    || detail.sampleRows.length > 0
    || detail.samplePoints.length > 0
    || detail.rowCount !== null
    || detail.pointCount !== null;
}

function hasStructuredResult(detail: NonNullable<ReturnType<typeof toDisplayStep>['codeInterpreterDetail']>) {
  return Boolean(
    detail.summary
    || detail.computedInsights.length > 0
    || detail.derivedEvidence.length > 0,
  );
}

function compactReactInput(value: Record<string, unknown> | null) {
  if (!value) return null;
  const omitted = new Set(['code', 'analysis_code', 'database_evidence', 'history', 'insight_requests']);
  const entries = Object.entries(value).filter(([key, entryValue]) => (
    !omitted.has(key)
    && entryValue !== null
    && entryValue !== undefined
    && entryValue !== ''
    && !(Array.isArray(entryValue) && entryValue.length === 0)
    && !(typeof entryValue === 'object' && !Array.isArray(entryValue) && Object.keys(entryValue as object).length === 0)
  ));
  return entries.length > 0 ? Object.fromEntries(entries) : null;
}

function insightCoverageItems(coverage: NonNullable<ReturnType<typeof toDisplayStep>['insightDetail']>['coverage']) {
  if (!coverage) return [];
  return [
    { label: 'Verified', status: 'complete', values: coverage.verified || [] },
    { label: 'Missing', status: 'attention', values: coverage.missing || [] },
    { label: 'Unavailable', status: 'attention', values: coverage.unavailable || [] },
    { label: 'Rejected', status: 'error', values: coverage.rejected || [] },
    { label: 'Partial', status: 'running', values: coverage.partial || [] },
  ].filter((item) => item.values.length > 0 || item.label === 'Verified');
}

function insightStatusClass(status: string) {
  if (status === 'verified') return 'complete';
  if (status === 'unavailable' || status === 'partial') return 'attention';
  if (status === 'rejected') return 'error';
  return 'running';
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? value.toLocaleString() : String(value);
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function deduplicateStructuredEntries(
  values: Record<string, unknown>,
  existingValues: unknown[] = [],
): Array<[string, unknown]> {
  const seen = new Set(existingValues.map(structuredValueKey).filter((value): value is string => Boolean(value)));
  return Object.entries(values).filter(([, value]) => {
    const key = structuredValueKey(value);
    if (!key) return true;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function structuredValueKey(value: unknown) {
  if (!value || typeof value !== 'object') return null;
  try {
    return JSON.stringify(value);
  } catch {
    return null;
  }
}
