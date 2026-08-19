import { Activity, AlertCircle, ArrowRight, Check, CheckCircle2, ChevronDown, Clipboard, Code2, FileText, LineChart, ListChecks, Loader2, PanelRightClose, PanelRightOpen, Table2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { MarkdownContent } from './FinalAnswer';
import { toDisplayStep } from '../lib/traceDisplay';
import { elapsedSecondsForTrace } from '../lib/traceTiming';
import type { CalculationTrace, FinalAnswer, TraceSpan, TraceStep } from '../types';
import { useI18n } from '../i18n';

type Props = {
  steps: TraceStep[];
  selectedNodeId: string | null;
  answer?: FinalAnswer | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
};

type InspectorSelection =
  | { kind: 'step'; step: TraceStep; stepIndex: number }
  | { kind: 'llm'; call: TraceSpan; parent: TraceStep; stepIndex: number; callIndex: number };

export function InspectorPanel({ steps, selectedNodeId, answer, collapsed, onToggleCollapsed }: Props) {
  const { t } = useI18n();
  if (steps.length === 0 && !answer) return null;

  const selection = resolveInspectorSelection(steps, selectedNodeId);
  const displayStep = selection?.kind === 'step' ? toDisplayStep(selection.step) : null;
  const nodeTitle = selection?.kind === 'llm' ? selection.call.title : displayStep?.title || 'Answer';
  const nodeCategory = selection?.kind === 'llm' ? 'LLM call' : displayStep?.category || 'Run detail';
  const nodeStatus = selection?.kind === 'llm' ? selection.call.status : displayStep?.status;

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
            {selection && <i aria-hidden="true" />}
            {selection?.kind === 'llm' ? (
              <span>{t('Call')} {selection.callIndex + 1} / {(selection.parent.children || []).length}</span>
            ) : selection ? (
              <span>{t('Step')} {selection.stepIndex + 1} / {steps.length}</span>
            ) : null}
          </div>
          <div className="inspector-title-row">
            {selection?.kind !== 'llm' && (
              <span className="inspector-title-icon" aria-hidden="true"><Activity size={16} /></span>
            )}
            <div>
              <h2>{t(nodeTitle)}</h2>
              <p>{t(nodeCategory)}</p>
            </div>
          </div>
        </div>
        <div className="inspector-header-actions">
          {nodeStatus && <span className={`status-line compact ${nodeStatus}`}>{t(statusLabel(nodeStatus))}</span>}
          <button type="button" onClick={onToggleCollapsed} aria-label={t('Collapse run details')}>
            <PanelRightClose size={18} />
          </button>
        </div>
      </header>

      <div className="inspector-body">
        {selection?.kind === 'step' && displayStep && <StepDetail step={displayStep} />}
        {selection?.kind === 'llm' && <LLMCallDetail selection={selection} />}
        {!selection && answer?.references && answer.references.length > 0 && (
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

      {step.codeInterpreterDetail && (
        <CodeInterpreterPreview detail={step.codeInterpreterDetail} insightDetail={step.insightDetail} />
      )}

      {step.forecastDetail && <ForecastPreview detail={step.forecastDetail} />}

      {step.anomalyDetail && <AnomalyPreview detail={step.anomalyDetail} />}

      {step.insightDetail && !step.codeInterpreterDetail && <InsightPreview detail={step.insightDetail} />}
    </>
  );
}

function ReactStepCard({ step }: { step: ReturnType<typeof toDisplayStep> }) {
  const { t } = useI18n();
  const detail = step.reactDetail;
  const actionInput = compactReactInput(detail.actionInput);
  const observation = detail.observation;
  const isDecisionPending = step.status === 'running' && !detail.action;
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
      {isDecisionPending ? (
        <div className="react-decision-pending" role="status" aria-live="polite">
          <Loader2 className="spin" size={15} aria-hidden="true" />
          <div>
            <strong>{t('ReAct is deciding the next action')}</strong>
            <span>{t('The result will update this step in place.')}</span>
          </div>
        </div>
      ) : (
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
      )}
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

function LLMCallDetail({ selection }: { selection: Extract<InspectorSelection, { kind: 'llm' }> }) {
  const { t } = useI18n();
  const { call, parent, stepIndex } = selection;
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    if (call.status !== 'running') return;
    const interval = window.setInterval(() => setNowMs(Date.now()), 250);
    return () => window.clearInterval(interval);
  }, [call.status]);
  const duration = elapsedSecondsForTrace(call, nowMs);
  const input = call.inputSummary;
  const output = call.outputSummary;
  const parentDisplay = toDisplayStep(parent);
  const parentTitle = parent.phase === 'reasoning' && !parent.tool
    ? t('ReAct decision')
    : parent.tool || t(parentDisplay.title);
  return (
    <div className="llm-call-detail" aria-live="polite">
      <section className={`inspector-card llm-call-overview ${call.status}`}>
        <div className="inspector-card-title">
          <span className="llm-call-status-icon" aria-hidden="true">
            {call.status === 'running'
              ? <Loader2 className="spin" size={14} />
              : call.status === 'error'
                ? <AlertCircle size={14} />
                : <Check size={14} />}
          </span>
          <h3>{t('Model invocation')}</h3>
          <span className={`status-line compact ${call.status}`}>{t(statusLabel(call.status))}</span>
        </div>
        {call.summary && <p className="llm-call-summary">{t(call.summary)}</p>}
        <div className="llm-call-metrics">
          <LLMMetric label={t('Duration')} value={duration === null ? '—' : `${duration.toFixed(1)}s`} />
          <LLMMetric label={t('Input tokens')} value={formatCount(call.tokenUsage?.inputTokens)} />
          <LLMMetric label={t('Output tokens')} value={formatCount(call.tokenUsage?.outputTokens)} />
          <LLMMetric label={t('Total tokens')} value={formatCount(call.tokenUsage?.totalTokens)} />
        </div>
        {call.error && <p className="llm-call-failure"><AlertCircle size={13} /> {call.error}</p>}
      </section>

      <section className="inspector-card llm-content-card">
        <div className="inspector-card-title">
          <FileText size={16} />
          <h3>{t('Call content')}</h3>
        </div>
        <p className="llm-content-note">{t('Text is bounded for display; binary and data URL payloads are omitted.')}</p>
        {call.inputPreview?.length ? (
          <details className="llm-content-section" open>
            <summary>
              <span><ChevronDown size={14} className="collapsible-chevron" /> {t('Input messages')}</span>
              <small>{call.inputPreview.length}</small>
            </summary>
            <div className="llm-message-list">
              {call.inputPreview.map((message, index) => (
                <LLMContentBlock
                  key={`${message.role}-${index}`}
                  label={message.role}
                  content={message.content}
                  copyLabel={t('Copy input message')}
                />
              ))}
            </div>
          </details>
        ) : (
          <p className="llm-content-empty">{t('No input content was captured.')}</p>
        )}
        {call.outputPreview ? (
          <details className="llm-content-section" open>
            <summary>
              <span><ChevronDown size={14} className="collapsible-chevron" /> {t('Output content')}</span>
              <small>{call.outputSummary?.format ? formatLabel(call.outputSummary.format) : t('Text')}</small>
            </summary>
            <LLMContentBlock
              label={call.outputSummary?.format ? formatLabel(call.outputSummary.format) : t('Output')}
              content={call.outputPreview}
              copyLabel={t('Copy output content')}
            />
          </details>
        ) : call.status === 'running' ? (
          <div className="llm-output-pending" role="status">
            <Loader2 className="spin" size={13} />
            <span>{t('Waiting for output')}</span>
          </div>
        ) : (
          <p className="llm-content-empty">{t('No output content was captured.')}</p>
        )}
      </section>

      <section className="inspector-card llm-io-card">
        <div className="inspector-card-title">
          <Activity size={16} />
          <h3>{t('Invocation I/O')}</h3>
        </div>
        <div className="llm-io-grid">
          <div className="llm-io-panel">
            <span>{t('Input summary')}</span>
            <strong>{input?.messageCount === undefined ? '—' : t('{count} messages', { count: input.messageCount })}</strong>
            <dl>
              <LLMSummaryRow label={t('Characters')} value={formatCount(input?.characterCount)} />
              <LLMSummaryRow label={t('Roles')} value={input?.roles?.length ? input.roles.join(' · ') : '—'} />
              <LLMSummaryRow label={t('Multimodal parts')} value={formatCount(input?.multimodalPartCount)} />
            </dl>
          </div>
          <div className="llm-io-panel">
            <span>{t('Output summary')}</span>
            <strong>{output?.format ? t(formatLabel(output.format)) : call.status === 'running' ? t('Waiting for output') : '—'}</strong>
            <dl>
              <LLMSummaryRow label={t('Characters')} value={formatCount(output?.characterCount)} />
              <LLMSummaryRow label={t('Multimodal parts')} value={formatCount(output?.multimodalPartCount)} />
            </dl>
          </div>
        </div>
      </section>

      <section className="inspector-card llm-parent-card">
        <div>
          <span>{t('Parent step')}</span>
          <strong>{parentTitle}</strong>
        </div>
        <span>{t('Step')} {stepIndex + 1}</span>
      </section>
    </div>
  );
}

function LLMContentBlock({ label, content, copyLabel }: { label: string; content: string; copyLabel: string }) {
  const displayContent = formattedLLMContent(content);
  return (
    <article className="llm-content-block">
      <header>
        <span>{label}</span>
        <button
          type="button"
          className="inspector-icon-button"
          onClick={() => void navigator.clipboard?.writeText(content)}
          aria-label={copyLabel}
          title={copyLabel}
        >
          <Clipboard size={13} />
        </button>
      </header>
      <pre><code>{displayContent}</code></pre>
    </article>
  );
}

function formattedLLMContent(content: string): string {
  const trimmed = content.trim();
  if (!trimmed || (!trimmed.startsWith('{') && !trimmed.startsWith('['))) return content;
  try {
    return JSON.stringify(JSON.parse(trimmed), null, 2);
  } catch {
    return content;
  }
}

function LLMMetric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function LLMSummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function formatCount(value: number | undefined): string {
  return value === undefined ? '—' : value.toLocaleString();
}

function resolveInspectorSelection(steps: TraceStep[], selectedNodeId: string | null): InspectorSelection | null {
  if (selectedNodeId) {
    for (let stepIndex = 0; stepIndex < steps.length; stepIndex += 1) {
      const step = steps[stepIndex];
      if (step.id === selectedNodeId) return { kind: 'step', step, stepIndex };
      const callIndex = (step.children || []).findIndex((call) => call.id === selectedNodeId);
      if (callIndex >= 0) {
        return { kind: 'llm', call: step.children![callIndex], parent: step, stepIndex, callIndex };
      }
    }
  }

  const stepIndex = steps.length - 1;
  if (stepIndex < 0) return null;
  const step = steps[stepIndex];
  const children = step.children || [];
  for (let callIndex = children.length - 1; callIndex >= 0; callIndex -= 1) {
    if (children[callIndex].status === 'running') {
      return { kind: 'llm', call: children[callIndex], parent: step, stepIndex, callIndex };
    }
  }

  let latest: InspectorSelection = { kind: 'step', step, stepIndex };
  let latestTimestamp = timestampOf(step.updatedAt);
  children.forEach((call, callIndex) => {
    const timestamp = timestampOf(call.updatedAt);
    if (timestamp >= latestTimestamp) {
      latest = { kind: 'llm', call, parent: step, stepIndex, callIndex };
      latestTimestamp = timestamp;
    }
  });
  return latest;
}

function timestampOf(value: string): number {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : 0;
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

function CodeInterpreterPreview({
  detail,
  insightDetail,
}: {
  detail: NonNullable<ReturnType<typeof toDisplayStep>['codeInterpreterDetail']>;
  insightDetail: ReturnType<typeof toDisplayStep>['insightDetail'];
}) {
  const { t } = useI18n();
  return (
    <section className="inspector-card tool-detail-card code-detail-card">
      {detail.analysisGoal && (
        <div className="code-analysis-goal">
          <span>{t('Analysis goal')}</span>
          <p>{detail.analysisGoal}</p>
        </div>
      )}

      {insightDetail && <CalculationMethods detail={insightDetail} />}

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
  return (
    <section className="inspector-card key-insight-preview">
      <InsightContent detail={detail} />
    </section>
  );
}

function CalculationMethods({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['insightDetail']> }) {
  const { t } = useI18n();
  const methods = detail.produced.flatMap((insight) => {
    const method = calculationMethodFromTrace(insight.calculation_trace);
    return method ? [{ insightId: insight.insight_id, name: insight.name, method }] : [];
  });
  if (methods.length === 0) return null;
  return (
    <div className="tool-result-section calculation-methods">
      <div className="sample-table-caption">
        <span>
          <Code2 size={14} />
          {t('Calculation methods')}
        </span>
      </div>
      <div className="calculation-method-list">
        {methods.map((item) => (
          <article key={item.insightId} className="calculation-method-item">
            <strong>{item.name}</strong>
            <p>{item.method}</p>
          </article>
        ))}
      </div>
    </div>
  );
}

function InsightContent({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['insightDetail']> }) {
  const { t } = useI18n();
  const coverageItems = insightCoverageItems(detail.coverage);
  return (
    <div className="key-insight-content">
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
              {hasCalculationTrace(insight.calculation_trace) && (
                <div className="tool-result-section">
                  <div className="sample-table-caption">
                    <span>
                      <Code2 size={14} />
                      {t('Calculation trace')}
                    </span>
                  </div>
                  <pre className="debug-json">{formatCalculationTrace(insight.calculation_trace)}</pre>
                </div>
              )}
            </details>
          ))}
        </div>
      ) : (
        <p className="sample-note">{t('No structured key insights were produced for this step.')}</p>
      )}
    </div>
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
  if (value.toLowerCase() === 'json') return 'JSON';
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

function hasCalculationTrace(trace: CalculationTrace | undefined): trace is CalculationTrace {
  if (typeof trace === 'string') return trace.trim().length > 0;
  if (Array.isArray(trace)) return trace.length > 0;
  return Boolean(trace && Object.keys(trace).length > 0);
}

function formatCalculationTrace(trace: CalculationTrace) {
  return typeof trace === 'string' ? trace : JSON.stringify(trace, null, 2);
}

function calculationMethodFromTrace(trace: CalculationTrace | undefined): string | null {
  const entries = Array.isArray(trace) ? trace : [trace];
  const methods = entries.flatMap((entry) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return [];
    const value = (entry as Record<string, unknown>).method;
    if (typeof value === 'string') return value.trim() ? [value.trim()] : [];
    if (typeof value === 'number' || typeof value === 'boolean') return [String(value)];
    if (value && typeof value === 'object') return [JSON.stringify(value)];
    return [];
  });
  return [...new Set(methods)].join('\n') || null;
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
