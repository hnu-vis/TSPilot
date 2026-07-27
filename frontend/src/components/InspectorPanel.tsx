import { Activity, AlertCircle, CheckCircle2, ChevronDown, Clipboard, Code2, Database, FileText, LineChart, ListChecks, Network, PanelRightClose, PanelRightOpen, Table2 } from 'lucide-react';
import { MarkdownContent } from './FinalAnswer';
import { toDisplayStep } from '../lib/traceDisplay';
import type { FinalAnswer, TraceStep } from '../types';

type Props = {
  steps: TraceStep[];
  selectedStepId: string | null;
  answer?: FinalAnswer | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
};

export function InspectorPanel({ steps, selectedStepId, answer, collapsed, onToggleCollapsed }: Props) {
  if (steps.length === 0 && !answer) return null;

  const selectedStep = selectedStepId
    ? steps.find((step) => step.id === selectedStepId) || null
    : null;
  const activeStep = selectedStep || steps[steps.length - 1] || null;
  const displayStep = activeStep ? toDisplayStep(activeStep) : null;

  if (collapsed) {
    return (
      <aside className="inspector-panel collapsed" aria-label="Run details collapsed">
        <button type="button" className="inspector-rail-button" onClick={onToggleCollapsed} aria-label="Show run details">
          <PanelRightOpen size={18} />
          <span>Detail</span>
        </button>
      </aside>
    );
  }

  return (
    <aside className="inspector-panel" aria-label="Run details">
      <header>
        <div>
          <p>Detail</p>
          <h2>{displayStep?.title || 'Answer'}</h2>
        </div>
        <div className="inspector-header-actions">
          <button type="button" onClick={onToggleCollapsed} aria-label="Collapse run details">
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
  return (
    <section className="inspector-card">
      <div className="inspector-card-title">
        <FileText size={16} />
        <h3>Answer references</h3>
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
  const visibleMetrics = step.metrics.filter((metric) => metric.value !== '0' && metric.value !== '0/0');
  return (
    <>
      {step.planDetail && <PlanPreview step={step} />}

      {step.sqlDetail ? <QueryRunSummary step={step} /> : !step.hasPrimaryDetail && <StepStatusCard step={step} />}

      {!step.hasPrimaryDetail && step.status !== 'error' && visibleMetrics.length > 0 && (
        <section className="metric-grid" aria-label="Step metrics">
          {visibleMetrics.map((metric) => (
            <div key={metric.label} className="metric-tile">
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </section>
      )}

      {step.sqlDetail?.query && <QueryPreview detail={step.sqlDetail} />}

      {step.sqlDetail && <DataPreview detail={step.sqlDetail} />}

      {step.codeInterpreterDetail && <CodeInterpreterPreview detail={step.codeInterpreterDetail} />}

      {step.forecastDetail && <ForecastPreview detail={step.forecastDetail} />}

      {step.anomalyDetail && <AnomalyPreview detail={step.anomalyDetail} />}

      {step.schemaLinkingDetail && <SchemaLinkingPreview detail={step.schemaLinkingDetail} />}

      {step.completionDetail && shouldShowCompletion(step.completionDetail) && (
        <CompletionPreview detail={step.completionDetail} />
      )}

      <ReactStepCard step={step} />

      {(step.artifactRefs.length > 0 || step.debugPayload) && (
        <AdvancedDetails step={step} />
      )}
    </>
  );
}

function ReactStepCard({ step }: { step: ReturnType<typeof toDisplayStep> }) {
  const detail = step.reactDetail;
  return (
    <details className="inspector-card react-step-card collapsible-card" open>
      <summary className="collapsible-summary">
        <span>
          <ChevronDown size={15} className="collapsible-chevron" />
          <Code2 size={16} />
          <strong>ReAct details</strong>
        </span>
      </summary>
      <div className="react-grid">
        <ReactBlock label="Thought" value={detail.thought || 'Waiting for model reasoning.'} />
        <ReactBlock label="Action" value={detail.action || step.category} />
        <ReactBlock label="Action Input" value={detail.actionInput} />
        <ReactBlock label="Observation" value={detail.observation} />
      </div>
    </details>
  );
}

function ReactBlock({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="react-block">
      <span>{label}</span>
      {typeof value === 'string' ? (
        <pre>{value}</pre>
      ) : value ? (
        <pre>{JSON.stringify(value, null, 2)}</pre>
      ) : (
        <pre>Pending</pre>
      )}
    </div>
  );
}

function StepStatusCard({ step }: { step: ReturnType<typeof toDisplayStep> }) {
  return (
    <section className="inspector-card">
      <div className="inspector-card-title">
        <StatusIcon status={step.status} />
        <h3>Status</h3>
      </div>
      <p className="step-summary">{step.summary}</p>
      <div className={`status-line ${step.status}`}>{statusLabel(step.status)}</div>
    </section>
  );
}

function PlanPreview({ step }: { step: ReturnType<typeof toDisplayStep> }) {
  const detail = step.planDetail;
  if (!detail) return null;
  const requiredOutputs = recordsFrom(detail.taskContract?.required_outputs);
  return (
    <section className="inspector-card plan-preview-card">
      <div className="inspector-card-title">
        <ListChecks size={16} />
        <h3>Todo list</h3>
        <span className={`status-line compact ${detail.planningComplete ? 'complete' : step.status}`}>
          {detail.completed}/{detail.total || detail.todos.length}
        </span>
      </div>

      {detail.inProgress && <p className="step-summary">Current: {detail.inProgress}</p>}

      {detail.todos.length > 0 && (
        <ol className="inspector-todo-list">
          {detail.todos.map((todo, index) => (
            <li key={`${todo.priority ?? index}-${todo.content}`} className={`inspector-todo-item ${todo.status}`}>
              <span className="inspector-todo-index">{todo.priority ?? index + 1}</span>
              <span className="inspector-todo-copy">
                <strong>{todo.content}</strong>
                <small>
                  {[formatLabel(todo.status), todo.taskType ? formatLabel(todo.taskType) : null]
                    .filter(Boolean)
                    .join(' · ')}
                </small>
                {todo.acceptanceCriteria && <em>{todo.acceptanceCriteria}</em>}
              </span>
            </li>
          ))}
        </ol>
      )}

      {requiredOutputs.length > 0 && (
        <div className="tool-result-section">
          <div className="sample-table-caption">
            <span>
              <CheckCircle2 size={14} />
              Required outputs
            </span>
          </div>
          <div className="chip-list compact">
            {requiredOutputs.slice(0, 12).map((output, index) => (
              <span key={stringFrom(output.id) || stringFrom(output.description) || index}>
                {stringFrom(output.description) || stringFrom(output.id) || `Output ${index + 1}`}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function QueryRunSummary({ step }: { step: ReturnType<typeof toDisplayStep> }) {
  const detail = step.sqlDetail;
  if (!detail) return null;
  const resultSize = [
    detail.rowCount !== null ? `${detail.rowCount.toLocaleString()} rows` : null,
    shouldShowPointCount(detail) ? `${detail.pointCount?.toLocaleString()} points` : null,
  ].filter(Boolean).join(' / ') || 'No visible rows';
  const dataShape = detail.columns.length > 0
    ? `${detail.columns.length} columns`
    : detail.sampleRows.length || detail.samplePoints.length
      ? 'Sample available'
      : 'No preview';
  return (
    <section className="inspector-card query-run-summary">
      <div className="inspector-card-title">
        <StatusIcon status={step.status} />
        <h3>Result</h3>
        <span className={`status-line compact ${step.status}`}>{statusLabel(step.status)}</span>
      </div>
      <p className="step-summary">{step.summary}</p>
      <div className="query-run-facts" aria-label="Result facts">
        <div>
          <Database size={14} />
          <span>{resultSize}</span>
        </div>
        <div>
          <Table2 size={14} />
          <span>{dataShape}</span>
        </div>
        {detail.queryLanguage && (
          <div>
            <Code2 size={14} />
            <span>{detail.queryLanguage.toUpperCase()}</span>
          </div>
        )}
      </div>
    </section>
  );
}

function CompletionPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['completionDetail']> }) {
  const status =
    detail.completed === true ? 'Complete' : detail.completed === false ? 'Needs more evidence' : 'Tracking';
  return (
    <section className="inspector-card">
      <div className="inspector-card-title">
        <CheckCircle2 size={16} />
        <h3>Completion</h3>
      </div>
      <div className={`status-line ${detail.completed === false ? 'error' : 'complete'}`}>{status}</div>
      {detail.todoProgress && detail.todoProgress.total > 0 && (
        <p className="step-summary">
          Todo progress: {detail.todoProgress.completed}/{detail.todoProgress.total}
          {detail.todoProgress.inProgress ? ` · ${detail.todoProgress.inProgress}` : ''}
        </p>
      )}
      {detail.reason && <p className="step-summary">{detail.reason}</p>}
      {detail.missingItems.length > 0 && (
        <div className="chip-list compact">
          {detail.missingItems.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      )}
      {detail.nextActionHint && <p className="sample-note">{detail.nextActionHint}</p>}
    </section>
  );
}

function QueryPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['sqlDetail']> }) {
  const queryLanguage = detail.queryLanguage || 'query';
  const queryMarkdown = fencedQueryMarkdown(detail.query || '', detail.queryLanguage);
  return (
    <details className="inspector-card sql-query-preview collapsible-card" open>
      <summary className="collapsible-summary">
        <span>
          <ChevronDown size={15} className="collapsible-chevron" />
          <Code2 size={16} />
          <strong>Generated query</strong>
        </span>
        {detail.queryLanguage && <span className="query-language-badge">{detail.queryLanguage}</span>}
      </summary>
      <div className="query-actions">
        <button
          type="button"
          className="icon-text-button"
          onClick={() => void navigator.clipboard?.writeText(detail.query || '')}
          aria-label="Copy generated query"
          title="Copy query"
        >
          <Clipboard size={13} />
          <span>Copy {queryLanguage}</span>
        </button>
      </div>
      <div className="query-markdown-render">
        <MarkdownContent content={queryMarkdown} />
      </div>
    </details>
  );
}

function DataPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['sqlDetail']> }) {
  const rows = detail.sampleRows.length > 0 ? detail.sampleRows : detail.samplePoints;
  const tableColumns = detail.columns.length > 0 ? detail.columns : inferColumns(rows);
  return (
    <section className="inspector-card data-preview">
      <div className="inspector-card-title sql-detail-title">
        <Table2 size={16} />
        <h3>Returned data</h3>
        <span className="query-language-badge">{formatCounts(detail)}</span>
      </div>

      {detail.columns.length > 0 && (
        <div className="column-chip-list" aria-label="Result columns">
          {detail.columns.map((column) => (
            <span key={column}>{column}</span>
          ))}
        </div>
      )}

      {rows.length > 0 && tableColumns.length > 0 && (
        <div className="sample-table-section">
          <div className="sample-table-caption">
            <span>
              <Table2 size={14} />
              Sample data
            </span>
            <strong>{formatCounts(detail)}</strong>
          </div>
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
          {detail.truncated && <p className="sample-note">Preview only</p>}
        </div>
      )}

      {rows.length === 0 && (
        <div className="empty-data-preview">
          <Table2 size={16} />
          <span>No sample rows are visible for this result.</span>
        </div>
      )}
    </section>
  );
}

function CodeInterpreterPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['codeInterpreterDetail']> }) {
  return (
    <section className="inspector-card tool-detail-card">
      <div className="inspector-card-title sql-detail-title">
        <Code2 size={16} />
        <h3>Code interpreter</h3>
        {detail.codeType && <span className="query-language-badge">{detail.codeType}</span>}
      </div>

      {detail.analysisGoal && <p className="step-summary">{detail.analysisGoal}</p>}

      <div className="query-run-facts" aria-label="Code interpreter facts">
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
        {detail.codeHash && (
          <div>
            <Code2 size={14} />
            <span>{detail.codeHash}</span>
          </div>
        )}
      </div>

      {detail.code && (
        <ToolCodeBlock
          title="Python code"
          language="python"
          code={detail.code}
          copyLabel="Copy code"
        />
      )}

      <StructuredResult
        title="Result"
        summary={detail.summary}
        metrics={detail.metrics}
        details={detail.details}
        fallback={detail.result}
      />
    </section>
  );
}

function ForecastPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['forecastDetail']> }) {
  const rows = detail.points;
  return (
    <section className="inspector-card tool-detail-card">
      <div className="inspector-card-title sql-detail-title">
        <LineChart size={16} />
        <h3>Forecast result</h3>
        {detail.status && <span className="query-language-badge">{detail.status}</span>}
      </div>

      <div className="query-run-facts" aria-label="Forecast facts">
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

      {detail.plan && <KeyValuePreview title="Forecast plan" data={detail.plan} />}

      {rows.length > 0 ? (
        <SimpleRecordsTable title="Forecast points" records={rows} />
      ) : (
        <p className="sample-note">No direct forecast points were returned for this step.</p>
      )}
    </section>
  );
}

function AnomalyPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['anomalyDetail']> }) {
  return (
    <section className="inspector-card tool-detail-card">
      <div className="inspector-card-title sql-detail-title">
        <AlertCircle size={16} />
        <h3>Anomaly result</h3>
        {detail.detectorName && <span className="query-language-badge">{detail.detectorName}</span>}
      </div>

      <div className="query-run-facts" aria-label="Anomaly facts">
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
        <SimpleRecordsTable title="Anomaly points" records={detail.points} />
      ) : (
        <p className="sample-note">No anomaly points are visible for this step.</p>
      )}
      {detail.scores.length > 0 && <SimpleRecordsTable title="Scores" records={detail.scores} />}
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
  const hasMetrics = Object.keys(metrics).length > 0;
  const hasDetails = Object.keys(details).length > 0;
  return (
    <div className="tool-result-section">
      <div className="sample-table-caption">
        <span>
          <CheckCircle2 size={14} />
          {title}
        </span>
      </div>
      {summary && <MarkdownContent content={summary} />}
      {hasMetrics && <InspectorMetricGroup metrics={metrics} />}
      {hasDetails && <KeyValuePreview title="Details" data={details} />}
      {!summary && !hasMetrics && !hasDetails && fallback && <pre className="debug-json">{JSON.stringify(fallback, null, 2)}</pre>}
    </div>
  );
}

function InspectorMetricGroup({ metrics }: { metrics: Record<string, unknown> }) {
  return (
    <dl className="answer-metric-grid">
      {Object.entries(metrics).map(([key, value]) => (
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
  const codeMarkdown = `\`\`\`${language}\n${code}\n\`\`\``;
  return (
    <details className="answer-code-details tool-code-details" open>
      <summary>
        <span>
          <ChevronDown size={14} className="collapsible-chevron" />
          <Code2 size={14} />
          {title}
        </span>
        <button
          type="button"
          className="icon-text-button inline-copy-button"
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            void navigator.clipboard?.writeText(code);
          }}
          aria-label={copyLabel}
          title={copyLabel}
        >
          <Clipboard size={13} />
          <span>Copy</span>
        </button>
      </summary>
      <div className="query-markdown-render">
        <MarkdownContent content={codeMarkdown} />
      </div>
    </details>
  );
}

function SchemaLinkingPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['schemaLinkingDetail']> }) {
  return (
    <section className="inspector-card schema-linking-preview">
      <div className="inspector-card-title sql-detail-title">
        <Network size={16} />
        <h3>Schema linking</h3>
        {detail.confidence && <span className="query-language-badge">{detail.confidence}</span>}
      </div>

      {detail.sources.length > 0 && (
        <div className="schema-linking-group">
          <span className="schema-linking-label">Sources</span>
          {detail.sources.map((source) => (
            <div key={source.name} className="schema-source-row">
              <strong>{source.name}</strong>
              {source.kind && <span>{source.kind}</span>}
              {source.timeColumn && <code>{source.timeColumn}</code>}
            </div>
          ))}
        </div>
      )}

      {detail.requiredFilters.length > 0 && (
        <div className="schema-linking-group">
          <span className="schema-linking-label">Required filters</span>
          <div className="schema-chip-list">
            {detail.requiredFilters.map((filter) => (
              <code key={`${filter.column}-${filter.value}`}>
                {filter.column} {filter.operator} {filter.value}
              </code>
            ))}
          </div>
        </div>
      )}

      {detail.fieldMappings.length > 0 && (
        <div className="schema-linking-group">
          <span className="schema-linking-label">Field mappings</span>
          <div className="schema-mapping-list">
            {detail.fieldMappings.slice(0, 6).map((mapping) => (
              <div key={`${mapping.sourceName || 'source'}-${mapping.fieldName}-${mapping.role || 'role'}`} className="schema-mapping-row">
                <span>{mapping.sourceName || 'source'}</span>
                <strong>{mapping.fieldName}</strong>
                {mapping.role && <code>{mapping.role}</code>}
              </div>
            ))}
          </div>
        </div>
      )}

      {detail.sources.some((source) => source.valueColumns.length > 0 || source.dimensionColumns.length > 0) && (
        <div className="schema-linking-group">
          <span className="schema-linking-label">Linked fields</span>
          <div className="schema-chip-list">
            {detail.sources.flatMap((source) => [
              ...source.valueColumns.map((column) => ({ key: `${source.name}-value-${column}`, label: column, role: 'value' })),
              ...source.dimensionColumns.map((column) => ({ key: `${source.name}-dimension-${column}`, label: column, role: 'dimension' })),
            ]).slice(0, 10).map((field) => (
              <code key={field.key}>{field.label} · {field.role}</code>
            ))}
          </div>
        </div>
      )}

      {detail.ambiguousTerms.length > 0 && (
        <div className="schema-linking-group">
          <span className="schema-linking-label">Ambiguous</span>
          <div className="schema-chip-list">
            {detail.ambiguousTerms.map((item) => (
              <code key={item.term}>{item.term}: {item.candidates.join(', ')}</code>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function AdvancedDetails({ step }: { step: ReturnType<typeof toDisplayStep> }) {
  return (
    <details className="debug-details advanced-details">
      <summary>
        <ChevronDown size={15} />
        Advanced
      </summary>
      <div className="advanced-body">
        {step.artifactRefs.length > 0 && (
          <section className="advanced-section">
            <div className="advanced-section-title">
              <FileText size={14} />
              <h4>References</h4>
            </div>
            <div className="chip-list compact">
              {step.artifactRefs.map((reference) => (
                <span key={reference}>{reference}</span>
              ))}
            </div>
          </section>
        )}

        {step.debugPayload && (
          <section className="advanced-section">
            <div className="advanced-section-title">
              <Code2 size={14} />
              <h4>Raw event</h4>
            </div>
            <pre className="debug-json">{JSON.stringify(step.debugPayload, null, 2)}</pre>
          </section>
        )}
      </div>
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
  if (value === 'sql_query' || value === 'query_database') return 'Database evidence';
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

function shouldShowCompletion(detail: ReturnType<typeof toDisplayStep>['completionDetail']) {
  if (!detail) return false;
  if (detail.completed === false) return true;
  if (detail.missingItems.length > 0) return true;
  if (detail.nextActionHint) return true;
  return false;
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? value.toLocaleString() : String(value);
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function stringFrom(value: unknown) {
  return typeof value === 'string' && value.trim() ? value : null;
}

function recordsFrom(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object'))
    : [];
}

function fencedQueryMarkdown(query: string, queryLanguage: string | null) {
  const language = markdownLanguage(queryLanguage);
  return `\`\`\`${language}\n${query.trim()}\n\`\`\``;
}

function markdownLanguage(queryLanguage: string | null) {
  const normalized = (queryLanguage || '').trim().toLowerCase();
  if (!normalized) return '';
  if (['postgres', 'postgresql', 'timescaledb', 'questdb', 'clickhouse'].includes(normalized)) return 'sql';
  if (normalized === 'influxdb') return 'flux';
  if (normalized === 'prometheus') return 'promql';
  return normalized;
}
