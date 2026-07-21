import { AlertCircle, CheckCircle2, ChevronDown, Code2, FileText, Network, PanelRightClose, PanelRightOpen, Table2 } from 'lucide-react';
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
  return (
    <>
      <section className="inspector-card">
        <div className="inspector-card-title">
          <StatusIcon status={step.status} />
          <h3>{step.title}</h3>
        </div>
        <p className="step-summary">{step.summary}</p>
        <div className={`status-line ${step.status}`}>{statusLabel(step.status)}</div>
      </section>

      {step.metrics.length > 0 && (
        <section className="metric-grid" aria-label="Step metrics">
          {step.metrics.map((metric) => (
            <div key={metric.label} className="metric-tile">
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </section>
      )}

      {step.sqlDetail && <DataPreview detail={step.sqlDetail} />}

      {step.schemaLinkingDetail && <SchemaLinkingPreview detail={step.schemaLinkingDetail} />}

      {step.completionDetail && <CompletionPreview detail={step.completionDetail} />}

      {(step.sqlDetail?.query || step.artifactRefs.length > 0 || step.debugPayload) && (
        <AdvancedDetails step={step} />
      )}
    </>
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

function DataPreview({ detail }: { detail: NonNullable<ReturnType<typeof toDisplayStep>['sqlDetail']> }) {
  const rows = detail.sampleRows.length > 0 ? detail.sampleRows : detail.samplePoints;
  const tableColumns = detail.columns.length > 0 ? detail.columns : inferColumns(rows);
  return (
    <section className="inspector-card data-preview">
      <div className="inspector-card-title sql-detail-title">
        <Table2 size={16} />
        <h3>Data preview</h3>
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
    </section>
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
        {step.sqlDetail?.query && (
          <section className="advanced-section">
            <div className="advanced-section-title">
              <Code2 size={14} />
              <h4>Query</h4>
              {step.sqlDetail.queryLanguage && <span className="query-language-badge">{step.sqlDetail.queryLanguage}</span>}
            </div>
            <pre className="sql-code-block">{step.sqlDetail.query}</pre>
          </section>
        )}

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

function StatusIcon({ status }: { status: TraceStep['status'] }) {
  if (status === 'error') return <AlertCircle size={16} />;
  return <CheckCircle2 size={16} />;
}

function statusLabel(status: string) {
  if (status === 'running') return 'Running';
  if (status === 'complete') return 'Complete';
  if (status === 'error') return 'Needs attention';
  return 'Not started';
}

function formatLabel(value: string) {
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
    detail.pointCount !== null ? `${detail.pointCount.toLocaleString()} points` : null,
  ].filter(Boolean);
  return parts.length ? parts.join(' / ') : `${detail.sampleRows.length || detail.samplePoints.length} samples`;
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? value.toLocaleString() : String(value);
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}
