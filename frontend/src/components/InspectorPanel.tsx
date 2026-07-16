import { AlertCircle, CheckCircle2, ChevronDown, FileText, PanelRightClose, PanelRightOpen } from 'lucide-react';
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

      {step.artifactRefs.length > 0 && (
        <section className="inspector-card">
          <div className="inspector-card-title">
            <FileText size={16} />
            <h3>Artifacts from this step</h3>
          </div>
          <div className="chip-list">
            {step.artifactRefs.map((reference) => (
              <span key={reference}>{reference}</span>
            ))}
          </div>
        </section>
      )}

      {step.debugPayload && (
        <details className="debug-details">
          <summary>
            <ChevronDown size={15} />
            Developer details
          </summary>
          <pre>{JSON.stringify(step.debugPayload, null, 2)}</pre>
        </details>
      )}
    </>
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
