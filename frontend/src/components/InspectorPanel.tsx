import { PanelRightClose } from 'lucide-react';
import type { TraceStep } from '../types';

type Props = {
  step: TraceStep | null;
  onClose: () => void;
};

export function InspectorPanel({ step, onClose }: Props) {
  if (!step) return null;

  return (
    <aside className="inspector-panel" aria-label="Step details">
      <header>
        <div>
          <p>Inspector</p>
          <h2>{step.tool || step.phase || 'Step details'}</h2>
        </div>
        <button type="button" onClick={onClose} aria-label="Close inspector">
          <PanelRightClose size={18} />
        </button>
      </header>
      <div className="inspector-body">
        <Detail label="Status" value={step.status} />
        <Detail label="Summary" value={step.summary} />
        <Detail label="Iteration" value={String(step.iteration)} />
        {step.toolCall && <JsonBlock title="Tool input preview" value={step.toolCall} />}
        {step.toolResult && <JsonBlock title="Tool result preview" value={step.toolResult} />}
        {step.error && <pre className="error-block">{step.error}</pre>}
      </div>
    </aside>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function JsonBlock({ title, value }: { title: string; value: Record<string, unknown> }) {
  return (
    <section className="json-section">
      <h3>{title}</h3>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}
