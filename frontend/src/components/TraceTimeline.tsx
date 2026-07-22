import { AlertCircle, CheckCircle2, CircleDot, Loader2 } from 'lucide-react';
import { toDisplayStep } from '../lib/traceDisplay';
import type { TraceStep } from '../types';

type Props = {
  steps: TraceStep[];
  selectedId: string | null;
  onSelect: (id: string) => void;
};

export function TraceTimeline({ steps, selectedId, onSelect }: Props) {
  if (steps.length === 0) return null;
  return (
    <section className="trace-timeline" aria-label="Execution process">
      {steps.map((step) => {
        const displayStep = toDisplayStep(step);
        const StatusIcon = displayStep.status === 'running'
          ? Loader2
          : displayStep.status === 'error' || displayStep.status === 'attention'
            ? AlertCircle
            : CheckCircle2;
        return (
          <button
            key={step.id}
            type="button"
            className={`trace-step ${displayStep.status} ${selectedId === step.id ? 'selected' : ''}`}
            onClick={() => onSelect(step.id)}
          >
            <span className="trace-icon">
              {displayStep.status === 'running' ? <StatusIcon className="spin" size={15} /> : <StatusIcon size={15} />}
            </span>
            <span className="trace-copy">
              <strong>{displayStep.title}</strong>
              <small>{displayStep.summary}</small>
            </span>
            <CircleDot size={13} className="trace-open-icon" />
          </button>
        );
      })}
    </section>
  );
}
