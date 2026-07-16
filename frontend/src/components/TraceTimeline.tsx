import { AlertCircle, CheckCircle2, CircleDot, Loader2 } from 'lucide-react';
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
        const StatusIcon = step.status === 'running' ? Loader2 : step.status === 'error' ? AlertCircle : CheckCircle2;
        return (
          <button
            key={step.id}
            type="button"
            className={`trace-step ${selectedId === step.id ? 'selected' : ''}`}
            onClick={() => onSelect(step.id)}
          >
            <span className="trace-icon">
              {step.status === 'running' ? <StatusIcon className="spin" size={15} /> : <StatusIcon size={15} />}
            </span>
            <span className="trace-copy">
              <strong>{formatPhase(step.phase, step.tool)}</strong>
              <small>{step.summary}</small>
            </span>
            <CircleDot size={13} className="trace-open-icon" />
          </button>
        );
      })}
    </section>
  );
}

function formatPhase(phase: string, tool?: string) {
  if (tool) return tool;
  if (phase === 'tool_selection') return 'Data';
  if (phase === 'answer_assembly') return 'Answer';
  if (phase === 'analysis') return 'Analysis';
  if (phase === 'intent') return 'Plan';
  return phase || 'Step';
}
