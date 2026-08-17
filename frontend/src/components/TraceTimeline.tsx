import { AlertCircle, CheckCircle2, CircleDot, Loader2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { toDisplayStep } from '../lib/traceDisplay';
import type { TraceStep } from '../types';
import { useI18n } from '../i18n';

type Props = {
  steps: TraceStep[];
  selectedId: string | null;
  onSelect: (id: string) => void;
};

export function TraceTimeline({ steps, selectedId, onSelect }: Props) {
  const { t } = useI18n();
  const hasRunningStep = useMemo(() => steps.some((step) => step.status === 'running'), [steps]);
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    if (!hasRunningStep) return;
    const interval = window.setInterval(() => setNowMs(Date.now()), 100);
    return () => window.clearInterval(interval);
  }, [hasRunningStep]);

  if (steps.length === 0) return null;
  return (
    <section className="trace-timeline" aria-label={t('Execution process')}>
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
              <strong>{t(displayStep.title)}</strong>
              <small>{withDuration(displayStep.summary, step, nowMs)}</small>
            </span>
            <CircleDot size={13} className="trace-open-icon" />
          </button>
        );
      })}
    </section>
  );
}

function withDuration(summary: string, step: TraceStep, nowMs: number) {
  const elapsed = elapsedSecondsForStep(step, nowMs);
  if (elapsed === null) return summary;
  return `${summary} · ${elapsed.toFixed(1)}s`;
}

function elapsedSecondsForStep(step: TraceStep, nowMs: number): number | null {
  if (step.status === 'running' && step.startedAt) {
    const started = Date.parse(step.startedAt);
    if (Number.isFinite(started)) {
      return Math.max(0, Math.round((nowMs - started) / 100) / 10);
    }
  }
  if (typeof step.elapsedSeconds === 'number' && Number.isFinite(step.elapsedSeconds)) {
    return step.elapsedSeconds;
  }
  if (step.startedAt && step.completedAt) {
    const started = Date.parse(step.startedAt);
    const completed = Date.parse(step.completedAt);
    if (Number.isFinite(started) && Number.isFinite(completed) && completed >= started) {
      return Math.round((completed - started) / 100) / 10;
    }
  }
  return null;
}
