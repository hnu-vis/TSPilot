import type { TraceSpan, TraceStep } from '../types';

type TimedTrace = Pick<TraceStep | TraceSpan, 'status' | 'startedAt' | 'completedAt' | 'elapsedSeconds'>;

export function elapsedSecondsForTrace(trace: TimedTrace, nowMs: number): number | null {
  if (trace.status === 'running' && trace.startedAt) {
    const started = Date.parse(trace.startedAt);
    if (Number.isFinite(started)) {
      return Math.max(0, Math.round((nowMs - started) / 100) / 10);
    }
  }
  if (typeof trace.elapsedSeconds === 'number' && Number.isFinite(trace.elapsedSeconds)) {
    return trace.elapsedSeconds;
  }
  if (trace.startedAt && trace.completedAt) {
    const started = Date.parse(trace.startedAt);
    const completed = Date.parse(trace.completedAt);
    if (Number.isFinite(started) && Number.isFinite(completed) && completed >= started) {
      return Math.round((completed - started) / 100) / 10;
    }
  }
  return null;
}

export function runElapsedSeconds(steps: TraceStep[], nowMs = Date.now()): number | null {
  const timestamps = steps
    .flatMap((step) => [step.startedAt, step.completedAt])
    .filter((value): value is string => Boolean(value))
    .map((value) => Date.parse(value))
    .filter((value) => Number.isFinite(value));
  const running = steps.some((step) => step.status === 'running');
  if (timestamps.length > 0) {
    const started = Math.min(...timestamps);
    const completed = running ? nowMs : Math.max(...timestamps);
    if (completed >= started) return Math.round((completed - started) / 100) / 10;
  }
  const elapsedValues = steps
    .map((step) => step.elapsedSeconds)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  if (elapsedValues.length === 0) return null;
  return Math.round(elapsedValues.reduce((total, value) => total + value, 0) * 10) / 10;
}
