import {
  AlertCircle,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Loader2,
} from 'lucide-react';
import { Fragment, useId, useMemo, useState } from 'react';
import { toDisplayStep } from '../lib/traceDisplay';
import { elapsedSecondsForTrace, runElapsedSeconds } from '../lib/traceTiming';
import type { TraceSpan, TraceStep } from '../types';
import { useI18n } from '../i18n';

type Props = {
  steps: TraceStep[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  nowMs?: number;
};

export function TraceTimeline({ steps, selectedId, onSelect, nowMs = Date.now() }: Props) {
  const { t } = useI18n();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [isCollapsed, setIsCollapsed] = useState(false);
  const treeId = useId();
  const elapsed = runElapsedSeconds(steps, nowMs);
  const visibleSteps = useMemo(
    () => steps.filter(isVisibleTraceStep),
    [steps],
  );

  if (visibleSteps.length === 0) return null;
  return (
    <section className={`trace-timeline ${isCollapsed ? 'collapsed' : ''}`} aria-label={t('Execution process')}>
      <header className="trace-timeline-header">
        <button
          type="button"
          className="trace-timeline-toggle"
          aria-expanded={!isCollapsed}
          aria-controls={treeId}
          aria-label={t(isCollapsed ? 'Expand execution process' : 'Collapse execution process')}
          onClick={() => setIsCollapsed((current) => !current)}
        >
          <span>{t('Execution process')}</span>
          <span className="trace-timeline-meta">
            {elapsed !== null && <small>{t('Total {duration}', { duration: `${elapsed.toFixed(1)}s` })}</small>}
            <ChevronDown size={15} className="trace-timeline-chevron" aria-hidden="true" />
          </span>
        </button>
      </header>
      <ol id={treeId} className="trace-tree" hidden={isCollapsed}>
        {visibleSteps.map((step) => {
          const displayStep = toDisplayStep(step);
          const title = step.phase === 'policy_decision' && step.status === 'error'
            ? `${step.tool || t(displayStep.title)} · ${t('Rejected')}`
            : step.tool || t(displayStep.title);
          const calls = (step.children || []).filter((child) => child.kind === 'llm');
          if (isDecisionContainer(step) && calls.length > 0) {
            return (
              <Fragment key={step.id}>
                {calls.map((call) => (
                  <DecisionLLMNode
                    key={call.id}
                    call={call}
                    selected={selectedId === call.id}
                    onSelect={onSelect}
                    nowMs={nowMs}
                  />
                ))}
              </Fragment>
            );
          }
          const hasRunningCall = calls.some((call) => call.status === 'running');
          const hasSelectedCall = calls.some((call) => call.id === selectedId);
          const isOpen = expanded[step.id]
            ?? (step.status === 'running' || step.status === 'error' || hasRunningCall || hasSelectedCall);
          const duration = elapsedSecondsForTrace(step, nowMs);
          return (
            <li key={step.id} className={`trace-node ${displayStep.status} ${isOpen ? 'open' : ''}`}>
              <button
                type="button"
                className={`trace-step ${displayStep.status} ${selectedId === step.id ? 'selected' : ''}`}
                aria-expanded={calls.length > 0 ? isOpen : undefined}
                onClick={() => {
                  onSelect(step.id);
                  if (calls.length > 0) {
                    setExpanded((current) => ({ ...current, [step.id]: !isOpen }));
                  }
                }}
              >
                <span className="trace-icon"><TraceStatusIcon status={displayStep.status} /></span>
                <span className="trace-copy">
                  <span className="trace-title-row">
                    <strong>{title}</strong>
                    {duration !== null && <time>{duration.toFixed(1)}s</time>}
                  </span>
                  <small>{t(displayStep.summary)}</small>
                  {calls.length > 0 && (
                    <span className="trace-call-count">{t('{count} LLM calls', { count: calls.length })}</span>
                  )}
                </span>
                {calls.length > 0
                  ? <ChevronDown size={15} className="trace-chevron" aria-hidden="true" />
                  : <CircleDot size={13} className="trace-open-icon" aria-hidden="true" />}
              </button>

              {calls.length > 0 && isOpen && (
                <ol className="trace-llm-list" aria-label={t('LLM calls')}>
                  {calls.map((call) => (
                    <LLMTraceLeaf
                      key={call.id}
                      call={call}
                      selected={selectedId === call.id}
                      onSelect={onSelect}
                      nowMs={nowMs}
                    />
                  ))}
                </ol>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function DecisionLLMNode({
  call,
  selected,
  onSelect,
  nowMs,
}: {
  call: TraceSpan;
  selected: boolean;
  onSelect: (id: string) => void;
  nowMs: number;
}) {
  const { t } = useI18n();
  const duration = elapsedSecondsForTrace(call, nowMs);
  const totalTokens = call.tokenUsage?.totalTokens;
  const status = call.status === 'running' ? 'Running' : call.status === 'error' ? 'Failed' : 'Completed';
  return (
    <li className={`trace-node trace-llm-root-node ${call.status}`}>
      <button
        type="button"
        className={`trace-step trace-llm-root-call ${call.status} ${selected ? 'selected' : ''}`}
        aria-current={selected ? 'true' : undefined}
        aria-label={`${t(call.title)} · ${t(status)}`}
        onClick={() => onSelect(call.id)}
      >
        <span className="trace-icon"><TraceStatusIcon status={call.status} /></span>
        <span className="trace-copy">
          <span className="trace-title-row">
            <strong>{t(call.title)}</strong>
            {duration !== null && <time>{duration.toFixed(1)}s</time>}
          </span>
          <small>{call.summary ? t(call.summary) : t(status)}</small>
          {totalTokens !== undefined && totalTokens > 0 && (
            <span className="trace-call-count">{totalTokens.toLocaleString()} tokens</span>
          )}
          {call.error && <small className="trace-llm-error">{call.error}</small>}
        </span>
        <CircleDot size={13} className="trace-open-icon" aria-hidden="true" />
      </button>
    </li>
  );
}

function LLMTraceLeaf({
  call,
  selected,
  onSelect,
  nowMs,
}: {
  call: TraceSpan;
  selected: boolean;
  onSelect: (id: string) => void;
  nowMs: number;
}) {
  const { t } = useI18n();
  const duration = elapsedSecondsForTrace(call, nowMs);
  const totalTokens = call.tokenUsage?.totalTokens;
  const statusLabel = call.status === 'running' ? 'Running' : call.status === 'error' ? 'Failed' : 'Completed';
  return (
    <li className="trace-llm-node">
      <button
        type="button"
        className={`trace-llm-leaf ${call.status} ${selected ? 'selected' : ''}`}
        aria-current={selected ? 'true' : undefined}
        aria-label={`${t(call.title)} · ${t(statusLabel)}`}
        onClick={() => onSelect(call.id)}
      >
        <span className="trace-llm-icon" aria-hidden="true">
          {call.status === 'running'
            ? <Loader2 className="spin" size={13} />
            : call.status === 'error'
              ? <AlertCircle size={13} />
              : <Check size={13} />}
        </span>
        <span className="trace-llm-copy">
          <span className="trace-llm-title">
            <strong>{t(call.title)}</strong>
            {duration !== null && <time>{duration.toFixed(1)}s</time>}
          </span>
          {call.summary && call.summary !== call.title && <small>{t(call.summary)}</small>}
          {totalTokens !== undefined && totalTokens > 0 && (
            <span className="trace-token-count">{totalTokens.toLocaleString()} tokens</span>
          )}
          {call.error && <small className="trace-llm-error">{call.error}</small>}
        </span>
        <span className="sr-only">{t(statusLabel)}</span>
      </button>
    </li>
  );
}

function isDecisionContainer(step: TraceStep): boolean {
  return step.phase === 'reasoning' && !step.tool && !step.toolCall;
}

function isVisibleTraceStep(step: TraceStep): boolean {
  return Boolean(
    step.toolCall
    || step.toolResult
    || step.observation
    || step.children?.length
    || step.phase === 'reasoning',
  );
}

function TraceStatusIcon({ status }: { status: ReturnType<typeof toDisplayStep>['status'] }) {
  if (status === 'running') return <Loader2 className="spin" size={15} />;
  if (status === 'error' || status === 'attention') return <AlertCircle size={15} />;
  return <CheckCircle2 size={15} />;
}
