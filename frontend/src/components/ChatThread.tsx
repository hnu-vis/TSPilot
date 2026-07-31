import { ChevronDown, ListChecks, Bot, Loader2, User } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChatMessage, TraceStep } from '../types';
import { FinalAnswer } from './FinalAnswer';
import { TraceTimeline } from './TraceTimeline';

type Props = {
  messages: ChatMessage[];
  traceSteps: TraceStep[];
  selectedTraceStepId: string | null;
  onSelectTraceStep: (id: string) => void;
};

export function ChatThread({ messages, traceSteps, selectedTraceStepId, onSelectTraceStep }: Props) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const hasStreamingMessage = useMemo(() => messages.some((message) => message.isStreaming), [messages]);
  const hasRunningStep = useMemo(() => traceSteps.some((step) => step.status === 'running'), [traceSteps]);
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    if (!hasStreamingMessage && !hasRunningStep) return;
    const interval = window.setInterval(() => setNowMs(Date.now()), 100);
    return () => window.clearInterval(interval);
  }, [hasStreamingMessage, hasRunningStep]);
  const scrollKey = useMemo(() => (
    [
      messages.length,
      messages[messages.length - 1]?.id,
      messages[messages.length - 1]?.content,
      messages[messages.length - 1]?.isStreaming ? 'streaming' : 'settled',
      messages[messages.length - 1]?.answer?.summary,
      traceSteps.length,
      traceSteps[traceSteps.length - 1]?.id,
      traceSteps[traceSteps.length - 1]?.summary,
      traceSteps[traceSteps.length - 1]?.status,
    ].join('|')
  ), [messages, traceSteps]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ block: 'end' });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [scrollKey]);

  const lastAssistantIndex = findLastAssistantIndex(messages);
  const latestTodos = useMemo(() => latestTodoList(traceSteps), [traceSteps]);

  if (messages.length === 0 && traceSteps.length === 0) {
    return (
      <div className="empty-thread">
        <div className="empty-mark">TS</div>
        <h2>Ask about your time-series data</h2>
        <p>Retrieve data, inspect trends, detect anomalies, forecast future values, and review the execution process.</p>
      </div>
    );
  }

  return (
    <div className="chat-thread">
      {messages.map((message, index) => (
        <article key={message.id} className={`message ${message.role} ${message.answer ? 'has-answer' : ''}`}>
          <div className="avatar">{message.role === 'user' ? <User size={16} /> : <Bot size={16} />}</div>
          {index === lastAssistantIndex && latestTodos.length > 0 && message.answer && (
            <TodoList todos={latestTodos} />
          )}
          {index === lastAssistantIndex && traceSteps.length > 0 && message.answer && (
            <TraceTimeline steps={traceSteps} selectedId={selectedTraceStepId} onSelect={onSelectTraceStep} />
          )}
          <div className="bubble">
            {message.answer ? (
              <FinalAnswer answer={message.answer} tokenUsage={message.tokenUsage} elapsedSeconds={runElapsedSeconds(traceSteps)} />
            ) : message.isStreaming ? (
              <div className="live-answer">
                <div className="live-status">
                  <Loader2 className="spin" size={15} />
                  <span>{latestTraceSummary(traceSteps, nowMs) || message.content}</span>
                </div>
              </div>
            ) : (
              <p>{message.content}</p>
            )}
          </div>
          {index === lastAssistantIndex && latestTodos.length > 0 && !message.answer && (
            <TodoList todos={latestTodos} />
          )}
          {index === lastAssistantIndex && traceSteps.length > 0 && !message.answer && (
            <TraceTimeline steps={traceSteps} selectedId={selectedTraceStepId} onSelect={onSelectTraceStep} />
          )}
        </article>
      ))}
      <div ref={bottomRef} className="thread-bottom-anchor" aria-hidden="true" />
    </div>
  );
}

function findLastAssistantIndex(messages: ChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'assistant') return index;
  }
  return -1;
}

function latestTraceSummary(steps: TraceStep[], nowMs: number) {
  const latestRunning = [...steps].reverse().find((step) => step.status === 'running');
  const latest = latestRunning || steps[steps.length - 1];
  if (!latest?.summary) return '';
  const elapsed = elapsedSecondsForStep(latest, nowMs);
  if (elapsed === null) return latest.summary;
  return `${latest.summary} · ${elapsed.toFixed(1)}s`;
}

function runElapsedSeconds(steps: TraceStep[]): number | null {
  const timestamps = steps
    .flatMap((step) => [step.startedAt, step.completedAt])
    .filter((value): value is string => Boolean(value))
    .map((value) => Date.parse(value))
    .filter((value) => Number.isFinite(value));
  if (timestamps.length >= 2) {
    const started = Math.min(...timestamps);
    const completed = Math.max(...timestamps);
    if (completed >= started) {
      return Math.round((completed - started) / 100) / 10;
    }
  }
  const elapsedValues = steps
    .map((step) => step.elapsedSeconds)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  if (elapsedValues.length === 0) return null;
  return Math.round(elapsedValues.reduce((total, value) => total + value, 0) * 10) / 10;
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

type TodoItem = {
  content: string;
  task_type?: string;
  status: 'pending' | 'in_progress' | 'completed' | string;
  priority?: number;
};

function TodoList({ todos }: { todos: TodoItem[] }) {
  const [open, setOpen] = useState(true);
  const completed = todos.filter((todo) => todo.status === 'completed').length;
  const active = todos.find((todo) => todo.status === 'in_progress');

  return (
    <section className={`todo-panel ${open ? 'open' : ''}`}>
      <button className="todo-header" type="button" onClick={() => setOpen((value) => !value)}>
        <span className="todo-title">
          <ListChecks size={16} />
          <span>Todo list</span>
        </span>
        <span className="todo-meta">
          {completed}/{todos.length}
          {active ? ` · ${active.content}` : ''}
        </span>
        <ChevronDown className="todo-chevron" size={16} />
      </button>
      {open && (
        <ol className="todo-items">
          {todos.map((todo, index) => (
            <li key={`${todo.priority || index}-${todo.content}`} className={`todo-item ${todo.status}`}>
              <span className="todo-check" aria-hidden="true" />
              <span className="todo-copy">
                <strong>{todo.content}</strong>
                <small>{formatTodoStatus(todo.status, todo.task_type)}</small>
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function latestTodoList(steps: TraceStep[]): TodoItem[] {
  const hasTerminalError = steps.some((step) => step.status === 'error');
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const preview = asRecord(steps[index].toolResult?.payload_preview);
    const todos = preview && Array.isArray(preview.todos) ? preview.todos : null;
    if (!todos) continue;
    return todos
      .filter((todo): todo is Record<string, unknown> => Boolean(todo && typeof todo === 'object'))
      .map((todo) => ({
        content: typeof todo.content === 'string' ? todo.content : 'Untitled todo',
        task_type: typeof todo.task_type === 'string' ? todo.task_type : undefined,
        status: statusForTodo(todo, hasTerminalError),
        priority: typeof todo.priority === 'number' ? todo.priority : undefined,
      }));
  }
  return [];
}

function statusForTodo(todo: Record<string, unknown>, hasTerminalError: boolean) {
  const status = typeof todo.status === 'string' ? todo.status : 'pending';
  return hasTerminalError && status === 'in_progress' ? 'attention' : status;
}

function formatTodoStatus(status: string, taskType?: string) {
  const label = status === 'in_progress'
    ? 'In progress'
    : status === 'completed'
      ? 'Completed'
      : status === 'attention'
        ? 'Needs attention'
        : 'Pending';
  return taskType ? `${label} · ${taskType}` : label;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? value as Record<string, unknown> : null;
}
