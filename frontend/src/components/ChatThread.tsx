import { ChevronDown, ListChecks, Bot, Loader2, User } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChatMessage, TraceStep } from '../types';
import { FinalAnswer } from './FinalAnswer';
import { TraceTimeline } from './TraceTimeline';
import { useI18n } from '../i18n';
import { elapsedSecondsForTrace, runElapsedSeconds } from '../lib/traceTiming';

type Props = {
  messages: ChatMessage[];
  traceSteps: TraceStep[];
  selectedTraceNodeId: string | null;
  onSelectTraceNode: (id: string) => void;
};

export function ChatThread({ messages, traceSteps, selectedTraceNodeId, onSelectTraceNode }: Props) {
  const { t } = useI18n();
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const hasStreamingMessage = useMemo(() => messages.some((message) => message.isStreaming), [messages]);
  const hasRunningStep = useMemo(() => traceSteps.some((step) => (
    step.status === 'running' || step.children?.some((child) => child.status === 'running')
  )), [traceSteps]);
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
      traceSteps[traceSteps.length - 1]?.children?.length,
      traceSteps[traceSteps.length - 1]?.children?.at(-1)?.status,
      traceSteps[traceSteps.length - 1]?.children?.at(-1)?.summary,
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
        <h2>{t('Ask about your time-series data')}</h2>
        <p>{t('Retrieve data, inspect trends, detect anomalies, forecast future values, and review the execution process.')}</p>
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
            <TraceTimeline steps={traceSteps} selectedId={selectedTraceNodeId} onSelect={onSelectTraceNode} nowMs={nowMs} />
          )}
          <div className="bubble">
            {message.answer ? (
              <FinalAnswer answer={message.answer} tokenUsage={message.tokenUsage} elapsedSeconds={runElapsedSeconds(traceSteps, nowMs)} />
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
            <TraceTimeline steps={traceSteps} selectedId={selectedTraceNodeId} onSelect={onSelectTraceNode} nowMs={nowMs} />
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
  for (let stepIndex = steps.length - 1; stepIndex >= 0; stepIndex -= 1) {
    const runningCall = [...(steps[stepIndex].children || [])]
      .reverse()
      .find((call) => call.status === 'running');
    if (runningCall) {
      const elapsed = elapsedSecondsForTrace(runningCall, nowMs);
      const summary = runningCall.summary || runningCall.title;
      return elapsed === null ? summary : `${summary} · ${elapsed.toFixed(1)}s`;
    }
  }
  const latestRunning = [...steps].reverse().find((step) => step.status === 'running');
  const latest = latestRunning || steps[steps.length - 1];
  if (!latest?.summary) return '';
  const elapsed = elapsedSecondsForTrace(latest, nowMs);
  if (elapsed === null) return latest.summary;
  return `${latest.summary} · ${elapsed.toFixed(1)}s`;
}

type TodoItem = {
  content: string;
  task_type?: string;
  status: 'pending' | 'in_progress' | 'completed' | string;
  priority?: number;
};

function TodoList({ todos }: { todos: TodoItem[] }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(true);
  const completed = todos.filter((todo) => todo.status === 'completed').length;
  const active = todos.find((todo) => todo.status === 'in_progress');

  return (
    <section className={`todo-panel ${open ? 'open' : ''}`}>
      <button className="todo-header" type="button" onClick={() => setOpen((value) => !value)}>
        <span className="todo-title">
          <ListChecks size={16} />
          <span>{t('Todo list')}</span>
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
                <small>{formatTodoStatus(todo.status, todo.task_type, t)}</small>
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
    const todos = preview && Array.isArray(preview.todos)
      ? preview.todos
      : preview && Array.isArray(preview.todos_preview)
        ? preview.todos_preview
        : null;
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

function formatTodoStatus(status: string, taskType: string | undefined, t: (key: string) => string) {
  const label = status === 'in_progress'
    ? 'In progress'
    : status === 'completed'
      ? 'Completed'
      : status === 'attention'
        ? 'Needs attention'
        : 'Pending';
  const localized = t(label);
  return taskType ? `${localized} · ${taskType}` : localized;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? value as Record<string, unknown> : null;
}
