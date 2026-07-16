import { ChevronDown, ListChecks, Bot, Loader2, User } from 'lucide-react';
import { useMemo, useState } from 'react';
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
  if (messages.length === 0 && traceSteps.length === 0) {
    return (
      <div className="empty-thread">
        <div className="empty-mark">TS</div>
        <h2>Ask about your time-series data</h2>
        <p>Query databases, inspect trends, detect anomalies, forecast future values, and review the execution process.</p>
      </div>
    );
  }

  const lastAssistantIndex = findLastAssistantIndex(messages);
  const latestTodos = useMemo(() => latestTodoList(traceSteps), [traceSteps]);

  return (
    <div className="chat-thread">
      {messages.map((message, index) => (
        <article key={message.id} className={`message ${message.role}`}>
          <div className="avatar">{message.role === 'user' ? <User size={16} /> : <Bot size={16} />}</div>
          <div className="bubble">
            {message.answer ? (
              <FinalAnswer answer={message.answer} />
            ) : message.isStreaming ? (
              <div className="live-answer">
                <div className="live-status">
                  <Loader2 className="spin" size={15} />
                  <span>{latestTraceSummary(traceSteps) || message.content}</span>
                </div>
              </div>
            ) : (
              <p>{message.content}</p>
            )}
          </div>
          {index === lastAssistantIndex && latestTodos.length > 0 && (
            <TodoList todos={latestTodos} />
          )}
          {index === lastAssistantIndex && traceSteps.length > 0 && (
            <TraceTimeline steps={traceSteps} selectedId={selectedTraceStepId} onSelect={onSelectTraceStep} />
          )}
        </article>
      ))}
    </div>
  );
}

function findLastAssistantIndex(messages: ChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'assistant') return index;
  }
  return -1;
}

function latestTraceSummary(steps: TraceStep[]) {
  const latest = steps[steps.length - 1];
  return latest?.summary || '';
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
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const preview = asRecord(steps[index].toolResult?.payload_preview);
    const todos = preview && Array.isArray(preview.todos) ? preview.todos : null;
    if (!todos) continue;
    return todos
      .filter((todo): todo is Record<string, unknown> => Boolean(todo && typeof todo === 'object'))
      .map((todo) => ({
        content: typeof todo.content === 'string' ? todo.content : 'Untitled todo',
        task_type: typeof todo.task_type === 'string' ? todo.task_type : undefined,
        status: typeof todo.status === 'string' ? todo.status : 'pending',
        priority: typeof todo.priority === 'number' ? todo.priority : undefined,
      }));
  }
  return [];
}

function formatTodoStatus(status: string, taskType?: string) {
  const label = status === 'in_progress' ? 'In progress' : status === 'completed' ? 'Completed' : 'Pending';
  return taskType ? `${label} · ${taskType}` : label;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? value as Record<string, unknown> : null;
}
