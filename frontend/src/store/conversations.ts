import type { ChatMessage, Conversation, FinalAnswer, TokenUsage, TraceStep } from '../types';

const STORAGE_KEY = 'tspilot:v03:conversations';
const MAX_STORED_CONVERSATIONS = 12;
const MAX_STORED_MESSAGES_PER_CONVERSATION = 30;
const MAX_STORED_TRACE_STEPS = 40;
const MAX_STORED_TEXT_CHARS = 12000;

const now = () => new Date().toISOString();
const makeId = (prefix: string) => {
  const randomUUID = globalThis.crypto?.randomUUID;
  return `${prefix}_${randomUUID ? randomUUID.call(globalThis.crypto) : Math.random().toString(36).slice(2)}`;
};

function titleFromContent(content: string) {
  const normalized = content.replace(/\s+/g, ' ').trim();
  if (!normalized) return 'New chat';
  return normalized.length > 36 ? `${normalized.slice(0, 36)}...` : normalized;
}

export function createConversation(): Conversation {
  const timestamp = now();
  return {
    id: makeId('conv'),
    title: 'New chat',
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: [],
    traceSteps: [],
    selectedTraceStepId: null,
    selectedDatabaseId: null,
    selectedKnowledgeId: null,
    selectedModelId: null,
  };
}

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [createConversation()];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [createConversation()];
    const conversations = parsed
      .filter((item): item is Conversation => Boolean(item?.id))
      .map((item) => ({ ...item, selectedModelId: item.selectedModelId ?? null }));
    return conversations.length ? conversations : [createConversation()];
  } catch {
    return [createConversation()];
  }
}

export function saveConversations(conversations: Conversation[]) {
  const compact = conversations.map(compactConversationForStorage).slice(0, MAX_STORED_CONVERSATIONS);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
    return;
  } catch (error) {
    if (!isStorageQuotaError(error)) {
      console.warn('Unable to save conversations.', error);
      return;
    }
  }

  for (const limit of [6, 3, 1]) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(compact.slice(0, limit)));
      console.warn(`Conversation history was trimmed to the latest ${limit} item(s) because browser storage is full.`);
      return;
    } catch {
      // Try a smaller snapshot.
    }
  }
  try {
    localStorage.removeItem(STORAGE_KEY);
    console.warn('Conversation history was cleared because browser storage is full.');
  } catch {
    // Nothing else to do; saving must never crash the app.
  }
}

export function sortConversations(conversations: Conversation[]) {
  return [...conversations].sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
}

export function appendUserMessage(conversation: Conversation, content: string): Conversation {
  const timestamp = now();
  const message: ChatMessage = {
    id: makeId('msg'),
    role: 'user',
    content,
    createdAt: timestamp,
  };
  return {
    ...conversation,
    title: conversation.messages.length === 0 ? titleFromContent(content) : conversation.title,
    messages: [...conversation.messages, message],
    selectedTraceStepId: null,
    updatedAt: timestamp,
  };
}

export function appendAssistantPending(conversation: Conversation): Conversation {
  const timestamp = now();
  return {
    ...conversation,
    messages: [
      ...conversation.messages,
      {
        id: makeId('msg'),
        role: 'assistant',
        content: '正在分析请求...',
        createdAt: timestamp,
        isStreaming: true,
      },
    ],
    updatedAt: timestamp,
  };
}

export function appendAssistantAnswer(conversation: Conversation, answer: FinalAnswer, tokenUsage?: TokenUsage | null): Conversation {
  const timestamp = now();
  const content = answer.summary || answer.title || 'Answer generated.';
  const streamingIndex = findStreamingAssistantIndex(conversation);
  if (streamingIndex >= 0) {
    const messages = [...conversation.messages];
    messages[streamingIndex] = {
      ...messages[streamingIndex],
      content,
      answer,
      tokenUsage,
      isStreaming: false,
      createdAt: messages[streamingIndex].createdAt,
    };
    return {
      ...conversation,
      messages,
      updatedAt: timestamp,
    };
  }
  return {
    ...conversation,
    messages: [
      ...conversation.messages,
      {
        id: makeId('msg'),
        role: 'assistant',
        content,
        answer,
        tokenUsage,
        createdAt: timestamp,
      },
    ],
    updatedAt: timestamp,
  };
}

export function completeRunningTraceSteps(conversation: Conversation, completedAt = now()): Conversation {
  let changed = false;
  const traceSteps = conversation.traceSteps.map((step) => {
    if (step.status !== 'running') return step;
    changed = true;
    return {
      ...step,
      status: 'complete' as const,
      completedAt,
      elapsedSeconds: step.elapsedSeconds ?? elapsedSecondsBetween(step.startedAt, completedAt),
      updatedAt: completedAt,
    };
  });
  if (!changed) return conversation;
  return {
    ...conversation,
    traceSteps,
    updatedAt: completedAt,
  };
}

export function appendAssistantError(conversation: Conversation, content: string): Conversation {
  const timestamp = now();
  const streamingIndex = findStreamingAssistantIndex(conversation);
  if (streamingIndex >= 0) {
    const messages = [...conversation.messages];
    messages[streamingIndex] = {
      ...messages[streamingIndex],
      content,
      isStreaming: false,
    };
    return {
      ...conversation,
      messages,
      updatedAt: timestamp,
    };
  }
  return {
    ...conversation,
    messages: [
      ...conversation.messages,
      {
        id: makeId('msg'),
        role: 'assistant',
        content,
        createdAt: timestamp,
      },
    ],
    updatedAt: timestamp,
  };
}

export function upsertTraceStep(conversation: Conversation, step: TraceStep): Conversation {
  const existingIndex = conversation.traceSteps.findIndex((item) => item.id === step.id);
  const traceSteps = [...conversation.traceSteps];
  if (existingIndex >= 0) {
    traceSteps[existingIndex] = mergeTraceStep(traceSteps[existingIndex], step);
  } else {
    traceSteps.push(step);
  }
  return {
    ...conversation,
    traceSteps,
    updatedAt: now(),
  };
}

function mergeTraceStep(existing: TraceStep, incoming: TraceStep): TraceStep {
  const startedAt = existing.startedAt ?? keepValue(incoming.startedAt, existing.startedAt);
  const completedAt = keepValue(incoming.completedAt, existing.completedAt);
  const elapsedSeconds = incoming.elapsedSeconds
    ?? elapsedSecondsBetween(startedAt, incoming.completedAt ? completedAt : undefined)
    ?? existing.elapsedSeconds
    ?? elapsedSecondsBetween(startedAt, completedAt);
  return {
    ...existing,
    ...incoming,
    tool: keepValue(incoming.tool, existing.tool),
    thought: keepValue(incoming.thought, existing.thought),
    actionInput: keepValue(incoming.actionInput, existing.actionInput),
    observation: keepValue(incoming.observation, existing.observation),
    toolCall: keepValue(incoming.toolCall, existing.toolCall),
    toolResult: keepValue(incoming.toolResult, existing.toolResult),
    startedAt,
    completedAt,
    elapsedSeconds,
  };
}

function keepValue<T>(incoming: T | null | undefined, existing: T | undefined): T | undefined {
  if (incoming === null || incoming === undefined || incoming === '') return existing;
  return incoming;
}

function elapsedSecondsBetween(startedAt?: string, completedAt?: string): number | undefined {
  if (!startedAt || !completedAt) return undefined;
  const started = Date.parse(startedAt);
  const completed = Date.parse(completedAt);
  if (!Number.isFinite(started) || !Number.isFinite(completed) || completed < started) return undefined;
  return Math.round((completed - started) / 100) / 10;
}

function compactConversationForStorage(conversation: Conversation): Conversation {
  return {
    ...conversation,
    messages: conversation.messages
      .slice(-MAX_STORED_MESSAGES_PER_CONVERSATION)
      .map(compactMessageForStorage),
    traceSteps: conversation.traceSteps
      .slice(-MAX_STORED_TRACE_STEPS)
      .map(compactTraceStepForStorage),
  };
}

function compactMessageForStorage(message: ChatMessage): ChatMessage {
  return {
    ...message,
    content: truncateString(message.content),
    answer: message.answer ? compactUnknown(message.answer) as FinalAnswer : undefined,
    tokenUsage: message.tokenUsage ? compactUnknown(message.tokenUsage) as TokenUsage : undefined,
  };
}

function compactTraceStepForStorage(step: TraceStep): TraceStep {
  const toolCall = compactToolCall(step.toolCall);
  const toolResult = compactToolResult(step.toolResult);
  return {
    ...step,
    summary: truncateString(step.summary),
    thought: truncateOptionalString(step.thought),
    actionInput: compactUnknown(step.actionInput) as TraceStep['actionInput'],
    observation: compactUnknown(step.observation) as TraceStep['observation'],
    toolCall,
    toolResult,
  };
}

function compactToolCall(toolCall: TraceStep['toolCall']): TraceStep['toolCall'] {
  if (!toolCall) return toolCall;
  return compactUnknown({
    ...toolCall,
    action_input: undefined,
    input_preview: compactUnknown((toolCall as Record<string, unknown>).input_preview),
  }) as TraceStep['toolCall'];
}

function compactToolResult(toolResult: TraceStep['toolResult']): TraceStep['toolResult'] {
  if (!toolResult) return toolResult;
  return compactUnknown({
    ...toolResult,
    observation: undefined,
    payload: undefined,
    payload_preview: compactUnknown((toolResult as Record<string, unknown>).payload_preview),
  }) as TraceStep['toolResult'];
}

function compactUnknown(value: unknown): unknown {
  if (typeof value === 'string') return truncateString(value);
  if (Array.isArray(value)) return value.slice(0, 24).map(compactUnknown);
  if (!value || typeof value !== 'object') return value;
  const result: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (child === undefined) continue;
    result[key] = compactUnknown(child);
  }
  return result;
}

function truncateOptionalString(value: string | undefined): string | undefined {
  return typeof value === 'string' ? truncateString(value) : value;
}

function truncateString(value: string): string {
  if (value.length <= MAX_STORED_TEXT_CHARS) return value;
  return `${value.slice(0, MAX_STORED_TEXT_CHARS)}... [truncated ${value.length - MAX_STORED_TEXT_CHARS} chars]`;
}

function isStorageQuotaError(error: unknown): boolean {
  return (
    error instanceof DOMException
    && (
      error.name === 'QuotaExceededError'
      || error.name === 'NS_ERROR_DOM_QUOTA_REACHED'
      || error.code === 22
      || error.code === 1014
    )
  );
}

export function buildBackendHistory(conversation: Conversation) {
  return conversation.messages
    .filter((message) => !message.isStreaming && (message.role === 'user' || message.role === 'assistant'))
    .slice(-12)
    .map((message) => ({
      role: message.role,
      content: message.answer?.summary || message.content,
      timestamp: message.createdAt,
    }));
}

function findStreamingAssistantIndex(conversation: Conversation) {
  for (let index = conversation.messages.length - 1; index >= 0; index -= 1) {
    const message = conversation.messages[index];
    if (message.role === 'assistant' && message.isStreaming) return index;
  }
  return -1;
}
