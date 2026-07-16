import type { ChatMessage, Conversation, FinalAnswer, TraceStep } from '../types';

const STORAGE_KEY = 'tspilot:v02:conversations';

const now = () => new Date().toISOString();
const makeId = (prefix: string) => `${prefix}_${crypto.randomUUID?.() || Math.random().toString(36).slice(2)}`;

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
  };
}

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [createConversation()];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [createConversation()];
    const conversations = parsed.filter((item): item is Conversation => Boolean(item?.id));
    return conversations.length ? conversations : [createConversation()];
  } catch {
    return [createConversation()];
  }
}

export function saveConversations(conversations: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
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

export function appendAssistantAnswer(conversation: Conversation, answer: FinalAnswer): Conversation {
  const timestamp = now();
  const content = answer.summary || answer.title || 'Answer generated.';
  const streamingIndex = findStreamingAssistantIndex(conversation);
  if (streamingIndex >= 0) {
    const messages = [...conversation.messages];
    messages[streamingIndex] = {
      ...messages[streamingIndex],
      content,
      answer,
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
        createdAt: timestamp,
      },
    ],
    updatedAt: timestamp,
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
    traceSteps[existingIndex] = { ...traceSteps[existingIndex], ...step };
  } else {
    traceSteps.push(step);
  }
  return {
    ...conversation,
    traceSteps,
    updatedAt: now(),
  };
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
