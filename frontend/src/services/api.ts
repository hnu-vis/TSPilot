import type {
  DatabaseConfigInput,
  DatabaseConnectionTest,
  DatabasePreviewResponse,
  DatabaseResource,
  FinalAnswer,
  KnowledgeResource,
  StreamEvent,
} from '../types';

const API_BASE = '/api/v1';

export async function fetchDatabases(): Promise<DatabaseResource[]> {
  const response = await fetch(`${API_BASE}/resources/databases`);
  if (!response.ok) throw new Error(`Failed to load databases: ${response.status}`);
  const payload = await response.json();
  return payload.databases || [];
}

export async function fetchKnowledge(): Promise<KnowledgeResource[]> {
  const response = await fetch(`${API_BASE}/resources/knowledge`);
  if (!response.ok) throw new Error(`Failed to load knowledge resources: ${response.status}`);
  const payload = await response.json();
  return payload.knowledge || [];
}

export async function fetchModel(): Promise<string> {
  const response = await fetch(`${API_BASE}/resources/model`);
  if (!response.ok) return 'backend model';
  const payload = await response.json();
  return payload.model || 'backend model';
}

export async function fetchDatabasePreview(databaseId: string): Promise<DatabasePreviewResponse> {
  const response = await fetch(`${API_BASE}/resources/databases/${encodeURIComponent(databaseId)}/preview`);
  if (!response.ok) throw new Error(`Failed to preview database: ${response.status}`);
  return response.json();
}

export async function createDatabase(config: DatabaseConfigInput): Promise<DatabaseResource> {
  const response = await fetch(`${API_BASE}/resources/databases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error(`Failed to create database: ${response.status}`);
  const payload = await response.json();
  return payload.database;
}

export async function updateDatabase(databaseId: string, config: Partial<DatabaseConfigInput>): Promise<DatabaseResource> {
  const response = await fetch(`${API_BASE}/resources/databases/${encodeURIComponent(databaseId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error(`Failed to update database: ${response.status}`);
  const payload = await response.json();
  return payload.database;
}

export async function deleteDatabase(databaseId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/resources/databases/${encodeURIComponent(databaseId)}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error(`Failed to delete database: ${response.status}`);
}

export async function testDatabaseConnection(databaseId: string): Promise<DatabaseConnectionTest> {
  const response = await fetch(`${API_BASE}/resources/databases/${encodeURIComponent(databaseId)}/test`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error(`Failed to test database: ${response.status}`);
  return response.json();
}

export type ChatStreamRequest = {
  message: string;
  conversationId: string;
  database?: DatabaseResource | null;
  history: Array<{ role: string; content: string; timestamp?: string }>;
};

export async function streamChat(
  request: ChatStreamRequest,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      message: request.message,
      conversation_id: request.conversationId,
      stream: true,
      history: request.history,
      database_context: request.database
        ? {
            database_id: request.database.id,
            database_type: request.database.type,
            display_name: request.database.display_name || request.database.name,
          }
        : null,
    }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }
  if (!response.body) {
    throw new Error('Chat stream is empty.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split('\n\n');
    buffer = frames.pop() || '';
    for (const frame of frames) {
      const event = parseSseFrame(frame);
      if (event) onEvent(event);
    }
  }

  const trailing = `${buffer}${decoder.decode()}`;
  const event = parseSseFrame(trailing);
  if (event) onEvent(event);
}

function parseSseFrame(frame: string): StreamEvent | null {
  const lines = frame.split('\n');
  let event = '';
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart());
    }
  }
  if (!event || dataLines.length === 0) return null;
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) };
  } catch {
    return {
      event: 'error',
      data: { message: `Unable to parse ${event} event.` },
    };
  }
}

export function extractFinalAnswer(data: Record<string, unknown>): FinalAnswer | null {
  const answer = data.answer;
  if (answer && typeof answer === 'object') {
    return answer as FinalAnswer;
  }
  return null;
}
