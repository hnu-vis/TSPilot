import type {
  DatabaseConfigInput,
  DatabaseConnectionTest,
  DatabasePreviewResponse,
  DatabaseResource,
  InsightMemoryDetailResponse,
  InsightMemoryLearningSettingsResponse,
  InsightMemoryResponse,
  FinalAnswer,
  KnowledgeResource,
  AIModelConfigInput,
  ModelConnectionTest,
  ModelsConfig,
  ExternalMachineModelInput,
  StreamEvent,
  Visualization,
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

export async function fetchModelsConfig(): Promise<ModelsConfig> {
  const response = await fetch(`${API_BASE}/resources/models/config`);
  if (!response.ok) throw new Error(`Failed to load model configuration: ${response.status}`);
  return response.json();
}

export async function updateAIModelConfig(section: 'llm' | 'embedding', config: AIModelConfigInput): Promise<ModelsConfig> {
  const response = await fetch(`${API_BASE}/resources/models/ai/${section}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error(await responseError(response, 'Failed to save model configuration'));
  return response.json();
}

export async function activateAIModelConfig(section: 'llm' | 'embedding', connectionId: string): Promise<ModelsConfig> {
  const response = await fetch(`${API_BASE}/resources/models/ai/${section}/${encodeURIComponent(connectionId)}/activate`, { method: 'PATCH' });
  if (!response.ok) throw new Error(await responseError(response, 'Failed to activate model'));
  return response.json();
}

export async function deleteAIModelConfig(section: 'llm' | 'embedding', connectionId: string): Promise<ModelsConfig> {
  const response = await fetch(`${API_BASE}/resources/models/ai/${section}/${encodeURIComponent(connectionId)}`, { method: 'DELETE' });
  if (!response.ok) throw new Error(await responseError(response, 'Failed to remove model'));
  return response.json();
}

export async function updateMachineLearningConfig(config: { forecast_model: string; anomaly_detector: string }): Promise<ModelsConfig> {
  const response = await fetch(`${API_BASE}/resources/models/machine-learning`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error(await responseError(response, 'Failed to save machine learning configuration'));
  return response.json();
}

export async function updateExternalMachineModel(task: 'forecast' | 'anomaly', config: ExternalMachineModelInput): Promise<ModelsConfig> {
  const response = await fetch(`${API_BASE}/resources/models/machine-learning/external/${task}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error(await responseError(response, 'Failed to save external model'));
  return response.json();
}

export async function activateMachineModel(task: 'forecast' | 'anomaly', name: string): Promise<ModelsConfig> {
  const response = await fetch(`${API_BASE}/resources/models/machine-learning/${task}/${encodeURIComponent(name)}/activate`, { method: 'PATCH' });
  if (!response.ok) throw new Error(await responseError(response, 'Failed to activate machine learning model'));
  return response.json();
}

export async function deleteExternalMachineModel(task: 'forecast' | 'anomaly', name: string): Promise<ModelsConfig> {
  const response = await fetch(`${API_BASE}/resources/models/machine-learning/external/${task}/${encodeURIComponent(name)}`, { method: 'DELETE' });
  if (!response.ok) throw new Error(await responseError(response, 'Failed to remove external model'));
  return response.json();
}

export async function testExternalMachineModel(input: ExternalMachineModelInput & { task: 'forecast' | 'anomaly' }): Promise<ModelConnectionTest> {
  const response = await fetch(`${API_BASE}/resources/models/machine-learning/test`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input),
  });
  if (!response.ok) throw new Error(await responseError(response, 'Failed to test external model'));
  return response.json();
}

export async function testModelConnection(input: {
  kind: 'llm' | 'embedding';
  connection_id?: string;
  api_base: string;
  model: string;
  api_key?: string;
}): Promise<ModelConnectionTest> {
  const response = await fetch(`${API_BASE}/resources/models/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new Error(await responseError(response, 'Failed to test model connection'));
  return response.json();
}

export async function fetchDatabasePreview(databaseId: string, options?: { refresh?: boolean }): Promise<DatabasePreviewResponse> {
  const search = options?.refresh ? '?refresh=true' : '';
  const response = await fetch(`${API_BASE}/resources/databases/${encodeURIComponent(databaseId)}/preview${search}`);
  if (!response.ok) throw new Error(`Failed to preview database: ${response.status}`);
  return response.json();
}

export async function fetchInsightMemory(databaseId?: string | null): Promise<InsightMemoryResponse> {
  const path = databaseId
    ? `/resources/databases/${encodeURIComponent(databaseId)}/insight-memory`
    : '/resources/insight-memory';
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`Failed to load key insight memory: ${response.status}`);
  return response.json();
}

export async function fetchInsightMemoryDetail(memoryId: string, databaseId?: string | null): Promise<InsightMemoryDetailResponse> {
  const encodedId = encodeURIComponent(memoryId);
  const path = databaseId
    ? `/resources/databases/${encodeURIComponent(databaseId)}/insight-memory/${encodedId}`
    : `/resources/insight-memory/${encodedId}`;
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`Failed to load key insight memory detail: ${response.status}`);
  return response.json();
}

export async function fetchInsightMemoryLearningSettings(): Promise<InsightMemoryLearningSettingsResponse> {
  const response = await fetch(`${API_BASE}/resources/insight-memory-learning-settings`);
  if (!response.ok) throw new Error(`Failed to load Insight learning settings: ${response.status}`);
  return response.json();
}

export async function updateInsightMemoryLearningSettings(maxWaitSeconds: number): Promise<InsightMemoryLearningSettingsResponse> {
  const response = await fetch(`${API_BASE}/resources/insight-memory-learning-settings`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ max_wait_seconds: maxWaitSeconds }),
  });
  if (!response.ok) throw new Error(`Failed to update Insight learning settings: ${response.status}`);
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
  modelId?: string | null;
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
      model_id: request.modelId || null,
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

export async function fetchVisualizationData(dataRef: string): Promise<Visualization> {
  const response = await fetch(dataRef);
  if (!response.ok) throw new Error(`Failed to load visualization data: ${response.status}`);
  return response.json();
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

async function responseError(response: Response, fallback: string) {
  try {
    const payload = await response.json();
    return typeof payload.detail === 'string' ? payload.detail : `${fallback}: ${response.status}`;
  } catch {
    return `${fallback}: ${response.status}`;
  }
}

export function extractFinalAnswer(data: Record<string, unknown>): FinalAnswer | null {
  const answer = data.answer;
  if (answer && typeof answer === 'object') {
    return answer as FinalAnswer;
  }
  return null;
}
