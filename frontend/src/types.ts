export type Role = 'user' | 'assistant' | 'system';

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
  answer?: FinalAnswer;
  tokenUsage?: TokenUsage | null;
  isStreaming?: boolean;
};

export type TokenUsage = {
  totals?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    call_count?: number;
    counting_method?: string;
  };
  by_tool?: Record<string, {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    call_count?: number;
    counting_method?: string;
  }>;
  calls?: Array<Record<string, unknown>>;
};

export type FinalAnswer = {
  title?: string | null;
  summary: string;
  sections?: Array<{
    section_type: string;
    heading?: string | null;
    content: string;
    structured_payload?: Record<string, unknown> | null;
  }>;
  references?: Array<{
    source_type: string;
    source_id?: string | null;
    label: string;
    evidence?: Record<string, unknown> | null;
  }>;
  visualizations?: Array<Record<string, unknown>>;
};

export type TraceStatus = 'running' | 'complete' | 'error';

export type TraceStep = {
  id: string;
  iteration: number;
  agent: string;
  phase: string;
  status: TraceStatus;
  summary: string;
  tool?: string;
  thought?: string;
  actionInput?: Record<string, unknown>;
  observation?: Record<string, unknown>;
  toolCall?: Record<string, unknown>;
  toolResult?: Record<string, unknown>;
  error?: string;
  updatedAt: string;
};

export type DatabaseResource = {
  id: string;
  name: string;
  type: string;
  status?: string;
  host?: string | null;
  port?: number | null;
  database?: string | null;
  display_name?: string | null;
  config_source?: string | null;
  has_reference_dataset?: boolean;
  username?: string | null;
  ssl_enabled?: boolean;
};

export type DatabaseConfigInput = {
  name: string;
  type: string;
  host?: string | null;
  port?: number | null;
  database?: string | null;
  username?: string | null;
  password?: string | null;
  display_name?: string | null;
  ssl_enabled?: boolean | null;
};

export type DatabaseConnectionTest = {
  database: DatabaseResource;
  status: string;
  success: boolean;
  latency_ms?: number | null;
  version?: string | null;
  error?: string | null;
  profile_refresh?: Record<string, unknown> | null;
};

export type DatabasePreviewColumn = {
  name: string;
  data_type?: string | null;
  nullable?: boolean | null;
};

export type DatabasePreviewObject = {
  name: string;
  schema?: string | null;
  type?: string | null;
  row_count?: number | null;
  columns?: DatabasePreviewColumn[];
  field_values?: string[];
  sample_rows?: Array<Record<string, unknown>>;
};

export type DatabasePreviewPayload = {
  tables_or_measurements?: DatabasePreviewObject[];
  metrics?: DatabasePreviewObject[];
  fields?: Array<Record<string, unknown>>;
  labels_or_tags?: Array<Record<string, unknown>>;
  time_columns?: string[];
  metadata?: Record<string, unknown>;
};

export type DatabasePreviewResponse = {
  database: DatabaseResource;
  preview_kind: 'schema' | 'metrics' | 'reference_dataset' | 'error' | string;
  summary?: string;
  preview?: DatabasePreviewPayload;
  error?: string;
  profile_cache?: Record<string, unknown> | null;
};

export type KnowledgeResource = {
  id: string;
  name: string;
  type: string;
  status: string;
  root?: string;
  document_count?: number;
};

export type Conversation = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  traceSteps: TraceStep[];
  selectedTraceStepId: string | null;
  selectedDatabaseId: string | null;
  selectedKnowledgeId: string | null;
};

export type ResourceState = {
  databases: DatabaseResource[];
  knowledge: KnowledgeResource[];
  model: string;
};

export type StreamEvent = {
  event: string;
  data: Record<string, unknown>;
};
