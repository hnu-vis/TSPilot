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
