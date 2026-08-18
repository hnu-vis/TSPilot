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
  claims?: AnswerClaim[];
  visualizations?: Visualization[];
};

export type AnswerClaim = {
  claim_id: string;
  text: string;
  insight_ids?: string[];
  item_ids?: string[];
  analysis_ids?: string[];
  artifact_type?: string | null;
  artifact_ids?: string[];
  evidence_ids?: string[];
  visualization_ids?: string[];
};

export type VisualizationBinding = {
  binding_id: string;
  source_type: string;
  insight_id?: string | null;
  item_id?: string | null;
  related_item_ids?: string[];
  evidence_id?: string | null;
  source_ref?: string | null;
  locator?: Record<string, unknown>;
};

export type VisualizationPoint = {
  x?: unknown;
  y?: number | null;
  lower?: number | null;
  upper?: number | null;
  label?: string | null;
  binding_id?: string | null;
  metadata?: Record<string, unknown>;
};

export type VisualizationSeries = {
  series_id: string;
  name: string;
  role: string;
  unit?: string | null;
  points?: VisualizationPoint[];
};

export type Visualization = {
  schema_version: '3';
  visualization_id: string;
  data_ref?: string | null;
  purpose: string;
  priority: 'primary' | 'supporting';
  title: string;
  summary?: string | null;
  source_refs?: string[];
  required_roles?: string[];
  datasets: Array<{
    dataset_id: string;
    source_ref: string;
    data_ref?: string | null;
    row_count?: number | null;
    time_range?: { start?: unknown; end?: unknown } | null;
    dimensions?: Array<{ name: string; data_type: string; role: string; unit?: string | null }>;
    series?: VisualizationSeries[];
  }>;
  layers?: Array<{
    layer_id: string;
    mark: string;
    role: string;
    source_ref: string;
    encoding?: Record<string, string | string[]>;
    transform?: Array<{ type: 'filter'; field: string; operator: string; value?: unknown }>;
    presentation?: Record<string, unknown>;
    dataset_id: string;
    series_id?: string | null;
    points?: VisualizationPoint[];
    label?: string | null;
  }>;
  bindings?: VisualizationBinding[];
  layout?: 'overlay' | 'facets';
  presentation?: Record<string, unknown>;
  accessibility: {
    description: string;
    table_columns?: string[];
    table_rows?: Array<Record<string, unknown>>;
  };
};

export type InsightStatus = 'verified' | 'unavailable' | 'rejected' | 'partial' | string;

export type InsightEvidenceRef = {
  source_type: string;
  source_id: string;
  label?: string | null;
  locator?: Record<string, unknown>;
};

export type KeyInsight = {
  insight_id: string;
  insight_key?: string;
  name: string;
  insight_type: string;
  statement: string;
  value?: unknown;
  unit?: string | null;
  subject?: string | null;
  dimensions?: Record<string, unknown>;
  time_range?: Record<string, unknown> | null;
  method: string;
  evidence_refs?: InsightEvidenceRef[];
  calculation_trace?: Record<string, unknown>;
  status: InsightStatus;
  confidence?: number | null;
  quality_flags?: string[];
  unavailable_reason?: string | null;
  derived_from?: string[];
};

export type KeyInsightRequest = {
  insight_key?: string;
  name: string;
  insight_type: string;
  subject?: string | null;
  time_range?: Record<string, unknown> | null;
  dimensions?: Record<string, unknown>;
  requirements?: Record<string, unknown>;
  derived_from?: string[];
};

export type InsightCoverage = {
  requested?: string[];
  verified?: string[];
  missing?: string[];
  unavailable?: string[];
  rejected?: string[];
  partial?: string[];
};

export type MemoryCard = {
  id: string;
  kind: string;
  title: string;
  description: string;
  tags?: string[];
  updated_at?: string | null;
};

export type MemoryDetail = {
  id: string;
  card: MemoryCard;
  insight_request?: Record<string, unknown> | null;
  preferred_tool?: string | null;
  guidance?: string | null;
  examples?: string[];
};

export type InsightMemory = {
  cards: MemoryCard[];
  storage_path?: string | null;
  updated_at?: string | null;
};

export type InsightMemoryResponse = {
  database?: DatabaseResource;
  memory: InsightMemory;
  prompt_view?: Record<string, unknown>;
};

export type InsightMemoryDetailResponse = {
  database?: DatabaseResource;
  detail: MemoryDetail;
};

export type InsightMemoryLearningSettings = {
  max_wait_seconds: number;
  enabled: boolean;
  batch_size: number;
};

export type InsightMemoryLearningSettingsResponse = {
  settings: InsightMemoryLearningSettings;
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
  startedAt?: string;
  completedAt?: string;
  elapsedSeconds?: number;
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
  username?: string | null;
  ssl_enabled?: boolean;
  insight_memory_summary?: {
    definition_count: number;
    recipe_count: number;
    card_count: number;
    updated_at?: string | null;
  };
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
  preview_kind: 'schema' | 'metrics' | 'error' | string;
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
  selectedModelId: string | null;
};

export type ResourceState = {
  databases: DatabaseResource[];
  knowledge: KnowledgeResource[];
  model: string;
  models: AIModelEndpointConfig[];
};

export type AIModelEndpointConfig = {
  id: string;
  provider: string;
  api_base: string;
  model: string;
  api_key_configured: boolean;
  is_active: boolean;
  source: 'environment' | 'workspace';
  config_path?: string | null;
};

export type AIModelEndpointGroup = {
  active_id: string;
  models: AIModelEndpointConfig[];
};

export type ModelsConfig = {
  ai: {
    llm: AIModelEndpointGroup;
    embedding: AIModelEndpointGroup;
  };
  machine_learning: {
    forecast_model: string;
    anomaly_detector: string;
    forecast_options: string[];
    anomaly_options: string[];
    forecast_models: MachineModelConfig[];
    anomaly_models: MachineModelConfig[];
  };
  saved_id?: string;
};

export type MachineModelConfig = {
  id: string;
  name: string;
  source: 'built_in' | 'api';
  endpoint?: string | null;
  timeout_seconds?: number | null;
  api_key_configured: boolean;
  is_active: boolean;
  config_path?: string | null;
};

export type ExternalMachineModelInput = {
  name: string;
  endpoint: string;
  api_key?: string;
  timeout_seconds: number;
};

export type AIModelConfigInput = {
  id?: string;
  api_base: string;
  model: string;
  api_key?: string;
};

export type ModelConnectionTest = {
  success: boolean;
  latency_ms: number;
  message: string;
};

export type StreamEvent = {
  event: string;
  data: Record<string, unknown>;
};
