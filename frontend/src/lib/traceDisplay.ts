import type { FinalAnswer, TraceStep } from '../types';

export type DisplayMetric = {
  label: string;
  value: string;
};

export type SqlDetail = {
  queryLanguage: string | null;
  query: string | null;
  columns: string[];
  sampleRows: Record<string, unknown>[];
  samplePoints: Record<string, unknown>[];
  rowCount: number | null;
  pointCount: number | null;
  truncated: boolean;
};

export type SchemaLinkingDetail = {
  confidence: string | null;
  sources: Array<{
    name: string;
    kind: string | null;
    timeColumn: string | null;
    valueColumns: string[];
    dimensionColumns: string[];
    linkedColumns: Array<{ name: string; role: string | null }>;
  }>;
  fieldMappings: Array<{
    sourceName: string | null;
    fieldName: string;
    role: string | null;
    confidence: number | null;
  }>;
  requiredFilters: Array<{
    column: string;
    operator: string;
    value: string;
  }>;
  evidence: string[];
  ambiguousTerms: Array<{ term: string; candidates: string[] }>;
};

export type CompletionDetail = {
  completed: boolean | null;
  reason: string | null;
  missingItems: string[];
  nextActionHint: string | null;
  todoProgress: {
    total: number;
    completed: number;
    inProgress: string | null;
  } | null;
};

export type CodeInterpreterDetail = {
  analysisGoal: string | null;
  code: string | null;
  codeHash: string | null;
  codeType: string | null;
  inputEvidenceId: string | null;
  inputRowCount: number | null;
  runtimeMs: number | null;
  summary: string | null;
  metrics: Record<string, unknown>;
  details: Record<string, unknown>;
  result: Record<string, unknown> | null;
  inputColumns: string[];
};

export type ForecastDetail = {
  status: string | null;
  modelName: string | null;
  horizon: number | null;
  plan: Record<string, unknown> | null;
  points: Record<string, unknown>[];
  pointCount: number | null;
};

export type AnomalyDetail = {
  detectorName: string | null;
  points: Record<string, unknown>[];
  scores: Record<string, unknown>[];
  pointCount: number | null;
  spanCount: number | null;
};

export type ReactDetail = {
  thought: string | null;
  action: string | null;
  actionInput: Record<string, unknown> | null;
  observation: Record<string, unknown> | null;
};

export type DisplayStep = {
  id: string;
  title: string;
  category: string;
  status: DisplayStatus;
  summary: string;
  metrics: DisplayMetric[];
  artifactRefs: string[];
  sqlDetail: SqlDetail | null;
  codeInterpreterDetail: CodeInterpreterDetail | null;
  forecastDetail: ForecastDetail | null;
  anomalyDetail: AnomalyDetail | null;
  schemaLinkingDetail: SchemaLinkingDetail | null;
  completionDetail: CompletionDetail | null;
  reactDetail: ReactDetail;
  hasPrimaryDetail: boolean;
  debugPayload: Record<string, unknown> | null;
};

export type DisplayStatus = TraceStep['status'] | 'attention';

export type RunOverview = {
  status: DisplayStatus | 'idle';
  completedSteps: number;
  totalSteps: number;
  metrics: DisplayMetric[];
  outputs: string[];
  references: string[];
};

export function toDisplayStep(step: TraceStep): DisplayStep {
  const result = asRecord(step.toolResult);
  const call = asRecord(step.toolCall);
  const preview = asRecord(result?.payload_preview);
  const tool = step.tool || stringFrom(call?.tool) || step.phase;
  const metrics = metricsForTool(tool, preview, call);
  const artifactRefs = artifactRefsFor(preview, result);
  const completionDetail = completionDetailFor(preview);
  const status = statusForStep(step, completionDetail);
  const reactDetail = reactDetailFor(step, call, result);
  const sqlDetail = sqlDetailFor(tool, preview);
  const codeInterpreterDetail = codeInterpreterDetailFor(tool, preview, call, step);
  const forecastDetail = forecastDetailFor(tool, preview);
  const anomalyDetail = anomalyDetailFor(tool, preview);
  const schemaLinkingDetail = schemaLinkingDetailFor(tool, preview);
  const hasPrimaryDetail = Boolean(
    sqlDetail ||
    codeInterpreterDetail ||
    forecastDetail ||
    anomalyDetail ||
    schemaLinkingDetail ||
    completionDetail,
  );

  return {
    id: step.id,
    title: titleForTool(tool, step.phase),
    category: categoryForTool(tool, step.phase),
    status,
    summary: summaryForStep(step, preview),
    metrics,
    artifactRefs,
    sqlDetail,
    codeInterpreterDetail,
    forecastDetail,
    anomalyDetail,
    schemaLinkingDetail,
    completionDetail,
    reactDetail,
    hasPrimaryDetail,
    debugPayload: result || call ? { toolCall: call, toolResult: result } : null,
  };
}

function reactDetailFor(
  step: TraceStep,
  call: Record<string, unknown> | null,
  result: Record<string, unknown> | null,
): ReactDetail {
  return {
    thought: step.thought || stringFrom(call?.thought) || (step.phase === 'reasoning' ? step.summary : null),
    action: step.tool || stringFrom(call?.tool) || null,
    actionInput: asRecord(step.actionInput) || asRecord(call?.action_input) || asRecord(call?.input_preview),
    observation: asRecord(step.observation) || asRecord(result?.observation) || result,
  };
}

function statusForStep(step: TraceStep, completionDetail: CompletionDetail | null): DisplayStatus {
  if (step.status === 'running') return 'running';
  if (step.status === 'error') return 'error';
  if (completionDetail?.completed === false) return 'attention';
  return step.status;
}

export function buildRunOverview(steps: TraceStep[], answer?: FinalAnswer | null): RunOverview {
  const displaySteps = steps.map(toDisplayStep);
  const totalSteps = displaySteps.length;
  const completedSteps = displaySteps.filter((step) => step.status === 'complete').length;
  const hasError = displaySteps.some((step) => step.status === 'error');
  const hasRunning = displaySteps.some((step) => step.status === 'running');
  const status = totalSteps === 0 ? 'idle' : hasError ? 'error' : hasRunning ? 'running' : 'complete';
  const metrics = compactMetrics([
    { label: 'Steps', value: totalSteps ? `${completedSteps}/${totalSteps}` : '0' },
    firstMetric(displaySteps, 'Rows'),
    firstMetric(displaySteps, 'Points'),
    firstMetric(displaySteps, 'Series'),
    firstMetric(displaySteps, 'Facts'),
    firstMetric(displaySteps, 'Anomalies'),
    firstMetric(displaySteps, 'Forecast points'),
  ]);
  const outputs = outputTypes(answer);
  const references = (answer?.references || [])
    .map((reference) => reference.source_id || reference.label)
    .filter((value): value is string => Boolean(value));

  return {
    status,
    completedSteps,
    totalSteps,
    metrics,
    outputs,
    references,
  };
}

export function titleForTool(tool?: string, phase?: string) {
  if (tool === 'todowrite') return 'Plan the work';
  if (tool === 'sql_query' || tool === 'query_database') return 'Data retrieval';
  if (tool === 'anomaly') return 'Check anomalies';
  if (tool === 'forecast') return 'Forecast trend';
  if (tool === 'rag') return 'Retrieve knowledge';
  if (tool === 'skill') return 'Run workflow';
  if (phase === 'answer_assembly') return 'Assemble answer';
  if (phase === 'analysis') return 'Analyze evidence';
  if (phase === 'tool_selection') return 'Data retrieval';
  if (phase === 'intent') return 'Plan the work';
  return 'Process step';
}

function categoryForTool(tool?: string, phase?: string) {
  if (tool === 'todowrite' || phase === 'intent') return 'Plan';
  if (tool === 'sql_query' || tool === 'query_database' || phase === 'tool_selection') return 'Data';
  if (phase === 'answer_assembly') return 'Answer';
  if (tool === 'rag' || tool === 'skill') return 'Context';
  return 'Analysis';
}

function summaryForStep(step: TraceStep, preview: Record<string, unknown> | null) {
  const completionDetail = completionDetailFor(preview);
  if (completionDetail?.completed === false) {
    return completionDetail.reason || completionDetail.nextActionHint || 'Needs more evidence.';
  }
  const tool = step.tool;
  if ((tool === 'sql_query' || tool === 'query_database') && preview) {
    const rowCount = numberFrom(preview.row_count) ?? nestedNumber(preview, ['result_preview', 'row_count']) ?? nestedNumber(preview, ['summary_stats', 'rows_count']);
    const pointCount = numberFrom(preview.point_count) ?? nestedNumber(preview, ['result_preview', 'point_count']) ?? nestedNumber(preview, ['summary_stats', 'points_count']);
    const seriesCount = numberFrom(preview.series_count);
    const parts = [
      rowCount !== null ? `${rowCount} rows` : null,
      pointCount !== null ? `${pointCount} points` : null,
      seriesCount !== null ? `${seriesCount} series` : null,
    ].filter(Boolean);
    return parts.length ? `Retrieved ${parts.join(', ')}.` : step.summary;
  }
  if (tool === 'anomaly' && preview) {
    const anomalyCount = numberFrom(preview.anomaly_count) ?? numberFrom(preview.anomaly_point_count);
    return anomalyCount !== null ? `Checked the series and found ${anomalyCount} anomaly points.` : step.summary;
  }
  if (tool === 'forecast' && preview) {
    const count = numberFrom(preview.forecast_point_count);
    return count !== null ? `Generated ${count} forecast points.` : step.summary;
  }
  return step.summary;
}

function metricsForTool(
  tool: string | undefined,
  preview: Record<string, unknown> | null,
  call: Record<string, unknown> | null,
): DisplayMetric[] {
  if (!preview && !call) return [];
  const metrics: DisplayMetric[] = [];

  addMetric(metrics, 'Rows', numberFrom(preview?.row_count) ?? nestedNumber(preview, ['result_preview', 'row_count']) ?? nestedNumber(preview, ['summary_stats', 'rows_count']));
  addMetric(metrics, 'Points', numberFrom(preview?.point_count) ?? nestedNumber(preview, ['result_preview', 'point_count']) ?? nestedNumber(preview, ['summary_stats', 'points_count']));
  addMetric(metrics, 'Series', numberFrom(preview?.series_count) ?? nestedNumber(preview, ['summary_stats', 'series_count']));
  addMetric(metrics, 'Facts', numberFrom(preview?.verified_fact_count));
  addMetric(metrics, 'Samples', numberFrom(preview?.input_row_count));
  addMetric(metrics, 'Anomalies', numberFrom(preview?.anomaly_count) ?? numberFrom(preview?.anomaly_point_count));
  addMetric(metrics, 'Forecast points', numberFrom(preview?.forecast_point_count));
  addMetric(metrics, 'Todos', numberFrom(preview?.todo_total));
  const todoProgress = asRecord(preview?.todo_progress);
  if (todoProgress) {
    const completed = numberFrom(todoProgress.completed);
    const total = numberFrom(todoProgress.total);
    if (completed !== null && total !== null && total > 0) {
      addMetric(metrics, 'Todo progress', `${completed}/${total}`);
    }
  }
  addMetric(metrics, 'Visuals', numberFrom(preview?.visualization_count));

  return metrics;
}

function artifactRefsFor(preview: Record<string, unknown> | null, result: Record<string, unknown> | null) {
  const refs = [
    stringFrom(preview?.payload_ref),
    stringFrom(result?.payload_ref),
    stringFrom(preview?.evidence_id),
    stringFrom(preview?.analysis_id),
    stringFrom(preview?.forecast_id),
    stringFrom(preview?.anomaly_id),
  ];
  return Array.from(new Set(refs.filter((value): value is string => Boolean(value))));
}

function sqlDetailFor(tool: string | undefined, preview: Record<string, unknown> | null): SqlDetail | null {
  if (tool !== 'sql_query' && tool !== 'query_database') return null;
  if (!preview) return null;
  const sampleRows = recordsFrom(preview.sample_rows);
  const samplePoints = recordsFrom(preview.sample_points);
  const columns = stringsFrom(preview.columns);
  const query = stringFrom(preview.query);
  if (!query && columns.length === 0 && sampleRows.length === 0 && samplePoints.length === 0) return null;
  return {
    queryLanguage: stringFrom(preview.query_language),
    query,
    columns,
    sampleRows,
    samplePoints,
    rowCount: numberFrom(preview.row_count),
    pointCount: numberFrom(preview.point_count),
    truncated: Boolean(preview.truncated || preview.payload_truncated),
  };
}

function codeInterpreterDetailFor(
  tool: string | undefined,
  preview: Record<string, unknown> | null,
  call: Record<string, unknown> | null,
  step: TraceStep,
): CodeInterpreterDetail | null {
  if (tool !== 'code_interpreter') return null;
  const inputPreview = asRecord(call?.input_preview);
  const actionInput = asRecord(step.actionInput) || asRecord(call?.action_input);
  const code = stringFrom(inputPreview?.code_preview) || stringFrom(actionInput?.code);
  const result = asRecord(preview?.analysis_result) || asRecord(preview?.result_preview);
  const metrics = asRecord(preview?.analysis_metrics) || asRecord(result?.metrics) || {};
  const details = asRecord(preview?.analysis_details) || asRecord(result?.details) || {};
  if (!code && !result && !preview) return null;
  return {
    analysisGoal: stringFrom(preview?.analysis_goal) || stringFrom(inputPreview?.analysis_goal) || stringFrom(actionInput?.analysis_goal),
    code,
    codeHash: stringFrom(preview?.code_hash),
    codeType: stringFrom(preview?.code_type) || stringFrom(inputPreview?.code_type) || stringFrom(actionInput?.code_type),
    inputEvidenceId: stringFrom(preview?.input_evidence_id) || stringFrom(inputPreview?.evidence_id) || stringFrom(actionInput?.database_evidence),
    inputRowCount: numberFrom(preview?.input_row_count),
    runtimeMs: numberFrom(preview?.runtime_ms),
    summary: stringFrom(preview?.analysis_summary) || stringFrom(result?.summary) || stringFrom(preview?.summary),
    metrics,
    details,
    result,
    inputColumns: stringsFrom(preview?.input_columns),
  };
}

function forecastDetailFor(tool: string | undefined, preview: Record<string, unknown> | null): ForecastDetail | null {
  if (tool !== 'forecast' || !preview) return null;
  const points = recordsFrom(preview.forecast_points);
  const plan = asRecord(preview.forecast_plan);
  if (!plan && points.length === 0 && !preview.forecast_id) return null;
  return {
    status: stringFrom(preview.forecast_status) || stringFrom(preview.status),
    modelName: stringFrom(preview.model_name),
    horizon: numberFrom(preview.horizon),
    plan,
    points,
    pointCount: numberFrom(preview.forecast_point_count),
  };
}

function anomalyDetailFor(tool: string | undefined, preview: Record<string, unknown> | null): AnomalyDetail | null {
  if (tool !== 'anomaly' || !preview) return null;
  const points = recordsFrom(preview.anomaly_points);
  const scores = recordsFrom(preview.anomaly_scores);
  if (points.length === 0 && scores.length === 0 && !preview.anomaly_id) return null;
  return {
    detectorName: stringFrom(preview.detector_name),
    points,
    scores,
    pointCount: numberFrom(preview.anomaly_point_count),
    spanCount: numberFrom(preview.anomaly_span_count),
  };
}

function schemaLinkingDetailFor(tool: string | undefined, preview: Record<string, unknown> | null): SchemaLinkingDetail | null {
  if (tool !== 'sql_query' && tool !== 'query_database') return null;
  const linking = asRecord(preview?.schema_linking);
  if (!linking) return null;
  const sources = recordsFrom(linking.sources).map((source) => ({
    name: stringFrom(source.name) || '',
    kind: stringFrom(source.kind),
    timeColumn: stringFrom(source.time_column),
    valueColumns: stringsFrom(source.value_columns),
    dimensionColumns: stringsFrom(source.dimension_columns),
    linkedColumns: recordsFrom(source.columns)
      .map((column) => ({
        name: stringFrom(column.name) || '',
        role: stringFrom(column.role),
      }))
      .filter((column) => column.name),
  })).filter((source) => source.name);
  const fieldMappings = recordsFrom(linking.field_mappings).map((mapping) => ({
    sourceName: stringFrom(mapping.source_name),
    fieldName: stringFrom(mapping.field_name) || '',
    role: stringFrom(mapping.role),
    confidence: numberFrom(mapping.confidence),
  })).filter((mapping) => mapping.fieldName);
  const requiredFilters = recordsFrom(linking.required_filters).map((filter) => ({
    column: stringFrom(filter.column) || '',
    operator: stringFrom(filter.operator) || '=',
    value: stringFrom(filter.value) || String(filter.value ?? ''),
  })).filter((filter) => filter.column && filter.value);
  const ambiguousRecord = asRecord(linking.ambiguous_terms);
  const ambiguousTerms = ambiguousRecord
    ? Object.entries(ambiguousRecord).map(([term, candidates]) => ({
        term,
        candidates: stringsFrom(candidates),
      })).filter((item) => item.candidates.length > 0)
    : [];
  if (
    sources.length === 0
    && fieldMappings.length === 0
    && requiredFilters.length === 0
    && ambiguousTerms.length === 0
  ) {
    return null;
  }
  return {
    confidence: stringFrom(linking.confidence),
    sources,
    fieldMappings,
    requiredFilters,
    evidence: stringsFrom(linking.evidence),
    ambiguousTerms,
  };
}

function completionDetailFor(preview: Record<string, unknown> | null): CompletionDetail | null {
  if (!preview) return null;
  const verdict =
    asRecord(preview.completion_verdict) ||
    asRecord(preview.answerability_verdict) ||
    asRecord(preview.plan_requirement);
  const todoProgress = asRecord(preview.todo_progress);
  if (!verdict && !todoProgress) return null;
  const completed =
    typeof verdict?.completed === 'boolean'
      ? verdict.completed
      : typeof verdict?.can_answer === 'boolean'
        ? verdict.can_answer
        : typeof verdict?.requires_plan === 'boolean'
          ? !verdict.requires_plan
          : null;
  return {
    completed,
    reason: stringFrom(verdict?.reason),
    missingItems: stringsFrom(verdict?.missing_items).length
      ? stringsFrom(verdict?.missing_items)
      : stringsFrom(verdict?.deliverables),
    nextActionHint: stringFrom(verdict?.next_action_hint),
    todoProgress: todoProgress
      ? {
          total: numberFrom(todoProgress.total) || 0,
          completed: numberFrom(todoProgress.completed) || 0,
          inProgress: stringFrom(todoProgress.in_progress),
        }
      : null,
  };
}

function outputTypes(answer?: FinalAnswer | null) {
  const sectionTypes = (answer?.sections || []).map((section) => section.section_type);
  const visualCount = answer?.visualizations?.length || 0;
  const outputs = Array.from(new Set(sectionTypes.filter((type) => type !== 'summary' && type !== 'conclusion')));
  if (visualCount > 0) outputs.push(`${visualCount} visualizations`);
  return outputs;
}

function firstMetric(steps: DisplayStep[], label: string): DisplayMetric | null {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const metric = steps[index].metrics.find((item) => item.label === label);
    if (metric) return metric;
  }
  return null;
}

function compactMetrics(metrics: Array<DisplayMetric | null>): DisplayMetric[] {
  const seen = new Set<string>();
  return metrics.filter((metric): metric is DisplayMetric => {
    if (!metric || seen.has(metric.label)) return false;
    seen.add(metric.label);
    return true;
  });
}

function addMetric(metrics: DisplayMetric[], label: string, value: number | string | null | undefined) {
  if (value === null || value === undefined || value === '') return;
  metrics.push({ label, value: typeof value === 'number' ? value.toLocaleString() : value });
}

function nestedNumber(root: Record<string, unknown> | null | undefined, path: string[]) {
  let current: unknown = root;
  for (const key of path) {
    current = asRecord(current)?.[key];
  }
  return numberFrom(current);
}

function numberFrom(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function stringFrom(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value : null;
}

function stringsFrom(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    : [];
}

function recordsFrom(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => Boolean(asRecord(item)))
    : [];
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? value as Record<string, unknown> : null;
}
