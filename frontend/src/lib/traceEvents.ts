import type { StreamEvent, TraceSpan, TraceStatus, TraceStep } from '../types';

const TRACE_SPAN_EVENTS = new Set([
  'trace_span_start',
  'trace_span_end',
]);

export function isTraceSpanEvent(event: StreamEvent): boolean {
  return TRACE_SPAN_EVENTS.has(event.event);
}

export function traceSpanFromStreamEvent(event: StreamEvent): TraceSpan | null {
  if (!isTraceSpanEvent(event) || event.data.kind !== 'llm') return null;

  const id = nonEmptyString(event.data.span_id);
  const parentId = nonEmptyString(event.data.parent_id);
  const title = nonEmptyString(event.data.title);
  if (!id || !parentId || !title) return null;

  const timestamp = new Date().toISOString();
  const status = spanStatus(event);
  const summary = nonEmptyString(event.data.summary) || undefined;
  const error = nonEmptyString(event.data.error) || undefined;

  return {
    id,
    parentId,
    kind: 'llm',
    title,
    status,
    summary,
    tokenUsage: tokenUsage(event.data.token_usage),
    inputSummary: inputSummary(event.data.input_summary),
    outputSummary: outputSummary(event.data.output_summary),
    inputPreview: inputPreview(event.data.input_preview),
    outputPreview: textPreview(event.data.output_preview),
    error,
    startedAt: validTimestamp(event.data.started_at) || (status === 'running' ? timestamp : undefined),
    completedAt: validTimestamp(event.data.completed_at) || (status === 'running' ? undefined : timestamp),
    elapsedSeconds: elapsedSeconds(event.data),
    updatedAt: timestamp,
  };
}

function inputPreview(value: unknown): TraceSpan['inputPreview'] {
  if (!Array.isArray(value)) return undefined;
  const messages = value.flatMap((item) => {
    const message = asRecord(item);
    const role = nonEmptyString(message?.role);
    const content = typeof message?.content === 'string' ? message.content : null;
    return role && content !== null ? [{ role, content }] : [];
  });
  return messages.length ? messages : undefined;
}

function inputSummary(value: unknown): TraceSpan['inputSummary'] {
  const summary = asRecord(value);
  if (!summary) return undefined;
  const messageCount = finiteNumber(summary.message_count);
  const characterCount = finiteNumber(summary.character_count);
  const multimodalPartCount = finiteNumber(summary.multimodal_part_count);
  const roles = Array.isArray(summary.roles)
    ? summary.roles.filter((role): role is string => typeof role === 'string' && Boolean(role.trim()))
    : undefined;
  if (messageCount === undefined && characterCount === undefined && multimodalPartCount === undefined && !roles?.length) {
    return undefined;
  }
  return { messageCount, roles, characterCount, multimodalPartCount };
}

function outputSummary(value: unknown): TraceSpan['outputSummary'] {
  const summary = asRecord(value);
  if (!summary) return undefined;
  const characterCount = finiteNumber(summary.character_count);
  const multimodalPartCount = finiteNumber(summary.multimodal_part_count);
  const format = nonEmptyString(summary.format) || undefined;
  if (characterCount === undefined && multimodalPartCount === undefined && format === undefined) return undefined;
  return { characterCount, format, multimodalPartCount };
}

export function traceStepFromPolicyDecision(event: StreamEvent): TraceStep | null {
  if (event.event !== 'policy_decision') return null;

  const iteration = finiteNumber(event.data.iteration);
  const tool = nonEmptyString(event.data.tool);
  if (iteration === undefined || !tool) return null;

  const accepted = event.data.accepted === true;
  const timestamp = new Date().toISOString();
  const summary = nonEmptyString(event.data.summary)
    || (accepted ? 'Action accepted by policy.' : 'Action rejected before execution.');
  const startedAt = validTimestamp(event.data.started_at) || timestamp;
  const completedAt = validTimestamp(event.data.completed_at) || timestamp;

  return {
    id: `iteration-${iteration}`,
    iteration,
    agent: 'runtime',
    phase: 'policy_decision',
    status: accepted ? 'complete' : 'error',
    summary,
    tool,
    observation: event.data,
    toolResult: { ...event.data, success: accepted },
    error: accepted ? undefined : summary,
    startedAt,
    completedAt,
    elapsedSeconds: elapsedSeconds(event.data),
    updatedAt: timestamp,
  };
}

function spanStatus(event: StreamEvent): TraceStatus {
  if (event.event === 'trace_span_start') return 'running';
  return event.data.status === 'error' || event.data.status === 'failed' ? 'error' : 'complete';
}

function tokenUsage(value: unknown): TraceSpan['tokenUsage'] {
  const direct = asRecord(value);
  const nested = asRecord(direct?.provider) || asRecord(direct?.estimated) || direct;
  if (!nested) return undefined;

  const inputTokens = finiteNumber(nested.input_tokens) ?? finiteNumber(nested.prompt_tokens);
  const outputTokens = finiteNumber(nested.output_tokens) ?? finiteNumber(nested.completion_tokens);
  const totalTokens = finiteNumber(nested.total_tokens)
    ?? (inputTokens !== undefined || outputTokens !== undefined
      ? (inputTokens || 0) + (outputTokens || 0)
      : undefined);
  if (inputTokens === undefined && outputTokens === undefined && totalTokens === undefined) return undefined;
  return { inputTokens, outputTokens, totalTokens };
}

function elapsedSeconds(data: Record<string, unknown>): number | undefined {
  const seconds = finiteNumber(data.elapsed_seconds);
  if (seconds !== undefined) return seconds;
  const durationMs = finiteNumber(data.duration_ms);
  return durationMs === undefined ? undefined : Math.round(durationMs) / 1000;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function textPreview(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
}

function validTimestamp(value: unknown): string | undefined {
  return typeof value === 'string' && Number.isFinite(Date.parse(value)) ? value : undefined;
}
