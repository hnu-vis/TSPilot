import type { ChatMessage, Conversation, FinalAnswer, TokenUsage, TraceStep, Visualization } from '../types';

const INTERRUPTED_MESSAGE = 'The previous response was interrupted before completion.';

export function normalizeFinalAnswer(value: unknown): FinalAnswer | null {
  const answer = asRecord(value);
  if (!answer) return null;

  const summary = stringValue(answer.summary);
  const title = nullableStringValue(answer.title);
  if (summary === null && title === null) return null;

  return {
    summary: summary || '',
    title,
    sections: normalizeSections(answer.sections),
    references: normalizeReferences(answer.references),
    claims: normalizeClaims(answer.claims),
    visualizations: Array.isArray(answer.visualizations)
      ? answer.visualizations.filter((item): item is Visualization => Boolean(asRecord(item)))
      : [],
  };
}

export function normalizeConversation(value: unknown): Conversation | null {
  const conversation = asRecord(value);
  const id = stringValue(conversation?.id);
  if (!conversation || !id) return null;

  const createdAt = validTimestamp(conversation.createdAt) || new Date().toISOString();
  const updatedAt = validTimestamp(conversation.updatedAt) || createdAt;
  const messages = Array.isArray(conversation.messages)
    ? conversation.messages.map(normalizeMessage).filter((item): item is ChatMessage => Boolean(item))
    : [];
  const traceSteps = Array.isArray(conversation.traceSteps)
    ? conversation.traceSteps.map(normalizeTraceStep).filter((item): item is TraceStep => Boolean(item))
    : [];

  const hadInterruptedMessage = messages.some((message) => message.isStreaming);
  const normalizedMessages = messages.map((message) => message.isStreaming
    ? { ...message, content: INTERRUPTED_MESSAGE, isStreaming: false }
    : message);
  const normalizedTraceSteps = traceSteps.map((step) => step.status === 'running'
    ? {
        ...step,
        status: 'error' as const,
        summary: hadInterruptedMessage ? INTERRUPTED_MESSAGE : step.summary,
        error: hadInterruptedMessage ? INTERRUPTED_MESSAGE : step.error,
        completedAt: updatedAt,
        updatedAt,
      }
    : step);

  return {
    id,
    title: stringValue(conversation.title) || 'New chat',
    createdAt,
    updatedAt,
    messages: normalizedMessages,
    traceSteps: normalizedTraceSteps,
    selectedTraceStepId: nullableStringValue(conversation.selectedTraceStepId),
    selectedDatabaseId: nullableStringValue(conversation.selectedDatabaseId),
    selectedKnowledgeId: nullableStringValue(conversation.selectedKnowledgeId),
    selectedModelId: nullableStringValue(conversation.selectedModelId),
  };
}

function normalizeMessage(value: unknown): ChatMessage | null {
  const message = asRecord(value);
  const id = stringValue(message?.id);
  const role = message?.role;
  if (!message || !id || (role !== 'user' && role !== 'assistant' && role !== 'system')) return null;
  const answer = normalizeFinalAnswer(message.answer);
  const tokenUsage = asRecord(message.tokenUsage);
  return {
    id,
    role,
    content: stringValue(message.content) || answer?.summary || '',
    createdAt: validTimestamp(message.createdAt) || new Date().toISOString(),
    answer: answer || undefined,
    tokenUsage: tokenUsage ? tokenUsage as TokenUsage : undefined,
    isStreaming: message.isStreaming === true,
  };
}

function normalizeTraceStep(value: unknown): TraceStep | null {
  const step = asRecord(value);
  const id = stringValue(step?.id);
  if (!step || !id) return null;
  const status = step.status === 'complete' || step.status === 'error' || step.status === 'running'
    ? step.status
    : 'error';
  const updatedAt = validTimestamp(step.updatedAt) || new Date().toISOString();
  return {
    id,
    iteration: finiteNumber(step.iteration) || 0,
    agent: stringValue(step.agent) || 'runtime',
    phase: stringValue(step.phase) || 'unknown',
    status,
    summary: stringValue(step.summary) || '',
    tool: optionalString(step.tool),
    thought: optionalString(step.thought),
    actionInput: asRecord(step.actionInput) || undefined,
    observation: asRecord(step.observation) || undefined,
    toolCall: asRecord(step.toolCall) || undefined,
    toolResult: asRecord(step.toolResult) || undefined,
    error: optionalString(step.error),
    startedAt: validTimestamp(step.startedAt) || undefined,
    completedAt: validTimestamp(step.completedAt) || undefined,
    elapsedSeconds: finiteNumber(step.elapsedSeconds),
    updatedAt,
  };
}

function normalizeSections(value: unknown): FinalAnswer['sections'] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const section = asRecord(item);
    const sectionType = stringValue(section?.section_type);
    const content = stringValue(section?.content);
    if (!section || !sectionType || content === null) return [];
    return [{
      section_type: sectionType,
      heading: nullableStringValue(section.heading),
      content,
      structured_payload: asRecord(section.structured_payload),
    }];
  });
}

function normalizeReferences(value: unknown): FinalAnswer['references'] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const reference = asRecord(item);
    const sourceType = stringValue(reference?.source_type);
    const label = stringValue(reference?.label);
    if (!reference || !sourceType || !label) return [];
    return [{
      source_type: sourceType,
      source_id: nullableStringValue(reference.source_id),
      label,
      evidence: asRecord(reference.evidence),
    }];
  });
}

function normalizeClaims(value: unknown): FinalAnswer['claims'] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const claim = asRecord(item);
    const claimId = stringValue(claim?.claim_id);
    const text = stringValue(claim?.text);
    if (!claim || !claimId || !text) return [];
    return [{
      claim_id: claimId,
      text,
      insight_ids: stringArray(claim.insight_ids),
      item_ids: stringArray(claim.item_ids),
      analysis_ids: stringArray(claim.analysis_ids),
      artifact_type: nullableStringValue(claim.artifact_type),
      artifact_ids: stringArray(claim.artifact_ids),
      evidence_ids: stringArray(claim.evidence_ids),
      visualization_ids: stringArray(claim.visualization_ids),
    }];
  });
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function nullableStringValue(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function validTimestamp(value: unknown): string | null {
  return typeof value === 'string' && Number.isFinite(Date.parse(value)) ? value : null;
}
