import { describe, expect, it } from 'vitest';
import { traceSpanFromStreamEvent, traceStepFromPolicyDecision } from './traceEvents';

describe('trace span stream events', () => {
  it('normalizes an LLM completion without retaining a model name', () => {
    const span = traceSpanFromStreamEvent({
      event: 'trace_span_end',
      data: {
        kind: 'llm',
        span_id: 'llm-2',
        parent_id: 'iteration-1',
        title: 'SQL Generation',
        status: 'complete',
        duration_ms: 2100,
        model: 'hidden-model-name',
        input_summary: {
          message_count: 2,
          roles: ['system', 'user'],
          character_count: 8420,
          multimodal_part_count: 1,
        },
        output_summary: { character_count: 1106, format: 'json', multimodal_part_count: 0 },
        input_preview: [
          { role: 'system', content: 'Generate a grounded SQL query.' },
          { role: 'user', content: 'Question and schema context' },
        ],
        output_preview: '{"query":"SELECT value FROM energy"}',
        token_usage: { prompt_tokens: 1200, completion_tokens: 536, total_tokens: 1736 },
      },
    });

    expect(span).toMatchObject({
      id: 'llm-2',
      parentId: 'iteration-1',
      title: 'SQL Generation',
      status: 'complete',
      elapsedSeconds: 2.1,
      tokenUsage: { inputTokens: 1200, outputTokens: 536, totalTokens: 1736 },
      inputSummary: { messageCount: 2, roles: ['system', 'user'], characterCount: 8420, multimodalPartCount: 1 },
      outputSummary: { characterCount: 1106, format: 'json', multimodalPartCount: 0 },
      inputPreview: [
        { role: 'system', content: 'Generate a grounded SQL query.' },
        { role: 'user', content: 'Question and schema context' },
      ],
      outputPreview: '{"query":"SELECT value FROM energy"}',
    });
    expect(span).not.toHaveProperty('model');
  });

  it('rejects incomplete spans instead of guessing identity or parentage', () => {
    expect(traceSpanFromStreamEvent({
      event: 'trace_span_start',
      data: { kind: 'llm', title: 'Schema Linking' },
    })).toBeNull();
  });
});

describe('policy decision stream events', () => {
  it('represents a rejected proposal as a failed policy step, not a tool call', () => {
    const step = traceStepFromPolicyDecision({
      event: 'policy_decision',
      data: {
        iteration: 5,
        tool: 'anomaly',
        accepted: false,
        summary: "Forecast output is required. Required actions: ['forecast'].",
        started_at: '2026-08-18T14:47:19.652Z',
        completed_at: '2026-08-18T14:47:19.652Z',
        duration_ms: 0,
      },
    });

    expect(step).toMatchObject({
      id: 'iteration-5',
      iteration: 5,
      phase: 'policy_decision',
      status: 'error',
      tool: 'anomaly',
      elapsedSeconds: 0,
    });
    expect(step?.toolCall).toBeUndefined();
    expect(step?.toolResult).toMatchObject({ success: false, accepted: false });
  });
});
