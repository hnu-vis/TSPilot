import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Conversation, TraceSpan, TraceStep } from '../types';
import {
  appendAssistantPending,
  completeRunningTraceSteps,
  createConversation,
  loadConversations,
  settleIncompleteStream,
  upsertTraceSpan,
  upsertTraceStep,
} from './conversations';

describe('conversation lifecycle', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('does not regress a completed tool step when a delayed running event arrives', () => {
    const completed = withStep(createConversation(), traceStep('complete'));
    const merged = upsertTraceStep(completed, {
      ...traceStep('running'),
      summary: 'late heartbeat',
      updatedAt: '2026-08-18T00:00:02Z',
    });

    expect(merged.traceSteps[0].status).toBe('complete');
  });

  it('starts tool timing at the real tool boundary instead of the earlier reasoning step', () => {
    const reasoning = withStep(createConversation(), {
      ...traceStep('running'),
      phase: 'reasoning',
      tool: undefined,
      toolCall: undefined,
      startedAt: '2026-08-18T00:00:00Z',
    });
    const started = upsertTraceStep(reasoning, {
      ...traceStep('running'),
      startedAt: '2026-08-18T00:00:02Z',
    });

    expect(started.traceSteps[0].startedAt).toBe('2026-08-18T00:00:02Z');
  });

  it('settles every running step and pending assistant when a stream ends early', () => {
    const pending = appendAssistantPending(withStep(createConversation(), traceStep('running')));
    const settled = settleIncompleteStream(pending, 'stream ended');

    expect(settled.messages.at(-1)).toMatchObject({ content: 'stream ended', isStreaming: false });
    expect(settled.traceSteps).toEqual([expect.objectContaining({ status: 'error', error: 'stream ended' })]);
  });

  it('progressively attaches and completes an LLM leaf under its tool parent', () => {
    const parent = withStep(createConversation(), traceStep('running'));
    const running = upsertTraceSpan(parent, {
      ...traceSpan('running'),
      inputSummary: { messageCount: 2, characterCount: 420 },
      inputPreview: [{ role: 'user', content: 'Generate a grounded query.' }],
    });
    const completed = upsertTraceSpan(running, {
      ...traceSpan('complete'),
      outputSummary: { characterCount: 96, format: 'json' },
      outputPreview: '{"query":"SELECT value"}',
      tokenUsage: { totalTokens: 1284 },
      elapsedSeconds: 1.4,
      completedAt: '2026-08-18T00:00:01.400Z',
      updatedAt: '2026-08-18T00:00:01.400Z',
    });
    const delayed = upsertTraceSpan(completed, traceSpan('running'));

    expect(completed.traceSteps[0].children).toEqual([
      expect.objectContaining({
        id: 'llm-schema-linking',
        status: 'complete',
        inputSummary: { messageCount: 2, characterCount: 420 },
        outputSummary: { characterCount: 96, format: 'json' },
        inputPreview: [{ role: 'user', content: 'Generate a grounded query.' }],
        outputPreview: '{"query":"SELECT value"}',
        tokenUsage: { totalTokens: 1284 },
      }),
    ]);
    expect(delayed.traceSteps[0].children?.[0].status).toBe('complete');
  });

  it('does not guess a parent tool for an orphan LLM span', () => {
    const conversation = withStep(createConversation(), traceStep('running'));
    const updated = upsertTraceSpan(conversation, { ...traceSpan('running'), parentId: 'unknown-tool' });

    expect(updated).toBe(conversation);
    expect(updated.traceSteps[0].children).toBeUndefined();
  });

  it('keeps a ReAct decision span separate from the tool selected in the same iteration', () => {
    const decision = withStep(createConversation(), {
      ...traceStep('running'),
      id: 'iteration-1:decision',
      phase: 'reasoning',
      tool: undefined,
      toolCall: undefined,
    });
    const withDecisionCall = upsertTraceSpan(decision, {
      ...traceSpan('complete'),
      parentId: 'iteration-1:decision',
      title: 'ReAct Decision',
    });
    const withTool = upsertTraceStep(withDecisionCall, {
      ...traceStep('running'),
      summary: 'sql_query',
      tool: 'sql_query',
      toolCall: { tool: 'sql_query' },
    });

    expect(withTool.traceSteps).toHaveLength(2);
    expect(withTool.traceSteps[0]).toMatchObject({ id: 'iteration-1:decision', phase: 'reasoning' });
    expect(withTool.traceSteps[0].children?.[0]).toMatchObject({ title: 'ReAct Decision' });
    expect(withTool.traceSteps[1]).toMatchObject({ id: 'iteration-1', tool: 'sql_query' });
    expect(withTool.traceSteps[1].children).toBeUndefined();
  });

  it('never turns a rejected policy step green when the request completes', () => {
    const rejected = withStep(createConversation(), {
      ...traceStep('error'),
      phase: 'policy_decision',
      toolCall: undefined,
      toolResult: { tool: 'forecast', accepted: false, success: false },
      error: 'Rejected before execution.',
    });

    const completed = completeRunningTraceSteps(rejected, '2026-08-18T00:00:02Z');

    expect(completed.traceSteps[0]).toMatchObject({
      phase: 'policy_decision',
      status: 'error',
      error: 'Rejected before execution.',
    });
    expect(completed.traceSteps[0].toolCall).toBeUndefined();
  });

  it('repairs partial and malformed persisted conversations before rendering', () => {
    vi.stubGlobal('localStorage', {
      getItem: () => JSON.stringify([{
        id: 'saved',
        title: 'Saved run',
        createdAt: '2026-08-18T00:00:00Z',
        updatedAt: '2026-08-18T00:00:01Z',
        messages: [{
          id: 'pending',
          role: 'assistant',
          content: 'working',
          createdAt: '2026-08-18T00:00:00Z',
          isStreaming: true,
          answer: { summary: 42, references: {} },
        }],
        traceSteps: [{ ...traceStep('running'), summary: { invalid: true } }],
      }]),
    });

    const [conversation] = loadConversations();

    expect(conversation.messages[0]).toMatchObject({ isStreaming: false });
    expect(conversation.messages[0].answer).toBeUndefined();
    expect(conversation.traceSteps[0]).toMatchObject({ status: 'error' });
    expect(typeof conversation.traceSteps[0].summary).toBe('string');
  });
});

function traceStep(status: TraceStep['status']): TraceStep {
  return {
    id: 'iteration-1',
    iteration: 1,
    agent: 'data_agent',
    phase: 'tool_call',
    status,
    summary: 'forecast',
    tool: 'forecast',
    toolCall: { tool: 'forecast' },
    startedAt: '2026-08-18T00:00:00Z',
    completedAt: status === 'running' ? undefined : '2026-08-18T00:00:01Z',
    updatedAt: '2026-08-18T00:00:01Z',
  };
}

function traceSpan(status: TraceSpan['status']): TraceSpan {
  return {
    id: 'llm-schema-linking',
    parentId: 'iteration-1',
    kind: 'llm',
    title: 'Schema Linking',
    status,
    summary: 'Identify grounded fields',
    startedAt: '2026-08-18T00:00:00Z',
    updatedAt: '2026-08-18T00:00:00Z',
  };
}

function withStep(conversation: Conversation, step: TraceStep): Conversation {
  return { ...conversation, traceSteps: [step] };
}
