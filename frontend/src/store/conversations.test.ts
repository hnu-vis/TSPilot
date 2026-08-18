import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Conversation, TraceStep } from '../types';
import {
  appendAssistantPending,
  createConversation,
  loadConversations,
  settleIncompleteStream,
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

  it('settles every running step and pending assistant when a stream ends early', () => {
    const pending = appendAssistantPending(withStep(createConversation(), traceStep('running')));
    const settled = settleIncompleteStream(pending, 'stream ended');

    expect(settled.messages.at(-1)).toMatchObject({ content: 'stream ended', isStreaming: false });
    expect(settled.traceSteps).toEqual([expect.objectContaining({ status: 'error', error: 'stream ended' })]);
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
    startedAt: '2026-08-18T00:00:00Z',
    completedAt: status === 'running' ? undefined : '2026-08-18T00:00:01Z',
    updatedAt: '2026-08-18T00:00:01Z',
  };
}

function withStep(conversation: Conversation, step: TraceStep): Conversation {
  return { ...conversation, traceSteps: [step] };
}
