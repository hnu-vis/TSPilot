import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { TraceStep } from '../types';
import { TraceTimeline } from './TraceTimeline';

describe('TraceTimeline', () => {
  it('renders running LLM calls as leaves under their tool parent', () => {
    const markup = renderToStaticMarkup(
      <TraceTimeline
        steps={[toolStep('running')]}
        selectedId="llm-2"
        onSelect={() => undefined}
        nowMs={Date.parse('2026-08-18T00:00:03Z')}
      />,
    );

    expect(markup).toContain('sql_query');
    expect(markup).toContain('Schema Linking');
    expect(markup).toContain('SQL Generation');
    expect(markup).toContain('1,284 tokens');
    expect(markup).toContain('aria-label="LLM calls"');
    expect(markup).toContain('aria-current="true"');
    expect(markup).toContain('trace-llm-leaf running selected');
    expect(markup).toContain('<button');
    expect(markup).not.toContain('qwen');
    expect(markup).not.toContain('gpt-');
    expect(markup).not.toContain('lucide-brain-circuit');
  });

  it('keeps completed tool children collapsed by default', () => {
    const markup = renderToStaticMarkup(
      <TraceTimeline
        steps={[toolStep('complete')]}
        selectedId={null}
        onSelect={() => undefined}
        nowMs={Date.parse('2026-08-18T00:00:07Z')}
      />,
    );

    expect(markup).toContain('2 LLM calls');
    expect(markup).not.toContain('Schema Linking');
    expect(markup).not.toContain('SQL Generation');
  });

  it('labels a policy-rejected proposal without presenting it as an execution success', () => {
    const markup = renderToStaticMarkup(
      <TraceTimeline
        steps={[{
          ...toolStep('error'),
          phase: 'policy_decision',
          tool: 'anomaly',
          summary: "Forecast output is required. Required actions: ['forecast'].",
          toolCall: undefined,
          toolResult: { accepted: false, success: false },
        }]}
        selectedId={null}
        onSelect={() => undefined}
        nowMs={Date.parse('2026-08-18T00:00:07Z')}
      />,
    );

    expect(markup).toContain('anomaly · Rejected');
    expect(markup).toContain('trace-step error');
    expect(markup).not.toContain('trace-step complete');
  });

  it('replaces the ReAct placeholder with a top-level LLM call', () => {
    const markup = renderToStaticMarkup(
      <TraceTimeline
        steps={[{
          id: 'iteration-2:decision',
          iteration: 2,
          agent: 'data_agent',
          phase: 'reasoning',
          status: 'running',
          summary: '正在选择下一步工具。',
          startedAt: '2026-08-18T00:00:00Z',
          children: [{
            id: 'iteration-2:llm:decision',
            parentId: 'iteration-2:decision',
            kind: 'llm',
            title: 'ReAct Decision',
            status: 'running',
            startedAt: '2026-08-18T00:00:00Z',
            updatedAt: '2026-08-18T00:00:00Z',
          }],
          updatedAt: '2026-08-18T00:00:00Z',
        }]}
        selectedId={null}
        onSelect={() => undefined}
        nowMs={Date.parse('2026-08-18T00:00:02Z')}
      />,
    );

    expect(markup).toContain('ReAct Decision');
    expect(markup).toContain('trace-llm-root-call running');
    expect(markup).not.toContain('Choose next action');
    expect(markup).not.toContain('trace-llm-list');
  });

  it('shows the decision placeholder only before the first LLM start arrives', () => {
    const markup = renderToStaticMarkup(
      <TraceTimeline
        steps={[{
          id: 'iteration-2:decision',
          iteration: 2,
          agent: 'data_agent',
          phase: 'reasoning',
          status: 'running',
          summary: '正在选择下一步工具。',
          startedAt: '2026-08-18T00:00:00Z',
          updatedAt: '2026-08-18T00:00:00Z',
        }]}
        selectedId={null}
        onSelect={() => undefined}
        nowMs={Date.parse('2026-08-18T00:00:00.100Z')}
      />,
    );

    expect(markup).toContain('Choose next action');
    expect(markup).not.toContain('ReAct Decision');
  });
});

function toolStep(status: TraceStep['status']): TraceStep {
  return {
    id: 'iteration-1',
    iteration: 1,
    agent: 'data_agent',
    phase: status === 'running' ? 'tool_call' : 'tool_result',
    status,
    summary: status === 'running' ? 'Querying time-series data' : 'Returned 366 rows',
    tool: 'sql_query',
    toolCall: { tool: 'sql_query' },
    startedAt: '2026-08-18T00:00:00Z',
    completedAt: status === 'complete' ? '2026-08-18T00:00:06.800Z' : undefined,
    children: [
      {
        id: 'llm-1',
        parentId: 'iteration-1',
        kind: 'llm',
        title: 'Schema Linking',
        status: 'complete',
        tokenUsage: { totalTokens: 1284 },
        elapsedSeconds: 1.4,
        updatedAt: '2026-08-18T00:00:01.400Z',
      },
      {
        id: 'llm-2',
        parentId: 'iteration-1',
        kind: 'llm',
        title: 'SQL Generation',
        summary: 'Generating a grounded query',
        status: status === 'running' ? 'running' : 'complete',
        startedAt: '2026-08-18T00:00:01.400Z',
        elapsedSeconds: status === 'complete' ? 2.1 : undefined,
        updatedAt: '2026-08-18T00:00:01.400Z',
      },
    ],
    updatedAt: '2026-08-18T00:00:03Z',
  };
}
