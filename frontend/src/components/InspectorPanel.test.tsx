import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { TraceStep } from '../types';
import { InspectorPanel } from './InspectorPanel';

describe('InspectorPanel LLM lifecycle', () => {
  it('automatically follows the latest running LLM call', () => {
    const step: TraceStep = {
      id: 'iteration-1:decision',
      iteration: 1,
      agent: 'data_agent',
      phase: 'reasoning',
      status: 'running',
      summary: 'Choosing the next tool.',
      startedAt: '2026-08-18T00:00:00Z',
      children: [
        {
          id: 'iteration-1:llm:decision',
          parentId: 'iteration-1:decision',
          kind: 'llm',
          title: 'ReAct Decision',
          summary: 'Choose the next action from the current task state',
          status: 'running',
          inputSummary: {
            messageCount: 2,
            roles: ['system', 'user'],
            characterCount: 8420,
            multimodalPartCount: 0,
          },
          inputPreview: [
            { role: 'system', content: 'Choose the next grounded action.' },
            { role: 'user', content: 'Analyze the selected time series.' },
          ],
          startedAt: '2026-08-18T00:00:00Z',
          updatedAt: '2026-08-18T00:00:00Z',
        },
        {
          id: 'iteration-1:llm:memory',
          parentId: 'iteration-1:decision',
          kind: 'llm',
          title: 'Memory Reranking',
          status: 'complete',
          elapsedSeconds: 1.2,
          tokenUsage: { totalTokens: 640 },
          updatedAt: '2026-08-18T00:00:01.200Z',
        },
      ],
      updatedAt: '2026-08-18T00:00:00Z',
    };

    const markup = renderToStaticMarkup(
      <InspectorPanel
        steps={[step]}
        selectedNodeId={null}
        collapsed={false}
        onToggleCollapsed={() => undefined}
      />,
    );

    expect(markup).toContain('ReAct Decision');
    expect(markup).toContain('Model invocation');
    expect(markup).toContain('2 messages');
    expect(markup).toContain('8,420');
    expect(markup).toContain('Waiting for output');
    expect(markup).toContain('Choose the next grounded action.');
    expect(markup).toContain('Analyze the selected time series.');
    expect(markup).not.toContain('Memory Reranking');
  });

  it('keeps an explicitly selected completed LLM call pinned with safe I/O metadata', () => {
    const step: TraceStep = {
      id: 'iteration-1',
      iteration: 1,
      agent: 'data_agent',
      phase: 'tool_call',
      status: 'running',
      summary: 'Querying data',
      tool: 'sql_query',
      children: [{
        id: 'llm-sql',
        parentId: 'iteration-1',
        kind: 'llm',
        title: 'SQL Generation',
        status: 'complete',
        inputSummary: { messageCount: 3, roles: ['system', 'user'], characterCount: 910 },
        outputSummary: { characterCount: 1106, format: 'json', multimodalPartCount: 0 },
        inputPreview: [
          { role: 'system', content: 'Generate SQL for the selected schema.' },
          { role: 'user', content: 'Return hourly BTC prices.' },
        ],
        outputPreview: '{"query":"SELECT time, price FROM btc"}',
        tokenUsage: { inputTokens: 240, outputTokens: 80, totalTokens: 320 },
        elapsedSeconds: 2.4,
        updatedAt: '2026-08-18T00:00:02.400Z',
      }, {
        id: 'llm-latest',
        parentId: 'iteration-1',
        kind: 'llm',
        title: 'Validation',
        status: 'running',
        updatedAt: '2026-08-18T00:00:03Z',
      }],
      updatedAt: '2026-08-18T00:00:03Z',
    };

    const markup = renderToStaticMarkup(
      <InspectorPanel
        steps={[step]}
        selectedNodeId="llm-sql"
        collapsed={false}
        onToggleCollapsed={() => undefined}
      />,
    );

    expect(markup).toContain('SQL Generation');
    expect(markup).toContain('JSON');
    expect(markup).toContain('1,106');
    expect(markup).toContain('320');
    expect(markup).toContain('sql_query');
    expect(markup).toContain('Generate SQL for the selected schema.');
    expect(markup).toContain('SELECT time, price FROM btc');
    expect(markup).toContain('Copy output content');
    expect(markup).not.toContain('Validation');
  });

  it('keeps a just-completed LLM call visible until a newer execution node arrives', () => {
    const step: TraceStep = {
      id: 'iteration-3',
      iteration: 3,
      agent: 'data_agent',
      phase: 'tool_call',
      status: 'running',
      summary: 'Preparing query',
      tool: 'sql_query',
      children: [{
        id: 'llm-complete',
        parentId: 'iteration-3',
        kind: 'llm',
        title: 'SQL Generation',
        status: 'complete',
        outputSummary: { characterCount: 88, format: 'json' },
        completedAt: '2026-08-18T00:00:02Z',
        updatedAt: '2026-08-18T00:00:02Z',
      }],
      updatedAt: '2026-08-18T00:00:02Z',
    };

    const markup = renderToStaticMarkup(
      <InspectorPanel
        steps={[step]}
        selectedNodeId={null}
        collapsed={false}
        onToggleCollapsed={() => undefined}
      />,
    );

    expect(markup).toContain('SQL Generation');
    expect(markup).toContain('JSON');
    expect(markup).toContain('88');
  });
});
