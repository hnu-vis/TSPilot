import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import App from './App';


describe('App persisted conversation compatibility', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders the workspace when saved history contains an obsolete visualization schema', () => {
    const stored = [{
      id: 'conv_old_visualization',
      title: 'Saved analysis',
      createdAt: '2026-08-14T00:00:00Z',
      updatedAt: '2026-08-14T00:00:00Z',
      selectedTraceStepId: null,
      selectedDatabaseId: null,
      selectedKnowledgeId: null,
      traceSteps: [],
      messages: [{
        id: 'answer_old_visualization',
        role: 'assistant',
        content: 'Saved answer',
        createdAt: '2026-08-14T00:00:00Z',
        answer: {
          summary: 'Saved answer',
          visualizations: [{
            schema_version: '2',
            visualization_id: 'old_chart',
            priority: 'primary',
            title: 'Old chart',
            dataset: { series: [] },
            layers: [],
            bindings: [],
            accessibility: { description: 'Old chart' },
          }],
        },
      }],
    }];
    vi.stubGlobal('localStorage', {
      getItem: (key: string) => key === 'tspilot:v03:conversations' ? JSON.stringify(stored) : null,
      setItem: () => undefined,
      removeItem: () => undefined,
    });

    const markup = renderToStaticMarkup(<App />);

    expect(markup).toContain('Saved answer');
    expect(markup).toContain('data schema is no longer supported');
    expect(markup).toContain('aria-label="Model"');
  });
});
