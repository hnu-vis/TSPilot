import { afterEach, describe, expect, it, vi } from 'vitest';
import { extractFinalAnswer, fetchVisualizationData } from './api';

afterEach(() => vi.unstubAllGlobals());

describe('extractFinalAnswer', () => {
  it('normalizes optional answer collections at the stream boundary', () => {
    const answer = extractFinalAnswer({
      answer: {
        summary: 'done',
        sections: null,
        references: { invalid: true },
        visualizations: ['invalid', { schema_version: '2' }],
      },
    });

    expect(answer).toMatchObject({ summary: 'done', sections: [], references: [] });
    expect(answer?.visualizations).toEqual([{ schema_version: '2' }]);
  });

  it('rejects payloads without displayable answer text', () => {
    expect(extractFinalAnswer({ answer: { summary: 42 } })).toBeNull();
  });
});

describe('fetchVisualizationData', () => {
  it('loads the complete V5 native option from its artifact data_ref', async () => {
    const payload = {
      schema_version: '5', chart_type: 'echarts', visualization_id: 'viz', title: 'Trend',
      option: { dataset: { source: [{ time: '2026-01-01', value: 10 }] }, series: { type: 'line' } },
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => payload });
    vi.stubGlobal('fetch', fetchMock);
    await expect(fetchVisualizationData('/api/v1/visualizations/viz/data')).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/visualizations/viz/data');
  });
});
