import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Visualization } from '../types';
import {
  bindingIdFromClickData,
  bindingLocations,
  isRenderableVisualization,
  VisualizationGallery,
  withTrustedDisplaySettings,
} from './VisualizationGallery';

export function visualization(overrides: Partial<Visualization> = {}): Visualization {
  return {
    schema_version: '5', chart_type: 'echarts', visualization_id: 'viz_test', purpose: 'verify trend',
    priority: 'primary', title: 'Price trend', source_refs: ['evidence:evi'],
    option: {
      dataset: [{ id: 'prices', source: [
        { timestamp: '2026-01-01T00:00:00Z', value: 10, bindingId: 'b1' },
        { timestamp: '2026-01-02T00:00:00Z', value: 12, bindingId: 'b2' },
      ] }],
      xAxis: { type: 'time' }, yAxis: { type: 'value' },
      series: [{ name: 'Price', type: 'line', datasetId: 'prices', encode: { x: 'timestamp', y: 'value' } }],
    },
    bindings: [{ binding_id: 'b1', source_type: 'insight', insight_id: 'i1' }],
    accessibility: { description: 'Price over time.', table_columns: ['timestamp', 'value'], table_rows: [] },
    ...overrides,
  };
}

describe('native ECharts V5 renderer', () => {
  it('passes native chart semantics through and adds only trusted display settings', () => {
    const chart = visualization();
    const option = withTrustedDisplaySettings(chart) as Record<string, any>;
    expect(option.dataset).toBe(chart.option.dataset);
    expect(option.series).toBe(chart.option.series);
    expect(option.useUTC).toBe(true);
    expect(option.aria.description).toBe('Price over time.');
  });

  it('finds dataset rows for evidence highlighting and click binding', () => {
    const chart = visualization();
    expect(bindingLocations(chart.option, 'b2')).toEqual([{ seriesIndex: 0, dataIndex: 1 }]);
    expect(bindingIdFromClickData({ value: 12, bindingId: 'b2' })).toBe('b2');
    expect(bindingIdFromClickData([12])).toBeNull();
  });

  it('keeps internal binding identifiers out of user-facing evidence detail', () => {
    const internalId = 'ins_ana_1a08a9492d679b29_max_drop_window_8965a0791047';
    const markup = renderToStaticMarkup(<VisualizationGallery
      visualizations={[visualization({
        title: `Window ${internalId}`,
        bindings: [{ binding_id: 'b1', source_type: 'insight', insight_id: internalId, source_ref: `insight:${internalId}` }],
        accessibility: { description: `Chart for ${internalId}.` },
      })]}
      activeBindingId="b1"
      onSelectBinding={() => undefined}
    />);
    expect(markup).toContain('Linked evidence');
    expect(markup).toContain('Window analysis result');
    expect(markup).not.toContain(internalId);
  });

  it('accepts V5 and rejects old V4 without a compatibility renderer', () => {
    expect(isRenderableVisualization(visualization())).toBe(true);
    expect(isRenderableVisualization({ schema_version: '4', chart_type: 'line' })).toBe(false);
    const markup = renderToStaticMarkup(<VisualizationGallery visualizations={[{ schema_version: '4' } as unknown as Visualization]} activeBindingId={null} onSelectBinding={() => undefined} />);
    expect(markup).toContain('schema is no longer supported');
  });
});
