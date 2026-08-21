import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Visualization } from '../types';
import { buildVisualizationOption, isRenderableVisualization, VisualizationGallery } from './VisualizationGallery';

export function visualization(overrides: Partial<Visualization> = {}): Visualization {
  return {
    schema_version: '4', chart_type: 'line', visualization_id: 'viz_test', purpose: 'verify trend',
    priority: 'primary', title: 'Price trend', source_refs: ['evidence:evi'], required_roles: ['history'],
    data_views: [{
      view_id: 'history', source_ref: 'view:evidence:evi:default',
      fields: [
        { name: 'timestamp', data_type: 'time', semantic_role: 'time' },
        { name: 'value', data_type: 'number', semantic_role: 'price' },
        { name: 'lower', data_type: 'number', semantic_role: 'lower' },
        { name: 'upper', data_type: 'number', semantic_role: 'upper' },
        { name: 'start', data_type: 'time', semantic_role: 'start' },
        { name: 'end', data_type: 'time', semantic_role: 'end' },
        { name: 'note', data_type: 'string', semantic_role: 'note' },
      ],
      records: [
        { record_id: 'r1', values: { timestamp: '2026-01-01T00:00:00Z', value: 10, lower: 9, upper: 11, start: '2026-01-01T00:00:00Z', end: '2026-01-02T00:00:00Z', note: 'rising' }, binding_id: 'b1' },
        { record_id: 'r2', values: { timestamp: '2026-01-02T00:00:00Z', value: 12, lower: 10, upper: 14, start: '2026-01-01T00:00:00Z', end: '2026-01-02T00:00:00Z', note: 'rising' } },
      ],
    }],
    x_axis: { axis_id: 'x', data_type: 'time', label: 'Time' },
    y_axes: [{ axis_id: 'price', measure: 'price', unit: 'USD', scale: 'linear' }],
    lines: [{ component_id: 'history', role: 'history', importance: 'primary', source_ref: 'history', view_id: 'history', x_field: 'timestamp', y_field: 'value', y_axis_id: 'price', line_style: 'solid', symbol: 'none' }],
    points: [{ component_id: 'peak', role: 'peak', importance: 'highlight', source_ref: 'history', view_id: 'history', x_field: 'timestamp', y_field: 'value', y_axis_id: 'price', symbol: 'diamond', size: 'medium' }],
    bands: [{ component_id: 'range', role: 'range', importance: 'support', source_ref: 'history', view_id: 'history', x_field: 'timestamp', lower_field: 'lower', upper_field: 'upper', y_axis_id: 'price' }],
    intervals: [{ component_id: 'window', role: 'window', importance: 'highlight', source_ref: 'history', view_id: 'history', start_field: 'start', end_field: 'end' }],
    reference_lines: [{ component_id: 'reference', role: 'reference', importance: 'support', source_ref: 'history', view_id: 'history', value_field: 'value', y_axis_id: 'price' }],
    annotations: [{ component_id: 'note', role: 'note', importance: 'support', source_ref: 'history', view_id: 'history', content_field: 'note', target: { target_type: 'chart' } }],
    legend: { visible: true, toggle_components: true, position: 'top' },
    tooltip: { mode: 'axis', show_source: true }, zoom: { enabled: true },
    bindings: [{ binding_id: 'b1', source_type: 'insight', insight_id: 'i1' }],
    accessibility: { description: 'Price over time.', table_columns: ['timestamp', 'value'], table_rows: [] },
    ...overrides,
  };
}

describe('LineChart V4 renderer', () => {
  it('builds lines, points, bands, intervals, references and annotations', () => {
    const option = buildVisualizationOption(visualization()) as Record<string, any>;
    const series = option.series as Array<Record<string, any>>;
    expect(series.some((item) => item.id === 'history' && item.type === 'line')).toBe(true);
    expect(series.some((item) => item.id === 'peak' && item.type === 'scatter')).toBe(true);
    expect(series.some((item) => item.id === 'range')).toBe(true);
    expect(series.find((item) => item.id === 'history')!.markArea.data).toHaveLength(2);
    expect(series.find((item) => item.id === 'history')!.markLine.data).toHaveLength(1);
    expect(option.graphic).toHaveLength(1);
    expect(option.dataZoom).toHaveLength(2);
  });

  it('preserves evidence bindings on ECharts data', () => {
    const option = buildVisualizationOption(visualization(), 'b1') as Record<string, any>;
    const history = option.series.find((item: any) => item.id === 'history');
    expect(history.data[0].bindingId).toBe('b1');
    expect(history.data[0].itemStyle.borderWidth).toBe(3);
  });

  it('keeps internal binding identifiers out of the user-facing evidence detail', () => {
    const internalId = 'ins_ana_1a08a9492d679b29_max_drop_window_8965a0791047';
    const markup = renderToStaticMarkup(<VisualizationGallery
      visualizations={[visualization({
        title: `Window ${internalId}`,
        bindings: [{
          binding_id: 'b1', source_type: 'insight', insight_id: internalId,
          source_ref: `insight:${internalId}`, evidence_id: 'evi_1234567890abcdef',
        }],
        accessibility: { description: `Chart for ${internalId}.` },
      })]}
      activeBindingId="b1"
      onSelectBinding={() => undefined}
    />);
    expect(markup).toContain('Linked evidence');
    expect(markup).toContain('Window analysis result');
    expect(markup).not.toContain(internalId);
    expect(markup).not.toContain('evi_1234567890abcdef');
  });

  it('rejects V3 instead of keeping a compatibility renderer', () => {
    expect(isRenderableVisualization({ schema_version: '3', datasets: [], layers: [] })).toBe(false);
    const markup = renderToStaticMarkup(<VisualizationGallery visualizations={[{ schema_version: '3' } as unknown as Visualization]} activeBindingId={null} onSelectBinding={() => undefined} />);
    expect(markup).toContain('schema is no longer supported');
  });
});
