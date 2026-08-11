import { describe, expect, it } from 'vitest';
import type { Visualization } from '../types';
import { buildVisualizationOption } from './VisualizationGallery';

function visualization(overrides: Partial<Visualization> = {}): Visualization {
  return {
    schema_version: '2',
    visualization_id: 'viz_test',
    template_id: 'timeseries.forecast',
    purpose: 'show history and forecast',
    priority: 'primary',
    title: 'Forecast',
    dataset: {
      dimensions: [
        { name: 'timestamp', data_type: 'time', role: 'x' },
        { name: 'value', data_type: 'number', role: 'y' },
      ],
      series: [
        {
          series_id: 'historical', name: 'Historical', role: 'historical',
          points: [{ x: '2026-01-01T00:00:00Z', y: 10 }, { x: '2026-01-02T00:00:00Z', y: 12 }],
        },
        {
          series_id: 'forecast', name: 'Forecast', role: 'forecast',
          points: [
            { x: '2026-01-03T00:00:00Z', y: 13, lower: 11, upper: 15, binding_id: 'forecast:1' },
            { x: '2026-01-04T00:00:00Z', y: 14, lower: 12, upper: 16, binding_id: 'forecast:2' },
          ],
        },
      ],
    },
    layers: [
      { kind: 'line', role: 'context', series_id: 'historical' },
      { kind: 'line', role: 'forecast', series_id: 'forecast' },
      { kind: 'band', role: 'confidence', series_id: 'forecast' },
      { kind: 'rule', role: 'forecast', points: [{ x: '2026-01-03T00:00:00Z', label: 'Forecast starts' }] },
    ],
    bindings: [
      { binding_id: 'forecast:1', source_type: 'prediction_point' },
      { binding_id: 'forecast:2', source_type: 'prediction_point' },
    ],
    layout: 'overlay',
    accessibility: { description: 'History and two predictions.' },
    ...overrides,
  };
}

describe('buildVisualizationOption', () => {
  it('builds one forecast view with historical, forecast, interval, and boundary layers', () => {
    const option = buildVisualizationOption(visualization()) as Record<string, any>;
    const series = option.series as Array<Record<string, any>>;

    expect(series.some((item) => item.name === 'Historical' && item.type === 'line')).toBe(true);
    expect(series.some((item) => item.name === 'Forecast' && item.lineStyle?.type === 'dashed')).toBe(true);
    expect(series.some((item) => String(item.name).includes('interval'))).toBe(true);
    expect(series[0].markLine.data[0].xAxis).toBe('2026-01-03T00:00:00Z');
    expect(option.xAxis[0].axisLabel.formatter(Date.UTC(2026, 0, 3, 0, 15))).toBe('1/3 00:15');
  });

  it('renders ranking as a horizontal bar chart', () => {
    const option = buildVisualizationOption(visualization({
      template_id: 'ranking.topk',
      dataset: {
        series: [{
          series_id: 'ranking', name: 'Top 3', role: 'ranking',
          points: [{ x: 'A', y: 30 }, { x: 'B', y: 20 }, { x: 'C', y: 10 }],
        }],
      },
      layers: [{ kind: 'bar', role: 'fact', series_id: 'ranking' }],
    })) as Record<string, any>;

    expect(option.xAxis[0].type).toBe('value');
    expect(option.yAxis[0].type).toBe('category');
    expect(option.series[0].data[0].value).toEqual([30, 'A']);
  });

  it('creates separate grids and axes for an unreadable shared scale', () => {
    const base = visualization();
    const option = buildVisualizationOption(visualization({
      layout: 'facets',
      dataset: {
        ...base.dataset,
        series: [
          { series_id: 'small', name: 'Small', role: 'comparison', points: [{ x: '2026-01-01', y: 1 }, { x: '2026-01-02', y: 1.1 }] },
          { series_id: 'large', name: 'Large', role: 'comparison', points: [{ x: '2026-01-01', y: 1_000_000 }, { x: '2026-01-02', y: 2_000_000 }] },
        ],
      },
      layers: [],
    })) as Record<string, any>;

    expect(option.grid).toHaveLength(2);
    expect(option.xAxis).toHaveLength(2);
    expect(option.series[1].xAxisIndex).toBe(1);
  });
});
