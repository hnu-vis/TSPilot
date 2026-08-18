import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Visualization } from '../types';
import { buildVisualizationOption, isRenderableVisualization, VisualizationGallery } from './VisualizationGallery';

const dimensions = [
  { name: 'timestamp', data_type: 'time', role: 'x' },
  { name: 'value', data_type: 'number', role: 'y' },
];

function visualization(overrides: Partial<Visualization> = {}): Visualization {
  return {
    schema_version: '3',
    visualization_id: 'viz_test',
    purpose: 'show history, forecast, uncertainty, and boundary',
    priority: 'primary',
    title: 'Forecast',
    source_refs: ['view:evidence:evi:default', 'view:forecast:fc:points'],
    required_roles: ['historical', 'forecast', 'confidence', 'boundary'],
    datasets: [
      {
        dataset_id: 'history', source_ref: 'view:evidence:evi:default', dimensions,
        series: [{ series_id: 'historical', name: 'Historical', role: 'historical', points: [
          { x: '2026-01-01T00:00:00Z', y: 10 }, { x: '2026-01-02T00:00:00Z', y: 12 },
        ] }],
      },
      {
        dataset_id: 'forecast', source_ref: 'view:forecast:fc:points', dimensions,
        series: [{ series_id: 'forecast', name: 'Forecast', role: 'forecast', points: [
          { x: '2026-01-03T00:00:00Z', y: 13, binding_id: 'forecast:1' },
          { x: '2026-01-04T00:00:00Z', y: 14, binding_id: 'forecast:2' },
        ] }],
      },
      {
        dataset_id: 'interval', source_ref: 'view:forecast:fc:interval', dimensions,
        series: [{ series_id: 'confidence', name: 'Confidence', role: 'confidence', points: [
          { x: '2026-01-03T00:00:00Z', lower: 11, upper: 15 },
          { x: '2026-01-04T00:00:00Z', lower: 12, upper: 16 },
        ] }],
      },
      {
        dataset_id: 'boundary', source_ref: 'insight:boundary', dimensions,
        series: [{ series_id: 'boundary', name: 'Boundary', role: 'boundary', points: [{ x: '2026-01-03T00:00:00Z' }] }],
      },
    ],
    layers: [
      { layer_id: 'l1', mark: 'line', role: 'historical', source_ref: 'view:evidence:evi:default', dataset_id: 'history', series_id: 'historical' },
      { layer_id: 'l2', mark: 'line', role: 'forecast', source_ref: 'view:forecast:fc:points', dataset_id: 'forecast', series_id: 'forecast' },
      { layer_id: 'l3', mark: 'band', role: 'confidence', source_ref: 'view:forecast:fc:interval', dataset_id: 'interval', series_id: 'confidence' },
      { layer_id: 'l4', mark: 'rule', role: 'boundary', source_ref: 'insight:boundary', dataset_id: 'boundary', series_id: 'boundary', points: [{ x: '2026-01-03T00:00:00Z', label: 'Forecast starts' }] },
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
  it('builds a generic layered forecast view', () => {
    const option = buildVisualizationOption(visualization()) as Record<string, any>;
    const series = option.series as Array<Record<string, any>>;

    expect(series.some((item) => item.name === 'Historical' && item.type === 'line')).toBe(true);
    expect(series.some((item) => item.name === 'Forecast' && item.type === 'line')).toBe(true);
    expect(series.some((item) => String(item.name).includes('interval'))).toBe(true);
    expect(series[0].markLine.data[0].xAxis).toBe('2026-01-03T00:00:00Z');
    expect(option.xAxis[0].axisLabel.formatter(Date.UTC(2026, 0, 3, 0, 15))).toBe('1/3 00:15');
  });

  it('renders a bar layer from its own dataset', () => {
    const option = buildVisualizationOption(visualization({
      datasets: [{
        dataset_id: 'ranking', source_ref: 'insight:top3',
        dimensions: [{ name: 'label', data_type: 'category', role: 'x' }, { name: 'value', data_type: 'number', role: 'y' }],
        series: [{ series_id: 'ranking', name: 'Top 3', role: 'ranking', points: [{ x: 'A', y: 30 }, { x: 'B', y: 20 }, { x: 'C', y: 10 }] }],
      }],
      layers: [{ layer_id: 'ranking', mark: 'bar', role: 'ranking', source_ref: 'insight:top3', dataset_id: 'ranking', series_id: 'ranking' }],
    })) as Record<string, any>;

    expect(option.xAxis[0].type).toBe('category');
    expect(option.series[0].type).toBe('bar');
    expect(option.series[0].data[0].value).toEqual(['A', 30]);
  });

  it('honors LLM-selected line presentation without changing grounded points', () => {
    const base = visualization();
    const layers = base.layers?.map((layer) => layer.role === 'historical' ? {
      ...layer,
      presentation: {
        smooth: true,
        step: 'middle',
        showSymbol: false,
        lineStyle: { color: '#b42318', width: 5, opacity: 0.6, type: 'dashed' },
        emphasis: { focus: 'self' },
      },
    } : layer);
    const option = buildVisualizationOption({ ...base, layers }) as Record<string, any>;
    const historical = option.series.find((item: Record<string, unknown>) => item.name === 'Historical');

    expect(historical.smooth).toBe(true);
    expect(historical.step).toBe('middle');
    expect(historical.lineStyle).toMatchObject({ color: '#b42318', width: 5, opacity: 0.6, type: 'dashed' });
    expect(historical.emphasis.focus).toBe('self');
    expect(historical.data.map((point: Record<string, unknown>) => point.value)).toEqual([
      ['2026-01-01T00:00:00Z', 10], ['2026-01-02T00:00:00Z', 12],
    ]);
  });

  it('creates separate grids and axes for layered facets', () => {
    const base = visualization();
    const option = buildVisualizationOption({ ...base, layout: 'facets' }) as Record<string, any>;
    expect(option.grid).toHaveLength(4);
    expect(option.xAxis).toHaveLength(4);
    expect(option.series.some((item: Record<string, any>) => item.xAxisIndex === 1)).toBe(true);
  });

  it('passes renderer-native marks, multi-field encodings, and presentation through grounded datasets', () => {
    const option = buildVisualizationOption(visualization({
      datasets: [{
        dataset_id: 'ohlc', source_ref: 'view:evidence:ohlc:default', dimensions,
        series: [{ series_id: 'ohlc', name: 'OHLC', role: 'ohlc', points: [{
          x: '2026-01-01T00:00:00Z', y: 10,
          metadata: { timestamp: '2026-01-01T00:00:00Z', open: 10, close: 12, low: 9, high: 13 },
        }] }],
      }],
      layers: [{
        layer_id: 'ohlc', mark: 'candlestick', role: 'ohlc',
        source_ref: 'view:evidence:ohlc:default', dataset_id: 'ohlc', series_id: 'ohlc',
        encoding: { x: 'timestamp', y: ['open', 'close', 'low', 'high'] },
        presentation: { itemStyle: { color: '#087f5b' }, emphasis: { focus: 'series' } },
      }],
    })) as Record<string, any>;

    expect(option.series[0].type).toBe('candlestick');
    expect(option.series[0].datasetId).toBe('ohlc_ohlc');
    expect(option.series[0].encode.y).toEqual(['open', 'close', 'low', 'high']);
    expect(option.series[0].emphasis.focus).toBe('series');
    expect(option.dataset[0].source[0].high).toBe(13);
  });

  it('applies chart-level presentation while grounded datasets and series remain authoritative', () => {
    const option = buildVisualizationOption(visualization({
      presentation: {
        visualMap: { type: 'continuous', calculable: true },
        dataZoom: [{ type: 'inside', yAxisIndex: [0] }],
        dataset: [{ id: 'forged', source: [{ value: 999 }] }],
        series: [{ type: 'pie', data: [999] }],
      },
    })) as Record<string, any>;

    expect(option.visualMap.type).toBe('continuous');
    expect(option.dataZoom[0].yAxisIndex).toEqual([0]);
    expect(option.dataset).toEqual([]);
    expect(option.series.some((item: Record<string, unknown>) => item.name === 'Historical')).toBe(true);
  });

  it('isolates an obsolete persisted visualization instead of crashing the page', () => {
    const obsolete = {
      schema_version: '2',
      visualization_id: 'old_chart',
      priority: 'primary',
      title: 'Old chart',
      dataset: { series: [] },
      layers: [],
      bindings: [],
      accessibility: { description: 'Old chart' },
    } as unknown as Visualization;

    expect(isRenderableVisualization(obsolete)).toBe(false);
    expect(() => renderToStaticMarkup(
      <VisualizationGallery activeBindingId={null} onSelectBinding={() => undefined} visualizations={[obsolete]} />,
    )).not.toThrow();
    expect(renderToStaticMarkup(
      <VisualizationGallery activeBindingId={null} onSelectBinding={() => undefined} visualizations={[obsolete]} />,
    )).toContain('data schema is no longer supported');
  });

  it('shows a loading state for a full-data artifact descriptor', () => {
    const descriptor = visualization({
      data_ref: '/api/v1/visualizations/viz_test/data',
      datasets: visualization().datasets.map((dataset) => ({ ...dataset, series: [], rows: [] })),
      layers: visualization().layers?.map((layer) => ({ ...layer, points: [] })),
    });
    const markup = renderToStaticMarkup(
      <VisualizationGallery activeBindingId={null} onSelectBinding={() => undefined} visualizations={[descriptor]} />,
    );
    expect(markup).toContain('Loading complete visualization data');
    expect(markup).not.toContain('answer-echart');
  });
});
