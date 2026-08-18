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

  it('localizes visible chart times while preserving the ISO series coordinates', () => {
    const input = visualization({
      datasets: [{
        dataset_id: 'history', source_ref: 'history', dimensions,
        series: [{
          series_id: 'history', name: '价格', role: 'history',
          points: [{ x: '2023-01-04T23:48:00+00:00', y: 16842.3425 }],
        }],
      }],
      layers: [{
        layer_id: 'history', mark: 'line', role: 'history', source_ref: 'history',
        dataset_id: 'history', series_id: 'history',
      }],
    });
    const option = buildVisualizationOption(input, null, false, 'zh-CN') as Record<string, any>;
    const tooltip = option.tooltip.formatter([{
      axisValue: '2023-01-04T23:48:00+00:00',
      marker: '',
      seriesName: '价格',
      value: ['2023-01-04T23:48:00+00:00', 16842.3425],
    }]);

    expect(option.xAxis[0].axisLabel.formatter(Date.UTC(2023, 0, 4, 23, 48))).toBe('1月4日 23:48');
    expect(tooltip).toContain('2023年1月4日 23:48（UTC）');
    expect(tooltip).not.toContain('2023-01-04T23:48:00+00:00');
    expect(option.series[0].data[0].value[0]).toBe('2023-01-04T23:48:00+00:00');
  });

  it('uses a renderer-owned compact layout without changing grounded series', () => {
    const input = visualization({
      presentation: {
        title: { text: 'Planner title' },
        toolbox: { show: true },
        legend: { textStyle: { fontSize: 18 } },
      },
    });
    const option = buildVisualizationOption(input, null, true) as Record<string, any>;

    expect(option.title.show).toBe(false);
    expect(option.toolbox.show).toBe(false);
    expect(option.grid[0]).toMatchObject({ left: 48, right: 12, top: 52, bottom: 48 });
    expect(option.legend.textStyle.fontSize).toBe(9);
    expect(option.xAxis[0].axisLabel.fontSize).toBe(9);
    expect(option.series[0].data.map((point: Record<string, unknown>) => point.value)).toEqual([
      ['2026-01-01T00:00:00Z', 10], ['2026-01-02T00:00:00Z', 12],
    ]);
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

  it('keeps the full time series while opening a scrollable viewport around a planned interval', () => {
    const points = Array.from({ length: 100 }, (_, index) => ({
      x: `2026-01-${String(Math.floor(index / 4) + 1).padStart(2, '0')}T${String((index % 4) * 6).padStart(2, '0')}:00:00Z`,
      y: 10 + index,
    }));
    const input = visualization({
      presentation: {
        dataZoom: [{
          type: 'slider',
          startValue: '2026-01-08T00:00:00Z',
          endValue: '2026-01-12T00:00:00Z',
        }],
      },
      datasets: [{
        dataset_id: 'history', source_ref: 'history', dimensions,
        series: [{ series_id: 'history', name: 'History', role: 'history', points }],
      }],
      layers: [{
        layer_id: 'history', mark: 'line', role: 'history', source_ref: 'history',
        dataset_id: 'history', series_id: 'history',
      }],
    });

    const desktop = buildVisualizationOption(input) as Record<string, any>;
    const compact = buildVisualizationOption(input, null, true) as Record<string, any>;

    for (const option of [desktop, compact]) {
      expect(option.dataZoom.map((zoom: Record<string, unknown>) => zoom.type)).toEqual(['inside', 'slider']);
      expect(option.dataZoom.every((zoom: Record<string, unknown>) => zoom.filterMode === 'filter')).toBe(true);
      expect(option.dataZoom.every((zoom: Record<string, unknown>) => (
        zoom.startValue === '2026-01-08T00:00:00Z'
        && zoom.endValue === '2026-01-12T00:00:00Z'
      ))).toBe(true);
      expect(option.series[0].data).toHaveLength(100);
    }
  });

  it('creates separate grids and axes for layered facets', () => {
    const base = visualization();
    const option = buildVisualizationOption({ ...base, layout: 'facets' }) as Record<string, any>;
    expect(option.grid).toHaveLength(4);
    expect(option.xAxis).toHaveLength(4);
    expect(option.series.some((item: Record<string, any>) => item.xAxisIndex === 1)).toBe(true);
  });

  it('preserves facet axis cardinality when presentation supplies fewer axes and grids', () => {
    const base = visualization();
    const option = buildVisualizationOption({
      ...base,
      layout: 'facets',
      presentation: {
        xAxis: { name: 'Shared time' },
        yAxis: [{ name: 'Only one presented axis' }],
        grid: [{ top: '10%', height: '80%' }],
      },
    }) as Record<string, any>;

    expect(option.grid).toHaveLength(4);
    expect(option.xAxis).toHaveLength(4);
    expect(option.yAxis).toHaveLength(4);
    expect(option.xAxis.every((axis: Record<string, unknown>, index: number) => axis.gridIndex === index)).toBe(true);
    expect(option.yAxis.every((axis: Record<string, unknown>, index: number) => axis.gridIndex === index)).toBe(true);
    expect(option.series.every((series: Record<string, number>) => series.xAxisIndex < option.xAxis.length)).toBe(true);
    expect(option.series.every((series: Record<string, number>) => series.yAxisIndex < option.yAxis.length)).toBe(true);
  });

  it('renders an LLM-authored shared-axis plan in one canvas even for a legacy facets payload', () => {
    const base = visualization();
    const layers = base.layers?.map((layer, index) => ({
      ...layer,
      presentation: { ...layer.presentation, xAxisIndex: 0, yAxisIndex: index === 2 ? 1 : 0 },
    }));
    const option = buildVisualizationOption({
      ...base,
      layout: 'facets',
      layers,
      presentation: {
        grid: [{ top: '8%', height: '48%' }, { top: '64%', height: '28%' }],
        xAxis: { type: 'time' },
        yAxis: [{ name: 'Value' }, { name: 'Secondary value' }],
      },
    }) as Record<string, any>;

    expect(option.grid).toHaveLength(1);
    expect(option.xAxis).toHaveLength(1);
    expect(option.yAxis).toHaveLength(2);
    expect(option.series.every((series: Record<string, number>) => series.xAxisIndex === 0)).toBe(true);
    expect(option.series.some((series: Record<string, number>) => series.yAxisIndex === 1)).toBe(true);
  });

  it('keeps incompatible single-value summaries inside the chart as annotations', () => {
    const option = buildVisualizationOption(visualization({
      layout: 'facets',
      presentation: {
        xAxis: { type: 'time' },
        yAxis: [{ name: 'Value' }, { name: 'Percent' }],
      },
      datasets: [
        {
          dataset_id: 'history', source_ref: 'history', dimensions,
          series: [{ series_id: 'history', name: 'History', role: 'history', points: [{ x: '2026-01-01', y: 10 }] }],
        },
        {
          dataset_id: 'summary', source_ref: 'summary',
          dimensions: [{ name: 'metric', data_type: 'category', role: 'x' }, { name: 'percent', data_type: 'number', role: 'y' }],
          series: [{ series_id: 'summary', name: 'Change', role: 'summary', unit: '%', points: [{ x: 'change', y: 12.5 }] }],
        },
      ],
      layers: [
        { layer_id: 'history', mark: 'line', role: 'history', source_ref: 'history', dataset_id: 'history', presentation: { xAxisIndex: 0, yAxisIndex: 0 } },
        { layer_id: 'summary', mark: 'bar', role: 'summary', source_ref: 'summary', dataset_id: 'summary', presentation: { xAxisIndex: 0, yAxisIndex: 1 } },
      ],
    })) as Record<string, any>;

    expect(option.grid).toHaveLength(1);
    expect(option.series).toHaveLength(1);
    expect(option.series[0].name).toBe('History');
    expect(option.yAxis).toHaveLength(1);
    expect(option.graphic[0].style.text).toContain('Change: 12.5 %');
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
