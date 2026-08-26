import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import type { Visualization } from '../types';
import {
  annotationLegendItems,
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
    const chart = visualization({ option: {
      ...visualization().option,
      legend: { show: true, data: ['Price'] },
      series: [{
        name: 'Price', type: 'line', datasetId: 'prices', encode: { x: 'timestamp', y: 'value' },
        markPoint: { data: [{ name: 'Peak' }] },
        markArea: { data: [[{ name: 'Interval' }, {}]] },
      }],
    } });
    const option = withTrustedDisplaySettings(chart) as Record<string, any>;
    expect(option.dataset).toBe(chart.option.dataset);
    expect(option.legend.show).toBe(false);
    expect(option.series[0].markPoint.itemStyle.color).toBe('#ee6666');
    expect(option.series[0].markArea.itemStyle).toMatchObject({ color: '#91cc75', opacity: 0.2 });
    expect(option.useUTC).toBe(true);
    expect(option.aria.description).toBe('Price over time.');
  });

  it('applies interactive external legend visibility to series and annotations', () => {
    const chart = visualization({ option: {
      ...visualization().option,
      legend: { show: true, data: ['Price'] },
      series: [{
        name: 'Price', type: 'line', datasetId: 'prices', encode: { x: 'timestamp', y: 'value' },
        markPoint: { data: [{ name: 'Peak' }] },
        markArea: { data: [[{ name: 'Interval' }, {}]] },
        markLine: { data: [{ name: 'Average' }] },
      }],
    } });
    const option = withTrustedDisplaySettings(
      chart,
      'en',
      new Set(['series:Price', 'point:Peak', 'interval:Interval', 'reference:Average']),
    ) as Record<string, any>;
    expect(option.legend.selected.Price).toBe(false);
    expect(option.series[0].markPoint.data).toEqual([]);
    expect(option.series[0].markArea.data).toEqual([]);
    expect(option.series[0].markLine.data).toEqual([]);
  });

  it('formats UTC time axes with an explicit localized date', () => {
    const option = withTrustedDisplaySettings(visualization(), 'zh-CN') as Record<string, any>;
    const formatter = option.xAxis.axisLabel.formatter as (value: number) => string;
    expect(formatter(Date.UTC(2026, 0, 2, 6, 30))).toBe('01月02日 06:30');
    expect(option.xAxis.axisLabel.hideOverlap).toBe(true);
  });

  it('finds dataset rows for evidence highlighting and click binding', () => {
    const chart = visualization();
    expect(bindingLocations(chart.option, 'b2')).toEqual([{ seriesIndex: 0, dataIndex: 1 }]);
    expect(bindingIdFromClickData({ value: 12, bindingId: 'b2' })).toBe('b2');
    expect(bindingIdFromClickData([12])).toBeNull();
  });

  it('builds a deduplicated legend for rendered Insight annotations', () => {
    const chart = visualization({
      option: {
        color: ['#5470c6'],
        dataset: [{ id: 'prices', source: [] }],
        xAxis: { type: 'time' }, yAxis: { type: 'value' },
        series: [{
          name: 'Price', type: 'line', datasetId: 'prices', encode: { x: 'timestamp', y: 'value' },
          markPoint: { data: [{ name: 'Monthly low' }, { name: 'Monthly low' }, { name: 'Rebound peak' }] },
          markArea: { data: [[{ name: 'Rebound interval' }, {}]] },
          markLine: { data: [{ name: 'Threshold' }] },
        }],
      },
    });
    expect(annotationLegendItems(chart.option)).toEqual([
      { kind: 'series', name: 'Price', color: '#5470c6' },
      { kind: 'point', name: 'Monthly low', color: '#ee6666' },
      { kind: 'point', name: 'Rebound peak', color: '#ee6666' },
      { kind: 'interval', name: 'Rebound interval', color: '#91cc75' },
      { kind: 'reference', name: 'Threshold', color: '#fac858' },
    ]);
    const markup = renderToStaticMarkup(<VisualizationGallery
      visualizations={[chart]} activeBindingId={null} onSelectBinding={() => undefined}
    />);
    expect(markup).toContain('answer-annotation-legend');
    expect(markup.match(/Monthly low/g)).toHaveLength(1);
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
