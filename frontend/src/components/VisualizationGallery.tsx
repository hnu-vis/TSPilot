import { useEffect, useMemo, useRef } from 'react';
import type { EChartsOption } from 'echarts';
import { BarChart, BoxplotChart, LineChart, ScatterChart } from 'echarts/charts';
import {
  AriaComponent,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TooltipComponent,
} from 'echarts/components';
import * as echarts from 'echarts/core';
import type { EChartsType } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import type { Visualization, VisualizationPoint, VisualizationSeries } from '../types';

echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  BoxplotChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkPointComponent,
  MarkLineComponent,
  MarkAreaComponent,
  AriaComponent,
  CanvasRenderer,
]);

type GalleryProps = {
  visualizations: Visualization[];
  activeBindingId: string | null;
  onSelectBinding: (bindingId: string) => void;
};

export function VisualizationGallery({ visualizations, activeBindingId, onSelectBinding }: GalleryProps) {
  const ordered = [...visualizations].sort((left, right) => (
    Number(right.priority === 'primary') - Number(left.priority === 'primary')
  ));
  const activeBinding = ordered
    .flatMap((visualization) => visualization.bindings || [])
    .find((binding) => binding.binding_id === activeBindingId);

  return (
    <div className="answer-visualization-gallery">
      <div className="answer-visualization-grid">
        {ordered.map((visualization) => (
          <VisualizationCard
            key={visualization.visualization_id}
            visualization={visualization}
            activeBindingId={activeBindingId}
            onSelectBinding={onSelectBinding}
          />
        ))}
      </div>
      {activeBinding && (
        <div className="visualization-link-detail" role="status">
          <strong>Linked evidence</strong>
          <span>{activeBinding.source_type}</span>
          {activeBinding.source_ref && <code>{activeBinding.source_ref}</code>}
          {activeBinding.evidence_id && <code>{activeBinding.evidence_id}</code>}
          {activeBinding.locator && Object.keys(activeBinding.locator).length > 0 && (
            <code>{JSON.stringify(activeBinding.locator)}</code>
          )}
        </div>
      )}
    </div>
  );
}

function VisualizationCard({ visualization, activeBindingId, onSelectBinding }: {
  visualization: Visualization;
  activeBindingId: string | null;
  onSelectBinding: (bindingId: string) => void;
}) {
  const isMetric = visualization.template_id === 'metric.single';
  const isTable = visualization.template_id === 'table.detail';
  const metric = visualization.dataset.metric;
  const rows = visualization.dataset.rows || [];
  const columns = visualization.dataset.columns?.length
    ? visualization.dataset.columns
    : Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  return (
    <article className={`answer-visualization-card is-${visualization.priority}`}>
      <div className="answer-visualization-card-header">
        <div>
          <strong>{visualization.title}</strong>
          {visualization.summary && <p>{visualization.summary}</p>}
        </div>
        <span>{formatTemplateLabel(visualization.template_id)}</span>
      </div>
      {isMetric && metric ? (
        <div className="answer-fact-metric" role="group" aria-label={visualization.title}>
          <strong>{formatCell(metric.value)}</strong>
          {metric.unit ? <span>{formatCell(metric.unit)}</span> : null}
          {metric.label ? <small>{formatCell(metric.label)}</small> : null}
        </div>
      ) : isTable ? (
        <AccessibleTable visualization={visualization} rows={rows} columns={columns} onSelectBinding={onSelectBinding} />
      ) : (
        <EChartView
          visualization={visualization}
          activeBindingId={activeBindingId}
          onSelectBinding={onSelectBinding}
        />
      )}
      {!isTable && (visualization.accessibility.table_rows?.length || 0) > 0 && (
        <details className="visualization-data-details">
          <summary>View chart data</summary>
          <AccessibleTable
            visualization={visualization}
            rows={visualization.accessibility.table_rows || []}
            columns={visualization.accessibility.table_columns || []}
            onSelectBinding={onSelectBinding}
          />
        </details>
      )}
    </article>
  );
}

function EChartView({ visualization, activeBindingId, onSelectBinding }: {
  visualization: Visualization;
  activeBindingId: string | null;
  onSelectBinding: (bindingId: string) => void;
}) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);
  const option = useMemo(() => buildVisualizationOption(visualization, activeBindingId), [visualization, activeBindingId]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;
    const chart = echarts.init(host, undefined, { renderer: 'canvas' });
    chartRef.current = chart;
    const handleClick = (params: { data?: unknown }) => {
      const data = isRecord(params.data) ? params.data : null;
      const bindingId = data && typeof data.bindingId === 'string' ? data.bindingId : null;
      if (bindingId) onSelectBinding(bindingId);
    };
    chart.on('click', handleClick);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(host);
    return () => {
      observer.disconnect();
      chart.off('click', handleClick);
      chart.dispose();
      chartRef.current = null;
    };
  }, [onSelectBinding]);

  useEffect(() => {
    chartRef.current?.setOption(option, { notMerge: true, lazyUpdate: true });
  }, [option]);

  return (
    <div
      ref={hostRef}
      className={`answer-echart${visualization.layout === 'facets' ? ' is-faceted' : ''}`}
      style={{ height: visualization.layout === 'facets' ? Math.max(360, (visualization.dataset.series?.length || 1) * 190 + 28) : 340 }}
      role="img"
      aria-label={visualization.accessibility.description || visualization.title}
    />
  );
}

export function buildVisualizationOption(visualization: Visualization, activeBindingId: string | null = null): EChartsOption {
  const template = visualization.template_id;
  const sourceSeries = visualization.dataset.series || [];
  const isRanking = template === 'ranking.topk';
  const isCategory = ['ranking.topk', 'category.comparison', 'distribution.histogram', 'distribution.boxplot'].includes(template);
  const isScatter = template === 'relationship.scatter';
  const faceted = visualization.layout === 'facets' && sourceSeries.length > 1;
  const grids = faceted
    ? sourceSeries.map((_, index) => ({ left: 66, right: 24, top: 34 + index * 190, height: 130 }))
    : [{ left: 62, right: 24, top: 36, bottom: 52, containLabel: false }];
  const xAxis = sourceSeries.map((_series, index) => ({
    type: isRanking ? 'value' : (isCategory ? 'category' : (isScatter ? 'value' : 'time')),
    gridIndex: faceted ? index : 0,
    axisLabel: { hideOverlap: true, color: '#667085', formatter: isCategory ? undefined : formatAxisTime },
    axisLine: { lineStyle: { color: '#d0d5dd' } },
    splitLine: { show: false },
    data: isCategory && !isRanking ? (sourceSeries[index]?.points || []).map((point) => String(point.x ?? '')) : undefined,
  }));
  const yAxis = sourceSeries.map((series, index) => ({
    type: isRanking ? 'category' : 'value',
    gridIndex: faceted ? index : 0,
    name: faceted ? series.name : undefined,
    nameTextStyle: { color: '#667085', align: 'left' },
    scale: true,
    axisLabel: { color: '#667085', formatter: formatAxisNumber },
    splitLine: { lineStyle: { color: '#eef1f4' } },
    data: isRanking ? (sourceSeries[index]?.points || []).map((point) => String(point.x ?? '')) : undefined,
  }));
  const seriesOptions: any[] = [];
  sourceSeries.forEach((series, index) => {
    seriesOptions.push(buildSeriesOption(visualization, series, index, faceted, activeBindingId));
    if (template === 'timeseries.forecast' && series.role === 'forecast') {
      seriesOptions.push(...confidenceBandSeries(series, index, faceted));
    }
  });
  const semanticLayers = visualization.layers || [];
  const factPoints = semanticLayers.flatMap((layer) => (
    ['fact', 'anomaly'].includes(layer.role) && layer.kind === 'point' && !layer.series_id ? layer.points || [] : []
  ));
  if (factPoints.length > 0) {
    seriesOptions.push({
      name: semanticLayers.find((layer) => ['fact', 'anomaly'].includes(layer.role))?.label || 'Highlight',
      type: 'scatter',
      xAxisIndex: 0,
      yAxisIndex: 0,
      symbolSize: (value: unknown, params: { data?: unknown }) => (
        isRecord(params.data) && params.data.bindingId === activeBindingId ? 14 : 10
      ),
      itemStyle: { color: '#dc6803', borderColor: '#fff', borderWidth: 2 },
      label: { show: true, position: 'top', color: '#7a2e0e', formatter: (params: { data?: unknown }) => isRecord(params.data) ? String(params.data.semanticLabel || '') : '' },
      data: factPoints.filter((point) => point.x !== null && point.x !== undefined).map(toEChartsPoint),
      z: 10,
    });
  }
  const rulePoints = semanticLayers.flatMap((layer) => layer.kind === 'rule' ? layer.points || [] : []);
  if (rulePoints.length > 0 && seriesOptions.length > 0) {
    const first = seriesOptions[0] as Record<string, unknown>;
    first.markLine = {
      symbol: 'none',
      label: { color: '#475467', formatter: '{b}' },
      lineStyle: { color: '#98a2b3', type: 'dashed' },
      data: rulePoints.map((point) => ({ name: point.label || '', xAxis: point.x, yAxis: point.x == null ? point.y : undefined })),
    };
  }
  const intervalPoints = semanticLayers.flatMap((layer) => layer.kind === 'area' ? layer.points || [] : []);
  if (intervalPoints.length >= 2 && seriesOptions.length > 0) {
    const first = seriesOptions[0] as Record<string, unknown>;
    first.markArea = {
      itemStyle: { color: 'rgba(220, 104, 3, 0.10)' },
      label: { color: '#7a2e0e' },
      data: [[{ name: intervalPoints[0].label || 'Interval', xAxis: intervalPoints[0].x }, { xAxis: intervalPoints[1].x }]],
    };
  }

  const option = {
    animationDuration: 260,
    aria: { enabled: true, decal: { show: true }, description: visualization.accessibility.description },
    color: ['#087f5b', '#6941c6', '#175cd3', '#dc6803', '#b42318'],
    grid: grids,
    tooltip: { trigger: isScatter ? 'item' : 'axis', confine: true, valueFormatter: formatAxisNumber },
    legend: { type: 'scroll', top: 2, textStyle: { color: '#475467' } },
    xAxis: faceted ? xAxis : xAxis.slice(0, 1),
    yAxis: faceted ? yAxis : yAxis.slice(0, 1),
    dataZoom: sourceSeries.some((series) => (series.points?.length || 0) > 80)
      ? [{ type: 'inside', filterMode: 'none' }, { type: 'slider', height: 18, bottom: 4 }]
      : [],
    series: seriesOptions,
  };
  return option as EChartsOption;
}

function buildSeriesOption(
  visualization: Visualization,
  series: VisualizationSeries,
  index: number,
  faceted: boolean,
  activeBindingId: string | null,
): any {
  const template = visualization.template_id;
  const points = series.points || [];
  if (template === 'distribution.boxplot') {
    return {
      name: series.name,
      type: 'boxplot',
      xAxisIndex: faceted ? index : 0,
      yAxisIndex: faceted ? index : 0,
      data: points.map((point) => [
        point.lower,
        numberFrom(point.metadata?.q1),
        numberFrom(point.metadata?.median) ?? point.y,
        numberFrom(point.metadata?.q3),
        point.upper,
      ]),
    };
  }
  if (['ranking.topk', 'category.comparison', 'distribution.histogram'].includes(template)) {
    return {
      name: series.name,
      type: 'bar',
      xAxisIndex: faceted ? index : 0,
      yAxisIndex: faceted ? index : 0,
      barMaxWidth: 34,
      itemStyle: { borderRadius: [5, 5, 0, 0] },
      emphasis: { focus: 'series' },
      data: points.map((point) => ({
        value: template === 'ranking.topk' ? [point.y, String(point.x ?? '')] : point.y,
        name: String(point.x ?? ''),
        bindingId: point.binding_id,
        semanticLabel: point.label,
      })),
    };
  }
  if (template === 'relationship.scatter') {
    return {
      name: series.name,
      type: 'scatter',
      xAxisIndex: faceted ? index : 0,
      yAxisIndex: faceted ? index : 0,
      symbolSize: 8,
      large: points.length > 1200,
      data: points.map(toEChartsPoint),
    };
  }
  return {
    name: series.name,
    type: 'line',
    xAxisIndex: faceted ? index : 0,
    yAxisIndex: faceted ? index : 0,
    showSymbol: points.length <= 40 || points.some((point) => point.binding_id),
    symbolSize: (value: unknown, params: { data?: unknown }) => (
      isRecord(params.data) && params.data.bindingId === activeBindingId ? 10 : 5
    ),
    smooth: false,
    connectNulls: false,
    lineStyle: series.role === 'forecast' ? { type: 'dashed', width: 2.5 } : { width: 2 },
    data: points.map(toEChartsPoint),
    emphasis: { focus: 'series' },
    z: series.role === 'forecast' ? 5 : 3,
  };
}

function confidenceBandSeries(series: VisualizationSeries, index: number, faceted: boolean): any[] {
  const bounded = (series.points || []).filter((point) => point.lower != null && point.upper != null);
  if (bounded.length === 0) return [];
  const axisIndex = faceted ? index : 0;
  return [
    {
      name: `${series.name} lower`,
      type: 'line',
      xAxisIndex: axisIndex,
      yAxisIndex: axisIndex,
      stack: `confidence_${index}`,
      symbol: 'none',
      lineStyle: { opacity: 0 },
      areaStyle: { opacity: 0 },
      tooltip: { show: false },
      data: bounded.map((point) => [point.x, point.lower]),
      silent: true,
    },
    {
      name: `${series.name} interval`,
      type: 'line',
      xAxisIndex: axisIndex,
      yAxisIndex: axisIndex,
      stack: `confidence_${index}`,
      symbol: 'none',
      lineStyle: { opacity: 0 },
      areaStyle: { color: 'rgba(105, 65, 198, 0.20)' },
      tooltip: { show: false },
      data: bounded.map((point) => [point.x, Number(point.upper) - Number(point.lower)]),
      silent: true,
    },
  ];
}

function AccessibleTable({ visualization, rows, columns, onSelectBinding }: {
  visualization: Visualization;
  rows: Array<Record<string, unknown>>;
  columns: string[];
  onSelectBinding: (bindingId: string) => void;
}) {
  const bindingById = new Map((visualization.bindings || []).map((binding) => [binding.binding_id, binding]));
  return (
    <div className="answer-visualization-table-wrap">
      <table className="answer-data-table answer-visualization-table">
        <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, rowIndex) => {
            const bindingId = typeof row.binding_id === 'string' ? row.binding_id : null;
            const binding = bindingId ? bindingById.get(bindingId) : undefined;
            return (
              <tr
                key={String(row.item_id || rowIndex)}
                onClick={() => binding && onSelectBinding(binding.binding_id)}
                onKeyDown={(event) => {
                  if ((event.key === 'Enter' || event.key === ' ') && binding) onSelectBinding(binding.binding_id);
                }}
                tabIndex={binding ? 0 : undefined}
              >
                {columns.map((column) => <td key={column}>{formatCell(row[column])}</td>)}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function toEChartsPoint(point: VisualizationPoint) {
  return {
    value: [chartValue(point.x), point.y],
    bindingId: point.binding_id,
    semanticLabel: point.label,
    itemStyle: point.binding_id ? { borderWidth: 2, borderColor: '#fff' } : undefined,
  };
}

function chartValue(value: unknown): string | number | null {
  if (typeof value === 'number' || typeof value === 'string') return value;
  return value == null ? null : String(value);
}

function formatAxisTime(value: unknown): string {
  const parsed = typeof value === 'number' && Number.isFinite(value)
    ? value
    : Date.parse(String(value ?? ''));
  if (!Number.isFinite(parsed)) return String(value ?? '');
  const date = new Date(parsed);
  return `${date.getUTCMonth() + 1}/${date.getUTCDate()} ${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')}`;
}

function formatAxisNumber(value: unknown): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '—';
  return new Intl.NumberFormat(undefined, {
    notation: Math.abs(numeric) >= 1_000_000 ? 'compact' : 'standard',
    maximumFractionDigits: 3,
  }).format(numeric);
}

function formatTemplateLabel(templateId: string) {
  return templateId.split('.').map((part) => part[0]?.toUpperCase() + part.slice(1)).join(' · ');
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number') return formatAxisNumber(value);
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function numberFrom(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
