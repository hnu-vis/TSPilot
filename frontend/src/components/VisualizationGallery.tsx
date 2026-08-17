import { useEffect, useMemo, useRef, useState } from 'react';
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
import { fetchVisualizationData } from '../services/api';
import { useI18n } from '../i18n';

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
  const { t } = useI18n();
  const [loaded, setLoaded] = useState<Record<string, Visualization>>({});
  const [loadErrors, setLoadErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    const controller = new AbortController();
    visualizations.forEach((visualization) => {
      if (!visualization.data_ref || loaded[visualization.visualization_id]) return;
      void fetchVisualizationData(visualization.data_ref)
        .then((payload) => {
          if (!controller.signal.aborted) {
            setLoaded((current) => ({ ...current, [visualization.visualization_id]: payload }));
          }
        })
        .catch((error) => {
          if (!controller.signal.aborted) {
            setLoadErrors((current) => ({
              ...current,
              [visualization.visualization_id]: error instanceof Error ? error.message : t('Unable to load chart data.'),
            }));
          }
        });
    });
    return () => controller.abort();
  }, [visualizations, loaded]);

  const hydrated = visualizations.map((visualization) => loaded[visualization.visualization_id] || visualization);
  const pendingIds = new Set(visualizations
    .filter((visualization) => visualization.data_ref && !loaded[visualization.visualization_id] && !loadErrors[visualization.visualization_id])
    .map((visualization) => visualization.visualization_id));
  const renderable = hydrated.filter((visualization) => !pendingIds.has(visualization.visualization_id) && isRenderableVisualization(visualization));
  const rejectedCount = hydrated.filter((visualization) => !pendingIds.has(visualization.visualization_id) && !isRenderableVisualization(visualization)).length;
  const ordered = [...renderable].sort((left, right) => (
    Number(right.priority === 'primary') - Number(left.priority === 'primary')
  ));
  const activeBinding = ordered
    .flatMap((visualization) => visualization.bindings || [])
    .find((binding) => binding.binding_id === activeBindingId);

  return (
    <div className="answer-visualization-gallery">
      {rejectedCount > 0 && (
        <div className="answer-visualization-unavailable" role="status">
          {t('{count} saved visualizations cannot be displayed because the data schema is no longer supported.', { count: rejectedCount })}
        </div>
      )}
      {Object.values(loadErrors).map((message) => (
        <div className="answer-visualization-unavailable" role="status" key={message}>{message}</div>
      ))}
      {pendingIds.size > 0 && (
        <div className="answer-visualization-unavailable" role="status">{t('Loading complete visualization data…')}</div>
      )}
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
          <strong>{t('Linked evidence')}</strong>
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

export function isRenderableVisualization(value: unknown): value is Visualization {
  if (!isRecord(value) || value.schema_version !== '3') return false;
  if (
    typeof value.visualization_id !== 'string'
    || typeof value.title !== 'string'
    || !['primary', 'supporting'].includes(String(value.priority))
    || !Array.isArray(value.datasets)
    || !Array.isArray(value.layers)
    || !Array.isArray(value.bindings)
    || !isRecord(value.accessibility)
    || typeof value.accessibility.description !== 'string'
  ) return false;

  const datasetsAreValid = value.datasets.every((dataset) => (
    isRecord(dataset)
    && typeof dataset.dataset_id === 'string'
    && typeof dataset.source_ref === 'string'
    && (dataset.series === undefined || Array.isArray(dataset.series))
    && (dataset.rows === undefined || Array.isArray(dataset.rows))
    && (dataset.columns === undefined || Array.isArray(dataset.columns))
  ));
  if (!datasetsAreValid) return false;

  const datasetIds = new Set(value.datasets.map((dataset) => String(dataset.dataset_id)));
  return value.layers.every((layer) => (
    isRecord(layer)
    && typeof layer.layer_id === 'string'
    && typeof layer.mark === 'string'
    && typeof layer.role === 'string'
    && typeof layer.dataset_id === 'string'
    && datasetIds.has(layer.dataset_id)
    && (layer.points === undefined || Array.isArray(layer.points))
  ));
}

function VisualizationCard({ visualization, activeBindingId, onSelectBinding }: {
  visualization: Visualization;
  activeBindingId: string | null;
  onSelectBinding: (bindingId: string) => void;
}) {
  const { t } = useI18n();
  const marks = (visualization.layers || []).map((layer) => layer.mark);
  const isMetric = marks.length === 1 && marks[0] === 'text';
  const isTable = marks.length > 0 && marks.every((mark) => mark === 'table');
  const metric = visualization.datasets.find((dataset) => dataset.metric)?.metric;
  const metricDisplay = metric ? normalizeMetricDisplay(metric) : null;
  const rows = visualization.datasets.flatMap((dataset) => dataset.rows || []);
  const configuredColumns = visualization.datasets.flatMap((dataset) => dataset.columns || []);
  const columns = configuredColumns.length
    ? Array.from(new Set(configuredColumns))
    : Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  return (
    <article className={`answer-visualization-card is-${visualization.priority}`}>
      <div className="answer-visualization-card-header">
        <div>
          <strong>{visualization.title}</strong>
          {visualization.summary && <p>{visualization.summary}</p>}
        </div>
        <span>{Array.from(new Set(marks)).map(formatMarkLabel).join(' · ')}</span>
      </div>
      {isMetric && metricDisplay ? (
        <div className="answer-insight-metric" role="group" aria-label={visualization.title}>
          {metricDisplay.value !== undefined && (
            <div className="answer-insight-metric-value">
              <strong>{formatCell(metricDisplay.value)}</strong>
              {metricDisplay.unit ? <span>{formatCell(metricDisplay.unit)}</span> : null}
            </div>
          )}
          {metricDisplay.label ? <small>{formatCell(metricDisplay.label)}</small> : null}
          {metricDisplay.context.length > 0 && (
            <dl className="answer-insight-metric-context">
              {metricDisplay.context.map(([key, value]) => (
                <div key={key}>
                  <dt>{formatFieldLabel(key)}</dt>
                  <dd>{formatCell(value)}</dd>
                </div>
              ))}
            </dl>
          )}
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
          <summary>{t('View chart data')}</summary>
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

function normalizeMetricDisplay(metric: Record<string, unknown>) {
  const nested = isRecord(metric.value) ? metric.value : null;
  if (!nested) {
    return {
      value: metric.value,
      unit: metric.unit,
      label: metric.label,
      context: [] as Array<[string, unknown]>,
    };
  }

  const entries = Object.entries(nested).filter(([, value]) => isDisplayScalar(value));
  const numericEntries = entries.filter(([, value]) => typeof value === 'number' && Number.isFinite(value));
  const primary = numericEntries.length === 1 ? numericEntries[0] : null;
  return {
    value: primary?.[1],
    unit: metric.unit,
    label: metric.label || (primary ? formatFieldLabel(primary[0]) : undefined),
    context: entries.filter(([key]) => key !== primary?.[0]),
  };
}

function isDisplayScalar(value: unknown): value is string | number | boolean | null {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value);
}

function formatFieldLabel(value: string) {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
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
      style={{ height: visualization.layout === 'facets' ? Math.max(360, Math.max(1, (visualization.layers || []).filter((layer) => layer.mark !== 'table').length) * 190 + 28) : 340 }}
      role="img"
      aria-label={visualization.accessibility.description || visualization.title}
    />
  );
}

export function buildVisualizationOption(visualization: Visualization, activeBindingId: string | null = null): EChartsOption {
  const layers = (visualization.layers || []).filter((layer) => layer.mark !== 'table');
  const datasets = new Map(visualization.datasets.map((dataset) => [dataset.dataset_id, dataset]));
  const faceted = visualization.layout === 'facets' && layers.length > 1;
  const grids = faceted
    ? layers.map((_, index) => ({ left: 66, right: 24, top: 34 + index * 190, height: 130 }))
    : [{ left: 62, right: 24, top: 36, bottom: 52, containLabel: false }];
  const xAxis = layers.map((layer, index) => ({
    type: axisType(datasets.get(layer.dataset_id)?.dimensions?.find((dimension) => dimension.role === 'x')?.data_type),
    gridIndex: faceted ? index : 0,
    axisLabel: { hideOverlap: true, color: '#667085', formatter: axisType(datasets.get(layer.dataset_id)?.dimensions?.find((dimension) => dimension.role === 'x')?.data_type) === 'time' ? formatAxisTime : undefined },
    axisLine: { lineStyle: { color: '#d0d5dd' } },
    splitLine: { show: false },
  }));
  const yAxis = layers.map((layer, index) => ({
    type: 'value',
    gridIndex: faceted ? index : 0,
    name: faceted ? layer.label || formatRoleLabel(layer.role) : undefined,
    nameTextStyle: { color: '#667085', align: 'left' },
    scale: true,
    axisLabel: { color: '#667085', formatter: formatAxisNumber },
    splitLine: { lineStyle: { color: '#eef1f4' } },
  }));
  const seriesOptions: any[] = [];
  layers.forEach((layer, index) => {
    const dataset = datasets.get(layer.dataset_id);
    const points = dataset?.series?.[0]?.points || layer.points || [];
    if (layer.mark === 'rule') return;
    if (layer.mark === 'band') {
      seriesOptions.push(...confidenceBandSeries({ series_id: layer.layer_id, name: layer.label || formatRoleLabel(layer.role), role: layer.role, points }, index, faceted));
      return;
    }
    seriesOptions.push(buildLayerSeriesOption(layer, points, index, faceted, activeBindingId));
  });
  const rulePoints = layers.flatMap((layer) => layer.mark === 'rule' ? layer.points || [] : []);
  if (rulePoints.length > 0 && seriesOptions.length > 0) {
    const first = seriesOptions[0] as Record<string, unknown>;
    first.markLine = {
      symbol: 'none',
      label: { color: '#475467', formatter: '{b}' },
      lineStyle: { color: '#98a2b3', type: 'dashed' },
      data: rulePoints.map((point) => ({ name: point.label || '', xAxis: point.x, yAxis: point.x == null ? point.y : undefined })),
    };
  }
  const intervalPoints = layers.flatMap((layer) => layer.mark === 'rect' ? layer.points || [] : []);
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
    tooltip: { trigger: layers.some((layer) => layer.mark === 'point') ? 'item' : 'axis', confine: true, valueFormatter: formatAxisNumber },
    legend: { type: 'scroll', top: 2, textStyle: { color: '#475467' } },
    xAxis: faceted ? xAxis : xAxis.slice(0, 1),
    yAxis: faceted ? yAxis : yAxis.slice(0, 1),
    dataZoom: layers.some((layer) => (datasets.get(layer.dataset_id)?.series?.[0]?.points?.length || 0) > 80)
      ? [{ type: 'inside', filterMode: 'none' }, { type: 'slider', height: 18, bottom: 4 }]
      : [],
    series: seriesOptions,
  };
  return option as EChartsOption;
}

function buildLayerSeriesOption(
  layer: NonNullable<Visualization['layers']>[number],
  points: VisualizationPoint[],
  index: number,
  faceted: boolean,
  activeBindingId: string | null,
): any {
  const name = layer.label || formatRoleLabel(layer.role);
  if (layer.mark === 'boxplot') {
    return {
      name,
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
  if (layer.mark === 'bar') {
    return {
      name,
      type: 'bar',
      xAxisIndex: faceted ? index : 0,
      yAxisIndex: faceted ? index : 0,
      barMaxWidth: 34,
      itemStyle: { borderRadius: [5, 5, 0, 0] },
      emphasis: { focus: 'series' },
      data: points.map(toEChartsPoint),
    };
  }
  if (['point', 'text'].includes(layer.mark)) {
    const style = semanticLayerStyle(layer.role);
    return {
      name,
      type: 'scatter',
      xAxisIndex: faceted ? index : 0,
      yAxisIndex: faceted ? index : 0,
      symbol: style.symbol,
      symbolSize: (value: unknown, params: { data?: unknown }) => isRecord(params.data) && params.data.bindingId === activeBindingId ? 15 : style.size,
      itemStyle: { color: style.color, borderColor: '#fff', borderWidth: 2 },
      label: { show: true, position: 'top', color: style.labelColor, formatter: (params: { data?: unknown }) => isRecord(params.data) ? String(params.data.semanticLabel || '') : '' },
      large: points.length > 1200,
      data: points.map(toEChartsPoint),
      z: 10,
    };
  }
  return {
    name,
    type: 'line',
    xAxisIndex: faceted ? index : 0,
    yAxisIndex: faceted ? index : 0,
    showSymbol: points.length <= 40 || points.some((point) => point.binding_id),
    symbolSize: (value: unknown, params: { data?: unknown }) => (
      isRecord(params.data) && params.data.bindingId === activeBindingId ? 10 : 5
    ),
    smooth: false,
    connectNulls: false,
    lineStyle: layer.role.includes('forecast') ? { type: 'dashed', width: 2.5 } : { width: 2 },
    areaStyle: layer.mark === 'area' ? { opacity: 0.16 } : undefined,
    data: points.map(toEChartsPoint),
    emphasis: { focus: 'series' },
    z: layer.role.includes('forecast') ? 5 : 3,
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

function formatMarkLabel(mark: string) {
  return mark[0]?.toUpperCase() + mark.slice(1);
}

function formatRoleLabel(role: string) {
  return role.replace(/[._-]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function axisType(dataType?: string): 'time' | 'value' | 'category' {
  if (dataType === 'time') return 'time';
  if (dataType === 'number') return 'value';
  return 'category';
}

function semanticLayerStyle(role: string) {
  const normalized = role.toLowerCase();
  if (normalized.includes('buy') || normalized.includes('start')) {
    return { color: '#087f5b', labelColor: '#05603f', symbol: 'triangle', size: 13 };
  }
  if (normalized.includes('sell') || normalized.includes('end')) {
    return { color: '#b42318', labelColor: '#7a271a', symbol: 'diamond', size: 13 };
  }
  if (normalized.includes('anomaly') || normalized.includes('excluded') || normalized.includes('outlier')) {
    return { color: '#dc6803', labelColor: '#7a2e0e', symbol: 'circle', size: 9 };
  }
  return { color: '#6941c6', labelColor: '#4a1fb8', symbol: 'circle', size: 10 };
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
