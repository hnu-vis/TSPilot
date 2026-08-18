import { useEffect, useMemo, useRef, useState } from 'react';
import * as echarts from 'echarts';
import type { EChartsOption } from 'echarts';
import type { EChartsType } from 'echarts/core';
import type { Visualization, VisualizationPoint, VisualizationSeries } from '../types';
import { fetchVisualizationData } from '../services/api';
import { useI18n } from '../i18n';

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
  return (
    <article className={`answer-visualization-card is-${visualization.priority}`}>
      <div className="answer-visualization-card-header">
        <div>
          <strong>{visualization.title}</strong>
          {visualization.summary && <p>{visualization.summary}</p>}
        </div>
        <span>{Array.from(new Set(marks)).map(formatMarkLabel).join(' · ')}</span>
      </div>
      <EChartView
        visualization={visualization}
        activeBindingId={activeBindingId}
        onSelectBinding={onSelectBinding}
      />
      {(visualization.accessibility.table_rows?.length || 0) > 0 && (
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
      const bindingId = data && typeof data.bindingId === 'string'
        ? data.bindingId
        : data && typeof data.__binding_id === 'string' ? data.__binding_id : null;
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
  const layers = visualization.layers || [];
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
  const rendererDatasets: Array<{ id: string; source: Array<Record<string, unknown>> }> = [];
  layers.forEach((layer, index) => {
    const dataset = datasets.get(layer.dataset_id);
    if (layer.mark === 'rule') return;
    const sourceSeries = dataset?.series?.length
      ? dataset.series
      : [{ series_id: layer.layer_id, name: layer.label || formatRoleLabel(layer.role), role: layer.role, points: layer.points || [] }];
    sourceSeries.forEach((series) => {
      if (layer.mark === 'band') {
        seriesOptions.push(...confidenceBandSeries(series, index, faceted));
        return;
      }
      if (!['line', 'area', 'bar', 'point', 'boxplot', 'rule', 'rect'].includes(layer.mark)) {
        const rendererDatasetId = `${layer.layer_id}_${series.series_id}`;
        rendererDatasets.push({
          id: rendererDatasetId,
          source: (series.points || []).map((point) => groundedPointRecord(point, layer.encoding)),
        });
        seriesOptions.push(buildRendererNativeSeriesOption(
          { ...layer, label: series.name }, rendererDatasetId, index, faceted,
        ));
        return;
      }
      seriesOptions.push(buildLayerSeriesOption(
        { ...layer, label: series.name }, series.points || [], index, faceted, activeBindingId,
      ));
    });
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

  const option: Record<string, unknown> = {
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
    dataset: rendererDatasets,
    series: seriesOptions,
  };
  const presented = deepMerge(option, visualization.presentation || {});
  // These are the immutable binding boundary. Even malformed or legacy
  // presentation payloads cannot replace verified renderer data.
  presented.dataset = rendererDatasets;
  presented.series = seriesOptions;
  return presented as EChartsOption;
}

function deepMerge(base: Record<string, unknown>, overlay: Record<string, unknown>): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...base };
  Object.entries(overlay).forEach(([key, value]) => {
    const current = merged[key];
    merged[key] = isRecord(current) && isRecord(value)
      ? deepMerge(current, value)
      : value;
  });
  return merged;
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
      ...(layer.presentation || {}),
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
      ...(layer.presentation || {}),
      name,
      type: 'bar',
      xAxisIndex: faceted ? index : 0,
      yAxisIndex: faceted ? index : 0,
      barMaxWidth: 34,
      itemStyle: { borderRadius: [5, 5, 0, 0] },
      emphasis: { focus: 'series' },
      data: points.map((point) => toEChartsPoint(point)),
    };
  }
  if (layer.mark === 'point') {
    const colorField = encodingField(layer.encoding?.color);
    const shapeField = encodingField(layer.encoding?.shape);
    const sizeField = encodingField(layer.encoding?.size);
    const opacityField = encodingField(layer.encoding?.opacity);
    return {
      ...(layer.presentation || {}),
      name,
      type: 'scatter',
      xAxisIndex: faceted ? index : 0,
      yAxisIndex: faceted ? index : 0,
      symbol: (_value: unknown, params: { data?: unknown }) => symbolForEncoding(params.data, shapeField, index),
      symbolSize: (_value: unknown, params: { data?: unknown }) => isRecord(params.data) && params.data.bindingId === activeBindingId
        ? 15
        : sizeForEncoding(params.data, sizeField),
      itemStyle: {
        color: (params: { data?: unknown }) => colorForEncoding(params.data, colorField, index),
        borderColor: '#fff',
        borderWidth: 2,
      },
      label: { show: true, position: 'top', color: '#475467', formatter: (params: { data?: unknown }) => isRecord(params.data) ? String(params.data.semanticLabel || '') : '' },
      large: points.length > 1200,
      data: points.map((point) => toEChartsPoint(point, layer, index, opacityField)),
      z: 10,
    };
  }
  const linePresentation = layer.presentation || {};
  const presentedLineStyle = isRecord(linePresentation.lineStyle) ? linePresentation.lineStyle : {};
  const presentedAreaStyle = isRecord(linePresentation.areaStyle) ? linePresentation.areaStyle : {};
  const presentedEmphasis = isRecord(linePresentation.emphasis) ? linePresentation.emphasis : {};
  const colorField = encodingField(layer.encoding?.color);
  const shapeField = encodingField(layer.encoding?.shape);
  const sizeField = encodingField(layer.encoding?.size);
  const opacityField = encodingField(layer.encoding?.opacity);
  return {
    ...linePresentation,
    name,
    type: 'line',
    xAxisIndex: faceted ? index : 0,
    yAxisIndex: faceted ? index : 0,
    showSymbol: linePresentation.showSymbol ?? (points.length <= 40 || points.some((point) => point.binding_id)),
    symbol: shapeField
      ? (_value: unknown, params: { data?: unknown }) => symbolForEncoding(params.data, shapeField, index)
      : linePresentation.symbol,
    symbolSize: (value: unknown, params: { data?: unknown }) => (
      isRecord(params.data) && params.data.bindingId === activeBindingId ? 10 : 5
    ),
    smooth: linePresentation.smooth ?? false,
    connectNulls: linePresentation.connectNulls ?? false,
    lineStyle: {
      ...presentedLineStyle,
      color: colorField
        ? colorForEncoding({ metadata: points[0]?.metadata || {} }, colorField, index)
        : presentedLineStyle.color,
      width: sizeField
        ? lineWidthForEncoding({ metadata: points[0]?.metadata || {} }, sizeField)
        : presentedLineStyle.width ?? 2,
      opacity: opacityField
        ? opacityForEncoding({ metadata: points[0]?.metadata || {} }, opacityField)
        : presentedLineStyle.opacity ?? 1,
    },
    areaStyle: layer.mark === 'area' ? { opacity: 0.16, ...presentedAreaStyle } : undefined,
    data: points.map((point) => toEChartsPoint(point, layer, index, opacityField)),
    emphasis: { focus: 'series', ...presentedEmphasis },
    z: linePresentation.z ?? 3 + index,
  };
}

function buildRendererNativeSeriesOption(
  layer: NonNullable<Visualization['layers']>[number],
  datasetId: string,
  index: number,
  faceted: boolean,
): Record<string, unknown> {
  // Presentation is merged first. Grounded binding properties below always win,
  // so renderer freedom cannot replace the dataset or its field encodings.
  return {
    ...(layer.presentation || {}),
    name: layer.label || formatRoleLabel(layer.role),
    type: layer.mark,
    datasetId,
    encode: layer.encoding || {},
    xAxisIndex: faceted ? index : 0,
    yAxisIndex: faceted ? index : 0,
  };
}

function groundedPointRecord(
  point: VisualizationPoint,
  encoding: Record<string, string | string[]> | undefined,
): Record<string, unknown> {
  const record: Record<string, unknown> = { ...(point.metadata || {}) };
  const xField = encodingField(encoding?.x);
  const yFields = encodingFields(encoding?.y || encoding?.value);
  if (xField && record[xField] === undefined) record[xField] = point.x;
  if (yFields.length === 1 && record[yFields[0]] === undefined) record[yFields[0]] = point.y;
  record.__binding_id = point.binding_id;
  record.__semantic_label = point.label;
  return record;
}

function encodingFields(value: string | string[] | undefined): string[] {
  return Array.isArray(value) ? value : typeof value === 'string' ? [value] : [];
}

function encodingField(value: string | string[] | undefined): string | undefined {
  return encodingFields(value)[0];
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

function toEChartsPoint(
  point: VisualizationPoint,
  layer?: NonNullable<Visualization['layers']>[number],
  layerIndex = 0,
  opacityField?: string,
) {
  const metadata = point.metadata || {};
  const colorField = encodingField(layer?.encoding?.color);
  return {
    value: [chartValue(point.x), point.y],
    bindingId: point.binding_id,
    semanticLabel: point.label,
    metadata,
    itemStyle: {
      color: colorForEncoding({ metadata }, colorField, layerIndex),
      opacity: opacityForEncoding({ metadata }, opacityField),
      borderWidth: point.binding_id ? 2 : undefined,
      borderColor: point.binding_id ? '#fff' : undefined,
    },
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

const VISUAL_COLORS = ['#087f5b', '#6941c6', '#175cd3', '#dc6803', '#b42318'];
const VISUAL_SYMBOLS = ['circle', 'triangle', 'diamond', 'rect', 'roundRect'];

function encodedValue(data: unknown, field?: string): unknown {
  return field && isRecord(data) && isRecord(data.metadata) ? data.metadata[field] : undefined;
}

function stableIndex(value: unknown, modulo: number, fallback: number): number {
  if (value === undefined || value === null) return fallback % modulo;
  const text = String(value);
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) hash = ((hash * 31) + text.charCodeAt(index)) >>> 0;
  return hash % modulo;
}

function colorForEncoding(data: unknown, field: string | undefined, fallback: number): string {
  return VISUAL_COLORS[stableIndex(encodedValue(data, field), VISUAL_COLORS.length, fallback)];
}

function symbolForEncoding(data: unknown, field: string | undefined, fallback: number): string {
  return VISUAL_SYMBOLS[stableIndex(encodedValue(data, field), VISUAL_SYMBOLS.length, fallback)];
}

function sizeForEncoding(data: unknown, field?: string): number {
  const value = Number(encodedValue(data, field));
  return Number.isFinite(value) ? Math.max(6, Math.min(24, Math.abs(value))) : 10;
}

function lineWidthForEncoding(data: unknown, field?: string): number {
  const value = Number(encodedValue(data, field));
  return Number.isFinite(value) ? Math.max(1, Math.min(8, Math.abs(value))) : 2;
}

function opacityForEncoding(data: unknown, field?: string): number {
  const value = Number(encodedValue(data, field));
  return Number.isFinite(value) ? Math.max(0.15, Math.min(1, value)) : 1;
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
