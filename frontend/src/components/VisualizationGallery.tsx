import { useEffect, useRef, useState, type CSSProperties } from 'react';
import * as echarts from 'echarts';
import type { ECharts, EChartsOption } from 'echarts';
import type { Visualization, VisualizationBinding } from '../types';
import { fetchVisualizationData } from '../services/api';
import { useI18n, type UiLocale } from '../i18n';
import { formatHumanTime, isIsoTimestamp } from '../lib/humanTime';
import { sanitizeUserFacingText } from '../lib/presentation';

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
    for (const visualization of visualizations) {
      if (!visualization.data_ref || loaded[visualization.visualization_id]) continue;
      void fetchVisualizationData(visualization.data_ref)
        .then((payload) => {
          if (!controller.signal.aborted) setLoaded((value) => ({ ...value, [visualization.visualization_id]: payload }));
        })
        .catch((error) => {
          if (!controller.signal.aborted) {
            setLoadErrors((value) => ({
              ...value,
              [visualization.visualization_id]: error instanceof Error ? error.message : t('Unable to load chart data.'),
            }));
          }
        });
    }
    return () => controller.abort();
  }, [visualizations, loaded, t]);

  const hydrated = visualizations.map((item) => loaded[item.visualization_id] || item);
  const pending = new Set(visualizations
    .filter((item) => item.data_ref && !loaded[item.visualization_id] && !loadErrors[item.visualization_id])
    .map((item) => item.visualization_id));
  const renderable = hydrated.filter((item) => !pending.has(item.visualization_id) && isRenderableVisualization(item));
  const rejected = hydrated.filter((item) => !pending.has(item.visualization_id) && !isRenderableVisualization(item)).length;
  const activeBinding = renderable.flatMap((item) => item.bindings).find((item) => item.binding_id === activeBindingId);

  return <div className="answer-visualization-gallery">
    {rejected > 0 && <div className="answer-visualization-unavailable" role="status">
      {t('{count} saved visualizations cannot be displayed because the data schema is no longer supported.', { count: rejected })}
    </div>}
    {Object.values(loadErrors).map((message) => <div className="answer-visualization-unavailable" role="status" key={message}>{message}</div>)}
    {pending.size > 0 && <div className="answer-visualization-unavailable" role="status">{t('Loading complete visualization data…')}</div>}
    <div className="answer-visualization-grid">
      {[...renderable].sort((a, b) => Number(a.priority !== 'primary') - Number(b.priority !== 'primary')).map((item) => (
        <VisualizationCard key={item.visualization_id} visualization={item} activeBindingId={activeBindingId} onSelectBinding={onSelectBinding} />
      ))}
    </div>
    {activeBinding && <BindingDetail binding={activeBinding} label={t('Linked evidence')} />}
  </div>;
}

export function isRenderableVisualization(value: unknown): value is Visualization {
  if (!isRecord(value) || value.schema_version !== '5' || value.chart_type !== 'echarts') return false;
  if (typeof value.visualization_id !== 'string' || typeof value.title !== 'string') return false;
  if (!isRecord(value.option) || !Array.isArray(value.bindings)) return false;
  if (!isRecord(value.accessibility) || typeof value.accessibility.description !== 'string') return false;
  const series = asArray(value.option.series);
  const datasets = asArray(value.option.dataset);
  return series.length > 0 && datasets.length > 0
    && series.every((item) => isRecord(item) && ['line', 'scatter', 'bar'].includes(String(item.type)))
    && datasets.every((item) => isRecord(item) && Array.isArray(item.source));
}

function VisualizationCard({ visualization, activeBindingId, onSelectBinding }: {
  visualization: Visualization;
  activeBindingId: string | null;
  onSelectBinding: (bindingId: string) => void;
}) {
  return <article className={`answer-visualization-card is-${visualization.priority}`}>
    <div className="answer-visualization-card-header">
      <div><strong>{sanitizeUserFacingText(visualization.title)}</strong>{visualization.summary && <p>{sanitizeUserFacingText(visualization.summary)}</p>}</div>
    </div>
    <AnnotationLegend option={visualization.option} />
    <EChartView visualization={visualization} activeBindingId={activeBindingId} onSelectBinding={onSelectBinding} />
    {(visualization.accessibility.table_rows?.length || 0) > 0 && <AccessibleTable visualization={visualization} onSelectBinding={onSelectBinding} />}
  </article>;
}

export type AnnotationLegendItem = {
  kind: 'point' | 'interval' | 'reference';
  name: string;
  color: string;
};

export function annotationLegendItems(option: Record<string, unknown>): AnnotationLegendItem[] {
  const items: AnnotationLegendItem[] = [];
  const seen = new Set<string>();
  const colors = asArray(option.color).filter((value): value is string => (
    typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value)
  ));
  const append = (kind: AnnotationLegendItem['kind'], name: unknown, seriesIndex: number) => {
    if (typeof name !== 'string' || !name.trim()) return;
    const safeName = sanitizeUserFacingText(name.trim());
    const color = colors[seriesIndex % colors.length] || '#5470c6';
    const key = `${seriesIndex}:${kind}:${safeName}`;
    if (!safeName || seen.has(key)) return;
    seen.add(key);
    items.push({ kind, name: safeName, color });
  };
  asArray(option.series).forEach((series, seriesIndex) => {
    if (!isRecord(series)) return;
    const markPoint = isRecord(series.markPoint) ? series.markPoint : {};
    for (const point of asArray(markPoint.data)) {
      if (isRecord(point)) append('point', point.name, seriesIndex);
    }
    const markArea = isRecord(series.markArea) ? series.markArea : {};
    for (const area of asArray(markArea.data)) {
      const start = Array.isArray(area) ? area[0] : area;
      if (isRecord(start)) append('interval', start.name, seriesIndex);
    }
    const markLine = isRecord(series.markLine) ? series.markLine : {};
    for (const reference of asArray(markLine.data)) {
      if (isRecord(reference)) append('reference', reference.name, seriesIndex);
    }
  });
  return items;
}

export function AnnotationLegend({ option }: { option: Record<string, unknown> }) {
  const items = annotationLegendItems(option);
  if (items.length === 0) return null;
  return <ul className="answer-annotation-legend" aria-label="Chart annotations">
    {items.map((item) => <li
      key={`${item.kind}:${item.name}:${item.color}`}
      style={{ '--annotation-color': item.color } as CSSProperties}
    >
      <i className={`is-${item.kind}`} aria-hidden="true" />
      <span>{item.name}</span>
    </li>)}
  </ul>;
}

function EChartView({ visualization, activeBindingId, onSelectBinding }: {
  visualization: Visualization;
  activeBindingId: string | null;
  onSelectBinding: (bindingId: string) => void;
}) {
  const { t, locale } = useI18n();
  const hostRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ECharts | null>(null);
  const [renderError, setRenderError] = useState(false);

  useEffect(() => {
    if (!hostRef.current) return undefined;
    let chart: ECharts;
    try {
      chart = echarts.init(hostRef.current, undefined, { renderer: 'canvas', locale: locale === 'zh-CN' ? 'ZH' : 'EN' });
      chartRef.current = chart;
      chart.setOption(withTrustedDisplaySettings(visualization, locale), { notMerge: true, lazyUpdate: false });
      chart.on('click', (params) => {
        const bindingId = bindingIdFromClickData(params.data);
        if (bindingId) onSelectBinding(bindingId);
      });
      setRenderError(false);
    } catch (error) {
      console.error('Unable to render visualization.', error);
      setRenderError(true);
      return undefined;
    }
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(hostRef.current);
    return () => { observer.disconnect(); chart.dispose(); chartRef.current = null; };
  }, [visualization, locale, onSelectBinding]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.dispatchAction({ type: 'downplay', seriesIndex: 'all' });
    for (const location of bindingLocations(visualization.option, activeBindingId)) {
      chart.dispatchAction({ type: 'highlight', ...location });
    }
  }, [visualization.option, activeBindingId]);

  if (renderError) return <div className="answer-visualization-unavailable" role="status">{t('Unable to render this visualization.')}</div>;
  return <div ref={hostRef} className="answer-visualization-chart" style={{ height: 380 }} role="img" aria-label={sanitizeUserFacingText(visualization.accessibility.description)} />;
}

export function withTrustedDisplaySettings(visualization: Visualization, locale: UiLocale = 'en-US'): EChartsOption {
  const option = visualization.option as EChartsOption;
  return {
    ...option,
    xAxis: withUtcDateAxes(option.xAxis, option.dataset, locale),
    useUTC: true,
    aria: {
      ...((isRecord(visualization.option.aria) ? visualization.option.aria : {}) as Record<string, unknown>),
      enabled: true,
      description: visualization.accessibility.description,
    },
  } as EChartsOption;
}

function withUtcDateAxes(xAxis: EChartsOption['xAxis'], dataset: EChartsOption['dataset'], locale: UiLocale) {
  const includeTime = visualizationTimeSpan(dataset) <= 7 * 24 * 60 * 60 * 1000;
  const axes = asArray(xAxis).map((axis) => {
    if (!isRecord(axis) || axis.type !== 'time') return axis;
    return {
      ...axis,
      axisLabel: {
        ...(isRecord(axis.axisLabel) ? axis.axisLabel : {}),
        hideOverlap: true,
        formatter: (value: string | number) => formatUtcAxisTick(value, locale, includeTime),
      },
    };
  });
  return Array.isArray(xAxis) ? axes : axes[0];
}

function visualizationTimeSpan(dataset: EChartsOption['dataset']): number {
  const timestamps = asArray(dataset).flatMap((item) => {
    if (!isRecord(item) || !Array.isArray(item.source)) return [];
    return item.source.flatMap((row) => {
      if (!isRecord(row)) return [];
      return Object.values(row).flatMap((value) => {
        if (typeof value !== 'string' || !isIsoTimestamp(value)) return [];
        const timestamp = Date.parse(value);
        return Number.isFinite(timestamp) ? [timestamp] : [];
      });
    });
  });
  if (timestamps.length < 2) return Number.POSITIVE_INFINITY;
  return Math.max(...timestamps) - Math.min(...timestamps);
}

function formatUtcAxisTick(value: string | number, locale: UiLocale, includeTime: boolean): string {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return String(value);
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  const day = String(date.getUTCDate()).padStart(2, '0');
  const datePart = locale === 'zh-CN' ? `${month}月${day}日` : `${month}-${day}`;
  if (!includeTime) return datePart;
  const hours = String(date.getUTCHours()).padStart(2, '0');
  const minutes = String(date.getUTCMinutes()).padStart(2, '0');
  return `${datePart} ${hours}:${minutes}`;
}

export function bindingLocations(option: Record<string, unknown>, bindingId: string | null) {
  if (!bindingId) return [];
  const datasets = asArray(option.dataset);
  const datasetIds = new Map(datasets.flatMap((dataset, index) => (
    isRecord(dataset) && typeof dataset.id === 'string' ? [[dataset.id, index] as const] : []
  )));
  const matches = new Map<number, number[]>();
  datasets.forEach((dataset, datasetIndex) => {
    if (!isRecord(dataset) || !Array.isArray(dataset.source)) return;
    dataset.source.forEach((row, rowIndex) => {
      if (isRecord(row) && row.bindingId === bindingId) matches.set(datasetIndex, [...(matches.get(datasetIndex) || []), rowIndex]);
    });
  });
  return asArray(option.series).flatMap((series, seriesIndex) => {
    if (!isRecord(series)) return [];
    const datasetIndex = typeof series.datasetId === 'string'
      ? datasetIds.get(series.datasetId)
      : (typeof series.datasetIndex === 'number' ? series.datasetIndex : 0);
    return datasetIndex === undefined ? [] : (matches.get(datasetIndex) || []).map((dataIndex) => ({ seriesIndex, dataIndex }));
  });
}

export function bindingIdFromClickData(data: unknown): string | null {
  return isRecord(data) && typeof data.bindingId === 'string' ? data.bindingId : null;
}

function AccessibleTable({ visualization, onSelectBinding }: { visualization: Visualization; onSelectBinding: (id: string) => void }) {
  const { locale } = useI18n();
  const rows = visualization.accessibility.table_rows || [];
  const columns = visualization.accessibility.table_columns || [];
  return <details className="visualization-data-details"><summary>Data</summary><div className="answer-visualization-table-wrap"><table className="answer-data-table answer-visualization-table"><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index} onClick={() => typeof row.bindingId === 'string' && onSelectBinding(row.bindingId)}>{columns.map((column) => <td key={column}>{formatValue(row[column], locale)}</td>)}</tr>)}</tbody></table></div></details>;
}

function BindingDetail({ binding, label }: { binding: VisualizationBinding; label: string }) {
  return <div className="visualization-link-detail" role="status"><strong>{label}</strong><span>{formatBindingType(binding.source_type)}</span></div>;
}

function formatBindingType(value: string) {
  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatValue(value: unknown, locale: UiLocale): string {
  if (typeof value === 'number') return new Intl.NumberFormat(locale === 'zh-CN' ? 'zh-CN' : 'en-US', { maximumFractionDigits: 4 }).format(value);
  if (typeof value === 'string' && isIsoTimestamp(value)) return formatHumanTime(value, locale);
  if (typeof value === 'string') return sanitizeUserFacingText(value);
  return value == null ? '—' : String(value);
}

function asArray(value: unknown): unknown[] {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value : [value];
}

function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}
