import { useEffect, useMemo, useRef, useState } from 'react';
import * as echarts from 'echarts';
import type { ECharts, EChartsOption, SeriesOption } from 'echarts';
import type { Visualization, VisualizationBinding, VisualizationComponent } from '../types';
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

  return (
    <div className="answer-visualization-gallery">
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
    </div>
  );
}

export function isRenderableVisualization(value: unknown): value is Visualization {
  if (!isRecord(value) || value.schema_version !== '4' || value.chart_type !== 'line') return false;
  if (typeof value.visualization_id !== 'string' || typeof value.title !== 'string') return false;
  if (!Array.isArray(value.data_views) || !Array.isArray(value.lines) || value.lines.length === 0) return false;
  if (!Array.isArray(value.y_axes) || !isRecord(value.x_axis) || !Array.isArray(value.bindings)) return false;
  if (!isRecord(value.accessibility) || typeof value.accessibility.description !== 'string') return false;
  const views = new Set(value.data_views.flatMap((item) => isRecord(item) && typeof item.view_id === 'string' ? [item.view_id] : []));
  return value.data_views.every((item) => isRecord(item) && Array.isArray(item.fields) && Array.isArray(item.records))
    && value.lines.every((item) => isRecord(item) && typeof item.component_id === 'string' && views.has(String(item.view_id)));
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
    <EChartView visualization={visualization} activeBindingId={activeBindingId} onSelectBinding={onSelectBinding} />
    {(visualization.accessibility.table_rows?.length || 0) > 0 && <AccessibleTable visualization={visualization} onSelectBinding={onSelectBinding} />}
  </article>;
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
  const compact = false;
  const option = useMemo(
    () => buildVisualizationOption(visualization, activeBindingId, compact, locale),
    [visualization, activeBindingId, locale],
  );

  useEffect(() => {
    if (!hostRef.current) return undefined;
    let chart: ECharts;
    try {
      chart = echarts.init(hostRef.current, undefined, { renderer: 'canvas' });
      chartRef.current = chart;
      chart.setOption(option, { notMerge: true, lazyUpdate: false });
      chart.on('click', (params) => {
        const data: unknown = params.data;
        if (isRecord(data) && typeof data.bindingId === 'string') onSelectBinding(data.bindingId);
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
  }, [option, onSelectBinding]);

  if (renderError) return <div className="answer-visualization-unavailable" role="status">{t('Unable to render this visualization.')}</div>;
  return <div ref={hostRef} className="answer-visualization-chart" style={{ height: 380 }} aria-label={sanitizeUserFacingText(visualization.accessibility.description)} />;
}

export function buildVisualizationOption(
  visualization: Visualization,
  activeBindingId: string | null = null,
  compact = false,
  locale: UiLocale = 'en',
): EChartsOption {
  const views = new Map(visualization.data_views.map((view) => [view.view_id, view]));
  const axisIndex = new Map(visualization.y_axes.map((axis, index) => [axis.axis_id, index]));
  const lines = visualization.lines.map((line) => {
    const data = componentData(views.get(line.view_id)?.records || [], line.x_field, line.y_field, activeBindingId);
    return {
      id: line.component_id,
      name: line.label || line.role,
      type: 'line',
      yAxisIndex: axisIndex.get(line.y_axis_id) || 0,
      data,
      showSymbol: line.symbol !== 'none',
      symbol: line.symbol === 'none' ? 'circle' : line.symbol,
      lineStyle: { type: line.line_style, width: line.importance === 'primary' ? 2.5 : 2 },
      emphasis: { focus: 'series' },
      ...(line.presentation || {}),
    } as unknown as SeriesOption;
  });
  const points = visualization.points.map((point) => ({
    id: point.component_id,
    name: point.label || point.role,
    type: 'scatter',
    yAxisIndex: axisIndex.get(point.y_axis_id) || 0,
    data: componentData(views.get(point.view_id)?.records || [], point.x_field, point.y_field, activeBindingId),
    symbol: point.symbol,
    symbolSize: point.size === 'large' ? 16 : point.size === 'small' ? 8 : 12,
    ...(point.presentation || {}),
  } as unknown as SeriesOption));
  const bands = visualization.bands.flatMap((band) => bandSeries(band, views.get(band.view_id)?.records || [], axisIndex));
  const guideHost = lines[0] as Record<string, unknown> | undefined;
  if (guideHost) attachGuides(guideHost, visualization, views, axisIndex);
  const annotations = chartAnnotations(visualization, views);
  const dense = visualization.data_views.some((view) => (view.row_count || view.records.length) > 80);

  return {
    animation: !compact,
    aria: { enabled: true, decal: { show: true }, description: visualization.accessibility.description },
    grid: { left: compact ? 48 : 64, right: visualization.y_axes.length > 1 ? 72 : 28, top: visualization.legend.visible ? 64 : 28, bottom: visualization.zoom.enabled || dense ? 72 : 42 },
    legend: { show: visualization.legend.visible, top: visualization.legend.position === 'top' ? 8 : undefined, bottom: visualization.legend.position === 'bottom' ? 8 : undefined, selectedMode: visualization.legend.toggle_components },
    tooltip: visualization.tooltip.mode === 'none' ? { show: false } : { trigger: visualization.tooltip.mode, valueFormatter: (value: unknown) => formatValue(value, locale) },
    xAxis: { type: visualization.x_axis.data_type, name: visualization.x_axis.label || undefined, axisLabel: { formatter: (value: unknown) => formatValue(value, locale) } },
    yAxis: visualization.y_axes.map((axis, index) => ({ type: axis.scale === 'log' ? 'log' : 'value', name: axis.label || axis.measure, position: index === 0 ? 'left' : 'right', axisLabel: { formatter: (value: unknown) => `${formatValue(value, locale)}${axis.unit ? ` ${axis.unit}` : ''}` } })),
    dataZoom: visualization.zoom.enabled || dense ? [
      { type: 'inside', startValue: zoomValue(visualization.zoom.start), endValue: zoomValue(visualization.zoom.end) },
      { type: 'slider', height: 20, bottom: 18, startValue: zoomValue(visualization.zoom.start), endValue: zoomValue(visualization.zoom.end) },
    ] : [],
    series: [...bands, ...lines, ...points],
    graphic: annotations,
  };
}

function componentData(records: Visualization['data_views'][number]['records'], xField: string, yField: string, active: string | null) {
  return records.flatMap((record) => {
    const x = record.values[xField];
    const y = record.values[yField];
    if (x === null || x === undefined || typeof y !== 'number') return [];
    if (typeof x !== 'string' && typeof x !== 'number') return [];
    return [{ value: [x, y], bindingId: record.binding_id, itemStyle: record.binding_id && record.binding_id === active ? { borderColor: '#111827', borderWidth: 3 } : undefined }];
  });
}

function bandSeries(band: Visualization['bands'][number], records: Visualization['data_views'][number]['records'], axes: Map<string, number>): SeriesOption[] {
  const base = records.flatMap((record) => {
    const x = record.values[band.x_field]; const lower = record.values[band.lower_field];
    return x != null && typeof lower === 'number' ? [[x, lower]] : [];
  });
  const range = records.flatMap((record) => {
    const x = record.values[band.x_field]; const lower = record.values[band.lower_field]; const upper = record.values[band.upper_field];
    return x != null && typeof lower === 'number' && typeof upper === 'number' ? [[x, upper - lower]] : [];
  });
  const stack = `band_${band.component_id}`;
  return [
    { id: `${band.component_id}_base`, name: band.label || band.role, type: 'line', stack, yAxisIndex: axes.get(band.y_axis_id) || 0, data: base, symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, tooltip: { show: false } } as SeriesOption,
    { id: band.component_id, name: band.label || band.role, type: 'line', stack, yAxisIndex: axes.get(band.y_axis_id) || 0, data: range, symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { opacity: 0.18 } } as SeriesOption,
  ];
}

function attachGuides(host: Record<string, unknown>, visualization: Visualization, views: Map<string, Visualization['data_views'][number]>, axes: Map<string, number>) {
  const markLines: Array<Record<string, unknown>> = visualization.reference_lines.flatMap((item) => {
    const record = views.get(item.view_id)?.records[0]; const value = record?.values[item.value_field];
    return typeof value === 'number' ? [{ name: item.label || item.role, yAxis: value, yAxisIndex: axes.get(item.y_axis_id) || 0 }] : [];
  });
  const markAreas: Array<Array<Record<string, unknown>>> = visualization.intervals.flatMap((item) => (
    (views.get(item.view_id)?.records || []).flatMap((record) => {
      const start = record.values[item.start_field]; const end = record.values[item.end_field];
      return start != null && end != null ? [[{ name: item.label || item.role, xAxis: start }, { xAxis: end }]] : [];
    })
  ));
  const markPoints: Array<Record<string, unknown>> = [];
  for (const item of visualization.annotations) {
    const records = (views.get(item.view_id)?.records || []).slice(0, 12);
    for (const record of records) {
      const content = record.values[item.content_field];
      if (content == null) continue;
      const label = { formatter: String(content) };
      if (item.target.target_type === 'x' && item.target.x_field) {
        const x = record.values[item.target.x_field];
        if (x != null) markLines.push({ xAxis: x, label, bindingId: record.binding_id });
      } else if (item.target.target_type === 'xy' && item.target.x_field && item.target.y_field) {
        const x = record.values[item.target.x_field]; const y = record.values[item.target.y_field];
        if (x != null && typeof y === 'number') markPoints.push({ coord: [x, y], label, bindingId: record.binding_id });
      } else if (item.target.target_type === 'interval' && item.target.start_field && item.target.end_field) {
        const start = record.values[item.target.start_field]; const end = record.values[item.target.end_field];
        if (start != null && end != null) markAreas.push([{ name: String(content), xAxis: start }, { xAxis: end }]);
      }
    }
  }
  host.markLine = { silent: false, data: markLines };
  host.markArea = { silent: false, data: markAreas };
  host.markPoint = { data: markPoints };
}

function chartAnnotations(visualization: Visualization, views: Map<string, Visualization['data_views'][number]>) {
  let index = 0;
  const seen = new Set<string>();
  return visualization.annotations.flatMap((item) => {
    if (item.target.target_type !== 'chart') return [];
    return (views.get(item.view_id)?.records || []).slice(0, 6).flatMap((record) => {
      const content = record.values[item.content_field];
      if (content == null) return [];
      const text = sanitizeUserFacingText(String(content));
      if (seen.has(text)) return [];
      seen.add(text);
      const top = 30 + index++ * 24;
      return [{ type: 'text', right: 18, top, style: { text, fill: '#374151', fontSize: 12, backgroundColor: 'rgba(255,255,255,.88)', padding: [4, 6] } }];
    });
  });
}

function AccessibleTable({ visualization, onSelectBinding }: { visualization: Visualization; onSelectBinding: (id: string) => void }) {
  const rows = visualization.accessibility.table_rows || [];
  const columns = visualization.accessibility.table_columns || [];
  return <details className="visualization-data-details"><summary>Data</summary><div className="answer-visualization-table-wrap"><table className="answer-data-table answer-visualization-table"><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index} onClick={() => typeof row.binding_id === 'string' && onSelectBinding(row.binding_id)}>{columns.map((column) => <td key={column}>{formatValue(row[column], 'en')}</td>)}</tr>)}</tbody></table></div></details>;
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

function zoomValue(value: unknown): string | number | Date | undefined {
  return typeof value === 'string' || typeof value === 'number' || value instanceof Date ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}
