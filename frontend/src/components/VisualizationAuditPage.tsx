import { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import type { ECharts } from 'echarts';
import type { Visualization } from '../types';
import { AnnotationLegend, withTrustedDisplaySettings } from './VisualizationGallery';

declare global {
  interface Window {
    __TSPILOT_RENDER_VISUALIZATION__?: (visualization: Visualization) => Promise<boolean>;
  }
}

type PendingRender = {
  resolve: (value: boolean) => void;
  reject: (reason?: unknown) => void;
};

export function VisualizationAuditPage() {
  const [visualization, setVisualization] = useState<Visualization | null>(null);
  const [ready, setReady] = useState(false);
  const chartHost = useRef<HTMLDivElement | null>(null);
  const chart = useRef<ECharts | null>(null);
  const pendingRender = useRef<PendingRender | null>(null);

  useEffect(() => {
    window.__TSPILOT_RENDER_VISUALIZATION__ = (next) => {
      pendingRender.current?.reject(new Error('A newer visualization replaced the pending render.'));
      setReady(false);
      setVisualization(next);
      return new Promise<boolean>((resolve, reject) => {
        pendingRender.current = { resolve, reject };
      });
    };
    return () => {
      pendingRender.current?.reject(new Error('Visualization audit page was unmounted.'));
      pendingRender.current = null;
      delete window.__TSPILOT_RENDER_VISUALIZATION__;
    };
  }, []);

  useEffect(() => {
    if (!visualization || !chartHost.current) return undefined;
    let instance: ECharts | null = null;
    const markReady = () => {
      setReady(true);
      pendingRender.current?.resolve(true);
      pendingRender.current = null;
    };
    try {
      chart.current?.dispose();
      instance = echarts.init(chartHost.current, undefined, { renderer: 'canvas' });
      chart.current = instance;
      instance.on('finished', markReady);
      const locale = document.documentElement.lang.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US';
      instance.setOption(withTrustedDisplaySettings(visualization, locale), { notMerge: true, lazyUpdate: false });
    } catch (error) {
      const renderError = error instanceof Error ? error : new Error(String(error));
      pendingRender.current?.reject(renderError);
      pendingRender.current = null;
      setReady(false);
    }
    return () => {
      instance?.off('finished', markReady);
      instance?.dispose();
      if (chart.current === instance) chart.current = null;
    };
  }, [visualization]);

  return (
    <main className="visualization-audit-page">
      <article
        className="visualization-audit-stage"
        data-visual-audit-ready={ready ? 'true' : 'false'}
      >
        {visualization ? (
          <>
            <header>
              <strong>{visualization.title}</strong>
              {visualization.summary && <p>{visualization.summary}</p>}
            </header>
            <AnnotationLegend option={visualization.option} />
            <div
              ref={chartHost}
              className="visualization-audit-chart"
              role="img"
              aria-label={visualization.accessibility.description || visualization.title}
            />
          </>
        ) : (
          <p>Waiting for a visualization candidate…</p>
        )}
      </article>
    </main>
  );
}
