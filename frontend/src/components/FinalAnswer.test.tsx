import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { FinalAnswer as FinalAnswerType, Visualization } from '../types';
import { FinalAnswer } from './FinalAnswer';

function answer(overrides: Partial<FinalAnswerType> = {}): FinalAnswerType {
  return {
    summary: '查询已完成。',
    ...overrides,
  };
}

describe('FinalAnswer', () => {
  it('does not expose internal claim-linking state below the conclusion', () => {
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({
      claims: [{ claim_id: 'claim_1', text: '内部绑定文本', insight_ids: ['insight_1'] }],
    })} />);

    expect(markup).not.toContain('linked');
    expect(markup).not.toContain('grounded');
    expect(markup).not.toContain('内部绑定文本');
  });

  it('renders the complete executed query as an open code block', () => {
    const query = 'from(bucket: "bitcoin")\n  |> range(start: -30d)\n  |> max()';
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({
      references: [{
        source_type: 'query',
        source_id: 'evi_1',
        label: '查询证据',
        evidence: { query_language: 'flux', query, row_count: 1 },
      }],
    })} />);

    expect(markup).toContain('实际执行的查询语句');
    expect(markup).toContain('<details class="answer-inline-details answer-query-details" open="">');
    expect(markup).toContain('<pre class="answer-query-code"><code>from(bucket: &quot;bitcoin&quot;)');
    expect(markup).toContain('|&gt; max()</code></pre>');
    expect(markup).not.toContain('[truncated');
  });

  it('deduplicates repeated logical insight references', () => {
    const duplicateInsight = {
      source_type: 'insight',
      label: '最大 7 天窗口标准差',
      evidence: { insight_key: 'max_7d_window_std', statement: '最大窗口。' },
    } as const;
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({
      references: [
        { ...duplicateInsight, source_id: 'insight_1' },
        { ...duplicateInsight, source_id: 'insight_2' },
      ],
    })} />);

    expect(markup.match(/最大 7 天窗口标准差/g)).toHaveLength(1);
    expect(markup).toContain('1 项依据');
  });

  it('renders an approved visualization as claim, evidence, interpretation, and source', () => {
    const visualization: Visualization = {
      schema_version: '3',
      visualization_id: 'viz_trend',
      purpose: 'verify the observed trend',
      priority: 'primary',
      title: 'Observed value over time',
      verification: {
        target_insight_ids: ['insight_trend'],
        verification_question: '完整序列是否在区间内上升？',
        interpretation: '从首个观测点到末个观测点读取完整序列。',
      },
      datasets: [{
        dataset_id: 'dataset_0',
        source_ref: 'semantic:trend',
        dimensions: [
          { name: 'timestamp', data_type: 'time', role: 'x' },
          { name: 'value', data_type: 'number', role: 'y' },
        ],
        series: [{
          series_id: 'series_0',
          name: 'Observed value',
          role: 'complete_series',
          points: [
            { x: '2026-01-01', y: 10 },
            { x: '2026-01-02', y: 12 },
          ],
        }],
      }],
      layers: [{
        layer_id: 'layer_0',
        mark: 'line',
        role: 'complete_series',
        source_ref: 'semantic:trend',
        encoding: { x: 'timestamp', y: 'value' },
        dataset_id: 'dataset_0',
      }],
      bindings: [],
      accessibility: { description: '完整观测序列。' },
    };
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({
      claims: [{
        claim_id: 'claim_trend',
        text: '价格在观察区间内上升。',
        insight_ids: ['insight_trend'],
        analysis_ids: ['ana_trend'],
        visualization_ids: ['viz_trend'],
      }],
      visualizations: [visualization],
    })} />);

    expect(markup).toContain('结论与视觉证据');
    expect(markup).toContain('价格在观察区间内上升。');
    expect(markup).toContain('完整序列是否在区间内上升？');
    expect(markup).toContain('从首个观测点到末个观测点读取完整序列。');
    expect(markup).toContain('Insight · insight_trend');
    expect(markup).toContain('Analysis · ana_trend');
    expect(markup).not.toContain('<h3>可视化</h3>');
  });
});
