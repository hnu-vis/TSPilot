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

  it('keeps the complete executed query available without opening technical details by default', () => {
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
    expect(markup).toContain('answer-supporting-section answer-evidence-section');
    expect(markup).not.toContain('answer-supporting-section answer-evidence-section" open');
    expect(markup).toContain('<details class="answer-inline-details answer-query-details">');
    expect(markup).not.toContain('answer-query-details" open');
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
    expect(markup).not.toContain('完整序列是否在区间内上升？');
    expect(markup).toContain('从首个观测点到末个观测点读取完整序列。');
    expect(markup).not.toContain('Insight · insight_trend');
    expect(markup).not.toContain('Analysis · ana_trend');
    expect(markup).not.toContain('<h3>可视化</h3>');
  });

  it('places a visualization directly after the section whose generated claim references it', () => {
    const visualization: Visualization = {
      schema_version: '3',
      visualization_id: 'viz_section',
      purpose: 'verify the finding',
      priority: 'primary',
      title: 'Observed trend',
      datasets: [],
      layers: [],
      bindings: [],
      accessibility: { description: 'Observed trend.' },
    };
    const finding = '价格在观察区间内上涨 20%。';
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({
      sections: [{ section_type: 'key_findings', heading: '关键发现', content: finding }],
      claims: [{
        claim_id: 'claim_section_1',
        text: finding,
        visualization_ids: ['viz_section'],
      }],
      visualizations: [visualization],
    })} />);

    expect(markup.match(/价格在观察区间内上涨 20%。/g)).toHaveLength(1);
    expect(markup).toContain('answer-claim-evidence-card section-linked');
    expect(markup.indexOf('关键发现')).toBeLessThan(markup.indexOf('Observed trend'));
    expect(markup).not.toContain('结论与视觉证据');
  });

  it('keeps execution telemetry out of the user-facing answer header', () => {
    const markup = renderToStaticMarkup(
      <FinalAnswer
        answer={answer()}
        elapsedSeconds={19}
        tokenUsage={{ totals: { total_tokens: 4665, call_count: 7 } }}
      />,
    );

    expect(markup).not.toContain('19.0s');
    expect(markup).not.toContain('4,665 tokens');
    expect(markup).not.toContain('7 calls');
  });

});
