import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { FinalAnswer as FinalAnswerType, Visualization } from '../types';
import { FinalAnswer } from './FinalAnswer';

function answer(overrides: Partial<FinalAnswerType> = {}): FinalAnswerType {
  return { summary: '查询已完成。', ...overrides };
}

function visualization(overrides: Partial<Visualization> = {}): Visualization {
  return {
    schema_version: '5', chart_type: 'echarts', visualization_id: 'viz', purpose: 'verify trend',
    priority: 'primary', title: 'Price trend',
    option: {
      dataset: { source: [{ time: '2026-01-01', value: 10 }, { time: '2026-01-02', value: 12 }] },
      xAxis: { type: 'time' }, yAxis: { type: 'value' },
      series: { type: 'line', encode: { x: 'time', y: 'value' } },
    },
    bindings: [],
    accessibility: { description: 'Price trend.' },
    ...overrides,
  };
}

describe('FinalAnswer', () => {
  it('keeps the complete executed query in collapsed evidence', () => {
    const query = 'from(bucket: "bitcoin")\n  |> range(start: -30d)\n  |> max()';
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({ references: [{
      source_type: 'query', source_id: 'evi_1', label: '查询证据',
      evidence: { query_language: 'flux', query, row_count: 1 },
    }] })} />);
    expect(markup).toContain('实际执行的查询语句');
    expect(markup).toContain('|&gt; max()</code></pre>');
    expect(markup).not.toContain('answer-query-details" open');
  });

  it('renders a V5 native ECharts chart as linked claim evidence', () => {
    const chart = visualization({
      visualization_id: 'viz_trend',
      verification: {
        target_insight_ids: ['insight_trend'],
        verification_question: '完整序列是否上升？',
        interpretation: '从首个点读到末个点。',
      },
    });
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({
      claims: [{ claim_id: 'claim_trend', text: '价格上升。', insight_ids: ['insight_trend'], visualization_ids: ['viz_trend'] }],
      visualizations: [chart],
    })} />);
    expect(markup).toContain('视觉证据');
    expect(markup).toContain('价格上升。');
    expect(markup).not.toContain('从首个点读到末个点。');
    expect(markup).not.toContain('<h3>可视化</h3>');
  });

  it('promotes only the most information-dense analysis into the conclusion and removes chart prose', () => {
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({
      summary: '存在明显的先升后降区间。',
      sections: [
        { section_type: 'analysis', heading: '结论', content: '区间为1月25日21:47至22:33。' },
        { section_type: 'analysis', heading: '区间特征', content: '起点23307.94，峰值23696.4752，终点23389.0217。' },
      ],
      visualizations: [visualization({ summary: '这是一段很长且重复的图表说明。' })],
    })} />);
    const conclusionEnd = markup.indexOf('</section>');
    const conclusionMarkup = markup.slice(0, conclusionEnd);
    expect(conclusionMarkup).toContain('起点23307.94，峰值23696.4752，终点23389.0217。');
    expect(markup).not.toContain('区间为1月25日21:47至22:33。');
    expect(markup).not.toContain('存在明显的先升后降区间。');
    expect(markup).not.toContain('这是一段很长且重复的图表说明。');
    expect((markup.match(/起点23307.94，峰值23696.4752，终点23389.0217。/g) || []).length).toBe(1);
  });

  it('shows one conclusion heading and keeps internal ids out of visible answer details', () => {
    const internalId = 'ins_ana_1a08a9492d679b29_max_drop_window_8965a0791047';
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({
      summary: '最大跌幅窗口已经确认。',
      sections: [{ section_type: 'analysis', heading: '结论', content: `窗口对应 ${internalId}。` }],
      references: [{
        source_type: 'insight', source_id: internalId, label: internalId,
        evidence: { insight_id: internalId, insight_key: 'max_drop_window', summary: `来自 ${internalId}` },
      }],
      claims: [{ claim_id: 'claim_1', text: '最大跌幅窗口已经确认。', insight_ids: [internalId], visualization_ids: ['viz_trend'] }],
      visualizations: [visualization({ visualization_id: 'viz_trend' })],
    })} />);
    expect((markup.match(/>结论</g) || []).length).toBe(1);
    expect(markup).toContain('视觉证据');
    expect(markup).toContain('图表说明');
    expect(markup).toContain('Max Drop Window');
    expect(markup).not.toContain(internalId);
  });

  it('keeps telemetry out of the answer header', () => {
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer()} elapsedSeconds={19} tokenUsage={{ totals: { total_tokens: 4665, call_count: 7 } }} />);
    expect(markup).not.toContain('4,665 tokens');
    expect(markup).not.toContain('7 calls');
  });
});
