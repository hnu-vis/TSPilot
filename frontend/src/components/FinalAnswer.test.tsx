import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { FinalAnswer as FinalAnswerType } from '../types';
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
      claims: [{ claim_id: 'claim_1', text: '内部绑定文本', fact_ids: ['fact_1'] }],
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

  it('deduplicates repeated logical fact references', () => {
    const duplicateFact = {
      source_type: 'fact',
      label: '最大 7 天窗口标准差',
      evidence: { fact_key: 'max_7d_window_std', statement: '最大窗口。' },
    } as const;
    const markup = renderToStaticMarkup(<FinalAnswer answer={answer({
      references: [
        { ...duplicateFact, source_id: 'fact_1' },
        { ...duplicateFact, source_id: 'fact_2' },
      ],
    })} />);

    expect(markup.match(/最大 7 天窗口标准差/g)).toHaveLength(1);
    expect(markup).toContain('1 项依据');
  });
});
