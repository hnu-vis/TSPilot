import { describe, expect, it } from 'vitest';
import { extractFinalAnswer } from './api';

describe('extractFinalAnswer', () => {
  it('normalizes optional answer collections at the stream boundary', () => {
    const answer = extractFinalAnswer({
      answer: {
        summary: 'done',
        sections: null,
        references: { invalid: true },
        visualizations: ['invalid', { schema_version: '2' }],
      },
    });

    expect(answer).toMatchObject({ summary: 'done', sections: [], references: [] });
    expect(answer?.visualizations).toEqual([{ schema_version: '2' }]);
  });

  it('rejects payloads without displayable answer text', () => {
    expect(extractFinalAnswer({ answer: { summary: 42 } })).toBeNull();
  });
});
