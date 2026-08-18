import { describe, expect, it } from 'vitest';
import { formatHumanTime, isIsoTimestamp } from './humanTime';

describe('human time presentation', () => {
  it('renders precise Chinese and English labels without changing the instant', () => {
    const source = '2023-01-04T23:48:00+00:00';

    expect(formatHumanTime(source, 'zh-CN')).toBe('2023年1月4日 23:48（UTC）');
    expect(formatHumanTime(source, 'en')).toContain('Jan 4, 2023');
    expect(formatHumanTime(source, 'en')).toBe('Jan 4, 2023 at 11:48 PM UTC');
    expect(formatHumanTime(source, 'en')).toContain('UTC');
    expect(source).toBe('2023-01-04T23:48:00+00:00');
  });

  it('preserves meaningful sub-minute precision and rejects ordinary text', () => {
    expect(formatHumanTime('2023-01-04T23:48:07.123Z', 'zh-CN')).toBe(
      '2023年1月4日 23:48:07.123（UTC）',
    );
    expect(formatHumanTime('2023-01-04T23:48:07.123Z', 'en')).toBe(
      'Jan 4, 2023 at 11:48:07.123 PM UTC',
    );
    expect(isIsoTimestamp('2023-01-04T23:48:00+00:00')).toBe(true);
    expect(isIsoTimestamp('January revenue')).toBe(false);
  });
});
