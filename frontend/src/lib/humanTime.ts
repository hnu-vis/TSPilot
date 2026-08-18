export type HumanTimeLocale = 'zh-CN' | 'en';

export type HumanTimeStyle = 'axis' | 'axis-compact' | 'long';

const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})$/i;

export function isIsoTimestamp(value: unknown): value is string {
  return typeof value === 'string' && ISO_TIMESTAMP.test(value.trim());
}

/**
 * Render a chart timestamp in UTC without changing its grounded source value.
 * UTC is deliberate: it prevents a browser's local timezone from silently
 * moving an analytical event to a different clock time.
 */
export function formatHumanTime(
  value: unknown,
  locale: HumanTimeLocale = 'en',
  style: HumanTimeStyle = 'long',
): string {
  const parsed = typeof value === 'number' && Number.isFinite(value)
    ? value
    : Date.parse(String(value ?? ''));
  if (!Number.isFinite(parsed)) return String(value ?? '');

  const date = new Date(parsed);
  const month = date.getUTCMonth() + 1;
  const day = date.getUTCDate();
  const hour = String(date.getUTCHours()).padStart(2, '0');
  const minute = String(date.getUTCMinutes()).padStart(2, '0');
  if (style === 'axis' || style === 'axis-compact') {
    const datePart = locale === 'zh-CN' ? `${month}月${day}日` : `${month}/${day}`;
    const separator = style === 'axis-compact' ? '\n' : ' ';
    return `${datePart}${separator}${hour}:${minute}`;
  }

  const includeSeconds = date.getUTCSeconds() !== 0 || date.getUTCMilliseconds() !== 0;
  if (locale === 'zh-CN') {
    const seconds = includeSeconds
      ? `:${String(date.getUTCSeconds()).padStart(2, '0')}${date.getUTCMilliseconds() ? `.${String(date.getUTCMilliseconds()).padStart(3, '0')}` : ''}`
      : '';
    return `${date.getUTCFullYear()}年${month}月${day}日 ${hour}:${minute}${seconds}（UTC）`;
  }
  const monthName = new Intl.DateTimeFormat('en', { timeZone: 'UTC', month: 'short' }).format(date);
  const hour12 = date.getUTCHours() % 12 || 12;
  const seconds = includeSeconds
    ? `:${String(date.getUTCSeconds()).padStart(2, '0')}${date.getUTCMilliseconds() ? `.${String(date.getUTCMilliseconds()).padStart(3, '0')}` : ''}`
    : '';
  const period = date.getUTCHours() < 12 ? 'AM' : 'PM';
  return `${monthName} ${day}, ${date.getUTCFullYear()} at ${hour12}:${minute}${seconds} ${period} UTC`;
}
