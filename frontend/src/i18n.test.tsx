import { describe, expect, it } from 'vitest';
import { translateUi } from './i18n';

describe('frontend i18n catalog', () => {
  it('translates interface copy without altering dynamic query content', () => {
    expect(translateUi('zh-CN', 'New chat')).toBe('新对话');
    expect(translateUi('en', 'New chat')).toBe('New chat');
    expect(translateUi('zh-CN', 'Delete database connection "{name}"?', { name: 'metrics-prod' }))
      .toBe('删除数据库连接“metrics-prod”？');
    expect(translateUi('zh-CN', '用户输入的 query 保持原样')).toBe('用户输入的 query 保持原样');
  });
});
