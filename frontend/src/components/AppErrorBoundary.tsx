import { Component, type ErrorInfo, type ReactNode } from 'react';

type Props = { children: ReactNode };
type State = { error: Error | null };

export class AppErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('TSPilot interface rendering failed.', error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="fatal-error" role="alert">
        <div className="fatal-error-card">
          <div className="brand-mark" aria-hidden="true">TS</div>
          <h1>页面暂时无法显示</h1>
          <p>界面遇到了未预期的数据或渲染错误。你的历史对话仍保存在浏览器中。</p>
          <details>
            <summary>错误信息</summary>
            <code>{this.state.error.message}</code>
          </details>
          <button type="button" onClick={() => window.location.reload()}>重新加载</button>
        </div>
      </main>
    );
  }
}
