import type { FinalAnswer as FinalAnswerType } from '../types';

export function FinalAnswer({ answer }: { answer: FinalAnswerType }) {
  const summary = answer.summary?.trim() || '我没有生成可展示的回答。';

  return (
    <div className="final-answer">
      <p className="answer-summary">{summary}</p>
    </div>
  );
}
