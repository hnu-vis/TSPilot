import type { FinalAnswer as FinalAnswerType } from '../types';

export function FinalAnswer({ answer }: { answer: FinalAnswerType }) {
  const summary = answer.summary?.trim() || '我没有生成可展示的回答。';
  const sections = (answer.sections || []).filter((section) => (
    section.content?.trim()
    && section.section_type !== 'summary'
    && section.section_type !== 'conclusion'
  ));

  return (
    <div className="final-answer">
      {answer.title && <h2>{answer.title}</h2>}
      <p className="answer-summary">{summary}</p>
      {sections.length > 0 && (
        <div className="answer-sections">
          {sections.map((section, index) => (
            <section key={`${section.section_type}-${index}`} className="answer-section">
              <h3>{section.heading || formatLabel(section.section_type)}</h3>
              <ContentBlock content={section.content} />
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

function ContentBlock({ content }: { content: string }) {
  const lines = content.split('\n').map((line) => line.trim()).filter(Boolean);
  const bulletLines = lines.filter((line) => line.startsWith('- '));
  if (lines.length > 1 && bulletLines.length === lines.length) {
    return (
      <ul>
        {bulletLines.map((line, index) => (
          <li key={`${line}-${index}`}>{line.slice(2)}</li>
        ))}
      </ul>
    );
  }
  return <p>{content}</p>;
}

function formatLabel(value: string) {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
