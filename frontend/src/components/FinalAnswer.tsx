import { CheckCircle2, Database, FileText } from 'lucide-react';
import type { FinalAnswer as FinalAnswerType } from '../types';

type MarkdownBlock =
  | { type: 'paragraph'; content: string }
  | { type: 'bulletList'; items: string[] }
  | { type: 'numberedList'; items: string[] }
  | { type: 'code'; language: string | null; content: string };

export function FinalAnswer({ answer }: { answer: FinalAnswerType }) {
  const summary = answer.summary?.trim() || '我没有生成可展示的回答。';
  const sections = (answer.sections || []).filter((section) => (
    section.content?.trim()
    && section.section_type !== 'summary'
    && section.section_type !== 'conclusion'
  ));
  const references = answer.references || [];

  return (
    <div className="final-answer">
      <header className="answer-header">
        <div className="answer-header-title">
          <CheckCircle2 size={17} />
          <span>Answer</span>
        </div>
        {references.length > 0 && (
          <span className="answer-reference-count">
            <Database size={13} />
            {references.length} reference{references.length > 1 ? 's' : ''}
          </span>
        )}
      </header>

      <section className="answer-summary-block" aria-label="Answer summary">
        <MarkdownContent content={summary} variant="summary" />
      </section>

      {sections.length > 0 && (
        <div className="answer-sections">
          {sections.map((section, index) => (
            <section key={`${section.section_type}-${index}`} className="answer-section">
              <span className="answer-section-type">{formatLabel(section.section_type)}</span>
              <h3>{section.heading || formatLabel(section.section_type)}</h3>
              <ContentBlock content={section.content} />
            </section>
          ))}
        </div>
      )}

      {references.length > 0 && (
        <footer className="answer-reference-strip" aria-label="Answer references">
          <FileText size={14} />
          <div>
            {references.slice(0, 4).map((reference, index) => (
              <span key={`${reference.source_type}-${reference.source_id || index}`}>
                {formatLabel(reference.source_type)}
              </span>
            ))}
          </div>
        </footer>
      )}
    </div>
  );
}

function ContentBlock({ content }: { content: string }) {
  return <MarkdownContent content={content} />;
}

function MarkdownContent({
  content,
  variant = 'section',
}: {
  content: string;
  variant?: 'summary' | 'section';
}) {
  const blocks = parseMarkdownBlocks(content);
  return (
    <div className={`answer-markdown answer-markdown-${variant}`}>
      {blocks.map((block, index) => {
        if (block.type === 'code') {
          return (
            <pre className="answer-code-block" key={`code-${index}`}>
              {block.language && <span className="answer-code-language">{block.language}</span>}
              <code>{block.content}</code>
            </pre>
          );
        }
        if (block.type === 'bulletList') {
          return (
            <ul key={`bullets-${index}`}>
              {block.items.map((item, itemIndex) => <li key={`${item}-${itemIndex}`}>{item}</li>)}
            </ul>
          );
        }
        if (block.type === 'numberedList') {
          return (
            <ol key={`numbers-${index}`}>
              {block.items.map((item, itemIndex) => <li key={`${item}-${itemIndex}`}>{item}</li>)}
            </ol>
          );
        }
        return <p key={`paragraph-${index}`}>{block.content}</p>;
      })}
    </div>
  );
}

function parseMarkdownBlocks(content: string): MarkdownBlock[] {
  const lines = content.trim().split('\n');
  const blocks: MarkdownBlock[] = [];
  let index = 0;

  while (index < lines.length) {
    const rawLine = lines[index];
    const trimmedLine = rawLine.trim();
    if (!trimmedLine) {
      index += 1;
      continue;
    }

    const fenceMatch = trimmedLine.match(/^```([A-Za-z0-9_+.-]*)\s*$/);
    if (fenceMatch) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push({
        type: 'code',
        language: fenceMatch[1] || null,
        content: codeLines.join('\n').trimEnd(),
      });
      continue;
    }

    if (trimmedLine.startsWith('- ')) {
      const items: string[] = [];
      while (index < lines.length && lines[index].trim().startsWith('- ')) {
        items.push(lines[index].trim().slice(2));
        index += 1;
      }
      blocks.push({ type: 'bulletList', items });
      continue;
    }

    if (/^\d+[.)、]\s+/.test(trimmedLine)) {
      const items: string[] = [];
      while (index < lines.length && /^\d+[.)、]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+[.)、]\s+/, ''));
        index += 1;
      }
      blocks.push({ type: 'numberedList', items });
      continue;
    }

    const paragraphLines: string[] = [];
    while (
      index < lines.length
      && lines[index].trim()
      && !lines[index].trim().match(/^```([A-Za-z0-9_+.-]*)\s*$/)
      && !lines[index].trim().startsWith('- ')
      && !/^\d+[.)、]\s+/.test(lines[index].trim())
    ) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    blocks.push({ type: 'paragraph', content: paragraphLines.join('\n') });
  }

  return blocks.length > 0 ? blocks : [{ type: 'paragraph', content: content.trim() }];
}

function formatLabel(value: string) {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
