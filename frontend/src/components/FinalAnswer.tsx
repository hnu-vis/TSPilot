import { CheckCircle2, ChevronDown, Code2, Database } from 'lucide-react';
import type { FinalAnswer as FinalAnswerType, TokenUsage } from '../types';

type AnswerSection = NonNullable<FinalAnswerType['sections']>[number];
type QueryResultItem = {
  evidence_id?: string;
  purpose?: string;
  summary?: string;
  query_language?: string;
  query?: string;
  row_count?: number | null;
  columns?: string[];
  rows_preview?: Array<Record<string, unknown>>;
  sampled_for_prompt?: boolean;
  artifact_ref?: string;
};

type MarkdownBlock =
  | { type: 'paragraph'; content: string }
  | { type: 'bulletList'; items: string[] }
  | { type: 'numberedList'; items: string[] }
  | { type: 'code'; language: string | null; content: string };

export function FinalAnswer({ answer, tokenUsage }: { answer: FinalAnswerType; tokenUsage?: TokenUsage | null }) {
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
        <div className="answer-header-meta">
          {tokenUsage?.totals?.total_tokens ? (
            <span className="answer-reference-count">
              {tokenUsage.totals.total_tokens.toLocaleString()} tokens
              {typeof tokenUsage.totals.call_count === 'number' ? ` · ${tokenUsage.totals.call_count} calls` : ''}
            </span>
          ) : null}
          {references.length > 0 && (
            <span className="answer-reference-count">
              <Database size={13} />
              {references.length} reference{references.length > 1 ? 's' : ''}
            </span>
          )}
        </div>
      </header>

      <section className="answer-summary-block" aria-label="Answer summary">
        <MarkdownContent content={summary} variant="summary" />
      </section>

      {sections.length > 0 && (
        <div className="answer-sections">
          {sections.map((section, index) => (
            <section key={`${section.section_type}-${index}`} className="answer-section">
              {shouldShowSectionType(section) && (
                <span className="answer-section-type">{formatLabel(section.section_type)}</span>
              )}
              <h3>{section.heading || formatLabel(section.section_type)}</h3>
              <StructuredSection section={section} />
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

function StructuredSection({ section }: { section: AnswerSection }) {
  if (section.section_type === 'query_results') {
    const items = queryResultItems(section.structured_payload);
    if (items.length > 0) {
      return <QueryResultsView items={items} fallbackContent={section.content} />;
    }
  }
  if (section.section_type === 'analysis') {
    const metrics = metricGroups(section.structured_payload);
    return (
      <div className="answer-structured-stack">
        <MarkdownContent content={section.content} />
        {metrics.length > 0 && <MetricGroups metrics={metrics} />}
      </div>
    );
  }
  return <MarkdownContent content={section.content} />;
}

function QueryResultsView({ items, fallbackContent }: { items: QueryResultItem[]; fallbackContent: string }) {
  return (
    <div className="query-result-list">
      {items.length === 0 && <MarkdownContent content={fallbackContent} />}
      {items.map((item, index) => {
        const columns = item.columns || [];
        const previewRows = item.rows_preview || [];
        const previewColumns = columns.length > 0
          ? columns
          : Array.from(new Set(previewRows.flatMap((row) => Object.keys(row))));
        return (
          <article className="query-result-item" key={item.evidence_id || `${index}`}>
            <div className="query-result-heading">
              <span>{index + 1}</span>
              <strong>{item.purpose || `Query result ${index + 1}`}</strong>
              {previewRows.length > 0 && (
                <div className="query-result-preview">
                  <span>{previewLabel(Math.min(previewRows.length, 8), item.row_count)}</span>
                  {item.sampled_for_prompt && <span>Sampled</span>}
                </div>
              )}
            </div>
            {previewRows.length > 0 && previewColumns.length > 0 && (
              <DataPreviewTable
                columns={previewColumns.slice(0, 8)}
                rows={previewRows.slice(0, 8)}
              />
            )}
            {previewRows.length === 0 && item.summary && (
              <MarkdownContent content={item.summary} />
            )}
            {item.query && (
              <details className="answer-inline-details">
                <summary>
                  <ChevronDown size={14} className="collapsible-chevron" />
                  {queryLabel(item.query_language)}
                </summary>
                <pre>{item.query}</pre>
              </details>
            )}
          </article>
        );
      })}
    </div>
  );
}

function DataPreviewTable({
  columns,
  rows,
}: {
  columns: string[];
  rows: Array<Record<string, unknown>>;
}) {
  return (
    <div className="answer-table-wrap">
      <table className="answer-data-table">
        <thead>
          <tr>
            {columns.map((column) => <th key={column}>{column}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`${rowIndex}`}>
              {columns.map((column) => <td key={column}>{formatCell(row[column])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricGroups({ metrics }: { metrics: Array<Record<string, unknown>> }) {
  return (
    <div className="answer-metric-groups">
      {metrics.map((metricGroup, index) => (
        <dl className="answer-metric-grid" key={`${index}`}>
          {Object.entries(metricGroup).map(([key, value]) => (
            <div key={key}>
              <dt>{formatLabel(key)}</dt>
              <dd>{formatCell(value)}</dd>
            </div>
          ))}
        </dl>
      ))}
    </div>
  );
}

export function MarkdownContent({
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
          return <CollapsibleCodeBlock block={block} index={index} key={`code-${index}`} />;
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

function CollapsibleCodeBlock({ block, index }: { block: Extract<MarkdownBlock, { type: 'code' }>; index: number }) {
  const language = block.language?.trim();
  const lineCount = block.content ? block.content.split('\n').length : 0;
  const title = language ? language.toUpperCase() : `Code ${index + 1}`;
  return (
    <details className="answer-code-details">
      <summary>
        <span>
          <ChevronDown size={14} className="collapsible-chevron" />
          <Code2 size={14} />
          {title}
        </span>
        {lineCount > 0 && <strong>{lineCount} lines</strong>}
      </summary>
      <pre className="answer-code-block">
        <code>{block.content}</code>
      </pre>
    </details>
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

function shouldShowSectionType(section: AnswerSection) {
  const heading = (section.heading || '').trim().toLowerCase();
  const sectionType = formatLabel(section.section_type).trim().toLowerCase();
  return Boolean(heading && heading !== sectionType);
}

function queryResultItems(payload: Record<string, unknown> | null | undefined): QueryResultItem[] {
  const items = asRecord(payload)?.items;
  if (!Array.isArray(items)) return [];
  const normalized: QueryResultItem[] = [];
  for (const item of items) {
    const record = asRecord(item);
    if (!record) continue;
    normalized.push({
      evidence_id: asString(record.evidence_id),
      purpose: asString(record.purpose),
      summary: asString(record.summary),
      query_language: asString(record.query_language),
      query: asString(record.query),
      row_count: typeof record.row_count === 'number' ? record.row_count : null,
      columns: asStringArray(record.columns),
      rows_preview: asRecordArray(record.rows_preview),
      sampled_for_prompt: typeof record.sampled_for_prompt === 'boolean' ? record.sampled_for_prompt : false,
      artifact_ref: asString(record.artifact_ref),
    });
  }
  return normalized;
}

function metricGroups(payload: Record<string, unknown> | null | undefined): Array<Record<string, unknown>> {
  const metrics = asRecord(payload)?.metrics;
  if (!Array.isArray(metrics)) return [];
  return metrics
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => Boolean(item && Object.keys(item).length > 0));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function asRecordArray(value: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is Record<string, unknown> => Boolean(asRecord(item)));
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function queryLabel(language: string | undefined) {
  const normalized = (language || '').toLowerCase();
  if (normalized === 'flux') return 'Flux';
  if (normalized === 'promql') return 'PromQL';
  if (normalized.includes('sql')) return 'SQL';
  return 'Query';
}

function previewLabel(visibleRows: number, totalRows?: number | null) {
  if (typeof totalRows === 'number' && totalRows >= visibleRows) {
    return `Showing ${visibleRows.toLocaleString()} / ${totalRows.toLocaleString()} rows`;
  }
  return `Showing ${visibleRows.toLocaleString()} rows`;
}

function formatLabel(value: string) {
  if (value === 'query') return 'Database evidence';
  if (value === 'sql_query' || value === 'query_database') return 'Database evidence';
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
