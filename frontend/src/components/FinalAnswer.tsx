import {
  BookOpen,
  CheckCircle2,
  ChevronDown,
  Code2,
  Database,
  FileSearch,
  Lightbulb,
} from 'lucide-react';
import { useState } from 'react';
import type { ReactNode } from 'react';
import type { FinalAnswer as FinalAnswerType, TokenUsage } from '../types';

type AnswerSection = NonNullable<FinalAnswerType['sections']>[number];
type AnswerReference = NonNullable<FinalAnswerType['references']>[number];
type EvidenceItem = {
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
type AnswerLocale = 'zh' | 'en';
type AnswerCopy = {
  answer: string;
  emptyAnswer: string;
  conclusion: string;
  evidence: string;
  sources: string;
  source: string;
  referenceCount: (count: number) => string;
  queryResult: string;
  sampled: string;
  showingRows: (visible: number, total?: number | null) => string;
};

const ANSWER_COPY: Record<AnswerLocale, AnswerCopy> = {
  zh: {
    answer: '回答',
    emptyAnswer: '没有生成可展示的回答。',
    conclusion: '结论',
    evidence: '数据依据',
    sources: '方法与来源',
    source: '来源',
    referenceCount: (count) => `${count.toLocaleString()} 项依据`,
    queryResult: '查询结果',
    sampled: '采样预览',
    showingRows: (visible, total) => (
      typeof total === 'number' && total >= visible
        ? `显示 ${visible.toLocaleString()} / ${total.toLocaleString()} 行`
        : `显示 ${visible.toLocaleString()} 行`
    ),
  },
  en: {
    answer: 'Answer',
    emptyAnswer: 'No answer was generated.',
    conclusion: 'Conclusion',
    evidence: 'Data evidence',
    sources: 'Methods and sources',
    source: 'Source',
    referenceCount: (count) => `${count.toLocaleString()} ${count === 1 ? 'reference' : 'references'}`,
    queryResult: 'Query result',
    sampled: 'Sampled preview',
    showingRows: (visible, total) => (
      typeof total === 'number' && total >= visible
        ? `Showing ${visible.toLocaleString()} / ${total.toLocaleString()} rows`
        : `Showing ${visible.toLocaleString()} rows`
    ),
  },
};

type MarkdownBlock =
  | { type: 'paragraph'; content: string }
  | { type: 'bulletList'; items: string[] }
  | { type: 'numberedList'; items: string[]; start: number }
  | { type: 'code'; language: string | null; content: string };

export function FinalAnswer({
  answer,
  tokenUsage,
  elapsedSeconds,
}: {
  answer: FinalAnswerType;
  tokenUsage?: TokenUsage | null;
  elapsedSeconds?: number | null;
}) {
  const locale = answerLocale(answer);
  const copy = ANSWER_COPY[locale];
  const summary = answer.summary?.trim() || copy.emptyAnswer;
  const sections = supportingSections(answer.sections, summary);
  const evidenceItems = normalizeAnswerEvidence(answer);
  const references = answer.references || [];
  const supportingReferences = references.filter((reference) => reference.source_type !== 'query');

  return (
    <div className="final-answer">
      <header className="answer-header">
        <div className="answer-header-title">
          <CheckCircle2 size={17} />
          <span>{answer.title?.trim() || copy.answer}</span>
        </div>
        <div className="answer-header-meta">
          {typeof elapsedSeconds === 'number' && Number.isFinite(elapsedSeconds) ? (
            <span className="answer-reference-count">
              {elapsedSeconds.toFixed(1)}s
            </span>
          ) : null}
          {tokenUsage?.totals?.total_tokens ? (
            <span className="answer-reference-count">
              {tokenUsage.totals.total_tokens.toLocaleString()} tokens
              {typeof tokenUsage.totals.call_count === 'number' ? ` · ${tokenUsage.totals.call_count} calls` : ''}
            </span>
          ) : null}
          {references.length > 0 && (
            <span className="answer-reference-count">
              <Database size={13} />
              {copy.referenceCount(references.length)}
            </span>
          )}
        </div>
      </header>

      <section className="answer-summary-block" aria-label="Answer summary">
        <div className="answer-summary-label">
          <Lightbulb size={15} />
          <span>{copy.conclusion}</span>
        </div>
        <MarkdownContent content={summary} variant="summary" />
      </section>

      {sections.length > 0 && (
        <div className="answer-sections">
          {sections.map((section, index) => (
            <section key={`${section.section_type}-${index}`} className="answer-section">
              <SectionHeading section={section} />
              <StructuredSection section={section} summary={summary} />
            </section>
          ))}
        </div>
      )}

      {evidenceItems.length > 0 && (
        <section className="answer-section answer-evidence-section">
          <ContentHeading icon={<Database size={16} />} title={copy.evidence} />
          <EvidenceItemsView items={evidenceItems} fallbackContent="" copy={copy} />
        </section>
      )}

      {supportingReferences.length > 0 && (
        <section className="answer-section answer-sources-section">
          <ContentHeading icon={<BookOpen size={16} />} title={copy.sources} />
          <ReferenceList references={supportingReferences} copy={copy} />
        </section>
      )}
    </div>
  );
}

function ContentHeading({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="answer-content-heading">
      {icon}
      <h3>{title}</h3>
    </div>
  );
}

function SectionHeading({ section }: { section: AnswerSection }) {
  return (
    <div className="answer-content-heading">
      <FileSearch size={16} />
      <h3>{section.heading || formatLabel(section.section_type)}</h3>
    </div>
  );
}

function StructuredSection({ section, summary }: { section: AnswerSection; summary: string }) {
  if (section.section_type === 'query_results') {
    const items = evidenceItemsFromQueryResultsPayload(section.structured_payload);
    if (items.length > 0) {
      return <EvidenceItemsView items={items} fallbackContent={section.content} />;
    }
  }
  if (section.section_type === 'analysis') {
    const metrics = metricGroups(section.structured_payload);
    const content = sectionContentWithoutDuplicateSummary(section.content, summary);
    return (
      <div className="answer-structured-stack">
        {content && <MarkdownContent content={content} />}
        {metrics.length > 0 && <MetricGroups metrics={metrics} />}
      </div>
    );
  }
  return <MarkdownContent content={section.content} />;
}

function EvidenceItemsView({
  items,
  fallbackContent,
  copy = ANSWER_COPY.en,
}: {
  items: EvidenceItem[];
  fallbackContent: string;
  copy?: AnswerCopy;
}) {
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
              <strong>{item.purpose || `${copy.queryResult} ${index + 1}`}</strong>
              {previewRows.length > 0 && (
                <div className="query-result-preview">
                  <span>{copy.showingRows(Math.min(previewRows.length, 8), item.row_count)}</span>
                  {item.sampled_for_prompt && <span>{copy.sampled}</span>}
                </div>
              )}
            </div>
            {previewRows.length === 0 && item.summary && comparableText(item.summary) !== comparableText(item.purpose || '') && (
              <MarkdownContent content={item.summary} />
            )}
            {previewRows.length > 0 && previewColumns.length > 0 && (
              <DataPreviewTable
                columns={previewColumns.slice(0, 8)}
                rows={previewRows.slice(0, 8)}
              />
            )}
            {visibleQuery(item.query) && (
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

function ReferenceList({ references, copy }: { references: AnswerReference[]; copy: AnswerCopy }) {
  return (
    <div className="answer-reference-list">
      {references.map((reference, index) => (
        <ReferenceItem
          copy={copy}
          index={index}
          key={reference.source_id || `${reference.source_type}-${index}`}
          reference={reference}
        />
      ))}
    </div>
  );
}

function ReferenceItem({ reference, index, copy }: { reference: AnswerReference; index: number; copy: AnswerCopy }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="answer-reference-item" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <span className="answer-reference-icon"><FileSearch size={14} /></span>
        <span className="answer-reference-label">
          <strong>{reference.label || `${copy.source} ${index + 1}`}</strong>
          <small>{formatLabel(reference.source_type)}</small>
        </span>
        <ChevronDown size={15} className="collapsible-chevron" />
      </summary>
      {open && <ReferenceEvidence evidence={reference.evidence} sourceId={reference.source_id} />}
    </details>
  );
}

function ReferenceEvidence({ evidence, sourceId }: { evidence: Record<string, unknown> | null | undefined; sourceId?: string | null }) {
  const record = asRecord(evidence);
  const summary = asString(record?.summary);
  const details = Object.fromEntries(
    Object.entries(record || {}).filter(([key, value]) => key !== 'summary' && value !== null && value !== undefined),
  );
  return (
    <div className="answer-reference-body">
      {summary && <MarkdownContent content={summary} />}
      {sourceId && <div className="answer-source-id">{sourceId}</div>}
      {Object.keys(details).length > 0 && (
        <pre>{JSON.stringify(details, null, 2)}</pre>
      )}
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
            <ol key={`numbers-${index}`} start={block.start}>
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

    const numberedMatch = trimmedLine.match(/^(\d+)[.)、]\s+/);
    if (numberedMatch) {
      const items: string[] = [];
      const start = Number.parseInt(numberedMatch[1], 10);
      while (index < lines.length && /^\d+[.)、]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+[.)、]\s+/, ''));
        index += 1;
      }
      blocks.push({ type: 'numberedList', items, start: Number.isFinite(start) ? start : 1 });
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

function supportingSections(sections: FinalAnswerType['sections'] | undefined, summary: string): AnswerSection[] {
  const seenContent = new Set<string>();
  const excludedTypes = new Set(['summary', 'conclusion', 'query', 'query_results']);
  const normalizedSummary = comparableText(summary);
  const normalizedTitle = (section: AnswerSection) => comparableText(section.heading || formatLabel(section.section_type));

  return (sections || []).filter((section) => {
    if (excludedTypes.has(section.section_type)) return false;
    const content = section.content?.trim();
    if (!content) return false;
    const normalizedContent = comparableText(content);
    if (!normalizedContent || normalizedContent === normalizedSummary) return false;
    if (normalizedContent === normalizedTitle(section)) return false;
    if (seenContent.has(normalizedContent)) return false;
    seenContent.add(normalizedContent);
    return true;
  });
}

function sectionContentWithoutDuplicateSummary(content: string, summary: string) {
  const trimmed = content.trim();
  if (!trimmed) return '';
  return comparableText(trimmed) === comparableText(summary) ? '' : trimmed;
}

function normalizeAnswerEvidence(answer: FinalAnswerType): EvidenceItem[] {
  const primaryItems = [
    ...evidenceItemsFromQueryResultSections(answer.sections),
    ...evidenceItemsFromQuerySections(answer.sections),
  ];
  const referenceItems = evidenceItemsFromQueryReferences(answer.references);
  return mergeEvidenceItems(primaryItems, referenceItems);
}

function evidenceItemsFromQueryResultsPayload(payload: Record<string, unknown> | null | undefined): EvidenceItem[] {
  const items = asRecord(payload)?.items;
  if (!Array.isArray(items)) return [];
  const normalized: EvidenceItem[] = [];
  for (const item of items) {
    const record = asRecord(item);
    if (!record) continue;
    const evidenceItem = normalizeEvidenceRecord(record);
    if (!hasVisibleEvidence(evidenceItem)) continue;
    normalized.push(evidenceItem);
  }
  return normalized;
}

function evidenceItemsFromQueryResultSections(sections: FinalAnswerType['sections'] | undefined): EvidenceItem[] {
  const normalized: EvidenceItem[] = [];
  for (const section of sections || []) {
    if (section.section_type !== 'query_results') continue;
    normalized.push(...evidenceItemsFromQueryResultsPayload(section.structured_payload));
  }
  return normalized;
}

function evidenceItemsFromQuerySections(sections: FinalAnswerType['sections'] | undefined): EvidenceItem[] {
  const normalized: EvidenceItem[] = [];
  for (const section of sections || []) {
    if (section.section_type !== 'query') continue;
    const payload = asRecord(section.structured_payload);
    const fenced = firstCodeBlock(section.content);
    const evidenceItem: EvidenceItem = {
      purpose: section.heading || undefined,
      summary: section.heading || undefined,
      query_language: asString(payload?.query_language) || fenced.language || undefined,
      query: fenced.content || section.content?.trim() || undefined,
    };
    if (!hasVisibleEvidence(evidenceItem)) continue;
    normalized.push(evidenceItem);
  }
  return normalized;
}

function evidenceItemsFromQueryReferences(references: FinalAnswerType['references'] | undefined): EvidenceItem[] {
  const normalized: EvidenceItem[] = [];
  for (const reference of references || []) {
    if (reference.source_type !== 'query') continue;
    const evidence = asRecord(reference.evidence);
    if (!evidence) continue;
    const evidenceItem = normalizeEvidenceRecord(evidence, reference);
    if (!hasVisibleEvidence(evidenceItem)) continue;
    normalized.push(evidenceItem);
  }
  return normalized;
}

function normalizeEvidenceRecord(record: Record<string, unknown>, reference?: AnswerReference): EvidenceItem {
  return {
    evidence_id: asString(record.evidence_id) || asString(reference?.source_id),
    purpose: asString(record.purpose) || asString(reference?.label),
    summary: asString(record.summary),
    query_language: asString(record.query_language),
    query: asString(record.query),
    row_count: typeof record.row_count === 'number' ? record.row_count : null,
    columns: asStringArray(record.columns),
    rows_preview: asRecordArray(record.rows_preview),
    sampled_for_prompt: typeof record.sampled_for_prompt === 'boolean' ? record.sampled_for_prompt : false,
    artifact_ref: asString(record.artifact_ref),
  };
}

function hasVisibleEvidence(item: EvidenceItem) {
  return Boolean(
    visibleQuery(item.query)
    || (item.rows_preview?.length || 0) > 0
    || typeof item.row_count === 'number'
    || item.summary
    || item.artifact_ref
  );
}

function mergeEvidenceItems(primaryItems: EvidenceItem[], supplementalItems: EvidenceItem[]): EvidenceItem[] {
  const merged: EvidenceItem[] = [];
  const indexByKey = new Map<string, number>();

  for (const item of primaryItems) {
    const existingIndex = evidenceIdentityKeys(item).map((key) => indexByKey.get(key)).find((index) => typeof index === 'number');
    if (typeof existingIndex === 'number') {
      merged[existingIndex] = supplementEvidenceItem(merged[existingIndex], item);
      evidenceIdentityKeys(merged[existingIndex]).forEach((key) => indexByKey.set(key, existingIndex));
      continue;
    }
    const nextIndex = merged.length;
    merged.push(item);
    evidenceIdentityKeys(item).forEach((key) => indexByKey.set(key, nextIndex));
  }

  for (const item of supplementalItems) {
    const existingIndex = evidenceIdentityKeys(item).map((key) => indexByKey.get(key)).find((index) => typeof index === 'number');
    if (typeof existingIndex === 'number') {
      merged[existingIndex] = supplementEvidenceItem(merged[existingIndex], item);
      evidenceIdentityKeys(merged[existingIndex]).forEach((key) => indexByKey.set(key, existingIndex));
      continue;
    }
    const nextIndex = merged.length;
    merged.push(item);
    evidenceIdentityKeys(item).forEach((key) => indexByKey.set(key, nextIndex));
  }

  return merged;
}

function supplementEvidenceItem(primary: EvidenceItem, supplemental: EvidenceItem): EvidenceItem {
  return {
    evidence_id: primary.evidence_id || supplemental.evidence_id,
    purpose: genericEvidencePurpose(primary.purpose) ? supplemental.purpose || primary.purpose : primary.purpose || supplemental.purpose,
    summary: genericEvidencePurpose(primary.summary) ? supplemental.summary || primary.summary : primary.summary || supplemental.summary,
    query_language: primary.query_language || supplemental.query_language,
    query: visibleQuery(primary.query) ? primary.query : supplemental.query,
    row_count: typeof primary.row_count === 'number' ? primary.row_count : supplemental.row_count,
    columns: primary.columns?.length ? primary.columns : supplemental.columns,
    rows_preview: primary.rows_preview?.length ? primary.rows_preview : supplemental.rows_preview,
    sampled_for_prompt: primary.sampled_for_prompt || supplemental.sampled_for_prompt,
    artifact_ref: primary.artifact_ref || supplemental.artifact_ref,
  };
}

function genericEvidencePurpose(value: string | undefined) {
  const normalized = comparableText(value || '');
  return normalized === 'query' || normalized === '查询' || normalized === 'database evidence';
}

function evidenceIdentityKeys(item: EvidenceItem) {
  return [
    item.evidence_id && `id:${comparableText(item.evidence_id)}`,
    item.artifact_ref && `artifact:${comparableText(item.artifact_ref)}`,
    visibleQuery(item.query) && `query:${comparableText(item.query)}`,
  ].filter((key): key is string => Boolean(key));
}

function firstCodeBlock(content: string | undefined): { language?: string; content?: string } {
  const match = (content || '').match(/```([A-Za-z0-9_+.-]*)\s*\n([\s\S]*?)```/);
  if (!match) return {};
  return {
    language: match[1]?.trim() || undefined,
    content: match[2]?.trim() || undefined,
  };
}

function visibleQuery(query: string | undefined): query is string {
  if (!query) return false;
  return comparableText(query) !== '[query omitted]';
}

function metricGroups(payload: Record<string, unknown> | null | undefined): Array<Record<string, unknown>> {
  const metrics = asRecord(payload)?.metrics;
  if (!Array.isArray(metrics)) return [];
  return metrics
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => Boolean(item && Object.keys(item).length > 0));
}

function answerLocale(answer: FinalAnswerType): AnswerLocale {
  const visibleText = [
    answer.title,
    answer.summary,
    ...(answer.sections || []).flatMap((section) => [section.heading, section.content]),
  ].filter((value): value is string => typeof value === 'string').join(' ');
  return /[\u3400-\u9fff]/.test(visibleText) ? 'zh' : 'en';
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

function comparableText(value: string) {
  return value.replace(/\s+/g, ' ').trim().toLowerCase();
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
  if (value === 'sql_query') return 'Database evidence';
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
