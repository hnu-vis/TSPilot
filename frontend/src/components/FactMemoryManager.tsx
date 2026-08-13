import {
  AlertCircle,
  BookOpen,
  BrainCircuit,
  Braces,
  ChevronLeft,
  ChevronRight,
  Database,
  HardDrive,
  RefreshCw,
  Search,
  Wrench,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { fetchFactMemory, fetchFactMemoryDetail } from '../services/api';
import type { DatabaseResource, FactMemoryResponse, MemoryCard, MemoryDetail } from '../types';

type Props = {
  databases: DatabaseResource[];
  selectedDatabaseId: string | null;
};

type MemoryState = {
  loading: boolean;
  error: string | null;
  data: FactMemoryResponse | null;
};

export function FactMemoryManager({ databases, selectedDatabaseId }: Props) {
  const orderedDatabases = useMemo(
    () => orderFactMemoryDatabases(databases, selectedDatabaseId),
    [databases, selectedDatabaseId],
  );
  const [scope, setScope] = useState<'global' | 'database'>(
    selectedDatabaseId || databases.length ? 'database' : 'global',
  );
  const scopeWasSelected = useRef(false);
  const [memoryDatabaseId, setMemoryDatabaseId] = useState<string | null>(selectedDatabaseId || orderedDatabases[0]?.id || null);
  const [query, setQuery] = useState('');
  const [kindFilter, setKindFilter] = useState<'all' | 'definition' | 'recipe'>('all');
  const [page, setPage] = useState(0);
  const [reloadToken, setReloadToken] = useState(0);
  const [state, setState] = useState<MemoryState>({ loading: false, error: null, data: null });
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null);
  const [detailState, setDetailState] = useState<{ loading: boolean; error: string | null; detail: MemoryDetail | null }>({
    loading: false,
    error: null,
    detail: null,
  });
  const databaseId = scope === 'database' ? memoryDatabaseId : null;
  const activeDatabase = databases.find((database) => database.id === memoryDatabaseId) || null;

  useEffect(() => {
    const selectedDatabaseExists = orderedDatabases.some((database) => database.id === selectedDatabaseId);
    const currentDatabaseExists = orderedDatabases.some((database) => database.id === memoryDatabaseId);
    if (!currentDatabaseExists) {
      setMemoryDatabaseId(selectedDatabaseExists ? selectedDatabaseId : orderedDatabases[0]?.id || null);
    }
  }, [memoryDatabaseId, orderedDatabases, selectedDatabaseId]);

  useEffect(() => {
    if (!scopeWasSelected.current && databases.length) setScope('database');
  }, [databases.length]);

  useEffect(() => {
    let cancelled = false;
    setState((current) => ({ loading: true, error: null, data: current.data }));
    fetchFactMemory(databaseId)
      .then((data) => {
        if (!cancelled) setState({ loading: false, error: null, data });
      })
      .catch((error) => {
        if (!cancelled) {
          setState({
            loading: false,
            error: error instanceof Error ? error.message : 'Unable to load fact memory.',
            data: null,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [databaseId, reloadToken]);

  const cards = state.data?.memory.cards || [];
  const visibleCards = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const filtered = cards.filter((card) => {
      const kindMatches = kindFilter === 'all'
        || (kindFilter === 'definition' && card.kind === 'fact_definition')
        || (kindFilter === 'recipe' && card.kind === 'fact_recipe');
      if (!kindMatches) return false;
      if (!normalizedQuery) return true;
      return [card.title, card.description, card.kind, ...(card.tags || [])]
        .join(' ')
        .toLowerCase()
        .includes(normalizedQuery);
    });
    if (scope !== 'database' || kindFilter === 'definition') return filtered;
    return filtered.map((card, index) => ({ card, index })).sort((left, right) => {
      const leftPriority = left.card.kind === 'fact_recipe' ? 0 : 1;
      const rightPriority = right.card.kind === 'fact_recipe' ? 0 : 1;
      return leftPriority - rightPriority || left.index - right.index;
    }).map(({ card }) => card);
  }, [cards, kindFilter, query, scope]);
  const summary = state.data?.prompt_view?.summary;
  const summaryRecord = summary && typeof summary === 'object'
    ? summary as Record<string, unknown>
    : {};
  const pageSize = 10;
  const pageCount = Math.max(1, Math.ceil(visibleCards.length / pageSize));
  const pageCards = visibleCards.slice(page * pageSize, (page + 1) * pageSize);

  useEffect(() => {
    setPage(0);
  }, [databaseId, kindFilter, query]);

  useEffect(() => {
    if (page >= pageCount) setPage(pageCount - 1);
  }, [page, pageCount]);

  useEffect(() => {
    if (visibleCards.length === 0) {
      setSelectedCardId(null);
      return;
    }
    if (!selectedCardId || !pageCards.some((card) => card.id === selectedCardId)) {
      setSelectedCardId(pageCards[0]?.id || visibleCards[0].id);
    }
  }, [pageCards, selectedCardId, visibleCards]);

  useEffect(() => {
    if (!selectedCardId) {
      setDetailState({ loading: false, error: null, detail: null });
      return;
    }
    let cancelled = false;
    setDetailState((current) => ({ loading: true, error: null, detail: current.detail }));
    fetchFactMemoryDetail(selectedCardId, databaseId)
      .then((payload) => {
        if (!cancelled) setDetailState({ loading: false, error: null, detail: payload.detail });
      })
      .catch((error) => {
        if (!cancelled) {
          setDetailState({
            loading: false,
            error: error instanceof Error ? error.message : 'Unable to load memory detail.',
            detail: null,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedCardId, databaseId]);

  return (
    <section className="fact-memory-manager" aria-label="Fact memory">
      <div className="fact-memory-toolbar">
        <div className="fact-memory-scope">
          <div className="segmented-control" aria-label="Fact memory scope">
            <button
              type="button"
              className={scope === 'database' ? 'active' : ''}
              disabled={!memoryDatabaseId}
              onClick={() => {
                scopeWasSelected.current = true;
                setScope('database');
              }}
            >
              <Database size={14} />
              <span>Database memory</span>
            </button>
            <button
              type="button"
              className={scope === 'global' ? 'active' : ''}
              onClick={() => {
                scopeWasSelected.current = true;
                setScope('global');
              }}
            >
              <BrainCircuit size={14} />
              <span>System defaults</span>
            </button>
          </div>
          {scope === 'database' ? (
            <label className="fact-memory-database-select">
              <Database size={13} />
              <select
                value={memoryDatabaseId || ''}
                onChange={(event) => setMemoryDatabaseId(event.target.value || null)}
                aria-label="Select Fact Memory database"
              >
                {orderedDatabases.map((database) => (
                  <option key={database.id} value={database.id}>
                    {database.display_name || database.name || database.id} · {formatRecipeCount(database)}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <span className="fact-memory-scope-label">Built-in contracts</span>
          )}
        </div>
        <div className="fact-memory-tools">
          <label className="fact-memory-search">
            <Search size={14} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search memory" aria-label="Search fact memory" />
          </label>
          <div className="segmented-control fact-memory-kind-filter" aria-label="Memory card type">
            <button type="button" className={kindFilter === 'all' ? 'active' : ''} onClick={() => setKindFilter('all')}>All</button>
            <button type="button" className={kindFilter === 'definition' ? 'active' : ''} onClick={() => setKindFilter('definition')}>
              <BookOpen size={13} /><span>Definitions</span>
            </button>
            <button type="button" className={kindFilter === 'recipe' ? 'active' : ''} onClick={() => setKindFilter('recipe')}>
              <Wrench size={13} /><span>Recipes</span>
            </button>
          </div>
          <button
            className="database-source-add"
            type="button"
            title="Refresh memory"
            aria-label="Refresh memory"
            disabled={state.loading}
            onClick={() => setReloadToken((value) => value + 1)}
          >
            <RefreshCw size={14} className={state.loading ? 'spin' : ''} />
          </button>
        </div>
      </div>

      {scope === 'database' && !activeDatabase && (
        <div className="database-inline-error">
          <AlertCircle size={16} />
          <span>Select a database in chat or Database view to inspect scoped fact memory.</span>
        </div>
      )}
      {state.error && (
        <div className="database-inline-error">
          <AlertCircle size={16} />
          <span>{state.error}</span>
        </div>
      )}

      <div className="fact-memory-summary" aria-label="Memory summary">
        <span><strong>{visibleCards.length.toLocaleString()}</strong> visible <small>of {cards.length.toLocaleString()}</small></span>
        <span><strong>{String(summaryRecord.definition_count ?? cards.filter((card) => card.kind === 'fact_definition').length)}</strong> definitions</span>
        <span><strong>{String(summaryRecord.recipe_count ?? cards.filter((card) => card.kind === 'fact_recipe').length)}</strong> recipes</span>
        <span>Updated <strong>{formatMemoryTimestamp(state.data?.memory.updated_at)}</strong></span>
      </div>

      <section className="fact-memory-library" aria-labelledby="memory-library-title">
        <div className="fact-memory-section-heading">
          <div>
            <BrainCircuit size={15} />
            <h2 id="memory-library-title">Memory Library</h2>
          </div>
          {pageCount > 1 && (
            <div className="fact-memory-pagination" aria-label="Memory pages">
              <button type="button" aria-label="Previous memory page" disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>
                <ChevronLeft size={14} />
              </button>
              <span>{page + 1} / {pageCount}</span>
              <button type="button" aria-label="Next memory page" disabled={page >= pageCount - 1} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>
                <ChevronRight size={14} />
              </button>
            </div>
          )}
        </div>
        {pageCards.length > 0 ? (
          <div className="fact-memory-tile-grid">
              {pageCards.map((card) => (
                <MemoryCardItem
                  key={card.id}
                  card={card}
                  selected={selectedCardId === card.id}
                  onSelect={() => setSelectedCardId(card.id)}
                />
              ))}
          </div>
          ) : (
            <EmptyState label={
              cards.length
                ? 'No memory cards match the current filter.'
                : scope === 'database'
                  ? 'This database has not learned any Fact definitions or recipes yet.'
                  : 'No system-default Fact Memory is available.'
            } />
          )}
      </section>

      <section className="fact-memory-detail" aria-labelledby="selected-memory-title">
        <div className="fact-memory-section-heading">
          <div>
            <Braces size={15} />
            <h2 id="selected-memory-title">Selected Memory</h2>
          </div>
        </div>
          {detailState.loading ? (
            <EmptyState label="Loading memory detail." />
          ) : detailState.error ? (
            <EmptyState label={detailState.error} />
          ) : detailState.detail ? (
            <MemoryDetailCard detail={detailState.detail} />
          ) : (
            <EmptyState label="Select a memory card to load its detail." />
          )}
          <div className="fact-memory-storage">
            <span><HardDrive size={13} /> Storage</span>
            <code>{state.data?.memory.storage_path || 'Default in-memory definitions'}</code>
          </div>
      </section>
    </section>
  );
}

export function orderFactMemoryDatabases(databases: DatabaseResource[], selectedDatabaseId: string | null) {
  return databases.map((database, index) => ({ database, index })).sort((left, right) => {
    const leftSelected = left.database.id === selectedDatabaseId;
    const rightSelected = right.database.id === selectedDatabaseId;
    if (leftSelected !== rightSelected) return leftSelected ? -1 : 1;

    const recipeDifference = (right.database.fact_memory_summary?.recipe_count || 0)
      - (left.database.fact_memory_summary?.recipe_count || 0);
    if (recipeDifference) return recipeDifference;

    const updatedDifference = String(right.database.fact_memory_summary?.updated_at || '')
      .localeCompare(String(left.database.fact_memory_summary?.updated_at || ''));
    return updatedDifference || left.index - right.index;
  }).map(({ database }) => database);
}

function formatRecipeCount(database: DatabaseResource) {
  const count = database.fact_memory_summary?.recipe_count || 0;
  return `${count} ${count === 1 ? 'recipe' : 'recipes'}`;
}

function MemoryCardItem({ card, selected, onSelect }: { card: MemoryCard; selected: boolean; onSelect: () => void }) {
  const isRecipe = card.kind === 'fact_recipe';
  const source = card.updated_at ? 'Learned' : 'System';
  const tool = (card.tags || []).find((tag) => ['sql_query', 'code_interpreter'].includes(tag));
  const factType = (card.tags || []).find((tag) => tag !== tool && !['global', 'system'].includes(tag.toLowerCase()));
  return (
    <button
      type="button"
      className={`fact-memory-tile ${selected ? 'selected' : ''}`}
      onClick={onSelect}
      aria-pressed={selected}
      title={`${card.title}\n${card.description}`}
    >
      <div className="fact-memory-tile-kind">
        {isRecipe ? <Wrench size={11} /> : <BookOpen size={11} />}
        <span>{isRecipe ? 'Recipe' : 'Definition'}</span>
        <i className={source === 'Learned' ? 'learned' : ''} aria-label={source} title={source} />
      </div>
      <strong>{humanize(card.title)}</strong>
      <p>{card.description}</p>
      <small>{[humanize(tool), humanize(factType || source)].filter(Boolean).join(' · ')}</small>
    </button>
  );
}

function MemoryDetailCard({ detail }: { detail: MemoryDetail }) {
  const request = detail.fact_request || {};
  const requirements = asRecord(request.requirements);
  const tags = detail.card.tags || [];
  const isRecipe = detail.card.kind === 'fact_recipe';
  const source = detail.card.updated_at ? 'Learned' : 'System default';
  const contractRows = compactRows([
    ['Fact type', request.fact_type || tags[0]],
    ['Subject', request.subject],
    ['Result shape', request.result_shape],
    ['Expected items', request.expected_item_count],
    ['Scope', tags[Math.max(tags.length - 1, 0)]],
  ]);
  const generationRows = compactRows([
    ['Preferred tool', detail.preferred_tool],
    ['Required evidence', isRecipe ? undefined : tags.slice(1, -1)],
    ['Derivation', request.derivation],
    ['Time position', requirements.time_position],
    ['Operator', requirements.operator],
    ['Dimensions', request.dimensions],
    ['Dependencies', request.derived_from],
  ]);
  const verification = (detail.guidance || '').split(';').map((item) => item.trim()).filter(Boolean);
  return (
    <article className="fact-memory-detail-card">
      <header className="fact-memory-detail-header">
        <div>
          <span>{detail.card.kind === 'fact_recipe' ? 'Fact recipe' : 'Fact definition'}</span>
          <h3>{humanize(detail.card.title)}</h3>
        </div>
        <span className={`fact-memory-source ${source === 'Learned' ? 'learned' : ''}`}>{source}</span>
      </header>
      <p className="fact-memory-detail-description">{detail.card.description}</p>

      <div className="fact-memory-detail-grid">
        <DetailBlock title="Fact contract" rows={contractRows} empty="This definition does not create a Fact request directly." />
        <DetailBlock title="Generation rule" rows={generationRows} empty="Generation behavior is defined by the runtime contract." />
      </div>

      <div className="fact-memory-verification">
        <h4>Verification</h4>
        {verification.length ? (
          <ul>{verification.map((item) => <li key={item}>{item}</li>)}</ul>
        ) : (
          <p>{detail.guidance || 'Use current request evidence and the runtime Fact contract for verification.'}</p>
        )}
      </div>

      <details className="fact-memory-raw-contract">
        <summary>Technical details and raw contract</summary>
        <dl>
          <div><dt>Memory ID</dt><dd><code>{detail.id}</code></dd></div>
          <div><dt>Tags</dt><dd>{(detail.card.tags || []).join(' · ') || 'None'}</dd></div>
        </dl>
        {detail.fact_request && <pre className="debug-json">{JSON.stringify(detail.fact_request, null, 2)}</pre>}
        {detail.examples && detail.examples.length > 0 && (
          <div className="chip-list compact">
            {detail.examples.slice(0, 4).map((item) => <span key={item}>{item}</span>)}
          </div>
        )}
      </details>
    </article>
  );
}

function DetailBlock({ title, rows, empty }: { title: string; rows: Array<[string, unknown]>; empty: string }) {
  return (
    <section className="fact-memory-detail-block">
      <h4>{title}</h4>
      {rows.length ? (
        <dl>
          {rows.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{formatContractValue(value)}</dd>
            </div>
          ))}
        </dl>
      ) : <p>{empty}</p>}
    </section>
  );
}

function compactRows(rows: Array<[string, unknown]>): Array<[string, unknown]> {
  return rows.filter(([, value]) => value !== undefined && value !== null && value !== ''
    && (!Array.isArray(value) || value.length > 0)
    && (typeof value !== 'object' || Array.isArray(value) || Object.keys(value as object).length > 0));
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function formatContractValue(value: unknown): string {
  if (Array.isArray(value)) return value.map((item) => humanize(String(item))).join(' · ');
  if (value && typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${humanize(key)}: ${humanize(String(item))}`)
      .join(' · ');
  }
  return humanize(String(value));
}

function humanize(value?: string | null): string {
  if (!value) return '';
  return value.replace(/_/g, ' ').replace(/\s+/g, ' ').trim().replace(/^./, (letter: string) => letter.toUpperCase());
}

function formatMemoryTimestamp(value?: string | null) {
  if (!value) return 'Built in';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function EmptyState({ label }: { label: string }) {
  return (
    <div className="database-empty-state">
      <AlertCircle size={16} />
      <span>{label}</span>
    </div>
  );
}
