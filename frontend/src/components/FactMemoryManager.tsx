import {
  AlertCircle,
  BookOpen,
  BrainCircuit,
  Braces,
  Database,
  FileText,
  HardDrive,
  RefreshCw,
  Search,
  Wrench,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
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
  const [scope, setScope] = useState<'global' | 'database'>('global');
  const [query, setQuery] = useState('');
  const [kindFilter, setKindFilter] = useState<'all' | 'definition' | 'recipe'>('all');
  const [reloadToken, setReloadToken] = useState(0);
  const [state, setState] = useState<MemoryState>({ loading: false, error: null, data: null });
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null);
  const [detailState, setDetailState] = useState<{ loading: boolean; error: string | null; detail: MemoryDetail | null }>({
    loading: false,
    error: null,
    detail: null,
  });
  const databaseId = scope === 'database' ? selectedDatabaseId : null;
  const activeDatabase = databases.find((database) => database.id === selectedDatabaseId) || null;

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
    return cards.filter((card) => {
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
  }, [cards, kindFilter, query]);
  const summary = state.data?.prompt_view?.summary;
  const summaryRecord = summary && typeof summary === 'object'
    ? summary as Record<string, unknown>
    : {};

  useEffect(() => {
    if (cards.length === 0) {
      setSelectedCardId(null);
      return;
    }
    if (!selectedCardId || !cards.some((card) => card.id === selectedCardId)) {
      setSelectedCardId(cards[0].id);
    }
  }, [cards, selectedCardId]);

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
            <button type="button" className={scope === 'global' ? 'active' : ''} onClick={() => setScope('global')}>
              <BrainCircuit size={14} />
              <span>Global</span>
            </button>
            <button
              type="button"
              className={scope === 'database' ? 'active' : ''}
              disabled={!selectedDatabaseId}
              onClick={() => setScope('database')}
            >
              <Database size={14} />
              <span>Database</span>
            </button>
          </div>
          <span className="fact-memory-scope-label">
            {scope === 'database' ? activeDatabase?.display_name || activeDatabase?.name || 'No database' : 'Shared definitions and recipes'}
          </span>
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

      <div className="fact-memory-meta">
        <div>
          <span>Visible</span>
          <strong>{visibleCards.length.toLocaleString()} <small>of {cards.length.toLocaleString()}</small></strong>
        </div>
        <div>
          <span>Definitions</span>
          <strong>{String(summaryRecord.definition_count ?? cards.filter((card) => card.kind === 'fact_definition').length)}</strong>
        </div>
        <div>
          <span>Recipes</span>
          <strong>{String(summaryRecord.recipe_count ?? cards.filter((card) => card.kind === 'fact_recipe').length)}</strong>
        </div>
        <div>
          <span>Updated</span>
          <strong>{formatMemoryTimestamp(state.data?.memory.updated_at)}</strong>
        </div>
      </div>

      <div className="fact-memory-grid">
        <section className="database-section fact-memory-index">
          <SectionTitle icon={BrainCircuit} title="Memory index" count={visibleCards.length} />
          {visibleCards.length > 0 ? (
            <div className="fact-definition-list">
              {visibleCards.map((card) => (
                <MemoryCardItem
                  key={card.id}
                  card={card}
                  selected={selectedCardId === card.id}
                  onSelect={() => setSelectedCardId(card.id)}
                />
              ))}
            </div>
          ) : (
            <EmptyState label={cards.length ? 'No memory cards match the current filter.' : 'No memory cards stored.'} />
          )}
        </section>

        <section className="database-section fact-memory-detail">
          <SectionTitle icon={FileText} title="Contract detail" count={detailState.detail ? 1 : 0} />
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
      </div>
    </section>
  );
}

function SectionTitle({ icon: Icon, title, count }: { icon: typeof BrainCircuit; title: string; count: number }) {
  return (
    <div className="database-section-title">
      <span><Icon size={15} /> {title}</span>
      <strong>{count}</strong>
    </div>
  );
}

function MemoryCardItem({ card, selected, onSelect }: { card: MemoryCard; selected: boolean; onSelect: () => void }) {
  return (
    <button type="button" className={`fact-memory-card ${selected ? 'selected' : ''}`} onClick={onSelect}>
      <div className="fact-memory-card-title">
        <span className="fact-memory-card-icon">{card.kind === 'fact_recipe' ? <Wrench size={14} /> : <BookOpen size={14} />}</span>
        <strong>{card.title}</strong>
        <span>{card.kind === 'fact_recipe' ? 'Recipe' : 'Definition'}</span>
      </div>
      <p>{card.description}</p>
      <div className="chip-list compact">
        {(card.tags || []).slice(0, 6).map((item) => <span key={item}>{item}</span>)}
      </div>
      {card.updated_at && <small>{card.updated_at}</small>}
    </button>
  );
}

function MemoryDetailCard({ detail }: { detail: MemoryDetail }) {
  return (
    <article className="fact-memory-card">
      <div className="fact-memory-card-title">
        <span className="fact-memory-card-icon"><Braces size={14} /></span>
        <strong>{detail.card.title}</strong>
        <span>{detail.card.kind === 'fact_recipe' ? 'Recipe' : 'Definition'}</span>
      </div>
      <p>{detail.guidance || detail.card.description}</p>
      <dl className="fact-memory-contract-meta">
        <div><dt>Memory ID</dt><dd><code>{detail.id}</code></dd></div>
        <div><dt>Preferred tool</dt><dd>{detail.preferred_tool || 'Defined by runtime contract'}</dd></div>
        <div><dt>Verification tags</dt><dd>{(detail.card.tags || []).join(' · ') || 'None'}</dd></div>
      </dl>
      {detail.fact_request && <pre className="debug-json">{JSON.stringify(detail.fact_request, null, 2)}</pre>}
      {detail.examples && detail.examples.length > 0 && (
        <div className="chip-list compact">
          {detail.examples.slice(0, 4).map((item) => <span key={item}>{item}</span>)}
        </div>
      )}
    </article>
  );
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
