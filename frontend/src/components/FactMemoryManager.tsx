import { AlertCircle, BrainCircuit, Database, FileText, RefreshCw } from 'lucide-react';
import { useEffect, useState } from 'react';
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
      <div className="fact-memory-header">
        <div>
          <span className="database-kicker">Long-term memory</span>
          <h2>Fact Memory</h2>
          <p>Memory cards summarize reusable fact-generation intent. Concrete numeric facts are regenerated from current evidence.</p>
        </div>
        <div className="fact-memory-actions">
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
          <button className="icon-text-button" type="button" disabled={state.loading} onClick={() => setReloadToken((value) => value + 1)}>
            <RefreshCw size={14} />
            <span>{state.loading ? 'Loading' : 'Refresh'}</span>
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
          <strong>{cards.length.toLocaleString()}</strong>
          <span>cards</span>
        </div>
        <div>
          <strong>{scope === 'database' ? activeDatabase?.display_name || activeDatabase?.name || 'Database' : 'Global'}</strong>
          <span>scope</span>
        </div>
        <div>
          <strong>{state.data?.memory.updated_at || 'default'}</strong>
          <span>updated</span>
        </div>
      </div>

      <div className="fact-memory-grid">
        <section className="database-section">
          <SectionTitle icon={BrainCircuit} title="Memory cards" count={cards.length} />
          {cards.length > 0 ? (
            <div className="fact-definition-list">
              {cards.map((card) => (
                <MemoryCardItem
                  key={card.id}
                  card={card}
                  selected={selectedCardId === card.id}
                  onSelect={() => setSelectedCardId(card.id)}
                />
              ))}
            </div>
          ) : (
            <EmptyState label="No memory cards stored." />
          )}
        </section>

        <section className="database-section">
          <SectionTitle icon={FileText} title="Selected detail" count={detailState.detail ? 1 : 0} />
          {detailState.loading ? (
            <EmptyState label="Loading memory detail." />
          ) : detailState.error ? (
            <EmptyState label={detailState.error} />
          ) : detailState.detail ? (
            <MemoryDetailCard detail={detailState.detail} />
          ) : (
            <EmptyState label="Select a memory card to load its detail." />
          )}
        </section>
      </div>

      <section className="database-section fact-memory-source-section">
        <SectionTitle icon={FileText} title="Storage" count={state.data?.memory.storage_path ? 1 : 0} />
        <div className="chip-list compact">
          {(state.data?.prompt_view?.summary && typeof state.data.prompt_view.summary === 'object')
            ? Object.entries(state.data.prompt_view.summary as Record<string, unknown>).slice(0, 6).map(([key, value]) => (
              <span key={key}>{key}: {String(value)}</span>
            ))
            : null}
        </div>
        {state.data?.memory.storage_path && <p className="sample-note">{state.data.memory.storage_path}</p>}
      </section>
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
        <strong>{card.title}</strong>
        <span>{card.kind}</span>
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
        <strong>{detail.card.title}</strong>
        <span>{detail.card.kind}</span>
      </div>
      <p>{detail.guidance || detail.card.description}</p>
      {detail.fact_request && <pre className="debug-json">{JSON.stringify(detail.fact_request, null, 2)}</pre>}
      {detail.examples && detail.examples.length > 0 && (
        <div className="chip-list compact">
          {detail.examples.slice(0, 4).map((item) => <span key={item}>{item}</span>)}
        </div>
      )}
    </article>
  );
}

function EmptyState({ label }: { label: string }) {
  return (
    <div className="database-empty-state">
      <AlertCircle size={16} />
      <span>{label}</span>
    </div>
  );
}
