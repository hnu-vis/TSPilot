import {
  AlertCircle,
  BookOpen,
  BrainCircuit,
  Braces,
  ChevronLeft,
  ChevronRight,
  Database,
  HardDrive,
  Clock3,
  LoaderCircle,
  RefreshCw,
  Search,
  Wrench,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchInsightMemory,
  fetchInsightMemoryDetail,
  fetchInsightMemoryLearningSettings,
  updateInsightMemoryLearningSettings,
} from '../services/api';
import type { DatabaseResource, InsightMemoryLearningSettings, InsightMemoryResponse, MemoryCard, MemoryDetail } from '../types';
import { useI18n } from '../i18n';
import { NotificationToast, type NotificationNotice } from './NotificationToast';

type Props = {
  databases: DatabaseResource[];
  selectedDatabaseId: string | null;
};

type MemoryState = {
  loading: boolean;
  error: string | null;
  data: InsightMemoryResponse | null;
};

type DurationUnit = 'seconds' | 'minutes' | 'hours';

export function InsightMemoryManager({ databases, selectedDatabaseId }: Props) {
  const { t, locale } = useI18n();
  const orderedDatabases = useMemo(
    () => orderInsightMemoryDatabases(databases, selectedDatabaseId),
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
  const [learningSettings, setLearningSettings] = useState<InsightMemoryLearningSettings | null>(null);
  const [durationAmount, setDurationAmount] = useState('');
  const [durationUnit, setDurationUnit] = useState<DurationUnit>('minutes');
  const [scheduleState, setScheduleState] = useState<{ loading: boolean; saving: boolean }>({
    loading: true,
    saving: false,
  });
  const [notice, setNotice] = useState<NotificationNotice | null>(null);
  const databaseId = scope === 'database' ? memoryDatabaseId : null;
  const activeDatabase = databases.find((database) => database.id === memoryDatabaseId) || null;

  useEffect(() => {
    let cancelled = false;
    fetchInsightMemoryLearningSettings()
      .then(({ settings }) => {
        if (cancelled) return;
        const duration = secondsToDurationInput(settings.max_wait_seconds);
        setLearningSettings(settings);
        setDurationAmount(String(duration.amount));
        setDurationUnit(duration.unit);
        setScheduleState({ loading: false, saving: false });
      })
      .catch((error) => {
        if (!cancelled) {
          setScheduleState({ loading: false, saving: false });
          setNotice({
            tone: 'error',
            title: t('Schedule unavailable'),
            message: error instanceof Error ? error.message : t('Unable to load automatic learning settings.'),
          });
        }
      });
    return () => { cancelled = true; };
  }, []);

  const saveLearningSchedule = async () => {
    const seconds = durationInputToSeconds(durationAmount, durationUnit);
    if (seconds === null) {
      setNotice({ tone: 'error', title: t('Check the duration'), message: t('Enter a duration greater than zero.') });
      return;
    }
    setScheduleState((current) => ({ ...current, saving: true }));
    try {
      const { settings } = await updateInsightMemoryLearningSettings(seconds);
      const duration = secondsToDurationInput(settings.max_wait_seconds);
      setLearningSettings(settings);
      setDurationAmount(String(duration.amount));
      setDurationUnit(duration.unit);
      setScheduleState({ loading: false, saving: false });
      setNotice({ tone: 'success', title: t('Schedule updated'), message: t('Schedule saved and active.') });
    } catch (error) {
      setScheduleState({ loading: false, saving: false });
      setNotice({
        tone: 'error',
        title: t('Schedule update failed'),
        message: error instanceof Error ? error.message : t('Unable to save automatic learning settings.'),
      });
    }
  };

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
    fetchInsightMemory(databaseId)
      .then((data) => {
        if (!cancelled) setState({ loading: false, error: null, data });
      })
      .catch((error) => {
        if (!cancelled) {
          setState({
            loading: false,
            error: error instanceof Error ? error.message : t('Unable to load key insight memory.'),
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
        || (kindFilter === 'definition' && card.kind === 'insight_definition')
        || (kindFilter === 'recipe' && card.kind === 'insight_recipe');
      if (!kindMatches) return false;
      if (!normalizedQuery) return true;
      return [card.title, card.description, card.kind, ...(card.tags || [])]
        .join(' ')
        .toLowerCase()
        .includes(normalizedQuery);
    });
    if (scope !== 'database' || kindFilter === 'definition') return filtered;
    return filtered.map((card, index) => ({ card, index })).sort((left, right) => {
      const leftPriority = left.card.kind === 'insight_recipe' ? 0 : 1;
      const rightPriority = right.card.kind === 'insight_recipe' ? 0 : 1;
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
    fetchInsightMemoryDetail(selectedCardId, databaseId)
      .then((payload) => {
        if (!cancelled) setDetailState({ loading: false, error: null, detail: payload.detail });
      })
      .catch((error) => {
        if (!cancelled) {
          setDetailState({
            loading: false,
            error: error instanceof Error ? error.message : t('Unable to load memory detail.'),
            detail: null,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedCardId, databaseId]);

  return (
    <section className="insight-memory-manager" aria-label={t('Key Insight Memory')}>
      <div className="insight-memory-toolbar">
        <div className="insight-memory-scope">
          <div className="segmented-control" aria-label={t('Insight memory scope')}>
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
              <span>{t('Database memory')}</span>
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
              <span>{t('System defaults')}</span>
            </button>
          </div>
          <div className="insight-memory-scope-context">
            {scope === 'database' ? (
              <label className="insight-memory-database-select">
                <Database size={13} />
                <select
                  value={memoryDatabaseId || ''}
                  onChange={(event) => setMemoryDatabaseId(event.target.value || null)}
                  aria-label={t('Select Key Insight Memory database')}
                >
                  {orderedDatabases.map((database) => (
                    <option key={database.id} value={database.id}>
                      {database.display_name || database.name || database.id} · {formatPlaybookCount(database, t)}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <span className="insight-memory-scope-label">{t('Built-in contracts')}</span>
            )}
          </div>
        </div>
        <div className="insight-memory-tools">
          <label className="insight-memory-search">
            <Search size={14} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t('Search memory')} aria-label={t('Search key insight memory')} />
          </label>
          <div className="segmented-control insight-memory-kind-filter" aria-label={t('Memory card type')}>
            <button type="button" className={kindFilter === 'all' ? 'active' : ''} onClick={() => setKindFilter('all')}>{t('All')}</button>
            <button type="button" className={kindFilter === 'definition' ? 'active' : ''} onClick={() => setKindFilter('definition')}>
              <BookOpen size={13} /><span>{t('Definitions')}</span>
            </button>
            <button type="button" className={kindFilter === 'recipe' ? 'active' : ''} onClick={() => setKindFilter('recipe')}>
              <Wrench size={13} /><span>{t('Playbooks')}</span>
            </button>
          </div>
          <button
            className="database-source-add"
            type="button"
            title={t('Refresh memory')}
            aria-label={t('Refresh memory')}
            disabled={state.loading}
            onClick={() => setReloadToken((value) => value + 1)}
          >
            <RefreshCw size={14} className={state.loading ? 'spin' : ''} />
          </button>
        </div>
      </div>

      <form
        className="insight-learning-schedule"
        aria-label={t('Automatic Key Insight learning schedule')}
        onSubmit={(event) => {
          event.preventDefault();
          void saveLearningSchedule();
        }}
      >
        <div className="insight-learning-schedule-copy">
          <Clock3 size={16} />
          <div>
            <strong>{t('Automatic learning')}</strong>
            <span>
              {learningSettings?.enabled === false
                ? t('Currently disabled by the server. This schedule will apply when automatic learning is enabled.')
                : t('Verified Key Insights wait up to this duration before consolidation. A full batch of {count} can run sooner.', { count: learningSettings?.batch_size ?? t('configured') })}
            </span>
          </div>
        </div>
        <div className="insight-learning-schedule-controls">
          <label>
            <span>{t('Maximum wait')}</span>
            <input
              type="number"
              min="0.01"
              step="any"
              value={durationAmount}
              disabled={scheduleState.loading || scheduleState.saving}
              onChange={(event) => {
                setDurationAmount(event.target.value);
              }}
              aria-label={t('Automatic learning maximum wait')}
            />
          </label>
          <select
            value={durationUnit}
            disabled={scheduleState.loading || scheduleState.saving}
            onChange={(event) => {
              setDurationUnit(event.target.value as DurationUnit);
            }}
            aria-label={t('Automatic learning wait unit')}
          >
            <option value="seconds">{t('seconds')}</option>
            <option value="minutes">{t('minutes')}</option>
            <option value="hours">{t('hours')}</option>
          </select>
          <button type="submit" disabled={scheduleState.loading || scheduleState.saving}>
            {scheduleState.saving && <LoaderCircle size={14} className="spin" />}
            {t(scheduleState.saving ? 'Saving…' : 'Save')}
          </button>
        </div>
      </form>

      {scope === 'database' && !activeDatabase && (
        <div className="database-inline-error">
          <AlertCircle size={16} />
          <span>{t('Select a database in chat or Database view to inspect scoped key insight memory.')}</span>
        </div>
      )}
      {state.error && (
        <div className="database-inline-error">
          <AlertCircle size={16} />
          <span>{state.error}</span>
        </div>
      )}

      <div className="insight-memory-summary" aria-label={t('Memory summary')}>
        <span><strong>{visibleCards.length.toLocaleString(locale)}</strong> {t('visible')} <small>{t('of')} {cards.length.toLocaleString(locale)}</small></span>
        <span><strong>{String(summaryRecord.definition_count ?? cards.filter((card) => card.kind === 'insight_definition').length)}</strong> {t('definitions')}</span>
        <span><strong>{String(summaryRecord.recipe_count ?? cards.filter((card) => card.kind === 'insight_recipe').length)}</strong> {t('playbooks')}</span>
        <span>{t('Updated')} <strong>{formatMemoryTimestamp(state.data?.memory.updated_at, locale, t)}</strong></span>
      </div>

      <section className={`insight-memory-library ${state.loading ? 'is-loading' : ''}`} aria-labelledby="memory-library-title" aria-busy={state.loading}>
        <div className="insight-memory-section-heading">
          <div>
            <BrainCircuit size={15} />
            <h2 id="memory-library-title">{t('Playbook Library')}</h2>
          </div>
          {pageCount > 1 && (
            <div className="insight-memory-pagination" aria-label={t('Memory pages')}>
              <button type="button" aria-label={t('Previous memory page')} disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>
                <ChevronLeft size={14} />
              </button>
              <span>{page + 1} / {pageCount}</span>
              <button type="button" aria-label={t('Next memory page')} disabled={page >= pageCount - 1} onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))}>
                <ChevronRight size={14} />
              </button>
            </div>
          )}
        </div>
        {pageCards.length > 0 ? (
          <div className="insight-memory-tile-grid">
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
                ? t('No memory cards match the current filter.')
                : scope === 'database'
                  ? t('This database has not learned any Key Insight definitions or playbooks yet.')
                  : t('No system-default Key Insight Memory is available.')
            } />
          )}
        {state.loading && (
          <div className="insight-memory-loading-overlay" role="status">
            <LoaderCircle size={18} className="spin" />
            <span>{t('Updating memory…')}</span>
          </div>
        )}
      </section>

      <section
        className={`insight-memory-detail ${detailState.loading ? 'is-loading' : ''}`}
        aria-labelledby="selected-memory-title"
        aria-busy={detailState.loading}
      >
        <div className="insight-memory-section-heading">
          <div>
            <Braces size={15} />
            <h2 id="selected-memory-title">{t('Selected Memory')}</h2>
          </div>
        </div>
          {detailState.error ? (
            <EmptyState label={detailState.error} />
          ) : detailState.detail ? (
            <MemoryDetailCard detail={detailState.detail} />
          ) : detailState.loading ? (
            <EmptyState label={t('Loading memory detail.')} />
          ) : (
            <EmptyState label={t('Select a memory card to load its detail.')} />
          )}
          <div className="insight-memory-storage">
            <span><HardDrive size={13} /> {t('Storage')}</span>
            <code>{state.data?.memory.storage_path || t('Default in-memory definitions')}</code>
          </div>
      </section>
      {notice && <NotificationToast {...notice} onDismiss={() => setNotice(null)} />}
    </section>
  );
}

export function orderInsightMemoryDatabases(databases: DatabaseResource[], selectedDatabaseId: string | null) {
  return databases.map((database, index) => ({ database, index })).sort((left, right) => {
    const leftSelected = left.database.id === selectedDatabaseId;
    const rightSelected = right.database.id === selectedDatabaseId;
    if (leftSelected !== rightSelected) return leftSelected ? -1 : 1;

    const recipeDifference = (right.database.insight_memory_summary?.recipe_count || 0)
      - (left.database.insight_memory_summary?.recipe_count || 0);
    if (recipeDifference) return recipeDifference;

    const updatedDifference = String(right.database.insight_memory_summary?.updated_at || '')
      .localeCompare(String(left.database.insight_memory_summary?.updated_at || ''));
    return updatedDifference || left.index - right.index;
  }).map(({ database }) => database);
}

export function secondsToDurationInput(seconds: number): { amount: number; unit: DurationUnit } {
  if (seconds >= 3600 && seconds % 3600 === 0) return { amount: seconds / 3600, unit: 'hours' };
  if (seconds >= 60 && seconds % 60 === 0) return { amount: seconds / 60, unit: 'minutes' };
  return { amount: seconds, unit: 'seconds' };
}

export function durationInputToSeconds(amount: string, unit: DurationUnit): number | null {
  const parsed = Number(amount);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  const multiplier = unit === 'hours' ? 3600 : unit === 'minutes' ? 60 : 1;
  const seconds = parsed * multiplier;
  return seconds <= 7 * 24 * 60 * 60 ? seconds : null;
}

function formatPlaybookCount(database: DatabaseResource, t: (key: string) => string) {
  const count = database.insight_memory_summary?.recipe_count || 0;
  return `${count} ${t(count === 1 ? 'playbook' : 'playbooks')}`;
}

function MemoryCardItem({ card, selected, onSelect }: { card: MemoryCard; selected: boolean; onSelect: () => void }) {
  const { t } = useI18n();
  const isPlaybook = card.kind === 'insight_recipe';
  const source = card.updated_at ? 'Learned' : 'System';
  const tool = (card.tags || []).find((tag) => ['sql_query', 'code_interpreter'].includes(tag));
  const insightType = (card.tags || []).find((tag) => tag !== tool && !['global', 'system'].includes(tag.toLowerCase()));
  return (
    <button
      type="button"
      className={`insight-memory-tile ${selected ? 'selected' : ''}`}
      onClick={onSelect}
      aria-pressed={selected}
      title={`${card.title}\n${card.description}`}
    >
      <div className="insight-memory-tile-kind">
        {isPlaybook ? <Wrench size={11} /> : <BookOpen size={11} />}
        <span>{t(isPlaybook ? 'Playbook' : 'Definition')}</span>
        <i className={source === 'Learned' ? 'learned' : ''} aria-label={t(source)} title={t(source)} />
      </div>
      <strong>{humanize(card.title)}</strong>
      <p>{card.description}</p>
      <small>{[humanize(tool), humanize(insightType || source)].filter(Boolean).join(' · ')}</small>
    </button>
  );
}

export function MemoryDetailCard({ detail }: { detail: MemoryDetail }) {
  const { t } = useI18n();
  const request = detail.insight_request || {};
  const requirements = asRecord(request.requirements);
  const tags = detail.card.tags || [];
  const isPlaybook = detail.card.kind === 'insight_recipe';
  const source = detail.card.updated_at ? 'Learned' : 'System default';
  const rawContract = compactMemoryContract(detail.insight_request);
  const contractRows = compactRows([
    [t('Key Insight type'), request.insight_type || tags[0]],
    [t('Subject'), request.subject],
    [t('Result shape'), request.result_shape],
    [t('Expected items'), request.expected_item_count],
    [t('Scope'), tags[Math.max(tags.length - 1, 0)]],
  ]);
  const generationRows = compactRows([
    [t('Tool'), detail.preferred_tool],
    [t('Method'), detail.calculation_trace?.method],
    [t('Required evidence'), isPlaybook ? undefined : tags.slice(1, -1)],
    [t('Time position'), requirements.time_position],
    [t('Operator'), requirements.operator],
    [t('Dimensions'), request.dimensions],
    [t('Dependencies'), request.derived_from],
  ]);
  const verification = (detail.guidance || '').split(';').map((item) => item.trim()).filter(Boolean);
  return (
    <article className="insight-memory-detail-card">
      <header className="insight-memory-detail-header">
        <div>
          <span>{t(detail.card.kind === 'insight_recipe' ? 'Key Insight Playbook' : 'Key Insight definition')}</span>
          <h3>{humanize(detail.card.title)}</h3>
        </div>
        <span className={`insight-memory-source ${source === 'Learned' ? 'learned' : ''}`}>{t(source)}</span>
      </header>
      <p className="insight-memory-detail-description">{detail.card.description}</p>

      <div className="insight-memory-detail-grid">
        <DetailBlock title={t('Key Insight contract')} rows={contractRows} empty={t('This definition does not create a Key Insight request directly.')} />
        <DetailBlock title={t('Playbook method')} rows={generationRows} empty={t('Generation behavior is defined by the runtime contract.')} />
      </div>

      <div className="insight-memory-verification">
        <h4>{t('Verification')}</h4>
        {verification.length ? (
          <ul>{verification.map((item) => <li key={item}>{item}</li>)}</ul>
        ) : (
          <p>{detail.guidance || t('Use current request evidence and the runtime Key Insight contract for verification.')}</p>
        )}
      </div>

      <details className="insight-memory-raw-contract">
        <summary>{t('Technical details and raw contract')}</summary>
        <dl>
          <div><dt>{t('Memory ID')}</dt><dd><code>{detail.id}</code></dd></div>
          <div><dt>{t('Tags')}</dt><dd>{(detail.card.tags || []).join(' · ') || t('None')}</dd></div>
        </dl>
        {rawContract && <pre className="debug-json">{JSON.stringify(rawContract, null, 2)}</pre>}
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
    <section className="insight-memory-detail-block">
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

export function compactMemoryContract(value: unknown): unknown {
  if (value === undefined || value === null || value === '') return undefined;
  if (Array.isArray(value)) {
    const items = value
      .map((item) => compactMemoryContract(item))
      .filter((item) => item !== undefined);
    return items.length ? items : undefined;
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => [key, compactMemoryContract(item)] as const)
      .filter(([, item]) => item !== undefined);
    return entries.length ? Object.fromEntries(entries) : undefined;
  }
  return value;
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

function formatMemoryTimestamp(value: string | null | undefined, locale: string, t: (key: string) => string) {
  if (!value) return t('Built in');
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(locale, {
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
