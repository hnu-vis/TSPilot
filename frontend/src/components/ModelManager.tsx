import {
  Activity,
  Check,
  CheckCircle2,
  ChevronDown,
  CloudCog,
  Cpu,
  KeyRound,
  LoaderCircle,
  MessageSquareCode,
  Orbit,
  Plus,
  Save,
  ScanSearch,
  ServerCog,
  Sparkles,
  Trash2,
  TriangleAlert,
  Timer,
  X,
  Zap,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  activateAIModelConfig,
  activateMachineModel,
  deleteExternalMachineModel,
  deleteAIModelConfig,
  fetchModelsConfig,
  testModelConnection,
  testExternalMachineModel,
  updateAIModelConfig,
  updateExternalMachineModel,
} from '../services/api';
import type { AIModelEndpointConfig, MachineModelConfig, ModelConnectionTest, ModelsConfig } from '../types';
import { useI18n } from '../i18n';
import { NotificationToast } from './NotificationToast';

type Tab = 'ai' | 'machine-learning';
type Section = 'llm' | 'embedding';
type Draft = { id?: string; apiBase: string; model: string; apiKey: string };
type MachineTask = 'forecast' | 'anomaly';
type MachineDraft = { name: string; endpoint: string; apiKey: string; timeoutSeconds: string; isNew: boolean };
type Feedback = { tone: 'success' | 'error'; message: string } | null;

const EMPTY_DRAFT: Draft = { apiBase: '', model: '', apiKey: '' };

export function ModelManager() {
  const { t } = useI18n();
  const [tab, setTab] = useState<Tab>('ai');
  const [config, setConfig] = useState<ModelsConfig | null>(null);
  const [expanded, setExpanded] = useState<Partial<Record<Section, string | 'new'>>>({});
  const [drafts, setDrafts] = useState<Partial<Record<Section, Draft>>>({});
  const [machineExpanded, setMachineExpanded] = useState<Partial<Record<MachineTask, string | 'new'>>>({});
  const [machineDrafts, setMachineDrafts] = useState<Partial<Record<MachineTask, MachineDraft>>>({});
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Feedback>(null);
  const [tests, setTests] = useState<Record<string, ModelConnectionTest>>({});
  const [confirmRemoveId, setConfirmRemoveId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchModelsConfig()
      .then((next) => {
        if (!cancelled) applyConfig(next, setConfig);
      })
      .catch((error) => {
        if (!cancelled) setFeedback({ tone: 'error', message: error instanceof Error ? error.message : t('Unable to load model configuration.') });
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  const configuredCount = useMemo(() => (
    (config?.ai.llm.models.length || 0)
    + (config?.ai.embedding.models.length || 0)
    + (config?.machine_learning.forecast_models.length || 0)
    + (config?.machine_learning.anomaly_models.length || 0)
  ), [config]);

  const openConnection = (section: Section, connection: AIModelEndpointConfig) => {
    const isClosing = expanded[section] === connection.id;
    setExpanded((current) => ({ ...current, [section]: isClosing ? undefined : connection.id }));
    setDrafts((current) => ({
      ...current,
      [section]: isClosing ? undefined : {
        id: connection.id,
        apiBase: connection.api_base,
        model: connection.model,
        apiKey: '',
      },
    }));
    setConfirmRemoveId(null);
  };

  const startNewConnection = (section: Section) => {
    setExpanded((current) => ({ ...current, [section]: 'new' }));
    setDrafts((current) => ({ ...current, [section]: EMPTY_DRAFT }));
    setConfirmRemoveId(null);
  };

  const closeConnection = (section: Section) => {
    setExpanded((current) => ({ ...current, [section]: undefined }));
    setDrafts((current) => ({ ...current, [section]: undefined }));
    setConfirmRemoveId(null);
  };

  const saveAI = async (section: Section) => {
    const draft = drafts[section];
    if (!draft) return;
    const busyKey = `save-${section}-${draft.id || 'new'}`;
    setBusy(busyKey);
    setFeedback(null);
    try {
      const next = await updateAIModelConfig(section, {
        ...(draft.id ? { id: draft.id } : {}),
        api_base: draft.apiBase,
        model: draft.model,
        ...(draft.apiKey.trim() ? { api_key: draft.apiKey.trim() } : {}),
      });
      applyConfig(next, setConfig);
      const savedId = next.saved_id || draft.id;
      if (savedId) {
        const saved = next.ai[section].models.find((item) => item.id === savedId);
        setExpanded((current) => ({ ...current, [section]: savedId }));
        if (saved) setDrafts((current) => ({ ...current, [section]: toDraft(saved) }));
      }
      setFeedback({ tone: 'success', message: t('{model} saved to the {kind} model library.', { model: draft.model, kind: t(section === 'llm' ? 'language' : 'embedding') }) });
    } catch (error) {
      setFeedback({ tone: 'error', message: error instanceof Error ? error.message : t('Unable to save model connection.') });
    } finally {
      setBusy(null);
    }
  };

  const testAI = async (section: Section) => {
    const draft = drafts[section];
    if (!draft) return;
    const testKey = `${section}:${draft.id || 'new'}`;
    setBusy(`test-${testKey}`);
    setFeedback(null);
    try {
      const result = await testModelConnection({
        kind: section,
        ...(draft.id ? { connection_id: draft.id } : {}),
        api_base: draft.apiBase,
        model: draft.model,
        ...(draft.apiKey.trim() ? { api_key: draft.apiKey.trim() } : {}),
      });
      setTests((current) => ({ ...current, [testKey]: result }));
    } catch (error) {
      setFeedback({ tone: 'error', message: error instanceof Error ? error.message : t('Unable to test model connection.') });
    } finally {
      setBusy(null);
    }
  };

  const activateAI = async (section: Section, connection: AIModelEndpointConfig) => {
    setBusy(`activate-${connection.id}`);
    setFeedback(null);
    try {
      const next = await activateAIModelConfig(section, connection.id);
      applyConfig(next, setConfig);
      setFeedback({ tone: 'success', message: t('{model} is now the active {kind} model.', { model: connection.model, kind: t(section === 'llm' ? 'language' : 'embedding') }) });
    } catch (error) {
      setFeedback({ tone: 'error', message: error instanceof Error ? error.message : t('Unable to activate model.') });
    } finally {
      setBusy(null);
    }
  };

  const removeAI = async (section: Section, connection: AIModelEndpointConfig) => {
    if (confirmRemoveId !== connection.id) {
      setConfirmRemoveId(connection.id);
      return;
    }
    setBusy(`remove-${connection.id}`);
    setFeedback(null);
    try {
      const next = await deleteAIModelConfig(section, connection.id);
      applyConfig(next, setConfig);
      closeConnection(section);
      setFeedback({ tone: 'success', message: t('{model} was removed from the workspace.', { model: connection.model }) });
    } catch (error) {
      setFeedback({ tone: 'error', message: error instanceof Error ? error.message : t('Unable to remove model.') });
    } finally {
      setBusy(null);
    }
  };

  const openMachineModel = (task: MachineTask, model: MachineModelConfig) => {
    if (model.source === 'built_in') return;
    const closing = machineExpanded[task] === model.id;
    setMachineExpanded((current) => ({ ...current, [task]: closing ? undefined : model.id }));
    setMachineDrafts((current) => ({
      ...current,
      [task]: closing ? undefined : {
        name: model.name,
        endpoint: model.endpoint || '',
        apiKey: '',
        timeoutSeconds: String(model.timeout_seconds || 30),
        isNew: false,
      },
    }));
    setConfirmRemoveId(null);
  };

  const addMachineModel = (task: MachineTask) => {
    setMachineExpanded((current) => ({ ...current, [task]: 'new' }));
    setMachineDrafts((current) => ({ ...current, [task]: { name: '', endpoint: '', apiKey: '', timeoutSeconds: '30', isNew: true } }));
    setConfirmRemoveId(null);
  };

  const closeMachineModel = (task: MachineTask) => {
    setMachineExpanded((current) => ({ ...current, [task]: undefined }));
    setMachineDrafts((current) => ({ ...current, [task]: undefined }));
    setConfirmRemoveId(null);
  };

  const saveExternalMachine = async (task: MachineTask) => {
    const draft = machineDrafts[task];
    if (!draft) return;
    setBusy(`save-machine-${task}`);
    setFeedback(null);
    try {
      const next = await updateExternalMachineModel(task, {
        name: draft.name,
        endpoint: draft.endpoint,
        timeout_seconds: Number(draft.timeoutSeconds),
        ...(draft.apiKey.trim() ? { api_key: draft.apiKey.trim() } : {}),
      });
      applyConfig(next, setConfig);
      const savedId = next.saved_id || draft.name;
      const saved = (task === 'forecast' ? next.machine_learning.forecast_models : next.machine_learning.anomaly_models).find((item) => item.id === savedId);
      if (saved) {
        setMachineExpanded((current) => ({ ...current, [task]: saved.id }));
        setMachineDrafts((current) => ({ ...current, [task]: toMachineDraft(saved) }));
      }
      setFeedback({ tone: 'success', message: t('{model} saved and registered from its model configuration file.', { model: draft.name }) });
    } catch (error) {
      setFeedback({ tone: 'error', message: error instanceof Error ? error.message : t('Unable to save external model.') });
    } finally { setBusy(null); }
  };

  const testExternalMachine = async (task: MachineTask) => {
    const draft = machineDrafts[task];
    if (!draft) return;
    const key = `machine:${task}:${draft.name || 'new'}`;
    setBusy(`test-${key}`);
    try {
      const result = await testExternalMachineModel({
        task, name: draft.name, endpoint: draft.endpoint, timeout_seconds: Number(draft.timeoutSeconds),
        ...(draft.apiKey.trim() ? { api_key: draft.apiKey.trim() } : {}),
      });
      setTests((current) => ({ ...current, [key]: result }));
    } catch (error) {
      setFeedback({ tone: 'error', message: error instanceof Error ? error.message : t('Unable to test external model.') });
    } finally { setBusy(null); }
  };

  const activateMachine = async (task: MachineTask, model: MachineModelConfig) => {
    setBusy(`activate-machine-${task}`);
    try {
      const next = await activateMachineModel(task, model.name);
      applyConfig(next, setConfig);
      setFeedback({ tone: 'success', message: t('{model} is now the active {kind} model.', { model: model.name, kind: t(task) }) });
    } catch (error) {
      setFeedback({ tone: 'error', message: error instanceof Error ? error.message : t('Unable to activate model.') });
    } finally { setBusy(null); }
  };

  const removeExternalMachine = async (task: MachineTask, model: MachineModelConfig) => {
    const key = `${task}:${model.id}`;
    if (confirmRemoveId !== key) { setConfirmRemoveId(key); return; }
    setBusy(`remove-machine-${task}`);
    try {
      const next = await deleteExternalMachineModel(task, model.name);
      applyConfig(next, setConfig);
      closeMachineModel(task);
      setFeedback({ tone: 'success', message: t('{model} and its configuration file were removed.', { model: model.name }) });
    } catch (error) {
      setFeedback({ tone: 'error', message: error instanceof Error ? error.message : t('Unable to remove external model.') });
    } finally { setBusy(null); }
  };

  return (
    <section className="model-manager" aria-label={t('Model configuration')}>
      <header className="model-hero">
        <div className="model-hero-copy">
          <div className="model-hero-icon"><Cpu size={22} /></div>
          <div>
            <span className="model-kicker">{t('MODEL ORCHESTRATION')}</span>
            <h2>{t('Connect intelligence to every workflow')}</h2>
            <p>{t('Build a reusable model library, then choose which connection powers each workflow.')}</p>
          </div>
        </div>
        <div className="model-overview" aria-label={t('Model status summary')}>
          <span><i className={config ? 'online' : ''} /> {t(config ? 'Configuration ready' : 'Waiting for backend')}</span>
          <strong>{configuredCount}<small> {t('configured models')}</small></strong>
        </div>
      </header>

      <div className="model-tabs" role="tablist" aria-label={t('Model configuration sections')}>
        <button type="button" role="tab" aria-selected={tab === 'ai'} className={tab === 'ai' ? 'active' : ''} onClick={() => setTab('ai')}>
          <Sparkles size={16} /><span>{t('AI & Embedding')}</span><small>{t('Connected model library')}</small>
        </button>
        <button type="button" role="tab" aria-selected={tab === 'machine-learning'} className={tab === 'machine-learning' ? 'active' : ''} onClick={() => setTab('machine-learning')}>
          <Activity size={16} /><span>{t('Machine Learning')}</span><small>{t('Forecast and anomaly')}</small>
        </button>
      </div>

      {feedback && (
        <NotificationToast
          tone={feedback.tone}
          title={t(feedback.tone === 'success' ? 'Operation completed' : 'Operation failed')}
          message={feedback.message}
          onDismiss={() => setFeedback(null)}
        />
      )}

      {loading ? (
        <div className="model-loading"><LoaderCircle size={20} className="spin" /> {t('Loading model configuration…')}</div>
      ) : tab === 'ai' ? (
        <div className="model-library" role="tabpanel">
          <ModelCollection
            section="llm"
            title={t('Large language models')}
            description={t('Models available for planning, reasoning, tool use, and final answers.')}
            icon={<MessageSquareCode size={20} />}
            models={config?.ai.llm.models || []}
            expandedId={expanded.llm}
            draft={drafts.llm}
            busy={busy}
            tests={tests}
            confirmRemoveId={confirmRemoveId}
            onOpen={(model) => openConnection('llm', model)}
            onAdd={() => startNewConnection('llm')}
            onClose={() => closeConnection('llm')}
            onDraftChange={(draft) => setDrafts((current) => ({ ...current, llm: draft }))}
            onSave={() => saveAI('llm')}
            onTest={() => testAI('llm')}
            onActivate={(model) => activateAI('llm', model)}
            onRemove={(model) => removeAI('llm', model)}
          />
          <ModelCollection
            section="embedding"
            title={t('Embedding models')}
            description={t('Vector models connected for semantic retrieval and Key Insight Memory learning.')}
            icon={<Orbit size={20} />}
            models={config?.ai.embedding.models || []}
            expandedId={expanded.embedding}
            draft={drafts.embedding}
            busy={busy}
            tests={tests}
            confirmRemoveId={confirmRemoveId}
            onOpen={(model) => openConnection('embedding', model)}
            onAdd={() => startNewConnection('embedding')}
            onClose={() => closeConnection('embedding')}
            onDraftChange={(draft) => setDrafts((current) => ({ ...current, embedding: draft }))}
            onSave={() => saveAI('embedding')}
            onTest={() => testAI('embedding')}
            onActivate={(model) => activateAI('embedding', model)}
            onRemove={(model) => removeAI('embedding', model)}
          />
        </div>
      ) : (
        <div className="model-library machine-model-library" role="tabpanel">
          <MachineModelCollection task="forecast" title={t('Prediction models')} description={t('Built-in and externally deployed models available to the forecast tool.')} icon={<Activity size={20} />} models={config?.machine_learning.forecast_models || []} expandedId={machineExpanded.forecast} draft={machineDrafts.forecast} busy={busy} tests={tests} confirmRemoveId={confirmRemoveId} onOpen={(model) => openMachineModel('forecast', model)} onAdd={() => addMachineModel('forecast')} onClose={() => closeMachineModel('forecast')} onDraftChange={(draft) => setMachineDrafts((current) => ({ ...current, forecast: draft }))} onSave={() => saveExternalMachine('forecast')} onTest={() => testExternalMachine('forecast')} onActivate={(model) => activateMachine('forecast', model)} onRemove={(model) => removeExternalMachine('forecast', model)} />
          <MachineModelCollection task="anomaly" title={t('Anomaly detection models')} description={t('Built-in and external detectors available for time-series anomaly analysis.')} icon={<ScanSearch size={20} />} models={config?.machine_learning.anomaly_models || []} expandedId={machineExpanded.anomaly} draft={machineDrafts.anomaly} busy={busy} tests={tests} confirmRemoveId={confirmRemoveId} onOpen={(model) => openMachineModel('anomaly', model)} onAdd={() => addMachineModel('anomaly')} onClose={() => closeMachineModel('anomaly')} onDraftChange={(draft) => setMachineDrafts((current) => ({ ...current, anomaly: draft }))} onSave={() => saveExternalMachine('anomaly')} onTest={() => testExternalMachine('anomaly')} onActivate={(model) => activateMachine('anomaly', model)} onRemove={(model) => removeExternalMachine('anomaly', model)} />
        </div>
      )}
    </section>
  );
}

function ModelCollection({ section, title, description, icon, models, expandedId, draft, busy, tests, confirmRemoveId, onOpen, onAdd, onClose, onDraftChange, onSave, onTest, onActivate, onRemove }: {
  section: Section; title: string; description: string; icon: ReactNode; models: AIModelEndpointConfig[];
  expandedId?: string; draft?: Draft; busy: string | null; tests: Record<string, ModelConnectionTest>; confirmRemoveId: string | null;
  onOpen: (model: AIModelEndpointConfig) => void; onAdd: () => void; onClose: () => void; onDraftChange: (draft: Draft) => void;
  onSave: () => void; onTest: () => void; onActivate: (model: AIModelEndpointConfig) => void; onRemove: (model: AIModelEndpointConfig) => void;
}) {
  const { t } = useI18n();
  return (
    <section className={`model-collection ${section}`} aria-labelledby={`${section}-models-title`}>
      <header className="model-collection-header">
        <div className={`model-collection-icon ${section}`}>{icon}</div>
        <div><h3 id={`${section}-models-title`}>{title}</h3><p>{description}</p></div>
        <span>{models.length} {t(models.length === 1 ? 'model' : 'models')}</span>
        <button type="button" onClick={onAdd}><Plus size={15} /> {t('Add model')}</button>
      </header>
      <div className="model-connection-grid">
        {models.map((model) => (
          <ConnectionCard
            key={model.id}
            section={section}
            connection={model}
            expanded={expandedId === model.id}
            draft={expandedId === model.id ? draft : undefined}
            busy={busy}
            test={tests[`${section}:${model.id}`]}
            confirmRemove={confirmRemoveId === model.id}
            onToggle={() => onOpen(model)}
            onClose={onClose}
            onDraftChange={onDraftChange}
            onSave={onSave}
            onTest={onTest}
            onActivate={() => onActivate(model)}
            onRemove={() => onRemove(model)}
          />
        ))}
        {expandedId === 'new' && draft && (
          <ConnectionCard section={section} expanded draft={draft} busy={busy} onToggle={onClose} onClose={onClose} onDraftChange={onDraftChange} onSave={onSave} onTest={onTest} test={tests[`${section}:new`]} />
        )}
      </div>
    </section>
  );
}

function ConnectionCard({ section, connection, expanded, draft, busy, test, confirmRemove, onToggle, onClose, onDraftChange, onSave, onTest, onActivate, onRemove }: {
  section: Section; connection?: AIModelEndpointConfig; expanded: boolean; draft?: Draft; busy: string | null; test?: ModelConnectionTest; confirmRemove?: boolean;
  onToggle: () => void; onClose: () => void; onDraftChange: (draft: Draft) => void; onSave: () => void; onTest: () => void;
  onActivate?: () => void; onRemove?: () => void;
}) {
  const { t } = useI18n();
  const testKey = `${section}:${draft?.id || 'new'}`;
  return (
    <article className={`model-connection-card ${expanded ? 'expanded' : ''} ${connection?.is_active ? 'active' : ''}`}>
      <button type="button" className="model-connection-summary" onClick={onToggle} aria-expanded={expanded}>
        <span className={`model-id-mark ${section}`}>{section === 'llm' ? <MessageSquareCode size={17} /> : <Orbit size={18} />}</span>
        <span className="model-id-copy"><strong>{connection?.model || t('New model connection')}</strong>{connection?.is_active && <small><Zap size={10} /> {t('Active')}</small>}</span>
        {connection && <span className={`model-key-dot ${connection.api_key_configured ? 'ready' : ''}`} title={t(connection.api_key_configured ? 'API key configured' : 'API key missing')} />}
        <ChevronDown size={16} className="model-card-chevron" />
      </button>
      {expanded && draft && (
        <div className="model-connection-details">
          <div className="model-connection-fields">
            <label className="model-field model-field-wide">
              <span>{t('Model ID')}</span>
              <input value={draft.model} onChange={(event) => onDraftChange({ ...draft, model: event.target.value })} placeholder={section === 'llm' ? 'e.g. gpt-5.4-mini' : 'e.g. text-embedding-3-small'} />
            </label>
            <label className="model-field model-field-wide">
              <span>{t('API Base URL')} <ServerCog size={13} /></span>
              <input type="url" value={draft.apiBase} onChange={(event) => onDraftChange({ ...draft, apiBase: event.target.value })} placeholder="https://api.example.com/v1" />
            </label>
            <label className="model-field model-field-wide">
              <span>{t('API Key')} <KeyRound size={13} /></span>
              <input type="password" autoComplete="new-password" value={draft.apiKey} onChange={(event) => onDraftChange({ ...draft, apiKey: event.target.value })} placeholder={t(connection?.api_key_configured ? 'Leave blank to keep current key' : 'Enter API key')} />
            </label>
          </div>
          {test && <div className={`model-test-result ${test.success ? 'success' : 'error'}`}>{test.success ? <CheckCircle2 size={15} /> : <TriangleAlert size={15} />}<span>{test.message}</span><small>{test.latency_ms} ms</small></div>}
          <footer className="model-connection-actions">
            <div>
              {connection && !connection.is_active && <button type="button" className="model-quiet-button" disabled={Boolean(busy)} onClick={onActivate}><Check size={14} /> {t('Set active')}</button>}
              {connection?.source === 'workspace' && <button type="button" className={`model-remove-button ${confirmRemove ? 'confirm' : ''}`} disabled={Boolean(busy)} onClick={onRemove}>{confirmRemove ? <TriangleAlert size={14} /> : <Trash2 size={14} />}{t(confirmRemove ? 'Confirm remove' : 'Remove')}</button>}
            </div>
            <div>
              <button type="button" className="model-icon-button" onClick={onClose} aria-label={t('Close model details')}><X size={15} /></button>
              <button type="button" className="model-secondary-button" disabled={Boolean(busy) || !draft.apiBase || !draft.model} onClick={onTest}>{busy === `test-${testKey}` ? <LoaderCircle size={15} className="spin" /> : <Zap size={15} />} {t('Test')}</button>
              <button type="button" className="model-primary-button" disabled={Boolean(busy) || !draft.apiBase || !draft.model} onClick={onSave}>{busy === `save-${section}-${draft.id || 'new'}` ? <LoaderCircle size={15} className="spin" /> : <Save size={15} />} {t('Save')}</button>
            </div>
          </footer>
        </div>
      )}
    </article>
  );
}

function MachineModelCollection({ task, title, description, icon, models, expandedId, draft, busy, tests, confirmRemoveId, onOpen, onAdd, onClose, onDraftChange, onSave, onTest, onActivate, onRemove }: {
  task: MachineTask; title: string; description: string; icon: ReactNode; models: MachineModelConfig[]; expandedId?: string; draft?: MachineDraft;
  busy: string | null; tests: Record<string, ModelConnectionTest>; confirmRemoveId: string | null;
  onOpen: (model: MachineModelConfig) => void; onAdd: () => void; onClose: () => void; onDraftChange: (draft: MachineDraft) => void;
  onSave: () => void; onTest: () => void; onActivate: (model: MachineModelConfig) => void; onRemove: (model: MachineModelConfig) => void;
}) {
  const { t } = useI18n();
  return (
    <section className={`model-collection machine-model-collection ${task}`} aria-labelledby={`${task}-models-title`}>
      <header className="model-collection-header">
        <div className={`model-collection-icon ${task}`}>{icon}</div>
        <div><h3 id={`${task}-models-title`}>{title}</h3><p>{description}</p></div>
        <span>{models.length} {t(models.length === 1 ? 'model' : 'models')}</span>
        <button type="button" onClick={onAdd}><Plus size={15} /> {t('Connect API model')}</button>
      </header>
      <div className="model-connection-grid">
        {models.map((model) => (
          <MachineConnectionCard key={model.id} task={task} model={model} expanded={expandedId === model.id} draft={expandedId === model.id ? draft : undefined} busy={busy} test={tests[`machine:${task}:${model.name}`]} confirmRemove={confirmRemoveId === `${task}:${model.id}`} onToggle={() => onOpen(model)} onClose={onClose} onDraftChange={onDraftChange} onSave={onSave} onTest={onTest} onActivate={() => onActivate(model)} onRemove={() => onRemove(model)} />
        ))}
        {expandedId === 'new' && draft && <MachineConnectionCard task={task} expanded draft={draft} busy={busy} test={tests[`machine:${task}:${draft.name || 'new'}`]} onToggle={onClose} onClose={onClose} onDraftChange={onDraftChange} onSave={onSave} onTest={onTest} />}
      </div>
    </section>
  );
}

function MachineConnectionCard({ task, model, expanded, draft, busy, test, confirmRemove, onToggle, onClose, onDraftChange, onSave, onTest, onActivate, onRemove }: {
  task: MachineTask; model?: MachineModelConfig; expanded: boolean; draft?: MachineDraft; busy: string | null; test?: ModelConnectionTest; confirmRemove?: boolean;
  onToggle: () => void; onClose: () => void; onDraftChange: (draft: MachineDraft) => void; onSave: () => void; onTest: () => void;
  onActivate?: () => void; onRemove?: () => void;
}) {
  const { t } = useI18n();
  const isApi = model?.source === 'api' || !model;
  const content = (
    <>
      <span className={`model-id-mark ${task}`}>{task === 'forecast' ? <Activity size={17} /> : <ScanSearch size={17} />}</span>
      <span className="model-id-copy"><strong>{model?.name || t('New external model')}</strong><small className={isApi ? 'api' : 'builtin'}>{isApi ? <CloudCog size={10} /> : <Cpu size={10} />}{isApi ? 'API' : t('Built-in')}</small>{model?.is_active && <small><Zap size={10} /> {t('Active')}</small>}</span>
    </>
  );
  return (
    <article className={`model-connection-card machine-connection-card ${expanded ? 'expanded' : ''} ${model?.is_active ? 'active' : ''}`}>
      {isApi ? (
        <button type="button" className="model-connection-summary" onClick={onToggle} aria-expanded={expanded}>{content}<span className={`model-key-dot ${model?.api_key_configured ? 'ready' : ''}`} /><ChevronDown size={16} className="model-card-chevron" /></button>
      ) : (
        <div className="model-connection-summary">{content}<span />{!model?.is_active ? <button type="button" className="model-inline-active" disabled={Boolean(busy)} onClick={onActivate}><Check size={13} /> {t('Use')}</button> : <CheckCircle2 size={16} className="model-builtin-check" />}</div>
      )}
      {expanded && draft && (
        <div className="model-connection-details">
          <div className="model-connection-fields machine-fields">
            <label className="model-field"><span>{t('Registry name')}</span><input value={draft.name} disabled={!draft.isNew} onChange={(event) => onDraftChange({ ...draft, name: event.target.value.toLowerCase() })} placeholder={task === 'forecast' ? 'external-forecast-v2' : 'production-detector'} /></label>
            <label className="model-field"><span>{t('Endpoint URL')} <CloudCog size={13} /></span><input type="url" value={draft.endpoint} onChange={(event) => onDraftChange({ ...draft, endpoint: event.target.value })} placeholder="https://models.example.com/predict" /></label>
            <label className="model-field"><span>{t('API Key')} <KeyRound size={13} /></span><input type="password" autoComplete="new-password" value={draft.apiKey} onChange={(event) => onDraftChange({ ...draft, apiKey: event.target.value })} placeholder={t(model?.api_key_configured ? 'Leave blank to keep current key' : 'Optional bearer token')} /></label>
            <label className="model-field"><span>{t('Timeout')} <Timer size={13} /></span><div className="model-timeout-input"><input type="number" min="1" max="300" value={draft.timeoutSeconds} onChange={(event) => onDraftChange({ ...draft, timeoutSeconds: event.target.value })} /><small>{t('seconds')}</small></div></label>
          </div>
          <div className="model-contract-note"><ServerCog size={14} /><span>Uses the TSPilot {task} JSON contract. The endpoint receives a normalized series and task parameters.</span>{model?.config_path && <code>{model.config_path}</code>}</div>
          {test && <div className={`model-test-result ${test.success ? 'success' : 'error'}`}>{test.success ? <CheckCircle2 size={15} /> : <TriangleAlert size={15} />}<span>{test.message}</span><small>{test.latency_ms} ms</small></div>}
          <footer className="model-connection-actions">
            <div>{model && !model.is_active && <button type="button" className="model-quiet-button" disabled={Boolean(busy)} onClick={onActivate}><Check size={14} /> {t('Set active')}</button>}{model && <button type="button" className={`model-remove-button ${confirmRemove ? 'confirm' : ''}`} disabled={Boolean(busy)} onClick={onRemove}>{confirmRemove ? <TriangleAlert size={14} /> : <Trash2 size={14} />}{t(confirmRemove ? 'Confirm remove' : 'Remove')}</button>}</div>
            <div><button type="button" className="model-icon-button" onClick={onClose} aria-label={t('Close external model details')}><X size={15} /></button><button type="button" className="model-secondary-button" disabled={Boolean(busy) || !draft.name || !draft.endpoint} onClick={onTest}>{busy === `test-machine:${task}:${draft.name || 'new'}` ? <LoaderCircle size={15} className="spin" /> : <Zap size={15} />} {t('Test')}</button><button type="button" className="model-primary-button" disabled={Boolean(busy) || !draft.name || !draft.endpoint} onClick={onSave}>{busy === `save-machine-${task}` ? <LoaderCircle size={15} className="spin" /> : <Save size={15} />} {t('Save')}</button></div>
          </footer>
        </div>
      )}
    </article>
  );
}

function applyConfig(next: ModelsConfig, setConfig: (value: ModelsConfig) => void) {
  setConfig(next);
}

function toDraft(config: AIModelEndpointConfig): Draft {
  return { id: config.id, apiBase: config.api_base, model: config.model, apiKey: '' };
}

function toMachineDraft(config: MachineModelConfig): MachineDraft {
  return { name: config.name, endpoint: config.endpoint || '', apiKey: '', timeoutSeconds: String(config.timeout_seconds || 30), isNew: false };
}
