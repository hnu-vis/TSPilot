import { Languages, Menu } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ChatThread } from './components/ChatThread';
import { Composer } from './components/Composer';
import { DatabaseManager } from './components/DatabaseManager';
import { InsightMemoryManager } from './components/InsightMemoryManager';
import { HistorySidebar, type WorkspaceView } from './components/HistorySidebar';
import { InspectorPanel } from './components/InspectorPanel';
import { ModelManager } from './components/ModelManager';
import {
  extractFinalAnswer,
  fetchDatabases,
  fetchKnowledge,
  fetchModelsConfig,
  streamChat,
} from './services/api';
import {
  appendAssistantAnswer,
  appendAssistantPending,
  appendUserMessage,
  buildBackendHistory,
  completeRunningTraceSteps,
  createConversation,
  failRunningTraceSteps,
  loadConversations,
  saveConversations,
  settleIncompleteStream,
  sortConversations,
  upsertTraceSpan,
  upsertTraceStep,
} from './store/conversations';
import {
  isTraceSpanEvent,
  traceSpanFromStreamEvent,
  traceStepFromPolicyDecision,
} from './lib/traceEvents';
import type { Conversation, ResourceState, StreamEvent, TraceStep } from './types';
import { useI18n } from './i18n';

const now = () => new Date().toISOString();

export default function App() {
  const { locale, setLocale, t } = useI18n();
  const [conversations, setConversations] = useState<Conversation[]>(() => (
    sortConversations(dedupeEmptyNewConversations(loadConversations()))
  ));
  const [activeId, setActiveId] = useState(() => conversations[0]?.id || createConversation().id);
  const [historyQuery, setHistoryQuery] = useState('');
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isHistoryCollapsed, setIsHistoryCollapsed] = useState(loadHistoryCollapsed);
  const [activeView, setActiveView] = useState<WorkspaceView>('chat');
  const [isInspectorCollapsed, setIsInspectorCollapsed] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [resources, setResources] = useState<ResourceState>({ databases: [], knowledge: [], model: 'backend model', models: [] });
  const [, setResourceError] = useState<string | null>(null);
  const activeAbortRef = useRef<AbortController | null>(null);
  const conversationsRef = useRef(conversations);

  const activeConversation = conversations.find((item) => item.id === activeId) || conversations[0];
  const latestAnswer = activeConversation?.messages
    .slice()
    .reverse()
    .find((message) => message.role === 'assistant' && message.answer)?.answer || null;
  const hasConversationContent = Boolean(
    activeView === 'chat'
    && activeConversation
    && (
      activeConversation.messages.length > 0
      || activeConversation.traceSteps.length > 0
    ),
  );
  const isDatabaseView = activeView === 'database';
  const isInsightMemoryView = activeView === 'insight-memory';
  const isModelView = activeView === 'model';
  const isInspectorVisible = activeView === 'chat' && hasConversationContent;

  const chatHasConversationContent = Boolean(
    activeConversation && (
      activeConversation.messages.length > 0
      || activeConversation.traceSteps.length > 0
    ),
  );

  useEffect(() => {
    conversationsRef.current = conversations;
    const saveTimer = window.setTimeout(() => saveConversations(conversations), 350);
    return () => window.clearTimeout(saveTimer);
  }, [conversations]);

  useEffect(() => {
    const flushConversations = () => saveConversations(conversationsRef.current);
    window.addEventListener('pagehide', flushConversations);
    return () => window.removeEventListener('pagehide', flushConversations);
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem('tspilot.history-collapsed', String(isHistoryCollapsed));
    } catch (error) {
      console.warn('Unable to save history sidebar state.', error);
    }
  }, [isHistoryCollapsed]);

  const loadResources = () => Promise.all([fetchDatabases(), fetchKnowledge(), fetchModelsConfig()]);

  const refreshResources = async () => {
    const [databases, knowledge, modelsConfig] = await loadResources();
    const activeModel = modelsConfig.ai.llm.models.find((item) => item.is_active) || modelsConfig.ai.llm.models[0];
    setResources({ databases, knowledge, model: activeModel?.model || 'backend model', models: modelsConfig.ai.llm.models });
    setResourceError(null);
    return databases;
  };

  useEffect(() => {
    let cancelled = false;
    loadResources()
      .then(([databases, knowledge, modelsConfig]) => {
        if (!cancelled) {
          const activeModel = modelsConfig.ai.llm.models.find((item) => item.is_active) || modelsConfig.ai.llm.models[0];
          setResources({ databases, knowledge, model: activeModel?.model || 'backend model', models: modelsConfig.ai.llm.models });
          setResourceError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) setResourceError(error instanceof Error ? error.message : t('Unable to load resources.'));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (activeView !== 'chat') return;
    let cancelled = false;
    fetchModelsConfig().then((modelsConfig) => {
      if (cancelled) return;
      const activeModel = modelsConfig.ai.llm.models.find((item) => item.is_active) || modelsConfig.ai.llm.models[0];
      setResources((current) => ({ ...current, model: activeModel?.model || 'backend model', models: modelsConfig.ai.llm.models }));
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [activeView]);

  const sortedConversations = useMemo(() => sortConversations(conversations), [conversations]);

  const updateConversation = (id: string, updater: (conversation: Conversation) => Conversation) => {
    setConversations((current) => sortConversations(current.map((conversation) => (
      conversation.id === id ? updater(conversation) : conversation
    ))));
  };

  const handleNewConversation = () => {
    const existingEmpty = conversations.find((conversation) => isEmptyNewConversation(conversation));
    if (existingEmpty) {
      setActiveId(existingEmpty.id);
      setActiveView('chat');
      setIsHistoryOpen(false);
      return;
    }
    const next = createConversation();
    setConversations((current) => sortConversations([next, ...current]));
    setActiveId(next.id);
    setActiveView('chat');
    setIsHistoryOpen(false);
  };

  const handleDeleteConversation = (id: string) => {
    setConversations((current) => {
      const next = current.filter((conversation) => conversation.id !== id);
      if (id === activeId) {
        const replacement = next[0] || createConversation();
        if (next.length === 0) {
          setActiveId(replacement.id);
          return [replacement];
        }
        setActiveId(replacement.id);
      }
      return sortConversations(next);
    });
  };

  const handleSelectTraceNode = (id: string) => {
    if (!activeConversation) return;
    setIsInspectorCollapsed(false);
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      selectedTraceStepId: id,
    }));
  };

  const handleResourceChange = (field: 'selectedDatabaseId' | 'selectedKnowledgeId' | 'selectedModelId', value: string | null) => {
    if (!activeConversation) return;
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      [field]: value,
      updatedAt: now(),
    }));
  };

  const handleSend = async (message: string) => {
    if (!activeConversation || isStreaming) return;
    const conversationId = activeConversation.id;
    const abortController = new AbortController();
    activeAbortRef.current = abortController;
    const withUserMessage = appendUserMessage(activeConversation, message);
    const withAssistantPending = appendAssistantPending(withUserMessage);
    setConversations((current) => sortConversations(current.map((conversation) => (
      conversation.id === conversationId ? { ...withAssistantPending, traceSteps: [], selectedTraceStepId: null } : conversation
    ))));
    setIsInspectorCollapsed(false);
    setIsStreaming(true);

    const selectedDatabase = resources.databases.find((database) => database.id === withUserMessage.selectedDatabaseId) || null;
    try {
      await streamChat(
        {
          message,
          conversationId,
          database: selectedDatabase,
          history: buildBackendHistory(withUserMessage),
          modelId: withUserMessage.selectedModelId || resources.models.find((item) => item.is_active)?.id || null,
        },
        (event) => handleStreamEvent(conversationId, event),
        abortController.signal,
      );
      updateConversation(conversationId, (conversation) => settleIncompleteStream(
        conversation,
        t('The response stream ended before a final answer.'),
      ));
    } catch (error) {
      if (isAbortError(error)) {
        updateConversation(conversationId, (conversation) => settleIncompleteStream(
          failRunningTraceSteps(conversation, t('Stopped by user.')),
          t('Stopped by user.'),
        ));
        return;
      }
      updateConversation(conversationId, (conversation) => settleIncompleteStream(
        failRunningTraceSteps(
          conversation,
          error instanceof Error ? error.message : t('The chat request failed.'),
        ),
        error instanceof Error ? error.message : t('The chat request failed.'),
      ));
    } finally {
      if (activeAbortRef.current === abortController) {
        activeAbortRef.current = null;
      }
      setIsStreaming(false);
    }
  };

  const handleStop = () => {
    activeAbortRef.current?.abort();
  };

  const handleStreamEvent = (conversationId: string, event: StreamEvent) => {
    if (isTraceSpanEvent(event)) {
      const span = traceSpanFromStreamEvent(event);
      if (span) {
        updateConversation(conversationId, (conversation) => upsertTraceSpan(conversation, span));
      }
      return;
    }

    if (event.event === 'step.start') {
      const iteration = numberFrom(event.data.step ?? event.data.iteration, 0);
      const id = stringFrom(event.data.id, `iteration-${iteration || Date.now()}`);
      const timestamp = now();
      const title = stringFrom(event.data.title, 'task');
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'reasoning',
        status: 'running',
        summary: stringFrom(event.data.detail, title),
        startedAt: stringFrom(event.data.started_at) || timestamp,
        elapsedSeconds: optionalNumberFrom(event.data.elapsed_seconds),
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'step.chunk') {
      if (event.data.output_type !== 'thought') return;
      const iteration = numberFrom(event.data.step ?? event.data.iteration, 0);
      const id = stringFrom(event.data.id, `iteration-${iteration || Date.now()}`);
      const content = stringFrom(event.data.content, '');
      if (!content) return;
      const timestamp = now();
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'reasoning',
        status: 'running',
        summary: content,
        thought: content,
        startedAt: stringFrom(event.data.started_at) || undefined,
        elapsedSeconds: optionalNumberFrom(event.data.elapsed_seconds),
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'step.meta') {
      const iteration = numberFrom(event.data.step ?? event.data.iteration, 0);
      const id = stringFrom(event.data.id, `iteration-${iteration || Date.now()}`);
      const action = stringFrom(event.data.action, '');
      const thought = stringFrom(event.data.thought, '');
      const timestamp = now();
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'tool_call',
        status: 'running',
        summary: thought || action || t('Processing step.'),
        tool: action || undefined,
        thought,
        actionInput: asRecord(event.data.action_input) || undefined,
        startedAt: stringFrom(event.data.started_at) || timestamp,
        elapsedSeconds: optionalNumberFrom(event.data.elapsed_seconds),
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'step.done') {
      const iteration = numberFrom(event.data.step ?? event.data.iteration, 0);
      const id = stringFrom(event.data.id, `iteration-${iteration || Date.now()}`);
      const success = stringFrom(event.data.status, 'done') !== 'failed';
      const observation = asRecord(event.data.observation);
      const timestamp = now();
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'tool_result',
        status: success ? 'complete' : 'error',
        summary: observation ? stringFrom(observation.summary, t(success ? 'Step completed.' : 'Step failed.')) : t(success ? 'Step completed.' : 'Step failed.'),
        tool: observation ? stringFrom(observation.tool_name, '') || stringFrom(observation.tool, '') : undefined,
        observation: observation || undefined,
        startedAt: stringFrom(event.data.started_at) || undefined,
        completedAt: stringFrom(event.data.completed_at) || timestamp,
        elapsedSeconds: optionalNumberFrom(event.data.elapsed_seconds),
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'thought') {
      const iteration = numberFrom(event.data.iteration, 0);
      const id = stringFrom(event.data.id, `iteration-${iteration || Date.now()}:decision`);
      const timestamp = now();
      const step: TraceStep = {
        id,
        iteration,
        agent: stringFrom(event.data.agent, 'data_agent'),
        phase: stringFrom(event.data.phase, 'reasoning'),
        status: statusFrom(event.data.status, 'running'),
        summary: stringFrom(event.data.message, t('Thinking about the next action.')),
        thought: stringFrom(event.data.thought, ''),
        startedAt: stringFrom(event.data.started_at) || undefined,
        elapsedSeconds: optionalNumberFrom(event.data.elapsed_seconds),
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'agent_step') {
      const iteration = numberFrom(event.data.iteration, 0);
      const phase = stringFrom(event.data.phase, 'intent');
      const id = stringFrom(
        event.data.id,
        phase === 'reasoning'
          ? `iteration-${iteration || Date.now()}:decision`
          : `iteration-${iteration || Date.now()}`,
      );
      const timestamp = now();
      const step: TraceStep = {
        id,
        iteration,
        agent: stringFrom(event.data.agent, 'data_agent'),
        phase,
        status: statusFrom(event.data.status, 'running'),
        summary: stringFrom(event.data.message, t('Processing request.')),
        tool: stringFrom(event.data.tool, '') || undefined,
        startedAt: stringFrom(event.data.started_at) || timestamp,
        elapsedSeconds: optionalNumberFrom(event.data.elapsed_seconds),
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'tool_call') {
      const iteration = numberFrom(event.data.iteration, 0);
      const id = `iteration-${iteration || Date.now()}`;
      const timestamp = now();
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'tool_call',
        status: 'running',
        summary: stringFrom(event.data.summary, t('Calling tool.')),
        tool: stringFrom(event.data.tool, 'tool'),
        thought: stringFrom(event.data.thought, ''),
        toolCall: event.data,
        actionInput: asRecord(event.data.action_input) || asRecord(event.data.input_preview) || undefined,
        startedAt: stringFrom(event.data.started_at) || timestamp,
        elapsedSeconds: optionalNumberFrom(event.data.elapsed_seconds),
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'tool_result') {
      const iteration = numberFrom(event.data.iteration, 0);
      const id = `iteration-${iteration || Date.now()}`;
      const success = Boolean(event.data.success);
      const timestamp = now();
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'tool_result',
        status: success ? 'complete' : 'error',
        summary: stringFrom(event.data.summary, t(success ? 'Tool completed.' : 'Tool failed.')),
        tool: stringFrom(event.data.tool, 'tool'),
        toolResult: event.data,
        observation: asRecord(event.data.observation) || event.data,
        startedAt: stringFrom(event.data.started_at) || undefined,
        completedAt: stringFrom(event.data.completed_at) || timestamp,
        elapsedSeconds: optionalNumberFrom(event.data.elapsed_seconds),
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'todo_updated') {
      const iteration = numberFrom(event.data.iteration ?? event.data.current_step, 0);
      const id = `todo-${iteration || Date.now()}`;
      const timestamp = now();
      const progress = asRecord(event.data.todo_progress);
      const completed = numberFrom(event.data.completed_count ?? progress?.completed, 0);
      const total = numberFrom(event.data.todo_total ?? progress?.total, 0);
      const step: TraceStep = {
        id,
        iteration,
        agent: 'runtime',
        phase: 'intent',
        status: 'complete',
        summary: total > 0 ? `${t('Todo list')} ${completed}/${total}` : t('Todo progress updated.'),
        tool: 'todowrite',
        toolResult: {
          tool: 'todowrite',
          success: true,
          summary: t('Todo progress updated.'),
          payload_preview: event.data,
        },
        completedAt: timestamp,
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'policy_decision') {
      const step = traceStepFromPolicyDecision(event);
      if (step) {
        updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      }
      return;
    }

    if (event.event === 'agent_decision_timeout' || event.event === 'runtime_deadline_exceeded') {
      const iteration = numberFrom(event.data.iteration, 0);
      const id = `${event.event}-${iteration || Date.now()}`;
      const timestamp = now();
      const message = stringFrom(event.data.message, event.event === 'agent_decision_timeout'
        ? t('Agent decision timed out.')
        : t('Request deadline exceeded.'));
      const step: TraceStep = {
        id,
        iteration,
        agent: 'runtime',
        phase: 'reasoning',
        status: 'error',
        summary: message,
        tool: event.event,
        toolResult: {
          tool: event.event,
          success: false,
          summary: message,
          payload_preview: event.data,
        },
        observation: event.data,
        completedAt: timestamp,
        updatedAt: timestamp,
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'final_answer') {
      const answer = extractFinalAnswer(event.data);
      if (!answer) return;
      updateConversation(conversationId, (conversation) => appendAssistantAnswer(
        completeRunningTraceSteps(conversation),
        answer,
        asRecord(event.data.token_usage),
      ));
      return;
    }

    if (event.event === 'terminate') {
      updateConversation(conversationId, (conversation) => completeRunningTraceSteps(conversation));
      return;
    }

    if (event.event === 'error') {
      const message = stringFrom(event.data.message, t('The backend returned an error.'));
      updateConversation(conversationId, (conversation) => settleIncompleteStream(
        failRunningTraceSteps(conversation, message),
        message,
      ));
    }
  };

  if (!activeConversation) {
    return null;
  }

  return (
    <div className={`app-shell ${hasConversationContent ? 'inspector-open' : ''} ${isInspectorCollapsed ? 'inspector-collapsed' : ''} ${isHistoryOpen ? 'history-open' : ''} ${isHistoryCollapsed ? 'history-collapsed' : ''}`}>
      <HistorySidebar
        conversations={sortedConversations}
        activeId={activeConversation.id}
        activeView={activeView}
        query={historyQuery}
        collapsed={isHistoryCollapsed}
        onQueryChange={setHistoryQuery}
        onToggleCollapsed={() => setIsHistoryCollapsed((collapsed) => !collapsed)}
        onNew={handleNewConversation}
        onViewChange={(view) => {
          setActiveView(view);
          setIsHistoryOpen(false);
        }}
        onSelect={(id) => {
          setActiveId(id);
          setActiveView('chat');
          setIsHistoryOpen(false);
        }}
        onRename={(id, title) => updateConversation(id, (conversation) => ({ ...conversation, title, updatedAt: now() }))}
        onDelete={handleDeleteConversation}
      />

      <main className={`workspace ${chatHasConversationContent ? '' : 'empty-workspace'} ${isDatabaseView || isInsightMemoryView || isModelView ? 'database-workspace' : ''}`}>
        <header className={`topbar ${activeView === 'chat' ? 'chat-topbar' : ''}`}>
          <button
            type="button"
            className="mobile-history-button"
            aria-label={t('Open chat history')}
            onClick={() => setIsHistoryOpen((open) => !open)}
          >
            <Menu size={17} />
          </button>
          <div className="topbar-title">
            <h1>{isDatabaseView ? t('Database') : isInsightMemoryView ? t('Key Insight Memory') : isModelView ? t('Model') : displayConversationTitle(activeConversation.title, t)}</h1>
            {activeView !== 'chat' && (
              <p>
                {isDatabaseView
                  ? t('Manage available data sources and inspect schema before analysis.')
                  : isInsightMemoryView
                    ? t('Manage reusable Key Insight definitions, playbooks, and verification guidance.')
                    : t('Configure model endpoints and select the engines used for time-series intelligence.')}
              </p>
            )}
          </div>
          <button
            type="button"
            className="topbar-language-button"
            aria-label={t(locale === 'zh-CN' ? 'Switch interface to English' : 'Switch interface to Chinese')}
            title={t('Interface language')}
            onClick={() => setLocale(locale === 'zh-CN' ? 'en' : 'zh-CN')}
          >
            <Languages size={16} aria-hidden="true" />
            <span>{locale === 'zh-CN' ? '中文' : 'EN'}</span>
          </button>
        </header>

        {isDatabaseView ? (
          <DatabaseManager
            databases={resources.databases}
            selectedDatabaseId={activeConversation.selectedDatabaseId}
            onSelectDatabase={(id) => handleResourceChange('selectedDatabaseId', id)}
            onDatabasesChange={refreshResources}
          />
        ) : isInsightMemoryView ? (
          <InsightMemoryManager
            databases={resources.databases}
            selectedDatabaseId={activeConversation.selectedDatabaseId}
          />
        ) : isModelView ? (
          <ModelManager />
        ) : (
          <section className={`thread-area ${hasConversationContent ? '' : 'empty-thread-area'}`}>
            <ChatThread
              messages={activeConversation.messages}
              traceSteps={activeConversation.traceSteps}
              selectedTraceNodeId={activeConversation.selectedTraceStepId}
              onSelectTraceNode={handleSelectTraceNode}
            />
            {!hasConversationContent && (
              <div className="empty-composer-slot">
                <Composer
                  disabled={isStreaming}
                  running={isStreaming}
                  databases={resources.databases}
                  knowledge={resources.knowledge}
                  selectedDatabaseId={activeConversation.selectedDatabaseId}
                  selectedKnowledgeId={activeConversation.selectedKnowledgeId}
                  model={resources.model}
                  models={resources.models}
                  selectedModelId={activeConversation.selectedModelId}
                  onSelectDatabase={(id) => handleResourceChange('selectedDatabaseId', id)}
                  onSelectKnowledge={(id) => handleResourceChange('selectedKnowledgeId', id)}
                  onSelectModel={(id) => handleResourceChange('selectedModelId', id)}
                  onSubmit={handleSend}
                  onStop={handleStop}
                />
              </div>
            )}
          </section>
        )}

        {activeView === 'chat' && hasConversationContent && (
          <Composer
            disabled={isStreaming}
            running={isStreaming}
            databases={resources.databases}
            knowledge={resources.knowledge}
            selectedDatabaseId={activeConversation.selectedDatabaseId}
            selectedKnowledgeId={activeConversation.selectedKnowledgeId}
            model={resources.model}
            models={resources.models}
            selectedModelId={activeConversation.selectedModelId}
            onSelectDatabase={(id) => handleResourceChange('selectedDatabaseId', id)}
            onSelectKnowledge={(id) => handleResourceChange('selectedKnowledgeId', id)}
            onSelectModel={(id) => handleResourceChange('selectedModelId', id)}
            onSubmit={handleSend}
            onStop={handleStop}
          />
        )}
      </main>

      {isInspectorVisible && (
        <InspectorPanel
          steps={activeConversation.traceSteps}
          selectedNodeId={activeConversation.selectedTraceStepId}
          answer={latestAnswer}
          collapsed={isInspectorCollapsed}
          onToggleCollapsed={() => setIsInspectorCollapsed((collapsed) => !collapsed)}
        />
      )}
    </div>
  );
}

function stringFrom(value: unknown, fallback = '') {
  return typeof value === 'string' && value.trim() ? value : fallback;
}

function numberFrom(value: unknown, fallback: number) {
  return typeof value === 'number' ? value : fallback;
}

function optionalNumberFrom(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function statusFrom(value: unknown, fallback: TraceStep['status']): TraceStep['status'] {
  if (value === 'complete' || value === 'error' || value === 'running') return value;
  return fallback;
}

function isEmptyNewConversation(conversation: Conversation) {
  return conversation.title === 'New chat'
    && conversation.messages.length === 0
    && conversation.traceSteps.length === 0;
}

function dedupeEmptyNewConversations(conversations: Conversation[]) {
  let hasEmptyNew = false;
  return conversations.filter((conversation) => {
    if (!isEmptyNewConversation(conversation)) return true;
    if (hasEmptyNew) return false;
    hasEmptyNew = true;
    return true;
  });
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? value as Record<string, unknown> : null;
}

function loadHistoryCollapsed() {
  try {
    return localStorage.getItem('tspilot.history-collapsed') === 'true';
  } catch {
    return false;
  }
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === 'AbortError';
}

function displayConversationTitle(title: string, t: (key: string) => string) {
  return title === 'New chat' ? t('New chat') : title;
}
