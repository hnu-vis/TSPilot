import { Menu } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ChatThread } from './components/ChatThread';
import { Composer } from './components/Composer';
import { DatabaseManager } from './components/DatabaseManager';
import { HistorySidebar, type WorkspaceView } from './components/HistorySidebar';
import { InspectorPanel } from './components/InspectorPanel';
import {
  extractFinalAnswer,
  fetchDatabases,
  fetchKnowledge,
  fetchModel,
  streamChat,
} from './services/api';
import {
  appendAssistantAnswer,
  appendAssistantError,
  appendAssistantPending,
  appendUserMessage,
  buildBackendHistory,
  createConversation,
  loadConversations,
  saveConversations,
  sortConversations,
  upsertTraceStep,
} from './store/conversations';
import type { Conversation, ResourceState, StreamEvent, TraceStep } from './types';

const now = () => new Date().toISOString();

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(() => (
    sortConversations(dedupeEmptyNewConversations(loadConversations()))
  ));
  const [activeId, setActiveId] = useState(() => conversations[0]?.id || createConversation().id);
  const [historyQuery, setHistoryQuery] = useState('');
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [activeView, setActiveView] = useState<WorkspaceView>('chat');
  const [isInspectorCollapsed, setIsInspectorCollapsed] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [resources, setResources] = useState<ResourceState>({ databases: [], knowledge: [], model: 'backend model' });
  const [, setResourceError] = useState<string | null>(null);
  const activeAbortRef = useRef<AbortController | null>(null);

  const activeConversation = conversations.find((item) => item.id === activeId) || conversations[0];
  const selectedTraceStep = activeConversation?.selectedTraceStepId
    ? activeConversation.traceSteps.find((step) => step.id === activeConversation.selectedTraceStepId) || null
    : null;
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
  const isInspectorVisible = activeView === 'chat' && hasConversationContent;

  const chatHasConversationContent = Boolean(
    activeConversation && (
      activeConversation.messages.length > 0
      || activeConversation.traceSteps.length > 0
    ),
  );

  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  const loadResources = () => Promise.all([fetchDatabases(), fetchKnowledge(), fetchModel()]);

  const refreshResources = async () => {
    const [databases, knowledge, model] = await loadResources();
    setResources({ databases, knowledge, model });
    setResourceError(null);
    return databases;
  };

  useEffect(() => {
    let cancelled = false;
    loadResources()
      .then(([databases, knowledge, model]) => {
        if (!cancelled) {
          setResources({ databases, knowledge, model });
          setResourceError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) setResourceError(error instanceof Error ? error.message : 'Unable to load resources.');
      });
    return () => {
      cancelled = true;
    };
  }, []);

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

  const handleSelectTraceStep = (id: string) => {
    if (!activeConversation) return;
    setIsInspectorCollapsed(false);
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      selectedTraceStepId: id,
    }));
  };

  const handleResourceChange = (field: 'selectedDatabaseId' | 'selectedKnowledgeId', value: string | null) => {
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
        },
        (event) => handleStreamEvent(conversationId, event),
        abortController.signal,
      );
    } catch (error) {
      if (isAbortError(error)) {
        updateConversation(conversationId, (conversation) => appendAssistantError(
          markLatestRunningStepErrored(conversation, 'Stopped by user.'),
          '已停止。',
        ));
        return;
      }
      updateConversation(conversationId, (conversation) => appendAssistantError(
        conversation,
        error instanceof Error ? error.message : 'The chat request failed.',
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
    if (event.event === 'step.start') {
      const iteration = numberFrom(event.data.step ?? event.data.iteration, 0);
      const id = stringFrom(event.data.id, `iteration-${iteration || Date.now()}`);
      const title = stringFrom(event.data.title, 'task');
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'tool_call',
        status: 'running',
        summary: stringFrom(event.data.detail, title),
        tool: title,
        updatedAt: now(),
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
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'reasoning',
        status: 'running',
        summary: content,
        thought: content,
        updatedAt: now(),
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'step.meta') {
      const iteration = numberFrom(event.data.step ?? event.data.iteration, 0);
      const id = stringFrom(event.data.id, `iteration-${iteration || Date.now()}`);
      const action = stringFrom(event.data.action, '');
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'tool_call',
        status: 'running',
        summary: action || 'Processing step.',
        tool: action || undefined,
        thought: stringFrom(event.data.thought, ''),
        actionInput: asRecord(event.data.action_input) || undefined,
        toolCall: event.data,
        updatedAt: now(),
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'step.done') {
      const iteration = numberFrom(event.data.step ?? event.data.iteration, 0);
      const id = stringFrom(event.data.id, `iteration-${iteration || Date.now()}`);
      const success = stringFrom(event.data.status, 'done') !== 'failed';
      const observation = asRecord(event.data.observation);
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'tool_result',
        status: success ? 'complete' : 'error',
        summary: observation ? stringFrom(observation.summary, success ? 'Step completed.' : 'Step failed.') : success ? 'Step completed.' : 'Step failed.',
        tool: observation ? stringFrom(observation.tool_name, '') : undefined,
        observation: observation || undefined,
        updatedAt: now(),
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'thought') {
      const iteration = numberFrom(event.data.iteration, 0);
      const id = `iteration-${iteration || Date.now()}`;
      const step: TraceStep = {
        id,
        iteration,
        agent: stringFrom(event.data.agent, 'data_agent'),
        phase: stringFrom(event.data.phase, 'reasoning'),
        status: statusFrom(event.data.status, 'running'),
        summary: stringFrom(event.data.message, 'Thinking about the next action.'),
        thought: stringFrom(event.data.thought, ''),
        updatedAt: now(),
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'agent_step') {
      const iteration = numberFrom(event.data.iteration, 0);
      const id = `iteration-${iteration || Date.now()}`;
      const step: TraceStep = {
        id,
        iteration,
        agent: stringFrom(event.data.agent, 'data_agent'),
        phase: stringFrom(event.data.phase, 'intent'),
        status: statusFrom(event.data.status, 'running'),
        summary: stringFrom(event.data.message, 'Processing request.'),
        updatedAt: now(),
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'tool_call') {
      const iteration = numberFrom(event.data.iteration, 0);
      const id = `iteration-${iteration || Date.now()}`;
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'tool_call',
        status: 'running',
        summary: stringFrom(event.data.summary, 'Calling tool.'),
        tool: stringFrom(event.data.tool, 'tool'),
        thought: stringFrom(event.data.thought, ''),
        toolCall: event.data,
        actionInput: asRecord(event.data.action_input) || asRecord(event.data.input_preview) || undefined,
        updatedAt: now(),
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'tool_result') {
      const iteration = numberFrom(event.data.iteration, 0);
      const id = `iteration-${iteration || Date.now()}`;
      const success = Boolean(event.data.success);
      const step: TraceStep = {
        id,
        iteration,
        agent: 'data_agent',
        phase: 'tool_result',
        status: success ? 'complete' : 'error',
        summary: stringFrom(event.data.summary, success ? 'Tool completed.' : 'Tool failed.'),
        tool: stringFrom(event.data.tool, 'tool'),
        toolResult: event.data,
        observation: asRecord(event.data.observation) || event.data,
        updatedAt: now(),
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'final_answer') {
      const answer = extractFinalAnswer(event.data);
      if (!answer) return;
      updateConversation(conversationId, (conversation) => appendAssistantAnswer(
        conversation,
        answer,
        asRecord(event.data.token_usage),
      ));
      return;
    }

    if (event.event === 'error') {
      const message = stringFrom(event.data.message, 'The backend returned an error.');
      updateConversation(conversationId, (conversation) => appendAssistantError(
        markLatestRunningStepErrored(conversation, message),
        message,
      ));
    }
  };

  if (!activeConversation) {
    return null;
  }

  return (
    <div className={`app-shell ${hasConversationContent ? 'inspector-open' : ''} ${isInspectorCollapsed ? 'inspector-collapsed' : ''} ${isHistoryOpen ? 'history-open' : ''}`}>
      <HistorySidebar
        conversations={sortedConversations}
        activeId={activeConversation.id}
        activeView={activeView}
        query={historyQuery}
        onQueryChange={setHistoryQuery}
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

      <main className={`workspace ${chatHasConversationContent ? '' : 'empty-workspace'} ${isDatabaseView ? 'database-workspace' : ''}`}>
        <header className="topbar">
          <button
            type="button"
            className="mobile-history-button"
            aria-label="Open chat history"
            onClick={() => setIsHistoryOpen((open) => !open)}
          >
            <Menu size={17} />
          </button>
          <div className="topbar-title">
            <h1>{isDatabaseView ? 'Database' : activeConversation.title}</h1>
            <p>{isDatabaseView ? 'Manage available data sources and inspect schema before analysis.' : 'Ask, inspect the agent process, and continue from previous context.'}</p>
          </div>
        </header>

        {isDatabaseView ? (
          <DatabaseManager
            databases={resources.databases}
            selectedDatabaseId={activeConversation.selectedDatabaseId}
            onSelectDatabase={(id) => handleResourceChange('selectedDatabaseId', id)}
            onDatabasesChange={refreshResources}
          />
        ) : (
          <section className={`thread-area ${hasConversationContent ? '' : 'empty-thread-area'}`}>
            <ChatThread
              messages={activeConversation.messages}
              traceSteps={activeConversation.traceSteps}
              selectedTraceStepId={activeConversation.selectedTraceStepId}
              onSelectTraceStep={handleSelectTraceStep}
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
                  onSelectDatabase={(id) => handleResourceChange('selectedDatabaseId', id)}
                  onSelectKnowledge={(id) => handleResourceChange('selectedKnowledgeId', id)}
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
            onSelectDatabase={(id) => handleResourceChange('selectedDatabaseId', id)}
            onSelectKnowledge={(id) => handleResourceChange('selectedKnowledgeId', id)}
            onSubmit={handleSend}
            onStop={handleStop}
          />
        )}
      </main>

      {isInspectorVisible && (
        <InspectorPanel
          steps={activeConversation.traceSteps}
          selectedStepId={selectedTraceStep?.id || null}
          answer={latestAnswer}
          collapsed={isInspectorCollapsed}
          onToggleCollapsed={() => setIsInspectorCollapsed((collapsed) => !collapsed)}
        />
      )}
    </div>
  );
}

function stringFrom(value: unknown, fallback: string) {
  return typeof value === 'string' && value.trim() ? value : fallback;
}

function numberFrom(value: unknown, fallback: number) {
  return typeof value === 'number' ? value : fallback;
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

function markLatestRunningStepErrored(conversation: Conversation, message: string): Conversation {
  for (let index = conversation.traceSteps.length - 1; index >= 0; index -= 1) {
    const step = conversation.traceSteps[index];
    if (step.status !== 'running') continue;
    const traceSteps = [...conversation.traceSteps];
    traceSteps[index] = {
      ...step,
      status: 'error',
      summary: message,
      error: message,
      updatedAt: now(),
    };
    return {
      ...conversation,
      traceSteps,
      selectedTraceStepId: conversation.selectedTraceStepId || step.id,
    };
  }
  return conversation;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? value as Record<string, unknown> : null;
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === 'AbortError';
}
