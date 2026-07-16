import { Menu } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { ChatThread } from './components/ChatThread';
import { Composer } from './components/Composer';
import { HistorySidebar } from './components/HistorySidebar';
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
  const [conversations, setConversations] = useState<Conversation[]>(() => sortConversations(loadConversations()));
  const [activeId, setActiveId] = useState(() => conversations[0]?.id || createConversation().id);
  const [historyQuery, setHistoryQuery] = useState('');
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isInspectorCollapsed, setIsInspectorCollapsed] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [resources, setResources] = useState<ResourceState>({ databases: [], knowledge: [], model: 'backend model' });
  const [, setResourceError] = useState<string | null>(null);

  const activeConversation = conversations.find((item) => item.id === activeId) || conversations[0];
  const selectedTraceStep = activeConversation?.selectedTraceStepId
    ? activeConversation.traceSteps.find((step) => step.id === activeConversation.selectedTraceStepId) || null
    : null;
  const latestAnswer = activeConversation?.messages
    .slice()
    .reverse()
    .find((message) => message.role === 'assistant' && message.answer)?.answer || null;
  const hasConversationContent = Boolean(
    activeConversation && (
      activeConversation.messages.length > 0
      || activeConversation.traceSteps.length > 0
    ),
  );

  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchDatabases(), fetchKnowledge(), fetchModel()])
      .then(([databases, knowledge, model]) => {
        if (!cancelled) setResources({ databases, knowledge, model });
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
    const next = createConversation();
    setConversations((current) => sortConversations([next, ...current]));
    setActiveId(next.id);
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
      );
    } catch (error) {
      updateConversation(conversationId, (conversation) => appendAssistantError(
        conversation,
        error instanceof Error ? error.message : 'The chat request failed.',
      ));
    } finally {
      setIsStreaming(false);
    }
  };

  const handleStreamEvent = (conversationId: string, event: StreamEvent) => {
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
        toolCall: event.data,
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
        updatedAt: now(),
      };
      updateConversation(conversationId, (conversation) => upsertTraceStep(conversation, step));
      return;
    }

    if (event.event === 'final_answer') {
      const answer = extractFinalAnswer(event.data);
      if (!answer) return;
      updateConversation(conversationId, (conversation) => appendAssistantAnswer(conversation, answer));
      return;
    }

    if (event.event === 'error') {
      updateConversation(conversationId, (conversation) => appendAssistantError(
        conversation,
        stringFrom(event.data.message, 'The backend returned an error.'),
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
        query={historyQuery}
        onQueryChange={setHistoryQuery}
        onNew={handleNewConversation}
        onSelect={(id) => {
          setActiveId(id);
          setIsHistoryOpen(false);
        }}
        onRename={(id, title) => updateConversation(id, (conversation) => ({ ...conversation, title, updatedAt: now() }))}
        onDelete={handleDeleteConversation}
      />

      <main className={`workspace ${hasConversationContent ? '' : 'empty-workspace'}`}>
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
            <h1>{activeConversation.title}</h1>
            <p>Ask, inspect the agent process, and continue from previous context.</p>
          </div>
        </header>

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
                databases={resources.databases}
                knowledge={resources.knowledge}
                selectedDatabaseId={activeConversation.selectedDatabaseId}
                selectedKnowledgeId={activeConversation.selectedKnowledgeId}
                model={resources.model}
                onSelectDatabase={(id) => handleResourceChange('selectedDatabaseId', id)}
                onSelectKnowledge={(id) => handleResourceChange('selectedKnowledgeId', id)}
                onSubmit={handleSend}
              />
            </div>
          )}
        </section>

        {hasConversationContent && (
          <Composer
            disabled={isStreaming}
            databases={resources.databases}
            knowledge={resources.knowledge}
            selectedDatabaseId={activeConversation.selectedDatabaseId}
            selectedKnowledgeId={activeConversation.selectedKnowledgeId}
            model={resources.model}
            onSelectDatabase={(id) => handleResourceChange('selectedDatabaseId', id)}
            onSelectKnowledge={(id) => handleResourceChange('selectedKnowledgeId', id)}
            onSubmit={handleSend}
          />
        )}
      </main>

      <InspectorPanel
        steps={activeConversation.traceSteps}
        selectedStepId={selectedTraceStep?.id || null}
        answer={latestAnswer}
        collapsed={isInspectorCollapsed}
        onToggleCollapsed={() => setIsInspectorCollapsed((collapsed) => !collapsed)}
      />
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
