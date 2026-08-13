import { BookOpen, BrainCircuit, Database, Edit3, MessageSquarePlus, PanelLeftClose, PanelLeftOpen, Search, Trash2 } from 'lucide-react';
import { useState } from 'react';
import type { Conversation } from '../types';

export type WorkspaceView = 'chat' | 'database' | 'fact-memory';

type Props = {
  conversations: Conversation[];
  activeId: string;
  activeView: WorkspaceView;
  query: string;
  collapsed: boolean;
  onQueryChange: (value: string) => void;
  onToggleCollapsed: () => void;
  onNew: () => void;
  onViewChange: (view: WorkspaceView) => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
};

export function HistorySidebar({
  conversations,
  activeId,
  activeView,
  query,
  collapsed,
  onQueryChange,
  onToggleCollapsed,
  onNew,
  onViewChange,
  onSelect,
  onRename,
  onDelete,
}: Props) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const normalizedQuery = query.trim().toLowerCase();
  const shown = conversations.filter((conversation) => {
    if (!normalizedQuery) return true;
    const haystack = `${conversation.title} ${conversation.messages.map((item) => item.content).join(' ')}`.toLowerCase();
    return haystack.includes(normalizedQuery);
  });
  const commitRename = (conversation: Conversation) => {
    const nextTitle = renameValue.trim();
    if (nextTitle && nextTitle !== conversation.title) onRename(conversation.id, nextTitle);
    setRenamingId(null);
  };

  return (
    <aside className={`history-sidebar ${collapsed ? 'collapsed' : ''}`} aria-label="Chat history">
      <div className="sidebar-brand">
        <div className="brand-mark">TS</div>
        <div className="sidebar-brand-copy">
          <h1>TSPilot</h1>
          <p>Time-series chat</p>
        </div>
        <button className="history-collapse-button" type="button" aria-label={collapsed ? 'Expand history sidebar' : 'Collapse history sidebar'} onClick={onToggleCollapsed}>
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
        </button>
      </div>
      <nav className="workspace-nav" aria-label="Workspace">
        {workspaceNavItems.map((item) => {
          const Icon = item.icon;
          const isActive = item.view === activeView;
          const isDisabled = !item.enabled;
          return (
            <button
              key={item.id}
              type="button"
              className={`workspace-nav-item ${isActive ? 'active' : ''}`}
              disabled={isDisabled}
              title={isDisabled ? item.hint : item.label}
              aria-label={item.label}
              onClick={() => {
                if (item.view) onViewChange(item.view);
              }}
            >
              <Icon size={16} />
              <span>{item.label}</span>
              {!item.enabled && <small>Soon</small>}
            </button>
          );
        })}
      </nav>
      <button className="new-chat-button" type="button" onClick={onNew}>
        <MessageSquarePlus size={17} />
        <span>New chat</span>
      </button>
      <label className="search-box">
        <Search size={15} />
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search chats"
          aria-label="Search conversations"
        />
      </label>
      <div className="conversation-list">
        {shown.map((conversation) => (
          <article
            key={conversation.id}
            className={`conversation-item ${conversation.id === activeId ? 'active' : ''}`}
          >
            {renamingId === conversation.id ? (
              <div className="conversation-rename-form">
                <input
                  autoFocus
                  value={renameValue}
                  onChange={(event) => setRenameValue(event.target.value)}
                  onBlur={() => commitRename(conversation)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      commitRename(conversation);
                    }
                    if (event.key === 'Escape') {
                      event.preventDefault();
                      setRenamingId(null);
                    }
                  }}
                  aria-label="Conversation title"
                />
              </div>
            ) : (
              <button type="button" onClick={() => onSelect(conversation.id)} className="conversation-main">
                <span>{conversation.title}</span>
                <small>{formatConversationTime(conversation.updatedAt)}</small>
              </button>
            )}
            <div className="conversation-actions">
              <button
                type="button"
                aria-label="Rename conversation"
                onClick={() => {
                  setDeleteConfirmId(null);
                  setRenamingId(conversation.id);
                  setRenameValue(conversation.title);
                }}
              >
                <Edit3 size={14} />
              </button>
              <button
                type="button"
                aria-label="Delete conversation"
                onClick={() => {
                  setRenamingId(null);
                  setDeleteConfirmId((current) => current === conversation.id ? null : conversation.id);
                }}
              >
                <Trash2 size={14} />
              </button>
            </div>
            {deleteConfirmId === conversation.id && (
              <div className="conversation-delete-popover">
                <span>Delete chat?</span>
                <button type="button" onClick={() => setDeleteConfirmId(null)}>Cancel</button>
                <button
                  type="button"
                  className="danger"
                  onClick={() => {
                    onDelete(conversation.id);
                    setDeleteConfirmId(null);
                  }}
                >
                  Delete
                </button>
              </div>
            )}
          </article>
        ))}
      </div>
    </aside>
  );
}

const workspaceNavItems: Array<{
  id: string;
  label: string;
  hint: string;
  icon: typeof Database;
  enabled: boolean;
  view?: WorkspaceView;
}> = [
  {
    id: 'database',
    label: 'Database',
    hint: 'Manage connections and inspect schema',
    icon: Database,
    enabled: true,
    view: 'database',
  },
  {
    id: 'fact-memory',
    label: 'Fact Memory',
    hint: 'Manage fact definitions and recipes',
    icon: BrainCircuit,
    enabled: true,
    view: 'fact-memory',
  },
  {
    id: 'knowledge',
    label: 'Knowledge base',
    hint: 'Knowledge management is planned',
    icon: BookOpen,
    enabled: false,
  },
  {
    id: 'skills',
    label: 'Skills',
    hint: 'Skill management is planned',
    icon: BookOpen,
    enabled: false,
  },
];

function formatConversationTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
