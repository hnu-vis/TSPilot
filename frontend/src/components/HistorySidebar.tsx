import { BookOpen, BrainCircuit, Cpu, Database, Edit3, MessageSquarePlus, PanelLeftClose, PanelLeftOpen, Search, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { useI18n } from '../i18n';
import type { Conversation } from '../types';

export type WorkspaceView = 'chat' | 'database' | 'insight-memory' | 'model';

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
  const { t, formatDate } = useI18n();
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const normalizedQuery = query.trim().toLowerCase();
  const shown = conversations.filter((conversation) => {
    if (!normalizedQuery) return true;
    const haystack = `${displayConversationTitle(conversation.title, t)} ${conversation.messages.map((item) => item.content).join(' ')}`.toLowerCase();
    return haystack.includes(normalizedQuery);
  });
  const commitRename = (conversation: Conversation) => {
    const nextTitle = renameValue.trim();
    if (nextTitle && nextTitle !== conversation.title) onRename(conversation.id, nextTitle);
    setRenamingId(null);
  };

  return (
    <aside className={`history-sidebar ${collapsed ? 'collapsed' : ''}`} aria-label={t('Chat history')}>
      <div className="sidebar-brand">
        <div className="brand-mark">TS</div>
        <div className="sidebar-brand-copy">
          <h1>TSPilot</h1>
          <p>{t('Time-series chat')}</p>
        </div>
        <button className="history-collapse-button" type="button" aria-label={t(collapsed ? 'Expand history sidebar' : 'Collapse history sidebar')} onClick={onToggleCollapsed}>
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
        </button>
      </div>
      <nav className="workspace-nav" aria-label={t('Workspace')}>
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
              title={t(isDisabled ? item.hint : item.label)}
              aria-label={t(item.label)}
              onClick={() => {
                if (item.view) onViewChange(item.view);
              }}
            >
              <Icon size={16} />
              <span>{t(item.label)}</span>
              {!item.enabled && <small>{t('Soon')}</small>}
            </button>
          );
        })}
      </nav>
      <button className="new-chat-button" type="button" onClick={onNew}>
        <MessageSquarePlus size={17} />
        <span>{t('New chat')}</span>
      </button>
      <label className="search-box">
        <Search size={15} />
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder={t('Search chats')}
          aria-label={t('Search conversations')}
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
                  aria-label={t('Conversation title')}
                />
              </div>
            ) : (
              <button type="button" onClick={() => onSelect(conversation.id)} className="conversation-main">
                <span>{displayConversationTitle(conversation.title, t)}</span>
                <small>{formatConversationTime(conversation.updatedAt, formatDate)}</small>
              </button>
            )}
            <div className="conversation-actions">
              <button
                type="button"
                aria-label={t('Rename conversation')}
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
                aria-label={t('Delete conversation')}
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
                <span>{t('Delete chat?')}</span>
                <button type="button" onClick={() => setDeleteConfirmId(null)}>{t('Cancel')}</button>
                <button
                  type="button"
                  className="danger"
                  onClick={() => {
                    onDelete(conversation.id);
                    setDeleteConfirmId(null);
                  }}
                >
                  {t('Delete')}
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
    id: 'insight-memory',
    label: 'Key Insight Memory',
    hint: 'Manage Key Insight definitions and playbooks',
    icon: BrainCircuit,
    enabled: true,
    view: 'insight-memory',
  },
  {
    id: 'model',
    label: 'Model',
    hint: 'Configure AI and machine learning models',
    icon: Cpu,
    enabled: true,
    view: 'model',
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

function formatConversationTime(value: string, formatDate: (value: Date, options?: Intl.DateTimeFormatOptions) => string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return formatDate(date, { month: 'short', day: 'numeric' });
}

function displayConversationTitle(title: string, t: (key: string) => string) {
  return title === 'New chat' ? t('New chat') : title;
}
