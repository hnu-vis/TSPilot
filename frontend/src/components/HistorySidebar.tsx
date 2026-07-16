import { Edit3, MessageSquarePlus, Search, Trash2 } from 'lucide-react';
import type { Conversation } from '../types';

type Props = {
  conversations: Conversation[];
  activeId: string;
  query: string;
  onQueryChange: (value: string) => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
};

export function HistorySidebar({
  conversations,
  activeId,
  query,
  onQueryChange,
  onNew,
  onSelect,
  onRename,
  onDelete,
}: Props) {
  const normalizedQuery = query.trim().toLowerCase();
  const shown = conversations.filter((conversation) => {
    if (!normalizedQuery) return true;
    const haystack = `${conversation.title} ${conversation.messages.map((item) => item.content).join(' ')}`.toLowerCase();
    return haystack.includes(normalizedQuery);
  });

  return (
    <aside className="history-sidebar" aria-label="Chat history">
      <div className="sidebar-brand">
        <div>
          <div className="brand-mark">TS</div>
        </div>
        <div>
          <h1>TSPilot</h1>
          <p>Time-series chat</p>
        </div>
      </div>
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
            <button type="button" onClick={() => onSelect(conversation.id)} className="conversation-main">
              <span>{conversation.title}</span>
              <small>{formatConversationTime(conversation.updatedAt)}</small>
            </button>
            <div className="conversation-actions">
              <button
                type="button"
                aria-label="Rename conversation"
                onClick={() => {
                  const nextTitle = window.prompt('Rename chat', conversation.title);
                  if (nextTitle?.trim()) onRename(conversation.id, nextTitle.trim());
                }}
              >
                <Edit3 size={14} />
              </button>
              <button
                type="button"
                aria-label="Delete conversation"
                onClick={() => {
                  if (window.confirm('Delete this chat?')) onDelete(conversation.id);
                }}
              >
                <Trash2 size={14} />
              </button>
            </div>
          </article>
        ))}
      </div>
    </aside>
  );
}

function formatConversationTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
