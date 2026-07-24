import { Send, Square } from 'lucide-react';
import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import type { DatabaseResource, KnowledgeResource } from '../types';
import { ModelChip, ResourceSelect } from './ResourceSelect';

type Props = {
  disabled: boolean;
  running?: boolean;
  databases: DatabaseResource[];
  knowledge: KnowledgeResource[];
  selectedDatabaseId: string | null;
  selectedKnowledgeId: string | null;
  model: string;
  onSelectDatabase: (id: string | null) => void;
  onSelectKnowledge: (id: string | null) => void;
  onSubmit: (message: string) => void;
  onStop?: () => void;
};

export function Composer({
  disabled,
  running = false,
  databases,
  knowledge,
  selectedDatabaseId,
  selectedKnowledgeId,
  model,
  onSelectDatabase,
  onSelectKnowledge,
  onSubmit,
  onStop,
}: Props) {
  const [value, setValue] = useState('');
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = '0px';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [value]);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    const message = value.trim();
    if (!message || disabled) return;
    setValue('');
    onSubmit(message);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form className={`composer ${focused ? 'focused' : ''}`} onSubmit={submit}>
      <div className="composer-input-row">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        placeholder="Ask TSPilot to retrieve, analyze, forecast, or explain your time-series data..."
        disabled={disabled}
        rows={1}
      />
      </div>
      <div className="composer-toolbar">
        <div className="composer-context">
          <ResourceSelect
            kind="database"
            value={selectedDatabaseId}
            items={databases}
            onChange={onSelectDatabase}
          />
        </div>
        <div className="composer-actions">
          <ModelChip model={model} />
          <button
            className={`send-button ${running ? 'stop-button' : ''}`}
            type={running ? 'button' : 'submit'}
            disabled={!running && (disabled || !value.trim())}
            aria-label={running ? 'Stop response' : 'Send message'}
            onClick={running ? onStop : undefined}
          >
            {running ? <Square size={15} fill="currentColor" /> : <Send size={18} />}
          </button>
        </div>
      </div>
    </form>
  );
}
