import { ChevronDown, Database, Library, Server } from 'lucide-react';
import type { DatabaseResource, KnowledgeResource } from '../types';

type DatabaseProps = {
  kind: 'database';
  value: string | null;
  items: DatabaseResource[];
  onChange: (id: string | null) => void;
};

type KnowledgeProps = {
  kind: 'knowledge';
  value: string | null;
  items: KnowledgeResource[];
  onChange: (id: string | null) => void;
};

type Props = DatabaseProps | KnowledgeProps;

export function ResourceSelect(props: Props) {
  const selected = props.value ? props.items.find((item) => item.id === props.value) : null;
  const Icon = props.kind === 'database' ? Database : Library;
  const label = selected?.name || (props.kind === 'database' ? 'No database' : 'No knowledge');

  return (
    <div className="resource-select">
      <button className={`chip ${selected ? 'selected' : ''}`} type="button">
        <Icon size={15} />
        <span>{label}</span>
        <ChevronDown size={14} />
      </button>
      <div className="resource-menu">
        <button type="button" onClick={() => props.onChange(null)} className={!selected ? 'active' : ''}>
          <span>{props.kind === 'database' ? 'No database' : 'No knowledge'}</span>
          <small>{props.kind === 'database' ? 'Ask without data context' : 'Do not attach retrieval context'}</small>
        </button>
        {props.items.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => props.onChange(item.id)}
            className={item.id === props.value ? 'active' : ''}
          >
            <span>{item.name}</span>
            <small>
              {props.kind === 'database'
                ? `${(item as DatabaseResource).type}${(item as DatabaseResource).database ? ` · ${(item as DatabaseResource).database}` : ''}`
                : `${(item as KnowledgeResource).document_count || 0} docs`}
            </small>
          </button>
        ))}
      </div>
    </div>
  );
}

export function ModelChip({ model }: { model: string }) {
  return (
    <span className="chip readonly">
      <Server size={15} />
      <span>{model}</span>
    </span>
  );
}
