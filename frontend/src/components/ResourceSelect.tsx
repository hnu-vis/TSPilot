import { ChevronDown, Cpu, Database, Library } from 'lucide-react';
import type { AIModelEndpointConfig, DatabaseResource, KnowledgeResource } from '../types';
import { useI18n } from '../i18n';

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
  const { t } = useI18n();
  const selected = props.value ? props.items.find((item) => item.id === props.value) : null;
  const Icon = props.kind === 'database' ? Database : Library;
  const label = selected?.name || t(props.kind === 'database' ? 'No database' : 'No knowledge');

  return (
    <div className="resource-select">
      <button className={`chip ${selected ? 'selected' : ''}`} type="button">
        <Icon size={15} />
        <span>{label}</span>
        <ChevronDown size={14} />
      </button>
      <div className="resource-menu">
        <button type="button" onClick={() => props.onChange(null)} className={!selected ? 'active' : ''}>
          <span>{t(props.kind === 'database' ? 'No database' : 'No knowledge')}</span>
          <small>{t(props.kind === 'database' ? 'Ask without data context' : 'Do not attach retrieval context')}</small>
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

export function ModelSelect({ models, value, fallbackLabel, onChange }: {
  models: AIModelEndpointConfig[];
  value: string | null;
  fallbackLabel: string;
  onChange: (id: string | null) => void;
}) {
  const { t } = useI18n();
  const active = models.find((item) => item.is_active) || models[0];
  const selected = models.find((item) => item.id === value) || active;
  return (
    <div className="resource-select model-select">
      <button className="chip selected" type="button" aria-label={t('Select conversation model')}>
        <Cpu size={15} />
        <span>{selected?.model || fallbackLabel}</span>
        <ChevronDown size={14} />
      </button>
      <div className="resource-menu model-menu">
        {models.map((item) => (
          <button key={item.id} type="button" onClick={() => onChange(item.id)} className={item.id === selected?.id ? 'active' : ''}>
            <span>{item.model}</span>
            <small>{t(item.is_active ? 'Workspace default' : 'Configured model')} · {item.provider}</small>
          </button>
        ))}
      </div>
    </div>
  );
}
