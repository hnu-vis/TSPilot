import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Columns3,
  Database,
  Pencil,
  PlugZap,
  Plus,
  RefreshCw,
  Rows3,
  Server,
  Table2,
  Trash2,
  X,
} from 'lucide-react';
import { type FormEvent, useEffect, useMemo, useState } from 'react';
import { DATABASE_CATALOG, DATABASE_TYPES, databaseCatalogEntry, databaseTypeLabel } from '../databaseCatalog';
import {
  createDatabase,
  deleteDatabase,
  fetchDatabasePreview,
  testDatabaseConnection,
  updateDatabase,
} from '../services/api';
import type {
  DatabaseConfigInput,
  DatabasePreviewObject,
  DatabasePreviewPayload,
  DatabasePreviewResponse,
  DatabaseResource,
} from '../types';
import { useI18n } from '../i18n';
import { NotificationToast, type NotificationNotice } from './NotificationToast';

type Props = {
  databases: DatabaseResource[];
  selectedDatabaseId: string | null;
  onSelectDatabase: (id: string | null) => void;
  onDatabasesChange: () => Promise<DatabaseResource[]>;
};

type PreviewState = {
  databaseId: string | null;
  loading: boolean;
  error: string | null;
  data: DatabasePreviewResponse | null;
};

type FormMode = 'create' | 'edit' | null;

const emptyForm = databaseDefaultForm('timescaledb');

export function DatabaseManager({ databases, selectedDatabaseId, onSelectDatabase, onDatabasesChange }: Props) {
  const { t } = useI18n();
  const firstAvailableId = databases[0]?.id || null;
  const selectedDatabaseExists = Boolean(selectedDatabaseId && databases.some((database) => database.id === selectedDatabaseId));
  const activeDatabaseId = selectedDatabaseExists ? selectedDatabaseId : firstAvailableId;
  const activeDatabase = databases.find((database) => database.id === activeDatabaseId) || null;
  const [reloadToken, setReloadToken] = useState(0);
  const [formMode, setFormMode] = useState<FormMode>(null);
  const [createStep, setCreateStep] = useState<'catalog' | 'form'>('catalog');
  const [formValue, setFormValue] = useState<DatabaseConfigInput>(emptyForm);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [refreshProfile, setRefreshProfile] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [testNotice, setTestNotice] = useState<NotificationNotice | null>(null);
  const [previewState, setPreviewState] = useState<PreviewState>({
    databaseId: null,
    loading: false,
    error: null,
    data: null,
  });

  useEffect(() => {
    if (selectedDatabaseId !== firstAvailableId && activeDatabaseId === firstAvailableId) {
      onSelectDatabase(firstAvailableId);
    }
  }, [activeDatabaseId, firstAvailableId, onSelectDatabase, selectedDatabaseId]);

  useEffect(() => {
    if (!activeDatabaseId) {
      setPreviewState({ databaseId: null, loading: false, error: null, data: null });
      return;
    }

    let cancelled = false;
    setPreviewState((current) => ({
      databaseId: activeDatabaseId,
      loading: true,
      error: null,
      data: current.databaseId === activeDatabaseId ? current.data : null,
    }));
    fetchDatabasePreview(activeDatabaseId, { refresh: refreshProfile })
      .then((data) => {
        if (!cancelled) setPreviewState({ databaseId: activeDatabaseId, loading: false, error: null, data });
      })
      .catch((error) => {
        if (!cancelled) {
          setPreviewState({
            databaseId: activeDatabaseId,
            loading: false,
            error: error instanceof Error ? error.message : t('Unable to load database schema.'),
            data: null,
          });
        }
      })
      .finally(() => {
        if (!cancelled) setRefreshProfile(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeDatabaseId, reloadToken]);

  const preview = previewState.data?.preview || null;
  const schemaObjects = useMemo(() => collectSchemaObjects(preview), [preview]);
  const fields = preview?.fields || [];
  const labelsOrTags = preview?.labels_or_tags || [];
  const metadataEntries = objectEntries(preview?.metadata);
  const [selectedObjectKey, setSelectedObjectKey] = useState<string | null>(null);
  const activeObject = useMemo(() => {
    if (schemaObjects.length === 0) return null;
    return schemaObjects.find((item) => schemaObjectKey(item) === selectedObjectKey) || schemaObjects[0];
  }, [schemaObjects, selectedObjectKey]);

  const openCreateForm = () => {
    setFormMode('create');
    setCreateStep('catalog');
    setFormValue(emptyForm);
    setActionError(null);
    setTestNotice(null);
  };

  const openEditForm = () => {
    if (!activeDatabase) return;
    setFormMode('edit');
    setFormValue(databaseToForm(activeDatabase));
    setActionError(null);
    setTestNotice(null);
  };

  const closeForm = () => {
    setFormMode(null);
    setCreateStep('catalog');
    setActionError(null);
  };

  const selectDatabaseType = (type: string) => {
    setFormValue(databaseDefaultForm(type));
    setCreateStep('form');
    setActionError(null);
  };

  const handleSubmitConfig = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setActionError(null);
    try {
      const payload = normalizeFormPayload(formValue, formMode);
      const saved = formMode === 'edit' && activeDatabase
        ? await updateDatabase(activeDatabase.id, payload)
        : await createDatabase(payload as DatabaseConfigInput);
      await onDatabasesChange();
      onSelectDatabase(saved.id);
      setFormMode(null);
      setReloadToken((value) => value + 1);
      setTestNotice({
        tone: 'success',
        title: t('Connection saved'),
        message: t('{name} is ready to use.', { name: saved.display_name || saved.name }),
      });
    } catch (error) {
      setActionError(error instanceof Error ? error.message : t('Unable to save database configuration.'));
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteConfig = async () => {
    if (!activeDatabase) return;
    const confirmed = window.confirm(t('Delete database connection "{name}"?', { name: activeDatabase.display_name || activeDatabase.name }));
    if (!confirmed) return;
    setSaving(true);
    setActionError(null);
    try {
      await deleteDatabase(activeDatabase.id);
      const nextDatabases = await onDatabasesChange();
      onSelectDatabase(nextDatabases[0]?.id || null);
      setFormMode(null);
      setReloadToken((value) => value + 1);
      setTestNotice({
        tone: 'success',
        title: t('Connection deleted'),
        message: t('The database connection was removed from the workspace.'),
      });
    } catch (error) {
      setActionError(error instanceof Error ? error.message : t('Unable to delete database configuration.'));
    } finally {
      setSaving(false);
    }
  };

  const handleTestConnection = async () => {
    if (!activeDatabase) return;
    setTesting(true);
    setActionError(null);
    setTestNotice(null);
    try {
      const result = await testDatabaseConnection(activeDatabase.id);
      await onDatabasesChange();
      setTestNotice(result.success
        ? { tone: 'success', title: t('Connection successful'), message: result.latency_ms ? t('Connection succeeded in {latency}ms.', { latency: result.latency_ms }) : t('Connection succeeded.') }
        : { tone: 'error', title: t('Connection failed'), message: result.error || t('Connection failed.') });
    } catch (error) {
      setTestNotice({
        tone: 'error',
        title: t('Connection failed'),
        message: error instanceof Error ? error.message : t('Unable to test database connection.'),
      });
    } finally {
      setTesting(false);
    }
  };

  useEffect(() => {
    if (!testNotice) return;
    const timeout = window.setTimeout(() => setTestNotice(null), 5000);
    return () => window.clearTimeout(timeout);
  }, [testNotice]);

  useEffect(() => {
    if (schemaObjects.length === 0) {
      setSelectedObjectKey(null);
      return;
    }
    if (!schemaObjects.some((item) => schemaObjectKey(item) === selectedObjectKey)) {
      setSelectedObjectKey(schemaObjectKey(schemaObjects[0]));
    }
  }, [schemaObjects, selectedObjectKey]);

  return (
    <section className="database-manager" aria-label={t('Database management')}>
      <div className="database-sidebar-panel">
        <div className="database-panel-heading">
          <span><Database size={16} /> {t('Sources')}</span>
          <strong>{databases.length}</strong>
        </div>
        <div className="database-source-actions" aria-label={t('Source actions')}>
          <button className="database-add-button" type="button" onClick={openCreateForm}>
            <Plus size={14} />
            <span>{t('Add connection')}</span>
          </button>
        </div>
        <div className="database-source-list">
          {databases.map((database) => (
            <button
              key={database.id}
              type="button"
              className={`database-source-item ${database.id === activeDatabaseId ? 'active' : ''}`}
              onClick={() => onSelectDatabase(database.id)}
            >
              <span className={`database-status-dot ${statusClass(database.status)}`} />
              <span className="database-source-copy">
                <strong>{database.display_name || database.name}</strong>
                <small>
                  {database.type}{database.database ? ` / ${database.database}` : ''} · {statusLabel(database.status)}
                </small>
              </span>
            </button>
          ))}
          {databases.length === 0 && (
            <div className="database-empty-state">
              <AlertCircle size={18} />
              <span>{t('No configured databases.')}</span>
            </div>
          )}
        </div>
      </div>

      <div className="database-detail-panel">
        {activeDatabase ? (
          <>
            <div className="database-detail-header">
              <div className="database-detail-identity">
                <div className="database-title-line">
                  <h2>{activeDatabase.display_name || activeDatabase.name}</h2>
                  <span className="database-kicker">{activeDatabase.type}</span>
                </div>
                <div className="database-header-meta" aria-label={t('Database summary')}>
                  <span><Server size={13} /> {formatEndpoint(activeDatabase)}</span>
                  <span className={`database-status-pill ${statusClass(activeDatabase.status)}`}>
                    <CheckCircle2 size={13} /> {statusLabel(activeDatabase.status)}
                  </span>
                  <span><Table2 size={13} /> {schemaObjects.length} {t('objects')}</span>
                  <span><Columns3 size={13} /> {fields.length || totalColumns(schemaObjects)} {t('fields')}</span>
                </div>
              </div>
              <div className="database-header-actions">
                <button className="icon-text-button" type="button" onClick={openEditForm}>
                  <Pencil size={14} />
                  <span>{t('Edit')}</span>
                </button>
                <button className="icon-text-button" type="button" disabled={testing} onClick={handleTestConnection}>
                  <PlugZap size={14} />
                  <span>{t(testing ? 'Testing' : 'Test')}</span>
                </button>
                <button
                  className="icon-text-button"
                  type="button"
                  disabled={previewState.loading}
                  onClick={() => {
                    setRefreshProfile(true);
                    setReloadToken((value) => value + 1);
                  }}
                >
                  <RefreshCw size={14} />
                  <span>{t(previewState.loading ? 'Loading' : 'Refresh')}</span>
                </button>
                <button className="icon-text-button danger" type="button" disabled={saving} onClick={handleDeleteConfig}>
                  <Trash2 size={15} />
                  <span>{t('Delete')}</span>
                </button>
              </div>
            </div>

            <div className="database-notice-stack" aria-live="polite">
              {actionError && !formMode && (
                <div className="database-inline-error">
                  <AlertCircle size={16} />
                  <span>{actionError}</span>
                </div>
              )}
              {previewState.error && (
                <div className="database-inline-error">
                  <AlertCircle size={16} />
                  <span>{previewState.error}</span>
                </div>
              )}
              {previewState.data?.preview_kind === 'error' && previewState.data.error && (
                <div className="database-inline-error">
                  <AlertCircle size={16} />
                  <span>{previewState.data.error}</span>
                </div>
              )}
            </div>

            <div className="database-content-stack">
              <section className="database-section database-schema-browser">
                <SectionTitle icon={Table2} title={t('Schema objects')} count={schemaObjects.length} />
                <SchemaObjectTable
                  items={schemaObjects}
                  activeKey={schemaObjectKey(activeObject)}
                  loading={previewState.loading}
                  onSelect={(item) => setSelectedObjectKey(schemaObjectKey(item))}
                />
              </section>

              <section className="database-section database-object-detail">
                {activeObject ? (
                  <ObjectDetail
                    item={activeObject}
                    fields={fields}
                    labelsOrTags={labelsOrTags}
                    metadataEntries={metadataEntries}
                  />
                ) : (
                  <EmptyPreview label={t('Select a schema object to inspect details.')} />
                )}
              </section>
            </div>
          </>
        ) : (
          <div className="database-empty-main">
            <Database size={24} />
            <h2>{t('No database selected')}</h2>
            <p>{t('Add a database connection, then inspect its schema here.')}</p>
            <button className="icon-text-button" type="button" onClick={openCreateForm}>
              <Plus size={14} />
              <span>{t('Add connection')}</span>
            </button>
          </div>
        )}
      </div>

      {formMode && (
        <div className="database-modal-backdrop" role="presentation">
          <div className={`database-modal ${formMode === 'create' && createStep === 'catalog' ? 'database-catalog-modal' : ''}`} role="dialog" aria-modal="true" aria-label={t(formMode === 'create' ? 'Add database connection' : 'Edit database connection')}>
            {actionError && (
              <div className="database-inline-error">
                <AlertCircle size={16} />
                <span>{actionError}</span>
              </div>
            )}
            {formMode === 'create' && createStep === 'catalog' ? (
              <DatabaseTypeCatalog onSelect={selectDatabaseType} onCancel={closeForm} />
            ) : (
              <DatabaseConfigForm
                mode={formMode}
                value={formValue}
                saving={saving}
                onChange={setFormValue}
                onBack={formMode === 'create' ? () => setCreateStep('catalog') : undefined}
                onCancel={closeForm}
                onSubmit={handleSubmitConfig}
              />
            )}
          </div>
        </div>
      )}

      {testNotice && (
        <NotificationToast
          {...testNotice}
          onDismiss={() => setTestNotice(null)}
        />
      )}
    </section>
  );
}

function DatabaseTypeCatalog({ onSelect, onCancel }: { onSelect: (type: string) => void; onCancel: () => void }) {
  const { t } = useI18n();
  return (
    <div className="database-type-catalog">
      <div className="database-catalog-header">
        <div>
          <strong>{t('Add a database connection')}</strong>
          <span>{t('Choose a time-series database to configure.')}</span>
        </div>
        <button type="button" className="database-form-icon-button" aria-label={t('Close database selector')} onClick={onCancel}>
          <X size={15} />
        </button>
      </div>
      <div className="database-type-grid" aria-label={t('Supported time-series databases')}>
        {DATABASE_CATALOG.map((database) => (
          <button key={database.type} type="button" className="database-type-card" onClick={() => onSelect(database.type)}>
            <span className="database-type-logo" aria-hidden="true">
              <img src={database.logoUrl} alt="" width="40" height="40" loading="lazy" decoding="async" />
            </span>
            <span className="database-type-copy">
              <strong>{database.label}</strong>
              <small>{database.type}</small>
            </span>
          </button>
        ))}
        <div
          className="database-type-card database-type-card-soon"
          role="note"
          aria-label={t('More databases are coming soon.')}
        >
          <span className="database-type-logo database-type-logo-soon" aria-hidden="true">
            <Plus size={20} strokeWidth={1.8} />
          </span>
          <span className="database-type-copy">
            <strong>{t('More databases')}</strong>
            <small>{t('Soon')}</small>
          </span>
        </div>
      </div>
    </div>
  );
}

function SectionTitle({ icon: Icon, title, count }: { icon: typeof Server; title: string; count: number }) {
  return (
    <div className="database-section-title">
      <span><Icon size={15} /> {title}</span>
      <strong>{count}</strong>
    </div>
  );
}

function DatabaseConfigForm({
  mode,
  value,
  saving,
  onChange,
  onBack,
  onCancel,
  onSubmit,
}: {
  mode: Exclude<FormMode, null>;
  value: DatabaseConfigInput;
  saving: boolean;
  onChange: (value: DatabaseConfigInput) => void;
  onBack?: () => void;
  onCancel: () => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const { t } = useI18n();
  const catalogEntry = databaseCatalogEntry(value.type);
  const extraFields = catalogEntry?.extraFields || [];
  const updateField = <Key extends keyof DatabaseConfigInput>(key: Key, nextValue: DatabaseConfigInput[Key]) => {
    onChange({ ...value, [key]: nextValue });
  };
  const updateExtraField = (key: string, nextValue: string | boolean | number | null) => {
    onChange({ ...value, extra: { ...(value.extra || {}), [key]: nextValue } });
  };
  const changeDatabaseType = (type: string) => {
    onChange({
      ...databaseDefaultForm(type),
      name: value.name,
      display_name: value.display_name,
    });
  };

  return (
    <form className="database-config-form" onSubmit={onSubmit}>
      <div className="database-config-form-header">
        <div className="database-config-title">
          {onBack && (
            <button type="button" className="database-form-icon-button" aria-label={t('Back to database types')} onClick={onBack}>
              <ArrowLeft size={15} />
            </button>
          )}
          <div>
            <strong>{t(mode === 'create' ? 'Add connection' : 'Edit connection')}</strong>
            <span>{mode === 'create' ? t('Configure {type}.', { type: databaseTypeLabel(value.type) }) : t('Update the selected database source config.')}</span>
          </div>
        </div>
        <button type="button" className="database-form-icon-button" aria-label={t('Close form')} onClick={onCancel}>
          <X size={15} />
        </button>
      </div>

      <div className="database-form-grid">
        <label>
          <span>{t('Name')}</span>
          <input
            required
            value={value.name}
            onChange={(event) => updateField('name', event.target.value)}
            placeholder="analytics-prod"
          />
        </label>
        <label>
          <span>{t('Display name')}</span>
          <input
            value={value.display_name || ''}
            onChange={(event) => updateField('display_name', event.target.value)}
            placeholder="Analytics production"
          />
        </label>
        <label>
          <span>{t('Type')}</span>
          {mode === 'create' ? (
            <span className="database-selected-type">{databaseTypeLabel(value.type)}</span>
          ) : (
            <select value={value.type} onChange={(event) => changeDatabaseType(event.target.value)}>
              {!DATABASE_TYPES.includes(value.type) && (
                <option value={value.type} disabled>{databaseTypeLabel(value.type)}</option>
              )}
              {DATABASE_TYPES.map((type) => <option key={type} value={type}>{databaseTypeLabel(type)}</option>)}
            </select>
          )}
        </label>
        <label>
          <span>{t('Host')}</span>
          <input
            required
            value={value.host || ''}
            onChange={(event) => updateField('host', event.target.value)}
            placeholder="localhost"
          />
        </label>
        <label>
          <span>{t('Port')}</span>
          <input
            required
            min={1}
            type="number"
            value={value.port ?? ''}
            onChange={(event) => updateField('port', event.target.value ? Number(event.target.value) : null)}
            placeholder="5432"
          />
        </label>
        <label>
          <span>{t('Database')}</span>
          <input
            value={value.database || ''}
            onChange={(event) => updateField('database', event.target.value)}
            placeholder="metrics"
          />
        </label>
        <label>
          <span>{t('Username')}</span>
          <input
            value={value.username || ''}
            onChange={(event) => updateField('username', event.target.value)}
            autoComplete="username"
          />
        </label>
        <label>
          <span>{t('Password')}</span>
          <input
            value={value.password || ''}
            type="password"
            onChange={(event) => updateField('password', event.target.value)}
            autoComplete={mode === 'create' ? 'new-password' : 'current-password'}
            placeholder={mode === 'edit' ? t('Leave blank to keep existing password') : ''}
          />
        </label>
      </div>

      {extraFields.length > 0 && <section className="database-type-settings" aria-label={t('Database-specific settings')}>
        <div className="database-type-settings-heading">
          <strong>{t('Database-specific settings')}</strong>
          <span>{t('Only settings used by {type} are shown.', { type: databaseTypeLabel(value.type) })}</span>
        </div>
        <div className="database-form-grid">
          {extraFields.map((field) => {
            const current = value.extra?.[field.key];
            if (field.input === 'checkbox') {
              return <label className="database-extra-checkbox" key={field.key}>
                <input
                  type="checkbox"
                  checked={Boolean(current)}
                  onChange={(event) => updateExtraField(field.key, event.target.checked)}
                />
                <span>{t(field.label)}</span>
              </label>;
            }
            return <label key={field.key}>
              <span>{t(field.label)}</span>
              {field.input === 'select' ? (
                <select
                  required={field.required}
                  value={String(current ?? '')}
                  onChange={(event) => updateExtraField(field.key, event.target.value)}
                >
                  {(field.options || []).map((option) => <option key={option.value} value={option.value}>{t(option.label)}</option>)}
                </select>
              ) : (
                <input
                  required={field.required && mode === 'create'}
                  type={field.input}
                  value={String(current ?? '')}
                  onChange={(event) => updateExtraField(field.key, event.target.value)}
                  placeholder={field.secret && mode === 'edit' ? t('Leave blank to keep existing value') : field.placeholder}
                  autoComplete={field.secret ? 'new-password' : undefined}
                />
              )}
            </label>;
          })}
        </div>
      </section>}

      <label className="database-checkbox-row">
        <input
          type="checkbox"
          checked={Boolean(value.ssl_enabled)}
          onChange={(event) => updateField('ssl_enabled', event.target.checked)}
        />
        <span>{t('Use SSL/TLS')}</span>
      </label>

      <div className="database-form-actions">
        <button type="button" className="icon-text-button" onClick={onCancel}>
          <X size={14} />
          <span>{t('Cancel')}</span>
        </button>
        <button type="submit" className="icon-text-button primary" disabled={saving}>
          <CheckCircle2 size={14} />
          <span>{t(saving ? 'Saving' : 'Save connection')}</span>
        </button>
      </div>
    </form>
  );
}

function SchemaObjectTable({
  items,
  activeKey,
  loading,
  onSelect,
}: {
  items: DatabasePreviewObject[];
  activeKey: string;
  loading: boolean;
  onSelect: (item: DatabasePreviewObject) => void;
}) {
  const { t } = useI18n();
  if (loading) return <EmptyPreview label={t('Loading schema preview.')} />;
  if (items.length === 0) return <EmptyPreview label={t('No schema objects returned.')} />;

  return (
    <div className="schema-object-list">
      {items.map((item) => {
        const key = schemaObjectKey(item);
        const columns = item.columns || [];
        const fieldValues = item.field_values || [];
        const fieldCount = columns.length || fieldValues.length;
        return (
          <button
            key={key}
            type="button"
            className={`schema-object-row ${key === activeKey ? 'selected' : ''}`}
            onClick={() => onSelect(item)}
          >
            <span className="schema-object-row-icon"><Database size={14} /></span>
            <span className="schema-object-row-copy">
              <strong>{item.name}</strong>
              <small>{item.schema || 'default'} · {item.type || objectTypeLabel(columns, fieldValues)}</small>
            </span>
            <span className="schema-object-row-counts">
              <strong>{fieldCount}</strong>
              <small>{t('fields')}</small>
            </span>
          </button>
        );
      })}
    </div>
  );
}

function ObjectDetail({
  item,
  fields,
  labelsOrTags,
  metadataEntries,
}: {
  item: DatabasePreviewObject;
  fields: Array<Record<string, unknown>>;
  labelsOrTags: Array<Record<string, unknown>>;
  metadataEntries: Array<[string, unknown]>;
}) {
  const { t, locale } = useI18n();
  const columns = item.columns || [];
  const fieldValues = item.field_values || [];
  const sampleRows = item.sample_rows || [];
  const relatedValues = [...fields, ...labelsOrTags].filter((value) => isRelatedSchemaValue(value, item.name));
  const contextValues = mergeContextRecords(
    relatedValues.length ? relatedValues : metadataEntriesToRecords(metadataEntries),
  );

  return (
    <>
      <div className="object-detail-header">
        <div>
          <span>{[item.schema, item.type].filter(Boolean).join(' / ') || t('schema object')}</span>
          <h3>{item.name}</h3>
        </div>
        {typeof item.row_count === 'number' && (
          <div className="object-detail-stats"><strong>{item.row_count.toLocaleString(locale)} {t('rows')}</strong></div>
        )}
      </div>

      <div className="object-detail-grid">
        <section className="object-detail-block columns-panel">
          <SectionTitle icon={Columns3} title={t('Columns')} count={columns.length || fieldValues.length} />
          <ColumnList columns={columns} fieldValues={fieldValues} />
        </section>

        <section className="object-detail-block samples-panel">
          <SectionTitle icon={Rows3} title={t('Sample Rows')} count={sampleRows.length} />
          <SampleRows rows={sampleRows} />
        </section>

        <section className="object-detail-block insights-panel">
          <SectionTitle icon={Database} title={t('Field Metadata')} count={contextValues.length} />
          <FieldMetadataList values={contextValues} />
        </section>
      </div>
    </>
  );
}

function FieldMetadataList({ values }: { values: Array<Record<string, unknown>> }) {
  const { t } = useI18n();
  return (
    <div className="database-field-metadata-list">
      {values.length ? values.map((value, index) => {
        const title = String(value.name || value.column || value.field || value.tag || value.label || `Item ${index + 1}`);
        const dataType = stringFromUnknown(value.data_type || value.type);
        const detail = objectEntries(value)
          .filter(([key]) => !['name', 'column', 'field', 'tag', 'label', 'data_type', 'type'].includes(key))
          .map(([key, entryValue]) => `${key}: ${formatValue(entryValue)}`)
          .join(' / ') || 'available';
        return (
          <div key={`${title}-${index}`} className="database-field-metadata-row">
            <span className="database-tree-field-heading">
              <strong title={title}>{title}</strong>
              {dataType && <em title={dataType}>{dataType}</em>}
            </span>
            <span title={detail}>{detail}</span>
          </div>
        );
      }) : (
        <div className="database-tree-empty">{t('No field metadata returned.')}</div>
      )}
    </div>
  );
}

function ColumnList({
  columns,
  fieldValues,
}: {
  columns: NonNullable<DatabasePreviewObject['columns']>;
  fieldValues: string[];
}) {
  const { t } = useI18n();
  if (columns.length === 0 && fieldValues.length === 0) return <EmptyPreview label={t('No columns returned.')} />;
  return (
    <div className="object-column-list">
      {columns.map((column) => (
        <div key={column.name} className="object-column-row">
          <strong title={column.name}>{column.name}</strong>
          <span title={column.data_type || 'unknown'}>{column.data_type || 'unknown'}</span>
        </div>
      ))}
      {columns.length === 0 && fieldValues.map((field) => (
        <div key={field} className="object-column-row">
          <strong title={field}>{field}</strong>
          <span>{t('field')}</span>
        </div>
      ))}
    </div>
  );
}

function SampleRows({ rows }: { rows: Array<Record<string, unknown>> }) {
  const { t } = useI18n();
  if (rows.length === 0) return <EmptyPreview label={t('No sample rows returned.')} />;
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  return (
    <div className="database-sample-wrap">
      <table className="sample-table database-sample-table">
        <thead>
          <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => {
                const value = formatValue(row[column]);
                return <td key={column} title={value}>{value}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EmptyPreview({ label }: { label: string }) {
  return (
    <div className="database-empty-state">
      <AlertCircle size={16} />
      <span>{label}</span>
    </div>
  );
}

function schemaObjectKey(item: DatabasePreviewObject | null) {
  if (!item) return '';
  return `${item.schema || ''}:${item.type || ''}:${item.name}`;
}

function collectSchemaObjects(preview: DatabasePreviewPayload | null): DatabasePreviewObject[] {
  if (!preview) return [];
  return [...(preview.tables_or_measurements || []), ...(preview.metrics || [])];
}

function totalColumns(objects: DatabasePreviewObject[]) {
  return objects.reduce((total, item) => total + (item.columns?.length || item.field_values?.length || 0), 0);
}

function databaseToForm(database: DatabaseResource): DatabaseConfigInput {
  return {
    name: database.name || database.id,
    type: database.type || 'timescaledb',
    host: database.host || '',
    port: database.port ?? null,
    database: database.database || '',
    username: database.username || '',
    password: '',
    display_name: database.display_name || '',
    ssl_enabled: Boolean(database.ssl_enabled),
    extra: { ...(database.extra || {}) },
  };
}

export function databaseDefaultForm(type: string): DatabaseConfigInput {
  const defaults = databaseCatalogEntry(type)?.defaults;
  return {
    name: '',
    type,
    host: defaults?.host || '',
    port: defaults?.port ?? null,
    database: defaults?.database || '',
    username: defaults?.username || '',
    password: '',
    display_name: '',
    ssl_enabled: Boolean(defaults?.ssl_enabled),
    extra: { ...(defaults?.extra || {}) },
  };
}

export function normalizeFormPayload(value: DatabaseConfigInput, mode: FormMode): Partial<DatabaseConfigInput> {
  const payload: Partial<DatabaseConfigInput> = {
    name: value.name.trim(),
    type: value.type.trim(),
    host: nullableString(value.host),
    port: value.port ?? null,
    database: nullableString(value.database),
    username: nullableString(value.username),
    display_name: nullableString(value.display_name),
    ssl_enabled: Boolean(value.ssl_enabled),
  };
  const password = nullableString(value.password);
  if (password || mode === 'create') {
    payload.password = password;
  }
  const secretFields = new Set(
    (databaseCatalogEntry(value.type)?.extraFields || [])
      .filter((field) => field.secret)
      .map((field) => field.key),
  );
  const extra = Object.fromEntries(
    Object.entries(value.extra || {}).filter(([key, raw]) => {
      if (raw === null || raw === undefined || raw === '') return false;
      return !(mode === 'edit' && secretFields.has(key) && !String(raw).trim());
    }),
  );
  if (Object.keys(extra).length > 0) payload.extra = extra;
  return payload;
}

function nullableString(value: string | null | undefined) {
  const trimmed = String(value || '').trim();
  return trimmed || null;
}

function objectTypeLabel(
  columns: NonNullable<DatabasePreviewObject['columns']>,
  fieldValues: string[],
) {
  if (columns.length > 0) return 'table';
  if (fieldValues.length > 0) return 'measurement';
  return 'object';
}

function formatEndpoint(database: DatabaseResource) {
  const host = database.host || 'local';
  return database.port ? `${host}:${database.port}` : host;
}

function statusClass(status: string | undefined) {
  const normalized = String(status || '').toLowerCase();
  if (normalized === 'connected' || normalized === 'available') return 'complete';
  if (normalized === 'disconnected' || normalized === 'error' || normalized === 'failed') return 'error';
  return 'running';
}

function statusLabel(status: string | undefined) {
  const normalized = String(status || '').trim();
  return normalized || 'unknown';
}

function objectEntries(value: unknown): Array<[string, unknown]> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
  return Object.entries(value as Record<string, unknown>);
}

function metadataEntriesToRecords(entries: Array<[string, unknown]>) {
  return entries.slice(0, 8).map(([name, value]) => ({ name, value }));
}

export function mergeContextRecords(records: Array<Record<string, unknown>>) {
  const merged = new Map<string, Record<string, unknown>>();
  records.forEach((record, index) => {
    const identity = contextRecordIdentity(record, index);
    const current = merged.get(identity);
    if (!current) {
      merged.set(identity, { ...record });
      return;
    }
    const next = { ...current };
    objectEntries(record).forEach(([key, value]) => {
      next[key] = mergeContextValue(next[key], value);
    });
    merged.set(identity, next);
  });
  return Array.from(merged.values());
}

function contextRecordIdentity(record: Record<string, unknown>, index: number) {
  const name = record.name || record.column || record.field || record.tag || record.label;
  if (name !== null && name !== undefined && String(name).trim()) {
    return String(name).trim().toLowerCase();
  }
  return `record-${index}`;
}

function mergeContextValue(current: unknown, incoming: unknown): unknown {
  if (current === undefined || current === null || current === '') return incoming;
  if (incoming === undefined || incoming === null || incoming === '') return current;
  if (valuesEqual(current, incoming)) return current;
  const values = [...asContextValues(current), ...asContextValues(incoming)];
  return values.filter((value, index) => (
    values.findIndex((candidate) => valuesEqual(candidate, value)) === index
  ));
}

function asContextValues(value: unknown) {
  return Array.isArray(value) ? value : [value];
}

function valuesEqual(left: unknown, right: unknown) {
  if (left === right) return true;
  try {
    return JSON.stringify(left) === JSON.stringify(right);
  } catch {
    return false;
  }
}

function stringFromUnknown(value: unknown) {
  return typeof value === 'string' || typeof value === 'number' ? String(value) : '';
}

function isRelatedSchemaValue(value: Record<string, unknown>, objectName: string) {
  const candidates = [value.table, value.measurement, value.metric, value.object, value.parent].map((item) => String(item || ''));
  return candidates.some((candidate) => candidate === objectName);
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'n/a';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
