import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
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

const DATABASE_TYPES = ['influxdb', 'timescaledb', 'prometheus', 'iotdb', 'questdb', 'clickhouse', 'openmldb'];

const emptyForm: DatabaseConfigInput = {
  name: '',
  type: 'timescaledb',
  host: '',
  port: null,
  database: '',
  username: '',
  password: '',
  display_name: '',
  ssl_enabled: false,
};

export function DatabaseManager({ databases, selectedDatabaseId, onSelectDatabase, onDatabasesChange }: Props) {
  const firstAvailableId = databases[0]?.id || null;
  const selectedDatabaseExists = Boolean(selectedDatabaseId && databases.some((database) => database.id === selectedDatabaseId));
  const activeDatabaseId = selectedDatabaseExists ? selectedDatabaseId : firstAvailableId;
  const activeDatabase = databases.find((database) => database.id === activeDatabaseId) || null;
  const [reloadToken, setReloadToken] = useState(0);
  const [formMode, setFormMode] = useState<FormMode>(null);
  const [formValue, setFormValue] = useState<DatabaseConfigInput>(emptyForm);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [sourcesCollapsed, setSourcesCollapsed] = useState(false);
  const [refreshProfile, setRefreshProfile] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [testMessage, setTestMessage] = useState<string | null>(null);
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
            error: error instanceof Error ? error.message : 'Unable to load database schema.',
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
  const profileRows = useMemo(() => collectDataProfileRows(preview), [preview]);
  const [selectedObjectKey, setSelectedObjectKey] = useState<string | null>(null);
  const activeObject = useMemo(() => {
    if (schemaObjects.length === 0) return null;
    return schemaObjects.find((item) => schemaObjectKey(item) === selectedObjectKey) || schemaObjects[0];
  }, [schemaObjects, selectedObjectKey]);

  const openCreateForm = () => {
    setFormMode('create');
    setFormValue(emptyForm);
    setActionError(null);
    setTestMessage(null);
  };

  const openEditForm = () => {
    if (!activeDatabase) return;
    setFormMode('edit');
    setFormValue(databaseToForm(activeDatabase));
    setActionError(null);
    setTestMessage(null);
  };

  const closeForm = () => {
    setFormMode(null);
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
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Unable to save database configuration.');
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteConfig = async () => {
    if (!activeDatabase) return;
    const confirmed = window.confirm(`Delete database connection "${activeDatabase.display_name || activeDatabase.name}"?`);
    if (!confirmed) return;
    setSaving(true);
    setActionError(null);
    try {
      await deleteDatabase(activeDatabase.id);
      const nextDatabases = await onDatabasesChange();
      onSelectDatabase(nextDatabases[0]?.id || null);
      setFormMode(null);
      setReloadToken((value) => value + 1);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Unable to delete database configuration.');
    } finally {
      setSaving(false);
    }
  };

  const handleTestConnection = async () => {
    if (!activeDatabase) return;
    setTesting(true);
    setActionError(null);
    setTestMessage(null);
    try {
      const result = await testDatabaseConnection(activeDatabase.id);
      await onDatabasesChange();
      setTestMessage(result.success
        ? `Connection succeeded${result.latency_ms ? ` in ${result.latency_ms}ms` : ''}.`
        : result.error || 'Connection failed.');
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Unable to test database connection.');
    } finally {
      setTesting(false);
    }
  };

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
    <section className={`database-manager ${sourcesCollapsed ? 'sources-collapsed' : ''}`} aria-label="Database management">
      <div className="database-sidebar-panel">
        <div className="database-panel-heading">
          {!sourcesCollapsed && (
            <>
              <span><Database size={16} /> Sources</span>
              <div className="database-panel-heading-actions">
                <strong>{databases.length}</strong>
              </div>
            </>
          )}
          <button
            type="button"
            className="database-source-add"
            aria-label={sourcesCollapsed ? 'Expand sources' : 'Collapse sources'}
            onClick={() => setSourcesCollapsed((collapsed) => !collapsed)}
          >
            {sourcesCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
        </div>
        {!sourcesCollapsed && (
          <>
            <div className="database-source-actions" aria-label="Source actions">
              <button className="icon-text-button primary" type="button" onClick={openCreateForm}>
                <Plus size={14} />
                <span>Add</span>
              </button>
              <button className="icon-text-button" type="button" disabled={!activeDatabase} onClick={openEditForm}>
                <Pencil size={14} />
                <span>Edit</span>
              </button>
              <button className="icon-text-button" type="button" disabled={!activeDatabase || testing} onClick={handleTestConnection}>
                <PlugZap size={14} />
                <span>{testing ? 'Testing' : 'Test'}</span>
              </button>
              <button className="icon-text-button danger" type="button" disabled={!activeDatabase || saving} onClick={handleDeleteConfig}>
                <Trash2 size={14} />
                <span>Delete</span>
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
                  <span>No configured databases.</span>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      <div className="database-detail-panel">
        {activeDatabase ? (
          <>
            <div className="database-detail-header">
              <div>
                <span className="database-kicker">{activeDatabase.type}</span>
                <h2>{activeDatabase.display_name || activeDatabase.name}</h2>
                <div className="database-header-meta" aria-label="Database summary">
                  <span><Server size={14} /> {formatEndpoint(activeDatabase)}</span>
                  <span className={`database-status-pill ${statusClass(activeDatabase.status)}`}>
                    <CheckCircle2 size={14} /> {statusLabel(activeDatabase.status)}
                  </span>
                  <span><Table2 size={14} /> {schemaObjects.length} objects</span>
                  <span><Columns3 size={14} /> {fields.length || totalColumns(schemaObjects)} fields</span>
                </div>
              </div>
              <div className="database-header-actions">
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
                  <span>{previewState.loading ? 'Loading' : 'Refresh'}</span>
                </button>
              </div>
            </div>

            {actionError && !formMode && (
              <div className="database-inline-error">
                <AlertCircle size={16} />
                <span>{actionError}</span>
              </div>
            )}
            {testMessage && (
              <div className={`database-inline-status ${testMessage.includes('succeeded') ? 'complete' : 'error'}`}>
                <PlugZap size={16} />
                <span>{testMessage}</span>
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

            <div className="database-content-stack">
              <section className="database-section database-schema-browser">
                <SectionTitle icon={Table2} title="Schema Objects" count={schemaObjects.length} />
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
                    profileRows={profileRows}
                    profileCache={previewState.data?.profile_cache || null}
                  />
                ) : (
                  <EmptyPreview label="Select a schema object to inspect details." />
                )}
              </section>
            </div>
          </>
        ) : (
          <div className="database-empty-main">
            <Database size={24} />
            <h2>No database selected</h2>
            <p>Add a database connection, then inspect its schema here.</p>
            <button className="icon-text-button" type="button" onClick={openCreateForm}>
              <Plus size={14} />
              <span>Add connection</span>
            </button>
          </div>
        )}
      </div>

      {formMode && (
        <div className="database-modal-backdrop" role="presentation">
          <div className="database-modal" role="dialog" aria-modal="true" aria-label={formMode === 'create' ? 'Add database connection' : 'Edit database connection'}>
            {actionError && (
              <div className="database-inline-error">
                <AlertCircle size={16} />
                <span>{actionError}</span>
              </div>
            )}
            <DatabaseConfigForm
              mode={formMode}
              value={formValue}
              saving={saving}
              onChange={setFormValue}
              onCancel={closeForm}
              onSubmit={handleSubmitConfig}
            />
          </div>
        </div>
      )}
    </section>
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
  onCancel,
  onSubmit,
}: {
  mode: Exclude<FormMode, null>;
  value: DatabaseConfigInput;
  saving: boolean;
  onChange: (value: DatabaseConfigInput) => void;
  onCancel: () => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const updateField = <Key extends keyof DatabaseConfigInput>(key: Key, nextValue: DatabaseConfigInput[Key]) => {
    onChange({ ...value, [key]: nextValue });
  };

  return (
    <form className="database-config-form" onSubmit={onSubmit}>
      <div className="database-config-form-header">
        <div>
          <strong>{mode === 'create' ? 'Add connection' : 'Edit connection'}</strong>
          <span>{mode === 'create' ? 'Create a database source config.' : 'Update the selected database source config.'}</span>
        </div>
        <button type="button" className="database-form-icon-button" aria-label="Close form" onClick={onCancel}>
          <X size={15} />
        </button>
      </div>

      <div className="database-form-grid">
        <label>
          <span>Name</span>
          <input
            required
            value={value.name}
            onChange={(event) => updateField('name', event.target.value)}
            placeholder="analytics-prod"
          />
        </label>
        <label>
          <span>Display name</span>
          <input
            value={value.display_name || ''}
            onChange={(event) => updateField('display_name', event.target.value)}
            placeholder="Analytics production"
          />
        </label>
        <label>
          <span>Type</span>
          <select value={value.type} onChange={(event) => updateField('type', event.target.value)}>
            {DATABASE_TYPES.map((type) => <option key={type} value={type}>{type}</option>)}
          </select>
        </label>
        <label>
          <span>Host</span>
          <input
            value={value.host || ''}
            onChange={(event) => updateField('host', event.target.value)}
            placeholder="localhost"
          />
        </label>
        <label>
          <span>Port</span>
          <input
            min={0}
            type="number"
            value={value.port ?? ''}
            onChange={(event) => updateField('port', event.target.value ? Number(event.target.value) : null)}
            placeholder="5432"
          />
        </label>
        <label>
          <span>Database</span>
          <input
            value={value.database || ''}
            onChange={(event) => updateField('database', event.target.value)}
            placeholder="metrics"
          />
        </label>
        <label>
          <span>Username</span>
          <input
            value={value.username || ''}
            onChange={(event) => updateField('username', event.target.value)}
            autoComplete="username"
          />
        </label>
        <label>
          <span>Password</span>
          <input
            value={value.password || ''}
            type="password"
            onChange={(event) => updateField('password', event.target.value)}
            autoComplete={mode === 'create' ? 'new-password' : 'current-password'}
            placeholder={mode === 'edit' ? 'Leave blank to keep existing password' : ''}
          />
        </label>
      </div>

      <label className="database-checkbox-row">
        <input
          type="checkbox"
          checked={Boolean(value.ssl_enabled)}
          onChange={(event) => updateField('ssl_enabled', event.target.checked)}
        />
        <span>Use SSL/TLS</span>
      </label>

      <div className="database-form-actions">
        <button type="button" className="icon-text-button" onClick={onCancel}>
          <X size={14} />
          <span>Cancel</span>
        </button>
        <button type="submit" className="icon-text-button primary" disabled={saving}>
          <CheckCircle2 size={14} />
          <span>{saving ? 'Saving' : 'Save connection'}</span>
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
  if (loading) return <EmptyPreview label="Loading schema preview." />;
  if (items.length === 0) return <EmptyPreview label="No schema objects returned." />;

  return (
    <div className="schema-table-wrap">
      <table className="schema-object-table">
        <thead>
          <tr>
            <th>Object</th>
            <th>Namespace</th>
            <th>Type</th>
            <th>Rows</th>
            <th>Fields</th>
            <th>Preview</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => {
            const key = schemaObjectKey(item);
            const columns = item.columns || [];
            const fieldValues = item.field_values || [];
            const fieldCount = columns.length || fieldValues.length;
            return (
              <tr
                key={key}
                className={key === activeKey ? 'selected' : ''}
                onClick={() => onSelect(item)}
              >
                <td>
                  <button
                    type="button"
                    className="schema-object-name"
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelect(item);
                    }}
                  >
                    <Database size={14} />
                    <span>{item.name}</span>
                  </button>
                </td>
                <td>{item.schema || 'default'}</td>
                <td>{item.type || objectTypeLabel(columns, fieldValues)}</td>
                <td>{typeof item.row_count === 'number' ? item.row_count.toLocaleString() : 'n/a'}</td>
                <td>{fieldCount.toLocaleString()}</td>
                <td>
                  <div className="schema-field-preview">
                    {summarizeFields(item).map((field, index) => <code key={`${field}-${index}`}>{field}</code>)}
                    {fieldCount === 0 && <span>n/a</span>}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ObjectDetail({
  item,
  fields,
  labelsOrTags,
  metadataEntries,
  profileRows,
  profileCache,
}: {
  item: DatabasePreviewObject;
  fields: Array<Record<string, unknown>>;
  labelsOrTags: Array<Record<string, unknown>>;
  metadataEntries: Array<[string, unknown]>;
  profileRows: Array<Record<string, unknown>>;
  profileCache: Record<string, unknown> | null;
}) {
  const columns = item.columns || [];
  const fieldValues = item.field_values || [];
  const sampleRows = item.sample_rows || [];
  const relatedValues = [...fields, ...labelsOrTags].filter((value) => isRelatedSchemaValue(value, item.name));

  return (
    <>
      <div className="object-detail-header">
        <div>
          <span>{[item.schema, item.type].filter(Boolean).join(' / ') || 'schema object'}</span>
          <h3>{item.name}</h3>
        </div>
        {typeof item.row_count === 'number' && <strong>{item.row_count.toLocaleString()} rows</strong>}
      </div>

      <div className="object-detail-grid">
        <div className="object-detail-block">
          <SectionTitle icon={Columns3} title="Columns" count={columns.length || fieldValues.length} />
          <ColumnList columns={columns} fieldValues={fieldValues} />
        </div>

        <div className="object-detail-block">
          <SectionTitle icon={Rows3} title="Sample Rows" count={sampleRows.length} />
          <SampleRows rows={sampleRows} />
        </div>

        <div className="object-detail-block compact">
          <SectionTitle icon={RefreshCw} title="Data Profile" count={profileRows.length} />
          <DataProfile rows={profileRows.filter((row) => String(row.measurement || row.source || '') === item.name)} cache={profileCache} />
        </div>

        {(relatedValues.length > 0 || metadataEntries.length > 0) && (
          <div className="object-detail-block compact">
            <SectionTitle icon={Database} title="Context" count={relatedValues.length + metadataEntries.length} />
            <KeyValueList values={relatedValues.length ? relatedValues : metadataEntriesToRecords(metadataEntries)} />
          </div>
        )}
      </div>
    </>
  );
}

function DataProfile({ rows, cache }: { rows: Array<Record<string, unknown>>; cache: Record<string, unknown> | null }) {
  const cacheText = cache
    ? `${formatValue(cache.source)} · ${formatValue(cache.generated_at)}`
    : 'n/a';
  if (rows.length === 0) {
    return (
      <div className="database-profile-block">
        <div className="database-profile-cache">Cache: {cacheText}</div>
        <EmptyPreview label="No data profile returned." />
      </div>
    );
  }
  return (
    <div className="database-profile-block">
      <div className="database-profile-cache">Cache: {cacheText}</div>
      <div className="database-profile-list">
        {rows.slice(0, 12).map((row, index) => (
          <div key={index} className="database-profile-row">
            <strong>{profileSeriesLabel(row)}</strong>
            <span>{formatValue(row.start)} → {formatValue(row.end)}</span>
            <small>{formatValue(row.point_count)} points</small>
          </div>
        ))}
      </div>
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
  if (columns.length === 0 && fieldValues.length === 0) return <EmptyPreview label="No columns returned." />;
  return (
    <div className="object-column-list">
      {columns.map((column) => (
        <div key={column.name} className="object-column-row">
          <strong>{column.name}</strong>
          <span>{column.data_type || 'unknown'}</span>
        </div>
      ))}
      {columns.length === 0 && fieldValues.map((field) => (
        <div key={field} className="object-column-row">
          <strong>{field}</strong>
          <span>field</span>
        </div>
      ))}
    </div>
  );
}

function KeyValueList({ values }: { values: Array<Record<string, unknown>> }) {
  if (values.length === 0) return <EmptyPreview label="No field, label, or tag details returned." />;
  return (
    <div className="database-kv-list">
      {values.slice(0, 24).map((value, index) => {
        const title = String(value.name || value.column || value.field || value.tag || value.label || `Item ${index + 1}`);
        const detail = objectEntries(value)
          .filter(([key]) => !['name', 'column', 'field', 'tag', 'label'].includes(key))
          .slice(0, 3)
          .map(([key, entryValue]) => `${key}: ${formatValue(entryValue)}`)
          .join(' / ');
        return (
          <div key={`${title}-${index}`} className="database-kv-row">
            <strong>{title}</strong>
            <span>{detail || 'available'}</span>
          </div>
        );
      })}
    </div>
  );
}

function SampleRows({ rows }: { rows: Array<Record<string, unknown>> }) {
  if (rows.length === 0) return <EmptyPreview label="No sample rows returned." />;
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).slice(0, 8);
  return (
    <div className="database-sample-wrap">
      <table className="sample-table database-sample-table">
        <thead>
          <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.slice(0, 8).map((row, index) => (
            <tr key={index}>
              {columns.map((column) => <td key={column}>{formatValue(row[column])}</td>)}
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

function collectDataProfileRows(preview: DatabasePreviewPayload | null): Array<Record<string, unknown>> {
  const metadata = preview?.metadata;
  if (!metadata || typeof metadata !== 'object') return [];
  const profile = (metadata as Record<string, unknown>).data_profile;
  if (!profile || typeof profile !== 'object' || Array.isArray(profile)) return [];
  const sources = (profile as Record<string, unknown>).sources;
  return Array.isArray(sources)
    ? sources.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object' && !Array.isArray(item)))
    : [];
}

function profileSeriesLabel(row: Record<string, unknown>) {
  const field = formatValue(row.field);
  const tags = row.tags && typeof row.tags === 'object' && !Array.isArray(row.tags)
    ? Object.entries(row.tags as Record<string, unknown>)
      .slice(0, 3)
      .map(([key, value]) => `${key}=${formatValue(value)}`)
      .join(', ')
    : '';
  return tags ? `${field} · ${tags}` : field;
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
  };
}

function normalizeFormPayload(value: DatabaseConfigInput, mode: FormMode): Partial<DatabaseConfigInput> {
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
  return payload;
}

function nullableString(value: string | null | undefined) {
  const trimmed = String(value || '').trim();
  return trimmed || null;
}

function summarizeFields(item: DatabasePreviewObject) {
  const columns = item.columns || [];
  if (columns.length > 0) {
    return columns.slice(0, 6).map((column) => (
      column.data_type ? `${column.name}: ${column.data_type}` : column.name
    ));
  }
  return (item.field_values || []).slice(0, 6);
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
