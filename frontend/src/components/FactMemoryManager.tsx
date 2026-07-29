import { AlertCircle, BrainCircuit, Code2, Database, FileText, RefreshCw } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { fetchFactMemory } from '../services/api';
import type { DatabaseResource, FactDefinition, FactMemoryResponse, FactRecipe } from '../types';

type Props = {
  databases: DatabaseResource[];
  selectedDatabaseId: string | null;
};

type MemoryState = {
  loading: boolean;
  error: string | null;
  data: FactMemoryResponse | null;
};

export function FactMemoryManager({ databases, selectedDatabaseId }: Props) {
  const [scope, setScope] = useState<'global' | 'database'>('global');
  const [reloadToken, setReloadToken] = useState(0);
  const [state, setState] = useState<MemoryState>({ loading: false, error: null, data: null });
  const databaseId = scope === 'database' ? selectedDatabaseId : null;
  const activeDatabase = databases.find((database) => database.id === selectedDatabaseId) || null;

  useEffect(() => {
    let cancelled = false;
    setState((current) => ({ loading: true, error: null, data: current.data }));
    fetchFactMemory(databaseId)
      .then((data) => {
        if (!cancelled) setState({ loading: false, error: null, data });
      })
      .catch((error) => {
        if (!cancelled) {
          setState({
            loading: false,
            error: error instanceof Error ? error.message : 'Unable to load fact memory.',
            data: null,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [databaseId, reloadToken]);

  const definitions = state.data?.memory.definitions || [];
  const recipes = state.data?.memory.recipes || [];
  const definitionsBySource = useMemo(() => groupBySource(definitions), [definitions]);

  return (
    <section className="fact-memory-manager" aria-label="Fact memory">
      <div className="fact-memory-header">
        <div>
          <span className="database-kicker">Long-term memory</span>
          <h2>Fact Memory</h2>
          <p>Fact definitions and recipes guide ReAct planning. Concrete numeric facts are regenerated from current evidence.</p>
        </div>
        <div className="fact-memory-actions">
          <div className="segmented-control" aria-label="Fact memory scope">
            <button type="button" className={scope === 'global' ? 'active' : ''} onClick={() => setScope('global')}>
              <BrainCircuit size={14} />
              <span>Global</span>
            </button>
            <button
              type="button"
              className={scope === 'database' ? 'active' : ''}
              disabled={!selectedDatabaseId}
              onClick={() => setScope('database')}
            >
              <Database size={14} />
              <span>Database</span>
            </button>
          </div>
          <button className="icon-text-button" type="button" disabled={state.loading} onClick={() => setReloadToken((value) => value + 1)}>
            <RefreshCw size={14} />
            <span>{state.loading ? 'Loading' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {scope === 'database' && !activeDatabase && (
        <div className="database-inline-error">
          <AlertCircle size={16} />
          <span>Select a database in chat or Database view to inspect scoped fact memory.</span>
        </div>
      )}
      {state.error && (
        <div className="database-inline-error">
          <AlertCircle size={16} />
          <span>{state.error}</span>
        </div>
      )}

      <div className="fact-memory-meta">
        <div>
          <strong>{definitions.length.toLocaleString()}</strong>
          <span>definitions</span>
        </div>
        <div>
          <strong>{recipes.length.toLocaleString()}</strong>
          <span>recipes</span>
        </div>
        <div>
          <strong>{scope === 'database' ? activeDatabase?.display_name || activeDatabase?.name || 'Database' : 'Global'}</strong>
          <span>scope</span>
        </div>
        <div>
          <strong>{state.data?.memory.updated_at || 'default'}</strong>
          <span>updated</span>
        </div>
      </div>

      <div className="fact-memory-grid">
        <section className="database-section">
          <SectionTitle icon={BrainCircuit} title="Definitions" count={definitions.length} />
          {definitions.length > 0 ? (
            <div className="fact-definition-list">
              {definitions.map((definition) => (
                <FactDefinitionCard key={`${definition.scope || 'global'}-${definition.fact_type}-${definition.source || 'source'}`} definition={definition} />
              ))}
            </div>
          ) : (
            <EmptyState label="No fact definitions stored." />
          )}
        </section>

        <section className="database-section">
          <SectionTitle icon={Code2} title="Recipes" count={recipes.length} />
          {recipes.length > 0 ? (
            <div className="fact-recipe-list">
              {recipes.map((recipe) => (
                <FactRecipeCard key={recipe.recipe_id} recipe={recipe} />
              ))}
            </div>
          ) : (
            <EmptyState label="No fact recipes stored." />
          )}
        </section>
      </div>

      <section className="database-section fact-memory-source-section">
        <SectionTitle icon={FileText} title="Sources" count={definitionsBySource.length} />
        <div className="chip-list compact">
          {definitionsBySource.map((item) => (
            <span key={item.source}>{item.source}: {item.count}</span>
          ))}
        </div>
        {state.data?.memory.storage_path && <p className="sample-note">{state.data.memory.storage_path}</p>}
      </section>
    </section>
  );
}

function SectionTitle({ icon: Icon, title, count }: { icon: typeof BrainCircuit; title: string; count: number }) {
  return (
    <div className="database-section-title">
      <span><Icon size={15} /> {title}</span>
      <strong>{count}</strong>
    </div>
  );
}

function FactDefinitionCard({ definition }: { definition: FactDefinition }) {
  return (
    <article className="fact-memory-card">
      <div className="fact-memory-card-title">
        <strong>{definition.fact_type}</strong>
        <span>{definition.preferred_tool || 'tool'}</span>
      </div>
      <p>{definition.description}</p>
      <div className="chip-list compact">
        {(definition.required_evidence || []).map((item) => <span key={item}>{item}</span>)}
      </div>
      {definition.report_guidance && <small>{definition.report_guidance}</small>}
    </article>
  );
}

function FactRecipeCard({ recipe }: { recipe: FactRecipe }) {
  return (
    <article className="fact-memory-card">
      <div className="fact-memory-card-title">
        <strong>{recipe.name}</strong>
        <span>{recipe.preferred_tool}</span>
      </div>
      <p>{recipe.fact_type}</p>
      <pre className="debug-json">{JSON.stringify(recipe.expected_result_schema || {}, null, 2)}</pre>
    </article>
  );
}

function EmptyState({ label }: { label: string }) {
  return (
    <div className="database-empty-state">
      <AlertCircle size={16} />
      <span>{label}</span>
    </div>
  );
}

function groupBySource(definitions: FactDefinition[]) {
  const counts = new Map<string, number>();
  definitions.forEach((definition) => {
    const source = definition.source || 'unknown';
    counts.set(source, (counts.get(source) || 0) + 1);
  });
  return Array.from(counts.entries()).map(([source, count]) => ({ source, count }));
}
