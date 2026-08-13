import { describe, expect, it } from 'vitest';
import type { DatabaseResource } from '../types';
import { orderFactMemoryDatabases } from './FactMemoryManager';

function database(id: string, recipeCount: number, updatedAt?: string): DatabaseResource {
  return {
    id,
    name: id,
    type: 'influxdb2',
    fact_memory_summary: {
      definition_count: 0,
      recipe_count: recipeCount,
      card_count: recipeCount,
      updated_at: updatedAt,
    },
  };
}

describe('orderFactMemoryDatabases', () => {
  it('keeps the selected database first and orders the rest by recipe count', () => {
    const databases = [database('empty', 0), database('rich', 8), database('selected', 1)];

    expect(orderFactMemoryDatabases(databases, 'selected').map((item) => item.id))
      .toEqual(['selected', 'rich', 'empty']);
  });

  it('uses the latest memory update for equal recipe counts without mutating input order', () => {
    const databases = [
      database('older', 3, '2026-08-12T10:00:00Z'),
      database('newer', 3, '2026-08-13T10:00:00Z'),
    ];

    expect(orderFactMemoryDatabases(databases, null).map((item) => item.id)).toEqual(['newer', 'older']);
    expect(databases.map((item) => item.id)).toEqual(['older', 'newer']);
  });
});
