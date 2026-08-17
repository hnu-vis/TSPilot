import { describe, expect, it } from 'vitest';
import type { DatabaseResource } from '../types';
import { durationInputToSeconds, orderInsightMemoryDatabases, secondsToDurationInput } from './InsightMemoryManager';

function database(id: string, recipeCount: number, updatedAt?: string): DatabaseResource {
  return {
    id,
    name: id,
    type: 'influxdb2',
    insight_memory_summary: {
      definition_count: 0,
      recipe_count: recipeCount,
      card_count: recipeCount,
      updated_at: updatedAt,
    },
  };
}

describe('orderInsightMemoryDatabases', () => {
  it('keeps the selected database first and orders the rest by recipe count', () => {
    const databases = [database('empty', 0), database('rich', 8), database('selected', 1)];

    expect(orderInsightMemoryDatabases(databases, 'selected').map((item) => item.id))
      .toEqual(['selected', 'rich', 'empty']);
  });

  it('uses the latest memory update for equal recipe counts without mutating input order', () => {
    const databases = [
      database('older', 3, '2026-08-12T10:00:00Z'),
      database('newer', 3, '2026-08-13T10:00:00Z'),
    ];

    expect(orderInsightMemoryDatabases(databases, null).map((item) => item.id)).toEqual(['newer', 'older']);
    expect(databases.map((item) => item.id)).toEqual(['older', 'newer']);
  });
});

describe('Insight learning schedule duration conversion', () => {
  it('uses the clearest exact unit for the persisted duration', () => {
    expect(secondsToDurationInput(7200)).toEqual({ amount: 2, unit: 'hours' });
    expect(secondsToDurationInput(600)).toEqual({ amount: 10, unit: 'minutes' });
    expect(secondsToDurationInput(45)).toEqual({ amount: 45, unit: 'seconds' });
  });

  it('converts editable values to seconds and rejects invalid ranges', () => {
    expect(durationInputToSeconds('1.5', 'hours')).toBe(5400);
    expect(durationInputToSeconds('0', 'minutes')).toBeNull();
    expect(durationInputToSeconds('169', 'hours')).toBeNull();
  });
});
