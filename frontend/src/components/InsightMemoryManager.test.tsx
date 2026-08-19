import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { DatabaseResource, MemoryDetail } from '../types';
import {
  compactMemoryContract,
  durationInputToSeconds,
  MemoryDetailCard,
  orderInsightMemoryDatabases,
  secondsToDurationInput,
} from './InsightMemoryManager';

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

describe('compactMemoryContract', () => {
  it('removes empty optional fields recursively while preserving meaningful false and zero values', () => {
    expect(compactMemoryContract({
      insight_key: 'turning_window',
      subject: null,
      dimensions: {},
      requirements: {
        include_window: true,
        expected_count: 0,
        filters: {},
      },
      derived_from: ['max_time', '', null],
      selection: {},
      enabled: false,
    })).toEqual({
      insight_key: 'turning_window',
      requirements: {
        include_window: true,
        expected_count: 0,
      },
      derived_from: ['max_time'],
      enabled: false,
    });
  });
});

describe('MemoryDetailCard', () => {
  it('shows tool and calculation method while omitting empty raw-contract fields', () => {
    const detail: MemoryDetail = {
      id: 'recipe.code_interpreter.analysis.turning_window',
      card: {
        id: 'recipe.code_interpreter.analysis.turning_window',
        kind: 'insight_recipe',
        title: 'peak turning segment',
        description: 'Identify a peak and its surrounding reversal window.',
        tags: ['analysis', 'code_interpreter', 'bitcoin'],
      },
      preferred_tool: 'code_interpreter',
      calculation_trace: {
        method: 'Scan local peak windows and retain rising-prefix/falling-suffix reversals.',
      },
      insight_request: {
        name: 'peak turning segment',
        insight_key: 'turning_window',
        insight_type: 'analysis',
        subject: null,
        dimensions: {},
        requirements: { include_window: true },
        derived_from: ['max_time', 'max_value'],
        selection: {},
      },
    };

    const markup = renderToStaticMarkup(<MemoryDetailCard detail={detail} />);

    expect(markup).toContain('Tool');
    expect(markup).toContain('Code interpreter');
    expect(markup).toContain('Method');
    expect(markup).toContain('Scan local peak windows and retain rising-prefix/falling-suffix reversals.');
    expect(markup).not.toContain('&quot;subject&quot;: null');
    expect(markup).not.toContain('&quot;dimensions&quot;: {}');
    expect(markup).not.toContain('&quot;selection&quot;: {}');
  });
});
