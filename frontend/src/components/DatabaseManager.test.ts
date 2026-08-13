import { describe, expect, it } from 'vitest';
import { DATABASE_CATALOG, DATABASE_TYPES } from '../databaseCatalog';
import { mergeContextRecords } from './DatabaseManager';

describe('database catalog', () => {
  it('provides one real brand asset for every supported connector', () => {
    expect(DATABASE_CATALOG).toHaveLength(32);
    expect(new Set(DATABASE_TYPES).size).toBe(DATABASE_TYPES.length);

    for (const database of DATABASE_CATALOG) {
      expect(database.label.length).toBeGreaterThan(1);
      expect(database.logoUrl).toMatch(/^\/database-logos\//);
      expect(database.sourceUrl).toMatch(/^https:\/\//);
      expect(database).not.toHaveProperty('mark');
      expect(database).not.toHaveProperty('hue');
    }
  });
});

describe('mergeContextRecords', () => {
  it('merges schema and value metadata for the same field', () => {
    const merged = mergeContextRecords([
      { table: 'coindesk', name: 'code', data_type: 'string', nullable: true },
      { table: 'coindesk', name: 'code', values: ['EUR', 'GBP', 'USD'] },
    ]);

    expect(merged).toEqual([{
      table: 'coindesk',
      name: 'code',
      data_type: 'string',
      nullable: true,
      values: ['EUR', 'GBP', 'USD'],
    }]);
  });

  it('combines distinct values without duplicating shared attributes', () => {
    const merged = mergeContextRecords([
      { table: 'metrics', name: 'region', values: ['eu', 'us'] },
      { table: 'metrics', name: 'region', values: ['us', 'apac'] },
    ]);

    expect(merged[0]).toMatchObject({
      table: 'metrics',
      name: 'region',
      values: ['eu', 'us', 'apac'],
    });
  });
});
