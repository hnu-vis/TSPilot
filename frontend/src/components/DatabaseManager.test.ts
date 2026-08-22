import { describe, expect, it } from 'vitest';
import { DATABASE_CATALOG, DATABASE_TYPES } from '../databaseCatalog';
import { databaseDefaultForm, mergeContextRecords, normalizeFormPayload } from './DatabaseManager';

describe('database catalog', () => {
  it('provides one real brand asset for every supported connector', () => {
    expect(DATABASE_CATALOG).toHaveLength(24);
    expect(new Set(DATABASE_TYPES).size).toBe(DATABASE_TYPES.length);
    expect(DATABASE_TYPES).toEqual([
      'influxdb',
      'influxdb3',
      'kdb',
      'prometheus',
      'timescaledb',
      'dolphindb',
      'druid',
      'questdb',
      'tdengine',
      'iotdb',
      'victoriametrics',
      'griddb',
      'arc',
      'm3db',
      'cratedb',
      'cnosdb',
      'arcadedb',
      'greptimedb',
      'db2',
      'riak_ts',
      'bangdb',
      'machbase',
      'openmldb',
      'opengemini',
    ]);

    for (const database of DATABASE_CATALOG) {
      expect(database.label.length).toBeGreaterThan(1);
      expect(database.logoUrl).toMatch(/^\/database-logos\//);
      expect(database.sourceUrl).toMatch(/^https:\/\//);
      expect(database).not.toHaveProperty('mark');
      expect(database).not.toHaveProperty('hue');
    }
  });

  it('uses the shared defaults and submits InfluxDB-specific fields', () => {
    const form = databaseDefaultForm('influxdb');
    expect(form).toMatchObject({
      type: 'influxdb',
      host: 'localhost',
      port: 8086,
      extra: { version: '2' },
    });

    const payload = normalizeFormPayload({
      ...form,
      name: 'metrics',
      extra: { version: '2', org: 'acme', bucket: 'telemetry', token: 'secret' },
    }, 'create');
    expect(payload.extra).toEqual({
      version: '2',
      org: 'acme',
      bucket: 'telemetry',
      token: 'secret',
    });
  });

  it('does not erase a stored secret when an edit leaves it blank', () => {
    const payload = normalizeFormPayload({
      ...databaseDefaultForm('influxdb3'),
      name: 'cloud',
      extra: { token: '' },
    }, 'edit');
    expect(payload.extra).toBeUndefined();
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
