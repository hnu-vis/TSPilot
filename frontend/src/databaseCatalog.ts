export type DatabaseCatalogEntry = {
  type: string;
  label: string;
  logoUrl: string;
  sourceUrl: string;
};

// Brand artwork comes from the product's official site or official GitHub
// organization. Keep identity metadata here so every database picker uses the
// same verified asset instead of generating an abbreviation in the UI.
export const DATABASE_CATALOG: readonly DatabaseCatalogEntry[] = [
  { type: 'influxdb', label: 'InfluxDB', logoUrl: '/database-logos/influxdb.svg', sourceUrl: 'https://www.influxdata.com/' },
  { type: 'timescaledb', label: 'TimescaleDB', logoUrl: '/database-logos/timescaledb.png', sourceUrl: 'https://github.com/timescale/timescaledb' },
  { type: 'prometheus', label: 'Prometheus', logoUrl: '/database-logos/prometheus.svg', sourceUrl: 'https://prometheus.io/' },
  { type: 'iotdb', label: 'Apache IoTDB', logoUrl: '/database-logos/iotdb.svg', sourceUrl: 'https://iotdb.apache.org/' },
  { type: 'questdb', label: 'QuestDB', logoUrl: '/database-logos/questdb.png', sourceUrl: 'https://github.com/questdb/questdb' },
  { type: 'clickhouse', label: 'ClickHouse', logoUrl: '/database-logos/clickhouse.png', sourceUrl: 'https://github.com/ClickHouse/ClickHouse' },
  { type: 'openmldb', label: 'OpenMLDB', logoUrl: '/database-logos/openmldb.jpg', sourceUrl: 'https://github.com/4paradigm/OpenMLDB' },
  { type: 'victoriametrics', label: 'VictoriaMetrics', logoUrl: '/database-logos/victoriametrics.png', sourceUrl: 'https://github.com/VictoriaMetrics/VictoriaMetrics' },
  { type: 'm3db', label: 'M3DB', logoUrl: '/database-logos/m3db.png', sourceUrl: 'https://github.com/m3db/m3' },
  { type: 'greptimedb', label: 'GreptimeDB', logoUrl: '/database-logos/greptimedb.png', sourceUrl: 'https://github.com/GreptimeTeam/greptimedb' },
  { type: 'tdengine', label: 'TDengine', logoUrl: '/database-logos/tdengine.png', sourceUrl: 'https://github.com/taosdata/TDengine' },
  { type: 'cnosdb', label: 'CnosDB', logoUrl: '/database-logos/cnosdb.png', sourceUrl: 'https://www.cnosdb.com/' },
  { type: 'arcadedb', label: 'ArcadeDB', logoUrl: '/database-logos/arcadedb.png', sourceUrl: 'https://arcadedb.com/' },
  { type: 'cratedb', label: 'CrateDB', logoUrl: '/database-logos/cratedb.png', sourceUrl: 'https://github.com/crate/crate' },
  { type: 'druid', label: 'Apache Druid', logoUrl: '/database-logos/druid.png', sourceUrl: 'https://druid.apache.org/' },
  { type: 'influxdb3', label: 'InfluxDB 3', logoUrl: '/database-logos/influxdb.svg', sourceUrl: 'https://www.influxdata.com/products/influxdb3/' },
  { type: 'griddb', label: 'GridDB', logoUrl: '/database-logos/griddb.png', sourceUrl: 'https://github.com/griddb/griddb' },
  { type: 'machbase', label: 'Machbase', logoUrl: '/database-logos/machbase.png', sourceUrl: 'https://github.com/machbase/neo-server' },
  { type: 'nsdb', label: 'NSDb', logoUrl: '/database-logos/nsdb.png', sourceUrl: 'https://github.com/radicalbit/NSDb' },
  { type: 'axibase', label: 'Axibase TSDB', logoUrl: '/database-logos/axibase.png', sourceUrl: 'https://github.com/axibase/atsd' },
  { type: 'opengemini', label: 'openGemini', logoUrl: '/database-logos/opengemini.png', sourceUrl: 'https://github.com/openGemini/openGemini' },
  { type: 'db2', label: 'IBM Db2', logoUrl: '/database-logos/db2.jpg', sourceUrl: 'https://www.ibm.com/products/db2' },
  { type: 'timestream', label: 'Amazon Timestream', logoUrl: '/database-logos/timestream.png', sourceUrl: 'https://aws.amazon.com/timestream/' },
  { type: 'riak_ts', label: 'Riak TS', logoUrl: '/database-logos/riak-ts.png', sourceUrl: 'https://github.com/basho/riak_ts' },
  { type: 'dolphindb', label: 'DolphinDB', logoUrl: '/database-logos/dolphindb.png', sourceUrl: 'https://github.com/dolphindb/DolphinDB' },
  { type: 'kdb', label: 'kdb+', logoUrl: '/database-logos/kdb.png', sourceUrl: 'https://kx.com/products/kdb/' },
  { type: 'raimadb', label: 'RaimaDB', logoUrl: '/database-logos/raimadb.png', sourceUrl: 'https://raima.com/product-overview/' },
  { type: 'extremedb', label: 'eXtremeDB', logoUrl: '/database-logos/extremedb.png', sourceUrl: 'https://www.mcobject.com/extremedb/' },
  { type: 'ittiadb', label: 'ITTIA DB', logoUrl: '/database-logos/ittiadb.png', sourceUrl: 'https://www.ittia.com/products/ittia-db' },
  { type: 'irondb', label: 'IRONdb', logoUrl: '/database-logos/irondb.png', sourceUrl: 'https://www.circonus.com/products/irondb/' },
  { type: 'bangdb', label: 'BangDB', logoUrl: '/database-logos/bangdb.png', sourceUrl: 'https://bangdb.com/' },
  { type: 'arc', label: 'Arc', logoUrl: '/database-logos/arc.svg', sourceUrl: 'https://www.basekick.net/' },
] as const;

export const DATABASE_TYPES = DATABASE_CATALOG.map(({ type }) => type);

const DATABASE_CATALOG_BY_TYPE = new Map(DATABASE_CATALOG.map((database) => [database.type, database]));

export function databaseTypeLabel(type: string) {
  return DATABASE_CATALOG_BY_TYPE.get(type)?.label || type;
}
