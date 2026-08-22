# Database Configuration Guide

TSPilot can manage database connections from the Web interface or from YAML
files in the local workspace. The Web interface is the recommended path for
normal use.

TSPilot currently exposes 24 database connection types. The shared catalog in
`configs/database_catalog.json` defines the supported types, their default
connection values, and any database-specific Web form fields. The backend and
frontend validate against the same catalog.

## Configure from the Web interface

1. Start the backend and frontend services.
2. Open **Database Management**.
3. Add a database and select its database type.
4. Enter the host, port, database-specific connection fields, and credentials.
5. Test the connection before selecting it in a chat.

The form fills the normal default host and port for the selected database. A
**Database-specific settings** section appears only when that connector needs
additional information, such as an InfluxDB organization, bucket, and token.
Secret fields are never returned to the browser after they are saved; leaving a
secret blank while editing keeps the stored value.

Saved credentials remain in the local workspace and are not read from a project
`.env` file. Do not commit real credentials to the repository.

## Configure with YAML

Project configuration files live under `configs/databases/`. The loader scans
that directory recursively for `.yaml` and `.yml` files, so configurations may
be grouped by database type.

Common fields:

```yaml
type: prometheus
name: prometheus-local
host: localhost
port: 9090
ssl_enabled: false
```

Database-specific fields can be added alongside the common fields. For example,
an InfluxDB 2 connection may include:

```yaml
type: influxdb
version: 2
name: influxdb-production
host: localhost
port: 8086
org: my-org
bucket: metrics
token: replace-with-a-local-token
ssl_enabled: false
```

## Supported connection types

| Database | YAML `type` | Default port | Additional settings in Web UI |
|---|---|---:|---|
| InfluxDB 2 | `influxdb` | 8086 | Organization, bucket, API token |
| InfluxDB 3 | `influxdb3` | 8181 | API token when authentication is enabled |
| kdb+ | `kdb` | 5000 | — |
| Prometheus | `prometheus` | 9090 | API path, environment proxy |
| TimescaleDB | `timescaledb` | 5432 | — |
| DolphinDB | `dolphindb` | 8848 | — |
| Apache Druid | `druid` | 8888 | — |
| QuestDB | `questdb` | 8812 | Optional HTTP mode |
| TDengine | `tdengine` | 6041 | — |
| Apache IoTDB | `iotdb` | 6667 | — |
| VictoriaMetrics | `victoriametrics` | 8428 | API path |
| GridDB | `griddb` | 8081 | Cluster |
| Arc | `arc` | 8000 | Bearer token when authentication is enabled |
| M3DB | `m3db` | 7201 | API path |
| CrateDB | `cratedb` | 4200 | — |
| CnosDB | `cnosdb` | 8902 | — |
| ArcadeDB | `arcadedb` | 2480 | — |
| GreptimeDB | `greptimedb` | 4000 | — |
| IBM Db2 | `db2` | 50000 | Schema when required |
| Riak TS | `riak_ts` | 8087 | — |
| BangDB | `bangdb` | 10101 | — |
| Machbase Neo | `machbase` | 5654 | — |
| OpenMLDB | `openmldb` | 9080 | Execution mode |
| openGemini | `opengemini` | 8086 | — |

All connectors use the common host, port, database, username, password, and
SSL fields where applicable. A listed connector means TSPilot provides the
configuration and query integration; the database server and its client/runtime
dependencies must still be installed separately.

The repository includes credential-free or local demonstration configurations
under:

- `configs/databases/influxdb/`
- `configs/databases/prometheus/`

## Configuration behavior

- `name` identifies the database connection in TSPilot.
- `type` selects the connector implementation.
- Only the 24 types in the shared catalog are accepted by the product API and connector factory.
- Host, port, authentication, SSL, and query options vary by database.
- Project YAML files are synchronized into the local runtime configuration.
- Removing a project YAML file removes its synchronized project connection from the runtime cache.
- A connector being available in TSPilot does not install or start the database service itself.

## Connection troubleshooting

- Confirm the target database is running and reachable from the backend host.
- Verify the connector-specific port and authentication method.
- Test the connection in **Database Management** and review the returned error.
- For containers or remote databases, make sure the configured host is reachable from the TSPilot process, not only from the browser.
- Keep real passwords and tokens in local workspace configuration rather than tracked example files.
