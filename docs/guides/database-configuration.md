# Database Configuration Guide

TSPilot can manage database connections from the Web interface or from YAML
files in the local workspace. The Web interface is the recommended path for
normal use.

## Configure from the Web interface

1. Start the backend and frontend services.
2. Open **Database Management**.
3. Add a database and select its database type.
4. Enter the host, port, database-specific connection fields, and credentials.
5. Test the connection before selecting it in a chat.

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

The repository includes credential-free or local demonstration configurations
under:

- `configs/databases/influxdb/`
- `configs/databases/prometheus/`

## Configuration behavior

- `name` identifies the database connection in TSPilot.
- `type` selects the connector implementation.
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
