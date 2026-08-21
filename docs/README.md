# TSPilot Documentation

This directory contains the small set of public documents that define TSPilot's
current architecture, integration contracts, and setup guidance.

## Architecture

- [System architecture](architecture/system.md) — request flow, tool boundaries, artifacts, and current product scope.

## Contracts

- [Chat API contract](contracts/api.md) — stable chat request and response boundary.
- [LineChart Visualization V4 contract](contracts/visualization.md) — visualization payload, semantic validation, persistence, and hydration.

## Guides

- [Database configuration guide](guides/database-configuration.md) — configure and test database connections from the Web interface or YAML.

Implementation notes, per-file specifications, and dated test reports are
development material rather than public product documentation. They are kept
locally under the ignored `develop_docs/` directory.
