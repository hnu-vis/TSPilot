# core/database/schema.py SPEC

## Purpose

Provide deterministic helpers to inspect schema and catalog metadata.

## Responsibilities

- inspect tables, measurements, fields, labels, and time columns
- support query planning and discovery

## Must not do

- execute arbitrary analysis
- write final narrative
