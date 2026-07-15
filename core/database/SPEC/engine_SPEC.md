# core/database/engine.py SPEC

## Purpose

Provide deterministic database execution helpers for `query_database`.

## Responsibilities

- execute database queries
- return backend results in a normalized shape
- expose adapter hooks for dialect differences

## Must not do

- decide user intent
- narrate facts
- assemble final answers
