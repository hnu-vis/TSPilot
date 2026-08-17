# core/database/engine.py SPEC

## Purpose

Provide single-attempt connector execution and backend-result normalization for `sql_query`.

## Responsibilities

- execute database queries
- return backend results in a normalized shape
- preserve connector errors for outer ReAct recovery

## Must not do

- decide user intent
- generate, repair, or retry queries
- narrate insights
- assemble final answers
