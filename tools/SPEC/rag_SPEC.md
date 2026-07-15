# tools/rag.py SPEC

## Purpose

Retrieve external knowledge that is not present in database evidence.

## Input

- `query: str`
- optional `database_context: DatabaseContext | null`
- optional `database_evidence: DatabaseEvidence | null`
- optional `filters: dict | null`

## Output

- retrieved passages
- summarized knowledge context

## Responsibilities

- retrieve business definitions
- support explanation and answer formatting

## Role

- extension tool, not a first-path core tool

## Must not do

- query databases directly
- invent final narrative
