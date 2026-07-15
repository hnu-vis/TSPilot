# tools/base.py SPEC

## Purpose

Define the shared tool interface contract.

## Responsibilities

- normalize tool input and output boundaries
- keep outer actions typed and comparable

## Expected shape

Every tool spec should define:

- purpose
- input contract
- output contract
- reads
- writes
- internal pipeline
- forbidden responsibilities

## Must not do

- execute business logic
- choose the next action
