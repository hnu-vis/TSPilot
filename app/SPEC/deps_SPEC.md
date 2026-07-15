# app/deps.py SPEC

## Purpose

Define dependency factories for the API layer.

## Responsibilities

- provide runtime dependencies
- provide model client wiring
- provide tool registry wiring
- provide storage / config accessors

## Must not do

- execute request-specific business logic
- hold mutable request state
