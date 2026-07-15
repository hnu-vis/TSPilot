# core/database/repair.py SPEC

## Purpose

Provide deterministic query repair helpers.

## Responsibilities

- inspect query errors
- suggest or apply safe repair steps
- retry only when the repair is safe and useful

## Must not do

- invent new facts
- choose user-facing analysis
- create final answer text
