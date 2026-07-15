# tools/todowrite.py SPEC

## Purpose

Maintain a structured todo list for multi-step requests.

## Input

- `todos: list[dict]`
- optional `message: str`
- optional `current_intent: str`
- optional `requested_fact_types: list[str]`
- optional `focus: str`
- optional `evidence_summary: dict | str | null`

Each todo item should minimally support:

- `content`
- `status`
- `priority`

## Output

- normalized todo state
- todo summary

## Responsibilities

- replace the current todo list
- enforce exactly one `in_progress` item at a time
- emit a progress-friendly observation

## Must not do

- query databases
- infer facts
- write final answer text
