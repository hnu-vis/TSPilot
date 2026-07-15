# tools/skill.py SPEC

## Purpose

Invoke a predefined domain workflow or packaged capability.

## Input

- `skill_name: str`
- `task_context: dict`
- optional `parameters: dict`

## Output

- skill result payload
- optional downstream instructions

## Responsibilities

- execute a named packaged workflow
- return structured results that can feed later tools

## Role

- extension tool, not a first-path core tool

## Must not do

- query databases directly
- invent final narrative
