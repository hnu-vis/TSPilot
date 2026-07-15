# tools/format_answer.py SPEC

## Purpose

Assemble the final user-facing answer from verified outputs.

## Input

- `summary_goal: str`
- optional `include_fact_ids: list[str]`
- optional `include_visualization_ids: list[str]`
- optional `section_plan: list[str]`

## Output

- `FinalAnswer`

## Reads

- request state
- verified facts
- latest evidence and analysis payloads
- visualization payloads

## Writes

- final answer draft
- final answer object

## Internal pipeline

1. collect verified facts and grounded evidence
2. collect any forecast or anomaly output
3. collect visualization payloads already grounded in evidence
4. organize sections and references
5. emit final answer

## Contract notes

- do not invent new facts
- do not query databases
- do not re-run analysis
- preserve visualization ids and evidence ids in references
- the runtime provides state access; the model should not pass full state snapshots back into the tool

## Must not do

- hidden reasoning beyond assembly
- business logic execution
