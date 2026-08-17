# `core/analysis/python_runner.py` SPEC

## Purpose

Provide the stable result contract and validation errors for Python row analysis.

## Responsibilities

- Execute row-oriented analysis helpers used by tests and sandbox-compatible flows.
- Validate that analysis code produces computation-only `computed_insights` and
  optional `derived_evidence` collections.
- Raise `AnalysisCodeError` for invalid imports, malformed results, unsafe behavior,
  or execution failures.

## Boundaries

- Must not access request state.
- Must not decide user intent.
- Must not register insights or assemble presentation output.
- Must keep the result shape compatible with `tools/code_interpreter.py`.
