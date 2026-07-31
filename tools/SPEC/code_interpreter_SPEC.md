# `tools/code_interpreter.py` SPEC

## Purpose

Run request-scoped analysis over database evidence and return a normalized
analysis artifact.

## Inputs

- `database_evidence`: evidence object or reference, defaulting to latest evidence
  from request state when available.
- `analysis_goal`: natural-language analysis goal.
- `analysis_request`: structured request for template-driven analysis.
- `code`: optional Python code for sandbox execution.
- `required_outputs`: output labels requested by the current task.
- `expected_result_schema`: optional validation contract.
- `constraints`: execution constraints such as timeout.
- `fact_requests`: requested facts to register from the tool output.

## Outputs

- `AnalysisResult` payload with stable `analysis_id`, input evidence id, row count,
  summary, metrics/details, diagnostics, and code type.
- Data facts are registered by the runtime from tool-produced requested outputs.

## Rules

- Must not query databases directly.
- Must not assemble final answers.
- Must only analyze evidence already present in request state or passed in input.
- Must preserve executable code only when code is actually supplied.
- Template-driven analysis should keep outputs scoped to requested facts and
  supporting values needed for those facts.
