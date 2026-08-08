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
- `fact_requests`: semantic Fact contracts. Each request has a stable `fact_key`;
  composite requests list parent Fact keys in `derived_from`.

## Outputs

- `AnalysisResult` payload with stable `analysis_id`, input evidence id, row count,
  summary, metrics/details, diagnostics, and code type.
- `result.facts` contains structured facts satisfying `fact_requests`, including
  `fact_key`, `value`, `statement`, `derived_from`, and `calculation_trace`.
- Verified parent Facts are available to generated code as `input_facts` and
  `fact_by_key`.

## Rules

- Must not query databases directly.
- Must not assemble final answers.
- Must only analyze evidence already present in request state or passed in input.
- Must preserve executable code only when code is actually supplied.
- Template-driven analysis should keep outputs scoped to requested facts and
  supporting values needed for those facts.
- A composite Fact is verified only when all parents are verified and its
  calculation trace is present.
