# `tools/code_interpreter.py` SPEC

## Purpose

Calculate exactly the requested analytical Key Insight values over grounded Evidence.

## Inputs

- `database_evidence`: an existing database Evidence object or reference.
- `analysis_goal`: natural-language computation goal.
- `insight_requests`: non-empty semantic contracts; output keys must match them exactly.
- `code`: optional Python. When absent, the internal LLM generates computation-only code.
- `constraints`: sandbox execution constraints such as timeout.

## Outputs

- `computed_insights`: immutable values/items, calculation traces, unavailable reasons,
  and optional derived Evidence references keyed by `insight_key`.
- `derived_evidence`: independently addressable complete tables or series needed to
  verify or reuse a calculation; absent for ordinary scalar/collection Insights.
- `produced_insights`: formal Key Insights created by the independent LLM Insight Binder.
- Execution identity, provenance, runtime diagnostics, and code hash.

## Rules

- Python must not write statements, Key Insight semantics, presentation objects,
  Data Views, chart roles, final-answer prose, or repair policy.
- The LLM Binder may add statements and item labels but must never calculate or
  modify Python-produced values.
- Output keys must preserve request order and exactly match `insight_requests`.
- Impossible calculations return `unavailable_reason`; placeholder values are forbidden.
- Complete derived datasets are registered as `derived_evidence:*` artifacts rather
  than embedded into an Insight or visualization contract.
- Authoritative Anomaly Artifacts must be consumed directly and cited in the affected
  calculation trace; Code Interpreter must not repeat anomaly detection or its audit schema.
- Must not query databases, assemble final answers, choose charts, or use deterministic
  business fallbacks.
