# core/visualization materialization SPEC

## Purpose

Project request-scoped artifacts into semantic views, then materialize an
LLM-authored visual-verification plan into renderer-independent V3 payloads.

## Contract

- Verified Key Insights define the conclusions to inspect; complete Evidence,
  Derived Evidence, Forecast, and Anomaly artifacts supply context.
- The caller supplies target Insight refs. The catalog parses each Insight's
  evidence lineage and resolves its related complete data views automatically;
  callers do not repeat contextual artifact refs.
- The planning LLM sees target-Insight semantic contracts and directly held
  scalar/located values plus reference-backed source contracts (shape, schema,
  count, coverage, lineage, and record paths). Large item collections stay
  reference-backed.
  Referenced data records are not copied into planning prompts.
- Semantic projection selects and renames existing values only. It never
  calculates a business result.
- Materialization reads authoritative request-scoped artifacts, applies only
  selection transforms, and preserves source lineage and stable bindings.
- Insight marks stay distinct from contextual series and retain Insight/item
  locators.
- The semantic validator independently checks required roles, verified target
  IDs, grounded encodings, and graphical data sufficiency.
- After those executable invariants pass, the candidate is published. The
  Candidate Semantic Audit and screenshot Render Audit are currently disabled
  in the publication path and cannot return `unavailable` or request a rewrite.
- Every line or area layer contains at least two grounded points per plotted
  series. A one-row scalar, method receipt, or boundary receipt is not a line.
- Invalid candidates are returned to the LLM planning repair loop. There is no
  heuristic or deterministic chart fallback.
- Materialization never queries a database, recomputes an Insight, or invents
  absent values.

## Output

`VisualizationPayload` schema version 3 contains explicit datasets, layers,
bindings, verification metadata, accessibility data, presentation options, and
source refs. The frontend maps this renderer-independent payload to production
ECharts options.
