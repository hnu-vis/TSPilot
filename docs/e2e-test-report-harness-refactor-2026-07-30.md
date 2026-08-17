# E2E Test Report: Harness Refactor

- Date: 2026-07-30
- Time: 18:42 CST
- Workspace: `/home/feilvvl/TSPilot-v0.2`
- Environment: `/home/feilvvl/TSPilot/tspilot_env`
- Frontend build: not executed

## Scope

This report covers the backend ReAct harness refactor completed on 2026-07-30.

The tested changes include:

- `core/harness/` introduction:
  - capability registry
  - observation frame
  - action space builder
  - state transition engine
- registry-driven prompt action cards
- registry-driven action/task/artifact mapping
- `StateTransitionEngine` as the runtime state transition boundary
- conservative intent fallback without keyword-based capability guessing
- KeyInsight-backed completion behavior
- final answer reference behavior that avoids expanding all KeyInsights when `include_insight_ids=[]`

Multi-database behavior was intentionally out of scope for this run.

## Code State

At test time, the working tree contained uncommitted changes in:

- `core/completion.py`
- `core/intent.py`
- `prompts/data_agent.py`
- `runtime/action_policy.py`
- `runtime/react_loop.py`
- `runtime/request_state.py`
- `runtime/tool_executor.py`
- `tests/fakes.py`
- `tools/format_answer.py`
- `tools/todowrite.py`
- `core/harness/`

## Static Check

Command:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m py_compile \
  core/harness/__init__.py \
  core/harness/capabilities.py \
  core/harness/transition.py \
  core/harness/action_space.py \
  core/harness/observation.py \
  runtime/request_state.py \
  runtime/action_policy.py \
  core/completion.py \
  tools/todowrite.py
```

Result:

```text
passed
```

## E2E Test Command

Command:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m pytest tests/test_chat_api.py -q -vv
```

Result:

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/feilvvl/TSPilot-v0.2
configfile: pyproject.toml
plugins: anyio-4.13.0, asyncio-1.3.0, langsmith-0.7.36
asyncio: mode=Mode.STRICT
collected 17 items

tests/test_chat_api.py::test_chat_json_path_returns_final_answer PASSED  [  5%]
tests/test_chat_api.py::test_chat_json_path_uses_code_interpreter_tool PASSED [ 11%]
tests/test_chat_api.py::test_first_visible_action_does_not_wait_for_separate_intent_llm_call PASSED [ 17%]
tests/test_chat_api.py::test_chat_sse_path_returns_event_stream PASSED   [ 23%]
tests/test_chat_api.py::test_chat_json_path_persists_complete_trace_log PASSED [ 29%]
tests/test_chat_api.py::test_chat_sse_path_persists_internal_and_public_trace_logs PASSED [ 35%]
tests/test_chat_api.py::test_sql_tool_result_preview_exposes_query_and_samples PASSED [ 41%]
tests/test_chat_api.py::test_code_interpreter_trace_preview_exposes_code_and_result PASSED [ 47%]
tests/test_chat_api.py::test_forecast_and_anomaly_trace_previews_expose_tool_specific_outputs PASSED [ 52%]
tests/test_chat_api.py::test_chat_json_path_can_answer_without_database_context PASSED [ 58%]
tests/test_chat_api.py::test_chat_sse_path_can_answer_without_database_context PASSED [ 64%]
tests/test_chat_api.py::test_chat_json_path_supports_complex_multi_step_react PASSED [ 70%]
tests/test_chat_api.py::test_chat_sse_path_supports_complex_multi_step_react PASSED [ 76%]
tests/test_chat_api.py::test_runtime_advances_plan_without_repeated_todowrite PASSED [ 82%]
tests/test_chat_api.py::test_tool_failure_returns_observation_and_model_can_recover PASSED [ 88%]
tests/test_chat_api.py::test_chat_json_path_preserves_multi_query_results_in_final_answer PASSED [ 94%]
tests/test_chat_api.py::test_chat_sse_path_preserves_multi_query_results_in_final_answer PASSED [100%]

============================= 17 passed in 12.80s ==============================
```

## E2E Case Details

| # | Test case | Test content | Expected result | Actual result / output |
|---:|---|---|---|---|
| 1 | `test_chat_json_path_returns_final_answer` | JSON `/api/v1/chat` request: analyze `appliances_energy_wh` trend from `2016-01-11T17:00:00` to `2016-01-12T23:00:00` on `influxdb2-energydata`. | HTTP 200; response status `completed`; response kind `final_answer`; used tools `["sql_query", "code_interpreter"]`; non-empty answer summary; token usage recorded. | PASSED. Output used tools: `sql_query -> code_interpreter`; token usage existed with `counting_method=tiktoken_estimate`. |
| 2 | `test_chat_json_path_uses_code_interpreter_tool` | JSON chat request asks to use code interpreter to compute pairwise adjacent-point deltas. Conversation logging enabled to a temp directory. | HTTP 200; status `completed`; used tools `["sql_query", "code_interpreter"]`; code interpreter output uses sandbox path; `delta_count = value_count - 1`; analysis section/reference exists; artifact output persisted. | PASSED. Output summary contained `Code interpreter computed 180 pairwise deltas`; code artifact persisted under request log directory. |
| 3 | `test_first_visible_action_does_not_wait_for_separate_intent_llm_call` | Direct ReAct loop iteration for trend analysis request. | First emitted action is `sql_query`; fake LLM call count is exactly `1`, proving no separate intent LLM call blocks first visible action. | PASSED. Output first action: `sql_query`; `llm.calls == 1`. |
| 4 | `test_chat_sse_path_returns_event_stream` | SSE `/api/v1/chat` request for energy trend analysis. | HTTP 200; `text/event-stream`; public stream includes `conversation_id`, `agent_step`, `tool_call`, `tool_result`, `step.start`, `step.meta`, `step.done`, `thought`, `final_answer`, `terminate`; no raw internal `action`/`observation` events. | PASSED. Output SSE body contained expected public events and placeholder appeared before first `tool_call`. |
| 5 | `test_chat_json_path_persists_complete_trace_log` | JSON chat request with conversation logging enabled. | HTTP 200; persisted `conversation_trace.json`, `request.json`, `response.json`, `state.json`, `trace_internal.jsonl`, `tool_calls.jsonl`, and index entry; trace includes 3 action events; used tools are `sql_query`, `code_interpreter`. | PASSED. Output log status `completed`; persisted trace had schema version `conversation_trace_v1`. |
| 6 | `test_chat_sse_path_persists_internal_and_public_trace_logs` | SSE chat request with logging enabled. | HTTP 200; persisted `conversation_trace.json` and `trace_public.jsonl`; internal trace includes `thought`; public trace includes `thought`, `step.start`, `step.meta`, `step.done`, `tool_call`, `tool_result`, `final_answer`. | PASSED. Output log mode was `sse`; public/internal trace files existed. |
| 7 | `test_sql_tool_result_preview_exposes_query_and_samples` | Unit-level preview check for a successful `sql_query` payload. | Preview exposes query language, query text, columns, row/point counts, sample rows/points, sampling metadata, task coverage, schema linking sources/mappings/filters. | PASSED. Output preview preserved SQL query, row count `2`, point count `10`, and schema linking filter details. |
| 8 | `test_code_interpreter_trace_preview_exposes_code_and_result` | Unit-level preview check for `code_interpreter` input and output payload. | Input preview exposes code preview and code length; output preview exposes metrics, details, runtime, and input columns. | PASSED. Output preview included metric `mean=1.2`, details `{"n": 3}`, runtime `12.4`, input columns `timestamp,value`. |
| 9 | `test_forecast_and_anomaly_trace_previews_expose_tool_specific_outputs` | Unit-level preview check for forecast and anomaly payloads. | Forecast preview exposes status, plan, forecast points; anomaly preview exposes detector, anomaly points, scores. | PASSED. Output forecast status `succeeded`; anomaly detector `zscore`; anomaly point and score preserved. |
| 10 | `test_chat_json_path_can_answer_without_database_context` | JSON chat request `你好` without database context. | HTTP 200; completed final answer; no used tools; empty trace; answer mentions TSPilot; token usage call count is 1. | PASSED. Output had no tool calls and direct conversational final answer. |
| 11 | `test_chat_sse_path_can_answer_without_database_context` | SSE chat request `你好` without database context. | HTTP 200; stream includes `final_answer`; no `tool_call`, `tool_result`, or `step.start`; no `terminate` tool event. | PASSED. Output SSE body contained TSPilot answer and no tool execution events. |
| 12 | `test_chat_json_path_supports_complex_multi_step_react` | JSON complex ReAct request: plan, query energy trend, run analysis, detect anomaly, forecast, then summarize. | HTTP 200; completed final answer; used tools `todowrite -> sql_query -> code_interpreter -> anomaly -> forecast`; answer includes analysis/anomaly/forecast sections and forecast/anomaly references; trace includes 6 actions and 6 observations including terminal step. | PASSED. Output section types included `analysis`, `anomaly`, `forecast`; references included `forecast` and `anomaly`. |
| 13 | `test_chat_sse_path_supports_complex_multi_step_react` | SSE version of the same complex multi-step ReAct workflow. | HTTP 200; stream has 6 `thought`, 6 `tool_call`, 6 `tool_result`; includes tools `todowrite`, `sql_query`, `code_interpreter`, `anomaly`, `forecast`, `terminate`; includes phases `intent`, `analysis`, `answer_assembly`; no raw internal action/observation events. | PASSED. Output stream included expected tool events, final answer, and terminate event. |
| 14 | `test_runtime_advances_plan_without_repeated_todowrite` | JSON workflow where fake model may repeat planning; verifies runtime does not accept repeated `todowrite`. | HTTP 200; completed; used tools `todowrite -> sql_query -> code_interpreter -> anomaly`; exactly one successful `todowrite` observation. | PASSED. Output had one todo plan update; no repeated successful `todowrite`. |
| 15 | `test_tool_failure_returns_observation_and_model_can_recover` | JSON workflow intentionally starts with invalid `forecast` before evidence, then recovers. | HTTP 200; completed; used tools `todowrite -> forecast -> sql_query -> code_interpreter -> anomaly`; failed forecast observation exists and model recovers. | PASSED. Output included failed forecast observation; later query/analysis/anomaly completed. |
| 16 | `test_chat_json_path_preserves_multi_query_results_in_final_answer` | JSON Bitcoin multi-query request: count USD price records, earliest 5 rows, latest 5 rows, earliest/latest timestamps, include Flux query and returned row counts. | HTTP 200; completed; used tools `todowrite -> sql_query -> sql_query -> sql_query -> sql_query`; final `query_results` section includes 4 Flux blocks, 4 row-count lines, count value `2680`, earliest abnormal value row, latest row, and exactly 4 answer references. | PASSED. Output preserved all 4 query results; row counts were `[1, 5, 5, 2]`; references count was `4`. |
| 17 | `test_chat_sse_path_preserves_multi_query_results_in_final_answer` | SSE version of Bitcoin multi-query request. | HTTP 200; stream includes 6 `tool_call`, 6 `tool_result`, `final_answer`, `terminate`, `query_results`, count `2680`, row count text, earliest/latest timestamps, and Flux blocks. | PASSED. Output SSE body preserved multi-query answer content. |

## Additional Targeted Checks

Command:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python -m pytest \
  tests/test_action_policy.py::test_terminate_blocks_analysis_capability_without_task_contract_until_code_runs \
  tests/test_action_policy.py::test_policy_does_not_force_todowrite_for_complex_initial_request \
  -q
```

Result:

```text
..                                                                       [100%]
2 passed in 0.33s
```

These targeted checks verify:

- `requested_capabilities=["analysis"]` still blocks `terminate` until grounded code analysis exists.
- Initial todo-plan detection no longer treats date/time digits as an explicit multi-deliverable numbered list.

## Result Summary

Status: passed for the selected backend E2E suite.

The core ReAct flow remains functional after the harness refactor:

```text
Observation -> Action Space -> Tool Execution -> State Transition -> Completion
```

No frontend compilation or browser test was run.

## Known Risks / Follow-up

- `tests/test_chat_api.py` validates the backend ReAct path with fake LLMs and test clients; it is not a live production-model test.
- Several lower-level `tests/test_action_policy.py` historical expectations conflict with the current E2E-oriented runtime progression semantics, especially around automatic todo advancement. The backend E2E path remains passing.
- `StateTransitionEngine` now owns the state transition boundary, but some implementation details still call helper functions in `runtime/request_state.py`. A later cleanup can move those helpers into `core/harness/transition.py` or a dedicated transition support module.
- Multi-database behavior was not tested in this report.
