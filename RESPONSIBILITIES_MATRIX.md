# TSPilot v0.2 Responsibilities Matrix

## Canonical rules

- exactly one outer `data_agent`
- exactly one outer ReAct loop
- one model turn = one ReAct block
- model output block: `Thought`, `Action`, `Action Input`
- `Observation` is runtime-owned
- request termination is runtime-owned after a terminal payload exists
- prompt context is bounded and runtime-compacted
- tools execute capabilities; they do not orchestrate
- deterministic core modules implement the hard logic behind tools
- presentation layers assemble outputs; they do not invent facts

## Control layer

### `app/server.py`

- Inputs: config, router registration, dependency factories
- Outputs: configured FastAPI app
- Can: wire routes, middleware, startup hooks
- Cannot: run agent logic, tool logic, or analysis

### `app/routes/chat.py`

- Inputs: `ChatRequest`
- Outputs: terminal `ChatResponse`, optional stream events
- Can: validate request, normalize aliases, forward to `data_agent`
- Cannot: orchestrate tools or infer facts

### `runtime/react_loop.py`

- Inputs: request state, conversation state, tool executor, prompt builder
- Outputs: updated state, trace events, terminal result
- Can: parse one ReAct block, validate it, invoke at most one outer action
- Can: enforce prompt budget, summarize old context, and keep the model-visible window bounded
- Cannot: contain domain logic

### `runtime/tool_executor.py`

- Inputs: action name, action input, request state
- Outputs: normalized `ToolObservation`
- Can: resolve tool, validate input contract, invoke tool, normalize output
- Cannot: choose the next action

## State layer

### `schemas/api.py`

- Purpose: external request/response contract
- Cannot: contain runtime logic

### `schemas/state.py`

- Purpose: request and conversation state contract
- Can: carry control, evidence, analysis, and presentation sub-state
- Cannot: perform orchestration by itself

### `schemas/database_context.py`

- Purpose: normalized database selection context
- Can: unify selected database id and type
- Cannot: infer backend behavior

### `schemas/tool.py`

- Purpose: tool call / observation / error wrappers
- Can: normalize tool interaction records
- Cannot: execute business logic

### `runtime/trace.py`

- Purpose: trace event contract
- Can: normalize thought/action/observation/final/error events
- Cannot: store business state

## Evidence layer

### `tools/sql_query.py`

- Inputs: request message, database context, time range, constraints, optional history
- Outputs: `DatabaseEvidence`
- Can: run LLM schema linking, generate a dialect-specific read-only query, execute it once, and normalize evidence
- Cannot: write facts, write final answer, decide charts

### `schemas/database.py`

- Purpose: database evidence models
- Can: describe schema, metric list, statistics, table, timeseries evidence
- Cannot: contain runtime logic

## Analysis layer

### `tools/code_interpreter.py`

- Inputs: database evidence, analysis goal, optional generated code and constraints
- Outputs: analysis artifact plus facts produced from requested outputs
- Can: execute scoped analysis over evidence and register tool-produced facts
- Cannot: query databases directly or assemble final answers

### `tools/forecast.py`

- Inputs: timeseries evidence, horizon, options
- Outputs: `ForecastResult` and visualization payload
- Can: normalize and call forecast adapter
- Cannot: infer unrelated facts

### `tools/anomaly.py`

- Inputs: timeseries evidence, options
- Outputs: `AnomalyResult` and visualization payload
- Can: normalize and call anomaly adapter
- Cannot: infer unrelated facts

### `core/timeseries/*`

- Purpose: deterministic normalization and adapter bridge
- Can: shape evidence for forecast/anomaly adapters
- Cannot: decide user intent

## Presentation layer

### `tools/format_answer.py`

- Inputs: a small assembly directive plus runtime state access
- Outputs: `FinalAnswer`
- Can: assemble sections, references, visualizations
- Cannot: perform hidden analysis

### `schemas/output.py`

- Purpose: final answer and reference models
- Must carry: summary, sections, references, visualizations
- Cannot: encode analysis logic

## Agent boundary

### `agents/data_agent.py`

- Inputs: `ChatRequest`, request state, conversation state
- Outputs: one ReAct block per turn, updated state, terminal response payload via the runtime loop
- Can: lightweight intent recognition, requested fact family selection, ReAct orchestration
- Cannot: low-level database execution, deterministic verification rules, presentation assembly

### `prompts/data_agent.py`

- Purpose: model-facing ReAct prompt template
- Can: define allowed actions, action input contracts, output style
- Cannot: contain backend-specific business logic
