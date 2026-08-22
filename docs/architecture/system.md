# TSPilot System Architecture

## Purpose

TSPilot provides one evidence-grounded workflow for querying, analyzing, and
visualizing time-series data through natural language. The system uses a single
outer data agent and a ReAct loop to select and execute one tool action at a
time.

## Request flow

1. The API validates a `ChatRequest` and initializes request and conversation state.
2. The data agent interprets the request and emits one ReAct action.
3. The runtime validates the action input and invokes the selected tool.
4. The tool returns a typed observation and registers any authoritative artifacts.
5. The runtime updates state and returns the observation to the data agent.
6. The loop continues until sufficient evidence exists for a final answer or a terminal error is reached.
7. `format_answer` assembles the evidence, analysis, references, and visualization identifiers into `FinalAnswer`.

Visualization can return `needs_sources` when the available artifacts do not
contain the data required by the requested chart. The outer loop then invokes
the appropriate owner tool and retries visualization with the new observation.

## Main tools

- `todowrite` organizes multi-step work.
- `sql_query` discovers schemas, generates read-only queries, and returns database evidence.
- `code_interpreter` calculates derived analytical results over existing evidence.
- `forecast` and `anomaly` produce specialized time-series artifacts.
- `visualization` turns existing data and Insights into a grounded interactive chart.
- `rag` and `skill` provide additional context when required.
- `format_answer` assembles the terminal response from verified state.

## Architectural boundaries

### Control

The API, ReAct loop, tool executor, and trace handling control execution. They do
not calculate business results.

### Evidence and analysis

Database results, derived analyses, forecasts, anomalies, and insights are
stored as typed, request-scoped artifacts. The tool that owns a calculation is
responsible for producing it; visualization does not invent missing data.

### Presentation

Final answers and visualizations reference authoritative artifacts.
Visualization receives concise descriptions of the available data and verified
Insights, chooses the most useful series and annotations, validates that every
visual element is supported, and stores the complete chart for frontend use.

## Core rules

- One model turn produces one outer ReAct action.
- Tool inputs and observations must satisfy typed contracts.
- Database access is read-only in the analytical workflow.
- Claims and visual components must remain traceable to source artifacts.
- Missing derived data is requested from the owning tool through `needs_sources`.
- Visualization receives one initial planning attempt and two model repair attempts; no deterministic substitute chart is generated.
- Unsupported historical visualization schemas are isolated instead of migrated at read time.
- The runtime owns request termination after a final answer or terminal error exists.

## Repository boundaries

- `app/` contains HTTP, streaming, settings, and resource endpoints.
- `agents/` contains outer orchestration.
- `runtime/` contains execution control and request state.
- `tools/` contains model-visible capabilities.
- `core/` contains database, analysis, time-series, insight, and visualization implementations.
- `schemas/` contains typed public and internal contracts.
- `prompts/` contains model-facing instructions and response schemas.
- `frontend/` contains the React interface and visualization renderer.

## Current scope

TSPilot v0.1 uses one outer data agent. The current production visualization
contract creates one primary line chart with one or two compatible series,
grounded Insight annotations, a visible legend and tooltip, and time-axis zoom.
Model and database connections are managed from the Web interface or local
workspace configuration.
