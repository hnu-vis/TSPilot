# TSPilot v0.2 File Specs

This document defines the planned code files for TSPilot v0.2 and points to the
specification document for each file.

For the hard architectural boundary between modules, see
[RESPONSIBILITIES_MATRIX.md](RESPONSIBILITIES_MATRIX.md).

## Planned file tree

```text
TSPilot-v0.2/
├── app/
│   ├── server.py
│   ├── deps.py
│   ├── routes/
│   │   └── chat.py
│   └── SPEC/
├── runtime/
│   ├── react_loop.py
│   ├── tool_executor.py
│   ├── request_state.py
│   ├── conversation_state.py
│   ├── trace.py
│   └── SPEC/
├── agents/
│   ├── base.py
│   ├── data_agent.py
│   └── SPEC/
├── tools/
│   ├── base.py
│   ├── registry.py
│   ├── todowrite.py
│   ├── query_database.py
│   ├── code_interpreter.py
│   ├── forecast.py
│   ├── anomaly.py
│   ├── rag.py
│   ├── skill.py
│   ├── format_answer.py
│   └── SPEC/
├── core/
│   ├── database/
│   │   ├── engine.py
│   │   ├── repair.py
│   │   ├── schema.py
│   │   └── SPEC/
│   ├── analysis/
│   │   ├── python_runner.py
│   │   └── SPEC/
│   ├── timeseries/
│   │   ├── normalization.py
│   │   ├── forecast_adapter.py
│   │   ├── anomaly_adapter.py
│   │   └── SPEC/
│   └── rag/
│       ├── retriever.py
│       └── SPEC/
├── prompts/
│   ├── data_agent.py
│   └── SPEC/
└── schemas/
    ├── api.py
    ├── state.py
    ├── database_context.py
    ├── tool.py
    ├── agent_turn.py
    ├── database.py
    ├── visualization.py
    ├── timeseries.py
    ├── output.py
    └── SPEC/
```

## Spec index

### app

- [server.py](app/SPEC/server_SPEC.md)
- [deps.py](app/SPEC/deps_SPEC.md)
- [routes/chat.py](app/SPEC/chat_route_SPEC.md)

### runtime

- [react_loop.py](runtime/SPEC/react_loop_SPEC.md)
- [tool_executor.py](runtime/SPEC/tool_executor_SPEC.md)
- [request_state.py](runtime/SPEC/request_state_SPEC.md)
- [conversation_state.py](runtime/SPEC/conversation_state_SPEC.md)
- [trace.py](runtime/SPEC/trace_SPEC.md)

### agents

- [base.py](agents/SPEC/base_SPEC.md)
- [data_agent.py](agents/SPEC/data_agent_SPEC.md)

### tools

- [base.py](tools/SPEC/base_SPEC.md)
- [registry.py](tools/SPEC/registry_SPEC.md)
- [todowrite.py](tools/SPEC/todowrite_SPEC.md)
- [query_database.py](tools/SPEC/query_database_SPEC.md)
- [code_interpreter.py](tools/SPEC/code_interpreter_SPEC.md)
- [forecast.py](tools/SPEC/forecast_SPEC.md)
- [anomaly.py](tools/SPEC/anomaly_SPEC.md)
- [rag.py](tools/SPEC/rag_SPEC.md)
- [skill.py](tools/SPEC/skill_SPEC.md)
- [format_answer.py](tools/SPEC/format_answer_SPEC.md)

### core/database

- [engine.py](core/database/SPEC/engine_SPEC.md)
- [repair.py](core/database/SPEC/repair_SPEC.md)
- [schema.py](core/database/SPEC/schema_SPEC.md)
- [schema linking](core/database/SPEC/schema_linking_SPEC.md)

### core/analysis

- [python_runner.py](core/analysis/SPEC/python_runner_SPEC.md)

### core/timeseries

- [normalization.py](core/timeseries/SPEC/normalization_SPEC.md)
- [forecast_adapter.py](core/timeseries/SPEC/forecast_adapter_SPEC.md)
- [anomaly_adapter.py](core/timeseries/SPEC/anomaly_adapter_SPEC.md)

### core/rag

- [retriever.py](core/rag/SPEC/retriever_SPEC.md)

### prompts

- [data_agent.py](prompts/SPEC/data_agent_prompt_SPEC.md)

### schemas

- [api.py](schemas/SPEC/api_SPEC.md)
- [state.py](schemas/SPEC/state_SPEC.md)
- [database_context.py](schemas/SPEC/database_context_SPEC.md)
- [tool.py](schemas/SPEC/tool_SPEC.md)
- [agent_turn.py](schemas/SPEC/agent_turn_SPEC.md)
- [database.py](schemas/SPEC/database_SPEC.md)
- [visualization.py](schemas/SPEC/visualization_SPEC.md)
- [timeseries.py](schemas/SPEC/timeseries_SPEC.md)
- [output.py](schemas/SPEC/output_SPEC.md)
