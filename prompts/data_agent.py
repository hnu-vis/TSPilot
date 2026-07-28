"""Prompt/context builder for the outer agent."""
from __future__ import annotations

import json

from schemas.state import ConversationStateModel, RequestStateModel


class DataAgentPromptBuilder:
    """Build the bounded model-visible context."""

    def build_system_prompt(self) -> str:
        return (
            "You are the single outer data_agent for TSPilot v0.2.\n"
            "Your job is to choose exactly one next action from the current state.\n"
            "Do not pre-commit a full workflow. Do not explain multiple future steps.\n"
            "Observation is runtime-owned and must never be emitted.\n"
            "Respond with exactly one JSON object and nothing else.\n"
            "Never emit a second JSON object. Never emit a bare tool input. Never emit markdown, prose, or trailing text.\n"
            "Output schema: "
            "{\"thought\": str, \"task_contract\": object|null, \"previous_observation_assessment\": object|null, \"action_intention\": str|null, \"action_reason\": str|null, \"action\": str, \"action_input\": object}.\n"
            "task_contract is an LLM-authored output contract for the user's visible deliverables. "
            "When state.task_contract is null and the request is data-oriented, include task_contract before or with the first non-terminal action. "
            "Use {\"source\":\"llm\",\"goal\":str,\"required_outputs\":[{\"id\":str,\"description\":str,\"output_type\":str|null,\"evidence_kind\":str|null,\"required\":bool,\"measures\":list[str],\"dimensions\":list[str],\"time_scope\":object|str|null,\"success_criteria\":str|null}],\"constraints\":object,\"assumptions\":list[str],\"evidence_quality_notes\":list[str]}. "
            "The contract must describe requested user-visible outputs, not internal tool stages. "
            "previous_observation_assessment must be null when there is no previous observation or the previous observation only created the todo plan. "
            "Otherwise, use {\"completed_active_todo\": bool, \"reason\": str|null, \"evidence_refs\": list[str], "
            "\"covered\": list[str], \"missing\": list[str], "
            "\"completed_todos\": list[int|str], \"next_active_todo\": int|str|null, "
            "\"next_action_reason\": str|null, \"can_answer\": bool|null}. "
            "Set completed_active_todo=true only when your Thought concludes the latest Observation satisfies the currently active todo from before this turn; observation itself is factual, not a completion verdict.\n"
            "Use completed_todos and next_active_todo to reconcile stale todo state against all accumulated evidence and task_contract coverage. "
            "completed_todos may name todo priorities or exact todo contents for non-answer steps already satisfied by grounded evidence. "
            "Never include an answer todo in completed_todos; the answer todo is completed only by terminate producing the final answer. "
            "When state.task_contract exists, previous_observation_assessment.covered and missing must refer to task_contract.required_outputs ids or descriptions. "
            "Do not set can_answer=true until every required task_contract output is covered or explicitly unavailable in terminate action_input.\n"
            "action_intention must name only this step's concrete purpose, <= 18 Chinese characters or <= 8 English words. "
            "action_reason must briefly explain why this step is needed now, <= 30 Chinese characters or <= 12 English words.\n"
            "Language policy: task.response_language is authoritative for all model-authored natural-language text. "
            "If it is \"zh\", write thought, action_intention, action_reason, todo text, sql_query message/purpose text, "
            "code_interpreter analysis_goal/result text, and terminate result/summary_goal/direct_answer in Simplified Chinese. "
            "If it is \"en\", write those fields in English. Keep JSON keys, action names, query code, identifiers, metric names, and database values unchanged.\n"
            "Allowed actions: todowrite, sql_query, code_interpreter, forecast, anomaly, rag, skill, terminate.\n"
            "Choose only the next best action for the current state.\n"
            "Decide from the current evidence gap, not from an imagined workflow.\n"
            "For tasks with an explicit numbered or bulleted deliverable list, verify each user-visible deliverable against observations before terminate. "
            "If a deliverable asks for its own query text, row count, raw records, extrema, bounds, or validation result and that item is not present in observations, call the appropriate tool instead of answering with a caveat.\n"
            "The context is grouped as task, state, evidence, outputs, recent_observations, and available_actions. "
            "Use the user's message, current plan, evidence, outputs, and observations as the source of task intent.\n"
            "Context is budgeted: evidence previews and recent observations may be sampled or summarized. "
            "Use diagnostics.prompt_sampling, summary_stats, data_completeness, artifact_ref, and query text to decide whether the visible preview is complete. "
            "When a task needs facts not present in the prompt preview, call sql_query or code_interpreter over the full artifact instead of guessing from the preview.\n"
            "Natural ReAct order is Thought_n, Action_n, Observation_n, then Thought_n+1 interprets Observation_n. "
            "Do not assume runtime already advanced the todo after Observation_n; explicitly assess the previous observation in Thought_n+1 before choosing Action_n+1.\n"
            "Do not emit any non-tool action or follow-up-question action.\n"
            "Prefer best-effort automatic recovery: re-query, refine field selection, continue deterministic analysis, and then answer with explicit caveats if needed.\n"
            "Task management follows the DB-GPT-style planning rule, adapted to TSPilot runtime progress ownership. "
            "For complex tasks that require 3 or more independently verifiable user-visible steps, use todowrite to create a structured task plan BEFORE starting work. "
            "This includes numbered or bulleted deliverables where separate results must be returned or verified, such as counts, earliest/latest raw rows, time bounds, query text, row counts, extrema, comparisons, validation checks, or final synthesis. "
            "Do not use todowrite for simple single-step tasks, even if the user uses numbering.\n"
            "Do not call todowrite when a todo plan already exists. Runtime advances plan status only after your next-turn assessment of the previous observation passes hard safety checks.\n"
            "Todo is visible process state, not a deterministic tool contract. After each observation, use your next Thought to judge whether the current task has enough evidence; if not, choose the next non-todowrite action.\n"
            "When a todo step is in_progress, treat its task_type as a progress label only. Choose the next action from the ReAct state, observations, and evidence gaps, not by mechanically matching the label.\n"
            "Task-first evidence rule:\n"
            "- Before choosing the next tool, translate the user's request into a task output contract: required measures, dimensions, time scope, grouping, comparisons, derived quantities, model outputs, and evidence quality notes.\n"
            "- After each observation, fill previous_observation_assessment as a gap assessment between that output contract and current evidence: covered, missing, can_answer, and next_action_reason. This is required even when there is no active todo to complete.\n"
            "- Treat tools as ways to produce or verify that contract. Do not choose a tool chain by template; choose the smallest next action that fills missing contract fields from grounded evidence.\n"
            "- A sql_query result is database evidence only. Use it for raw rows, database-returned columns, query text, row counts, grouping/filter validation, and simple database facts. "
            "Do not use the final answer to perform arithmetic, statistical analysis, anomaly/outlier reasoning, normalization, trend judgment, or multi-field derivation.\n"
            "- When required outputs include derived values or analysis such as percentage change, deltas, ratios, returns, rates, extrema derived from raw series, trend, volatility, correlation, windowed metrics, or outlier treatment, call code_interpreter over the full grounded evidence before terminate. "
            "The final answer must cite and organize the code_interpreter result instead of recomputing those values.\n"
            "- If evidence is grounded but some contract fields are missing, prefer the tool that can directly and reliably produce those missing fields. If a generated database expression repeatedly fails because it is too complex, simplify or split the evidence query instead of repeating fragile syntax, and state the changed strategy in next_action_reason/action_reason before retrying.\n"
            "- Set missing only for explicitly requested core outputs that cannot be answered from current evidence. Do not put optional drill-downs, caveats, nicer formatting, or quality notes in missing; mention those in the final answer when useful.\n"
            "- Do not set can_answer=true while missing contains core requested outputs. If the core request is answerable, set can_answer=true and keep missing empty. Only terminate with truly missing core outputs when action_input includes unavailable_outputs and unavailable_reason that explicitly name what cannot be computed and why.\n"
            "Tool contracts:\n"
            "- todowrite: create the initial full todo plan only when no plan exists and the task needs 3 or more independently verifiable user-visible steps. Plan user-visible deliverables, not internal tool stages. Do not create todo items for field confirmation, query generation, or query planning; those are internal to sql_query.\n"
            "- sql_query: query the selected datasource and return database evidence, including the actual backend query text when available. "
            "Use automatic message-based input for normal database requests. Use explicit query/query_language only when repairing a failed generated query, or when the user supplied an exact query. "
            "It is the primary database-analysis action for raw pulls, exact aggregates, grouping, ranking, bucketing, period checks, and validation queries. "
            "It may be called repeatedly when the last observation reveals missing filters, suspicious outliers, or another grounded SQL check is needed.\n"
            "- code_interpreter: execute Python code in a subprocess over existing full evidence artifacts. Use it when the user explicitly asks for code interpreter, or when any requested output requires arithmetic, derived metrics, statistical analysis, threshold/outlier policy, comparisons across returned fields, correlations, normalization, windowed calculations, custom loops, or multi-step dataframe-like computation. It may support a forecast/anomaly workflow with statistics, but it must not replace the registered forecast or anomaly tool when the user asks for prediction or anomaly detection.\n"
            "- anomaly: detect anomalies from existing time-series evidence only. Use detector_name only when the user names a supported detector; otherwise omit it and use the default registered detector.\n"
            "- forecast: generate a forecast plan/result from existing time-series evidence only. It accepts explicit step counts or duration-like horizons such as '1 day'. Use model_name only when the user names a supported forecast model; otherwise omit it and use the default registered model. A forecast observation with status 'succeeded' or 'requires_rolling' is forecast evidence that can be answered from.\n"
            "- rag: retrieve external or local knowledge only when database evidence alone is insufficient and the user explicitly needs extra knowledge.\n"
            "- skill: invoke a named packaged workflow only when the user explicitly asks for a packaged workflow or named skill.\n"
            "- terminate: end the ReAct loop when verified outputs already answer the task, and provide the final response payload. "
            "Do not terminate with phrases like 'if needed, continue' for an explicitly requested deliverable; perform the needed next action first. "
            "When no datasource is selected and the user asks a greeting, capability question, clarification, or other non-data question, use terminate directly and provide a concise result/direct_answer. "
            "When the user asks a data-analysis question without a datasource, use terminate to explain that a database/context is needed and suggest selecting one.\n"
            "Choose tools only from their current-state preconditions. Do not call a tool just because it appears in the user request.\n"
            "Action Input must be valid JSON.\n"
            "Use the exact action-input field names defined here.\n"
            "For todowrite, use: {\"message\": str, \"current_intent\": str|null, "
            "\"focus\": str|null, \"task_contract\": object|null, \"todos\": list[object], "
            "\"evidence_summary\": str|object|null}.\n"
            "Each todo item should use {\"content\": str, \"task_type\": \"query|code_interpreter|anomaly|forecast|answer|rag|skill|generic\", \"status\": \"pending|in_progress|completed\", \"priority\": int, \"acceptance_criteria\": str|null}.\n"
            "The todo output should include the complete latest todo list, not only a delta. Keep at most one in_progress step unless all steps are completed.\n"
            "Task types must stay narrow to the user's actual request. Do not add forecast steps unless the user explicitly asks for prediction. For database plans, split by requested result: count, earliest/latest rows, grouped results, time bounds, comparisons, and final answer. Do not split by internal preparation stages.\n"
            "For sql_query automatic planning, use: {\"message\": str, \"database_context\": object, \"time_range\": object|null, \"constraints\": object}. This is the normal path for database queries.\n"
            "For sql_query explicit analysis, use: {\"database_context\": object, \"query\": str, \"query_language\": str|null, \"purpose\": str|null, \"constraints\": object}. Only write read-only SELECT/WITH SQL, Flux without output/write functions, or read-only backend query language.\n"
            "After a sql_query observation, inspect its query, columns, counts, and sample rows/points. If the sample shows wrong entity filters, mixed units/categories, suspicious extreme values, or insufficient raw evidence, issue another explicit sql_query that corrects or validates the data. "
            "For data questions requiring exact grouping, ranking, period checks, or source validation, prefer an explicit sql_query when the database can return the needed evidence directly. "
            "For calculations over returned evidence, including extrema, boundary comparisons, percentage changes, median/quantile, threshold proportions, outlier policy, or comparison across categories, use code_interpreter after the SQL evidence is sufficiently grounded. Do not calculate final facts from prompt previews or inside terminate. "
            "If code_interpreter excludes, filters, winsorizes, flags, or otherwise treats outliers/anomalies, its result.details must include the explicit outlier_rule, threshold_or_formula, rationale, excluded_rows, and both raw_metrics and adjusted_metrics when an adjusted result is presented. Do not silently replace raw metrics with adjusted metrics. "
            "When outlier treatment changes level-based metrics such as start/end/max/min over prices or measurements, choose an outlier rule over that same value distribution or an explicit user threshold; do not use a first-difference/spike detector to clean level metrics unless the user explicitly asked for abrupt changes or jumps. "
            "The adjusted_metrics must be recomputed from exactly the rows left after removing details.excluded_rows, and excluded_rows must be the row list, not only a count. "
            "In code_interpreter, rows, points, and database_evidence rows/points/series are aliases for the same grounded evidence; choose one collection as the base input and do not concatenate aliases or double-count duplicate timestamp/value records. "
            "If the user asks for forecast or anomaly detection, code_interpreter output alone is not enough to answer; call the corresponding registered tool before terminate.\n"
            "For code_interpreter, use: {\"database_evidence\": str|object|null, \"analysis_goal\": str, \"code\": str, \"expected_result_schema\": object|null, \"constraints\": object}. The code may use rows, points, columns, database_evidence, metadata, diagnostics, Python imports, math, and statistics. It must assign result={\"summary\": str, \"metrics\": object, \"details\": object}; print output alone is not enough. "
            "In code_interpreter code, prefer the injected rows and points variables as the analysis input; do not assume database_evidence['data']['series'] exists. Use Python literals such as None/True/False, never JSON literals null/true/false. "
            "When the user needs precise or detailed analysis, set expected_result_schema to the exact nested fields the answer will need, such as {\"metrics\":{\"row_count\":\"int\",\"min_value\":\"number\",\"max_value\":\"number\"},\"details\":{\"findings\":[{\"label\":\"str\",\"value\":\"number\",\"evidence_ref\":\"str\"}]}}; the runtime rejects code_interpreter output that misses those fields or returns the wrong JSON types. "
            "When any outlier treatment is used, include expected_result_schema fields for details.outlier_rule, details.threshold_or_formula, details.rationale, details.excluded_rows, details.raw_metrics, and details.adjusted_metrics.\n"
            "For anomaly, use: {\"database_evidence\": str|object|null, \"detector_name\": str|null, \"series_name\": str|null, \"constraints\": object}. Omit detector_name unless the user requested a specific supported detector.\n"
            "For forecast, use: {\"database_evidence\": str|object|null, \"horizon\": int|str|object|null, \"model_name\": str|null, \"series_name\": str|null, \"constraints\": object}. Pass fuzzy user durations as strings when the user did not specify exact steps; the forecast tool resolves sampling interval and may return status='requires_rolling' with forecast_plan instead of points for long horizons. Omit model_name unless the user requested a specific supported model.\n"
            "For terminate, use: {\"result\": str|null, \"summary_goal\": str|null, \"direct_answer\": str|null, "
            "\"include_analysis_ids\": list[str], \"include_fact_ids\": list[str], \"include_visualization_ids\": list[str], \"section_plan\": list[str], "
            "\"unavailable_outputs\": list[str], \"unavailable_reason\": str|null}.\n"
            "Do not output markdown fences."
        )

    def build_context(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> dict:
        latest_evidence = (
            self._summarize_database_evidence(request_state.latest_database_evidence)
            if request_state.latest_database_evidence
            else None
        )
        latest_evidence_id = latest_evidence.get("evidence_id") if isinstance(latest_evidence, dict) else None
        prior_queries = [
            item
            for item in self._summarize_query_history(request_state)
            if item.get("evidence_id") != latest_evidence_id
        ]
        return {
            "task": {
                "message": request_state.message,
                "response_language": request_state.response_language,
                "database_context": (
                    request_state.database_context.model_dump(mode="json")
                    if request_state.database_context
                    else None
                ),
                "selected_database": request_state.selected_database,
                "selected_database_type": request_state.selected_database_type,
                "time_range": request_state.time_range,
                "constraints": request_state.constraints,
                "history": [message.model_dump(mode="json") for message in request_state.history[-4:]],
            },
            "state": {
                "execution": self._execution_state(request_state),
                "todo_list": request_state.todo_list,
                "plan_current_step": request_state.plan_current_step,
                "planning_complete": request_state.planning_complete,
                "focus": request_state.focus,
                "recent_todo_summary": conversation_state.recent_todo_summary,
                "prompt_context_summary": request_state.prompt_context_summary,
                "task_contract": (
                    request_state.task_contract.model_dump(mode="json")
                    if request_state.task_contract
                    else None
                ),
            },
            "evidence": {
                "latest": latest_evidence,
                "prior_queries": prior_queries[-5:],
            },
            "outputs": {
                "analysis_workspace": self._analysis_workspace(request_state),
                "latest_forecast": (
                    self._summarize_forecast_status(request_state.latest_forecast)
                    if request_state.latest_forecast
                    else None
                ),
                "latest_anomaly": (
                    self._summarize_anomaly_status(request_state.latest_anomaly)
                    if request_state.latest_anomaly
                    else None
                ),
                "latest_rag": request_state.latest_rag,
                "latest_skill": request_state.latest_skill,
                "verified_facts": [
                    fact.model_dump(mode="json")
                    for fact in request_state.verified_facts
                ],
                "visualizations": [
                    self._summarize_visualization(visualization)
                    for visualization in request_state.visualizations
                ],
            },
            "recent_observations": [
                {
                    "tool_name": observation.tool_name,
                    "success": observation.success,
                    "summary": observation.summary,
                    "error": observation.error,
                    "payload": self._summarize_observation_payload(observation.payload),
                }
                for observation in request_state.observations[-4:]
            ],
            "recent_react_transcript": [
                self._react_step_context(step)
                for step in request_state.react_transcript[-6:]
            ],
            "available_actions": self._available_actions(),
        }

    def _available_actions(self) -> list[dict]:
        return [
            {
                "action": "todowrite",
                "use_when": "Create the initial plan before starting work when todo_list is empty and the request needs 3 or more independently verifiable user-visible steps.",
                "input": "message, current_intent, focus, task_contract, todos, evidence_summary; todos may include acceptance_criteria",
            },
            {
                "action": "sql_query",
                "use_when": "Database evidence is missing, or a read-only follow-up query can compute the exact aggregation, grouping, ranking, filter validation, or diagnostic needed to answer correctly.",
                "input": "prefer message, database_context, time_range, constraints for automatic database querying; use query/query_language only for repair or for a user-supplied exact query",
            },
            {
                "action": "code_interpreter",
                "use_when": "Evidence is available and the user explicitly asks for code interpreter, or the analysis needs fuller Python features such as imports, iterators, correlation, normalization, windows, or custom multi-step computation.",
                "input": "database_evidence, analysis_goal, code, expected_result_schema, constraints",
            },
            {
                "action": "anomaly",
                "use_when": "Time-series evidence is available and the user specifically needs anomaly/spike/outlier detection.",
                "input": "database_evidence, detector_name, series_name, constraints",
            },
            {
                "action": "forecast",
                "use_when": "Time-series evidence is available and the user specifically asks for prediction or forecast.",
                "input": "database_evidence, horizon, model_name, series_name, constraints",
            },
            {
                "action": "rag",
                "use_when": "The user explicitly needs external or knowledge-base retrieval beyond database evidence.",
                "input": "query, filters",
            },
            {
                "action": "skill",
                "use_when": "The user explicitly asks for a named packaged workflow or skill.",
                "input": "skill_name, parameters",
            },
            {
                "action": "terminate",
                "use_when": "Enough evidence-backed outputs are available, or the request is conversational / cannot proceed without more context.",
                "input": "summary_goal, direct_answer, include_analysis_ids, include_fact_ids, include_visualization_ids, section_plan, unavailable_outputs, unavailable_reason",
            },
        ]

    def _execution_state(self, request_state: RequestStateModel) -> dict:
        last_success = next((item for item in reversed(request_state.observations) if item.success), None)
        last_failure = next((item for item in reversed(request_state.observations) if not item.success), None)
        last_call = request_state.tool_history[-1] if request_state.tool_history else None
        last_tool = last_call.tool_name if last_call else None
        return {
            "iteration": request_state.iteration,
            "max_iterations": request_state.max_iterations,
            "tool_sequence": [call.tool_name for call in request_state.tool_history],
            "last_tool": last_tool,
            "last_action_reason": last_call.reason if last_call else None,
            "last_successful_tool": last_success.tool_name if last_success else None,
            "last_failure": (
                {
                    "tool_name": last_failure.tool_name,
                    "summary": last_failure.summary,
                    "error": last_failure.error,
                }
                if last_failure
                else None
            ),
            "artifacts": {
                "has_database_evidence": request_state.latest_database_evidence is not None,
                "has_analysis": bool(request_state.analysis_artifacts),
                "has_forecast": request_state.latest_forecast is not None,
                "has_anomaly": request_state.latest_anomaly is not None,
                "has_final_answer": request_state.final_answer_draft is not None,
                "verified_fact_count": len(request_state.verified_facts),
                "visualization_count": len(request_state.visualizations),
                "analysis_count": len(request_state.analysis_artifacts),
            },
            "plan_progress_owner": "runtime" if request_state.todo_list else "none",
        }

    def build_user_prompt(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> str:
        context = self.build_context(request_state, conversation_state)
        return "\n".join(
            [
                "User Task:",
                json.dumps(context["task"], ensure_ascii=False, indent=2),
                "",
                "Available Tools:",
                json.dumps(context["available_actions"], ensure_ascii=False, indent=2),
                "",
                "Previous Thought/Action/Observation:",
                self._react_transcript(request_state),
                "",
                "Runtime State JSON:",
                json.dumps(context, ensure_ascii=False, indent=2),
            ]
        )

    def _react_transcript(self, request_state: RequestStateModel) -> str:
        if request_state.react_transcript:
            lines = []
            for step in request_state.react_transcript[-6:]:
                if step.question:
                    lines.append(f"Question: {self._truncate_text(step.question, 800)}")
                if step.thought:
                    lines.append(f"Thought: {self._truncate_text(step.thought, 800)}")
                if step.phase:
                    lines.append(f"Phase: {self._truncate_text(step.phase, 300)}")
                if step.action_intention:
                    lines.append(f"Action Intention: {self._truncate_text(step.action_intention, 300)}")
                if step.action_reason:
                    lines.append(f"Action Reason: {self._truncate_text(step.action_reason, 500)}")
                lines.append(f"Action: {step.action}")
                lines.append(
                    "Action Input: "
                    + self._truncate_text(json.dumps(step.action_input, ensure_ascii=False), 2000)
                )
                if step.observation is not None:
                    lines.append(
                        "Observation: "
                        + self._truncate_text(
                            json.dumps(
                                self._observation_context(step.observation),
                                ensure_ascii=False,
                            ),
                            3000,
                        )
                    )
            return "\n".join(lines) if lines else "(none)"

        if not request_state.tool_history and not request_state.observations:
            return "(none)"
        lines = []
        calls_by_iteration = {call.iteration: call for call in request_state.tool_history}
        observations_by_iteration: dict[int, list] = {}
        for observation in request_state.observations:
            iteration = self._observation_iteration(observation, calls_by_iteration)
            observations_by_iteration.setdefault(iteration, []).append(observation)
        for iteration in sorted(set(calls_by_iteration) | set(observations_by_iteration)):
            call = calls_by_iteration.get(iteration)
            if call is not None:
                if call.reason:
                    lines.append(f"Thought: {self._truncate_text(call.reason, 800)}")
                lines.append(f"Action: {call.tool_name}")
                lines.append(
                    "Action Input: "
                    + self._truncate_text(json.dumps(call.tool_input, ensure_ascii=False), 2000)
                )
            for observation in observations_by_iteration.get(iteration, []):
                lines.append(
                    "Observation: "
                    + self._truncate_text(
                        json.dumps(
                            {
                                "tool_name": observation.tool_name,
                                "success": observation.success,
                                "summary": observation.summary,
                                "error": observation.error,
                                "payload": self._summarize_observation_payload(observation.payload),
                            },
                            ensure_ascii=False,
                        ),
                        3000,
                    )
                )
        return "\n".join(lines) if lines else "(none)"

    def _react_step_context(self, step) -> dict:
        return {
            "iteration": step.iteration,
            "question": self._truncate_text(step.question, 800),
            "thought": self._truncate_text(step.thought, 800),
            "phase": self._truncate_text(step.phase, 300),
            "action_intention": self._truncate_text(step.action_intention, 300),
            "action_reason": self._truncate_text(step.action_reason, 500),
            "action": step.action,
            "action_input": self._bounded_value(
                step.action_input,
                max_string_chars=1200,
                max_list_items=8,
                max_dict_items=16,
            ),
            "observation": (
                self._observation_context(step.observation)
                if step.observation is not None
                else None
            ),
        }

    def _observation_context(self, observation) -> dict:
        context = {
            "tool_name": observation.tool_name,
            "success": observation.success,
            "summary": observation.summary,
            "payload": self._summarize_observation_payload(observation.payload),
        }
        if observation.error:
            context["error"] = observation.error
        if observation.payload_truncated:
            context["payload_truncated"] = True
        if observation.payload_ref:
            context["payload_ref"] = observation.payload_ref
        return context

    def _observation_iteration(self, observation, calls_by_iteration: dict[int, object]) -> int:
        for iteration in sorted(calls_by_iteration, reverse=True):
            if calls_by_iteration[iteration].tool_name == observation.tool_name:
                return iteration
        return max(calls_by_iteration.keys(), default=0)

    def _summarize_database_evidence(self, evidence) -> dict:
        payload = evidence.model_dump(mode="json")
        data = dict(payload.get("data") or {})
        diagnostics = dict(payload.get("diagnostics") or {})
        summary_stats = diagnostics.get("summary_stats") or {}
        if isinstance(data.get("points"), list):
            data["points"] = data["points"][:8]
        if isinstance(data.get("rows"), list):
            data["rows"] = data["rows"][:4]
        if isinstance(data.get("series"), list):
            series_preview = []
            for item in data["series"][:3]:
                if isinstance(item, dict):
                    series_preview.append(self._summarize_series_preview(item, point_limit=4))
            data["series"] = series_preview
        payload["data"] = data
        payload["query"] = self._truncate_text(payload.get("query"), 4000)
        payload["summary"] = self._truncate_text(payload.get("summary"), 1200)
        payload["metadata"] = self._bounded_value(payload.get("metadata") or {}, max_string_chars=600, max_list_items=8, max_dict_items=16)
        visible_diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "summary_stats", "prompt_sampling", "query_trace", "series_count"}
        }
        visible_diagnostics["prompt_sampling"] = self._prompt_sampling(
            full_counts=summary_stats,
            data=data,
            fallback=visible_diagnostics.get("prompt_sampling") if isinstance(visible_diagnostics.get("prompt_sampling"), dict) else None,
        )
        if "query_trace" in visible_diagnostics:
            visible_diagnostics["query_trace"] = self._bounded_value(
                visible_diagnostics["query_trace"],
                max_string_chars=1200,
                max_list_items=6,
                max_dict_items=16,
            )
        payload["diagnostics"] = visible_diagnostics
        if summary_stats:
            payload["summary_stats"] = summary_stats
        return payload

    def _summarize_query_history(self, request_state: RequestStateModel) -> list[dict]:
        """Expose prior database queries as stable model-visible context."""

        history = []
        for evidence in request_state.database_evidence_artifacts.values():
            item = self._summarize_database_evidence(evidence)
            data = item.get("data") if isinstance(item.get("data"), dict) else {}
            diagnostics = item.get("diagnostics") if isinstance(item.get("diagnostics"), dict) else {}
            summary_stats = diagnostics.get("summary_stats") or item.get("summary_stats") or {}
            row_count = summary_stats.get("rows_count")
            point_count = summary_stats.get("points_count")
            series_count = summary_stats.get("series_count")
            if row_count is None and isinstance(data.get("rows"), list):
                row_count = len(evidence.data.get("rows", [])) if isinstance(evidence.data, dict) else len(data["rows"])
            if point_count is None and isinstance(data.get("points"), list):
                point_count = len(evidence.data.get("points", [])) if isinstance(evidence.data, dict) else len(data["points"])
            if series_count is None and isinstance(data.get("series"), list):
                series_count = len(evidence.data.get("series", [])) if isinstance(evidence.data, dict) else len(data["series"])
            preview = {}
            if isinstance(data.get("rows"), list):
                preview["rows"] = data["rows"][:3]
            if isinstance(data.get("points"), list):
                preview["points"] = data["points"][:4]
            if isinstance(data.get("series"), list):
                preview["series"] = [
                    self._summarize_series_preview(series, point_limit=4)
                    for series in data["series"][:2]
                    if isinstance(series, dict)
                ]
            history.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "database": item.get("database"),
                    "query_language": item.get("query_language"),
                    "query": self._truncate_text(item.get("query"), 2500),
                    "summary": self._truncate_text(item.get("summary"), 800),
                    "result_type": item.get("result_type"),
                    "columns": (item.get("columns") or [])[:20],
                    "row_count": row_count,
                    "point_count": point_count,
                    "series_count": series_count,
                    "metadata": self._bounded_value(
                        item.get("metadata") or {},
                        max_string_chars=400,
                        max_list_items=6,
                        max_dict_items=12,
                    ),
                    "preview": preview,
                }
            )
        return history[-6:]

    def _analysis_workspace(self, request_state: RequestStateModel) -> dict:
        analyses = []
        for analysis in request_state.analysis_artifacts.values():
            payload = analysis.model_dump(mode="json")
            analyses.append(
                {
                    "analysis_id": payload.get("analysis_id"),
                    "goal": payload.get("analysis_goal"),
                    "summary": payload.get("summary"),
                    "status": payload.get("status"),
                    "code_type": payload.get("code_type"),
                    "code_hash": payload.get("code_hash"),
                    "input_ref": f"evidence:{payload.get('input_evidence_id')}",
                    "input_row_count": payload.get("input_row_count"),
                    "result_preview": self._bounded_value(
                        payload.get("result") or {},
                        max_string_chars=1000,
                        max_list_items=8,
                        max_dict_items=16,
                    ),
                }
            )
        return {
            "latest_analysis_id": request_state.latest_analysis_id,
            "analysis_count": len(analyses),
            "analyses": analyses[-8:],
        }

    def _summarize_forecast_status(self, forecast) -> dict:
        payload = forecast.model_dump(mode="json")
        forecast_points = payload.get("forecast_points", []) or []
        return {
            "forecast_id": payload.get("forecast_id"),
            "status": payload.get("status"),
            "horizon": payload.get("horizon"),
            "forecast_plan": payload.get("forecast_plan"),
            "summary": self._truncate_text(payload.get("summary"), 800),
            "forecast_point_count": len(forecast_points),
            "forecast_points_preview": forecast_points[:6],
            "visualization_count": len(payload.get("visualizations", []) or []),
        }

    def _summarize_anomaly_status(self, anomaly) -> dict:
        payload = anomaly.model_dump(mode="json")
        return {
            "anomaly_id": payload.get("anomaly_id"),
            "summary": self._truncate_text(payload.get("summary"), 800),
            "anomaly_point_count": len(payload.get("anomaly_points", []) or []),
            "visualization_count": len(payload.get("visualizations", []) or []),
        }

    def _summarize_observation_payload(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}
        summarized = {}
        preview = {}
        for key in (
            "recovery_hint",
            "error",
            "evidence_id",
            "result_type",
            "analysis_id",
            "analysis_goal",
            "input_evidence_id",
            "input_row_count",
            "forecast_id",
            "anomaly_id",
            "current_step",
            "planning_complete",
            "query",
            "query_language",
            "columns",
            "task_contract",
        ):
            if key in payload:
                summarized[key] = self._bounded_value(payload[key], max_string_chars=1200, max_list_items=12, max_dict_items=16)
        if isinstance(payload.get("data"), dict):
            data = dict(payload["data"])
            if isinstance(data.get("rows"), list):
                preview["rows"] = data["rows"][:3]
            if isinstance(data.get("points"), list):
                preview["points"] = data["points"][:4]
            if isinstance(data.get("series"), list):
                preview["series"] = [
                    self._summarize_series_preview(series, point_limit=4)
                    for series in data["series"][:2]
                    if isinstance(series, dict)
                ]
            if preview:
                summarized["data_preview"] = preview
        if isinstance(payload.get("result"), dict):
            summarized["result_preview"] = self._bounded_value(
                payload["result"],
                max_string_chars=1000,
                max_list_items=8,
                max_dict_items=16,
            )
        if isinstance(payload.get("diagnostics"), dict):
            diagnostics = dict(payload["diagnostics"])
            summarized_diagnostics = {}
            for key in ("summary_stats", "artifact_ref", "prompt_sampling"):
                if key in diagnostics:
                    summarized_diagnostics[key] = self._bounded_value(
                        diagnostics[key],
                        max_string_chars=1000,
                        max_list_items=8,
                        max_dict_items=16,
                    )
            if payload.get("error") and isinstance(diagnostics.get("query_trace"), dict):
                summarized_diagnostics["query_trace"] = self._bounded_value(
                    diagnostics["query_trace"],
                    max_string_chars=1000,
                    max_list_items=8,
                    max_dict_items=16,
                )
            summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
            summarized_diagnostics["prompt_sampling"] = self._prompt_sampling(
                full_counts=summary_stats,
                data=preview,
                fallback=diagnostics.get("prompt_sampling") if isinstance(diagnostics.get("prompt_sampling"), dict) else None,
            )
            summarized["diagnostics"] = summarized_diagnostics
        if isinstance(payload.get("todos"), list):
            summarized["todos"] = [
                {
                    key: todo.get(key)
                    for key in ("content", "task_type", "status", "priority", "acceptance_criteria")
                    if todo.get(key) not in (None, "", [], {})
                }
                for todo in payload["todos"][:8]
                if isinstance(todo, dict)
            ]
        if isinstance(payload.get("verified_facts"), list):
            summarized["verified_facts"] = [
                {
                    "fact_id": fact.get("fact_id"),
                    "fact_type": fact.get("fact_type"),
                    "statement": fact.get("statement"),
                }
                for fact in payload["verified_facts"][:6]
                if isinstance(fact, dict)
            ]
        if isinstance(payload.get("valid_actions"), list):
            summarized["valid_actions"] = payload["valid_actions"]
        return summarized

    def _summarize_series_preview(self, series: dict, *, point_limit: int) -> dict:
        item = {
            key: self._bounded_value(value, max_string_chars=400, max_list_items=6, max_dict_items=8)
            for key, value in series.items()
            if key not in {"points", "rows"}
        }
        points = series.get("points")
        if isinstance(points, list):
            item["points_count"] = series.get("points_count") or len(points)
            item["points"] = self._sample_edges(points, limit=point_limit)
        rows = series.get("rows")
        if isinstance(rows, list):
            item["rows_count"] = series.get("rows_count") or len(rows)
            item["rows"] = self._sample_edges(rows, limit=point_limit)
        return item

    def _prompt_sampling(self, *, full_counts: dict, data: dict, fallback: dict | None = None) -> dict:
        visible_counts = {
            "points_count": len(data.get("points", [])) if isinstance(data.get("points"), list) else None,
            "rows_count": len(data.get("rows", [])) if isinstance(data.get("rows"), list) else None,
            "series_count": len(data.get("series", [])) if isinstance(data.get("series"), list) else None,
        }
        counts = {key: value for key, value in (full_counts or {}).items() if value is not None}
        if not counts and fallback:
            counts = dict(fallback.get("full_counts") or {})
        return {
            "policy": "head_tail_edges",
            "sampled_for_prompt": any(
                isinstance(counts.get(key), int)
                and isinstance(visible_counts.get(key), int)
                and visible_counts[key] < counts[key]
                for key in ("points_count", "rows_count", "series_count")
            ),
            "full_counts": counts,
            "visible_counts": {key: value for key, value in visible_counts.items() if value is not None},
            "full_artifact_ref": (fallback or {}).get("full_artifact_ref"),
        }

    def _truncate_text(self, value, max_chars: int):
        if not isinstance(value, str):
            return value
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"

    def _bounded_value(
        self,
        value,
        *,
        max_string_chars: int = 800,
        max_list_items: int = 8,
        max_dict_items: int = 12,
    ):
        if isinstance(value, str):
            return self._truncate_text(value, max_string_chars)
        if isinstance(value, list):
            items = [
                self._bounded_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_list_items=max_list_items,
                    max_dict_items=max_dict_items,
                )
                for item in value[:max_list_items]
            ]
            if len(value) > max_list_items:
                items.append({"truncated_items": len(value) - max_list_items})
            return items
        if isinstance(value, dict):
            bounded = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= max_dict_items:
                    bounded["truncated_keys"] = len(value) - max_dict_items
                    break
                bounded[key] = self._bounded_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_list_items=max_list_items,
                    max_dict_items=max_dict_items,
                )
            return bounded
        return value

    def _sample_edges(self, items: list, *, limit: int) -> list:
        if len(items) <= limit:
            return items
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        return [*items[:head], *items[-tail:]]

    def _summarize_forecast(self, forecast) -> dict:
        payload = forecast.model_dump(mode="json")
        payload["forecast_points"] = payload.get("forecast_points", [])[:6]
        payload["visualizations"] = [
            self._summarize_visualization_from_dict(item)
            for item in payload.get("visualizations", [])[:3]
        ]
        diagnostics = dict(payload.get("diagnostics") or {})
        payload["diagnostics"] = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "snapshot_ref", "series_name"}
        }
        return payload

    def _summarize_anomaly(self, anomaly) -> dict:
        payload = anomaly.model_dump(mode="json")
        payload["anomaly_points"] = payload.get("anomaly_points", [])[:8]
        payload["scores"] = payload.get("scores", [])[:8]
        payload["visualizations"] = [
            self._summarize_visualization_from_dict(item)
            for item in payload.get("visualizations", [])[:3]
        ]
        diagnostics = dict(payload.get("diagnostics") or {})
        payload["diagnostics"] = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "snapshot_ref", "threshold"}
        }
        return payload

    def _summarize_visualization(self, visualization) -> dict:
        return self._summarize_visualization_from_dict(visualization.model_dump(mode="json"))

    def _summarize_visualization_from_dict(self, payload: dict) -> dict:
        return {
            "visualization_id": payload.get("visualization_id"),
            "visualization_type": payload.get("visualization_type"),
            "visualization_kind": payload.get("visualization_kind"),
            "renderer": payload.get("renderer"),
            "title": payload.get("title"),
            "summary": payload.get("summary"),
            "binding_fact_ids": payload.get("binding_fact_ids", [])[:6],
            "binding_evidence_ids": payload.get("binding_evidence_ids", [])[:6],
            "time_column": payload.get("time_column"),
            "primary_measure": payload.get("primary_measure"),
            "display_priority": payload.get("display_priority"),
            "row_count": payload.get("row_count"),
            "columns": payload.get("columns", [])[:8],
            "annotations_count": len(payload.get("annotations", []) or []),
            "rows_count": len(payload.get("rows", []) or []),
            "display_rows_count": len(payload.get("display_rows", []) or []),
            "chart_summary": self._summarize_chart(payload.get("chart")),
        }

    def _summarize_chart(self, chart: dict | None) -> dict | None:
        if not isinstance(chart, dict):
            return None
        if "x_axis_count" in chart and "x_axis_data" not in chart:
            return chart
        x_axis_data = list(chart.get("x_axis_data") or [])
        series_data = list(chart.get("series_data") or [])
        return {
            "x_axis_count": len(x_axis_data),
            "x_axis_preview": [*x_axis_data[:3], *x_axis_data[-3:]] if len(x_axis_data) > 6 else x_axis_data,
            "series": [
                {
                    "name": item.get("name"),
                    "points_count": len(item.get("data") or []),
                }
                for item in series_data[:4]
                if isinstance(item, dict)
            ],
        }
