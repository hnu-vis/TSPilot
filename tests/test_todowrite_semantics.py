from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.todowrite import TodoWriteInput, TodoWriteTool


class _TodoBindingLlm:
    async def ainvoke(self, _messages):
        return SimpleNamespace(content=json.dumps({
            "bindings": [
                {"index": 0, "task_type": "query", "reason": "Retrieves the series."},
                {"index": 1, "task_type": "anomaly", "reason": "Owns anomaly detection."},
                {"index": 2, "task_type": "forecast", "reason": "Owns prediction output."},
                {"index": 3, "task_type": "code_interpreter", "reason": "Calculates change."},
                {"index": 4, "task_type": "visualization", "reason": "Creates the chart."},
                {"index": 5, "task_type": "answer", "reason": "Synthesizes the result."},
            ],
        }))


@pytest.mark.asyncio
async def test_todo_capabilities_are_semantically_bound_not_positionally_zipped():
    request = TodoWriteInput(
        message="1. 查询序列；2. 检测异常；3. 预测6小时；4. 计算变化；5. 生成综合图；6. 总结。",
        todos=[
            {"title": "查询完整时间序列"},
            {"title": "检测并说明异常点"},
            {"title": "基于异常检测结果预测之后6小时"},
            {"title": "计算预测方向和起止变化幅度"},
            {"title": "生成历史、异常和预测综合图"},
            {"title": "总结最终结论"},
        ],
        task_contract={
            "required_outputs": [
                {"id": "todo_plan", "evidence_kind": "skill"},
                {"id": "history", "evidence_kind": "query"},
                {"id": "anomalies", "evidence_kind": "anomaly"},
                {"id": "prediction", "evidence_kind": "forecast"},
                {"id": "change", "evidence_kind": "analysis"},
                {"id": "chart", "output_type": "visualization"},
                {"id": "summary", "output_type": "conclusion"},
            ],
        },
    )

    result = await TodoWriteTool(llm=_TodoBindingLlm()).execute(request)

    assert [todo["task_type"] for todo in result["todos"]] == [
        "query", "anomaly", "forecast", "code_interpreter", "visualization", "answer",
    ]
    assert [todo["status"] for todo in result["todos"]] == [
        "in_progress", "pending", "pending", "pending", "pending", "pending",
    ]


@pytest.mark.asyncio
async def test_no_llm_compatibility_path_prefers_todo_meaning_over_contract_position():
    result = await TodoWriteTool().execute(TodoWriteInput(
        message="1. 查询序列；2. 检测异常；3. 预测6小时；4. 计算变化；5. 生成综合图；6. 总结。",
        todos=[
            {"title": "查询完整时间序列"},
            {"title": "检测并说明异常点"},
            {"title": "预测之后6小时"},
            {"title": "计算预测方向和起止变化幅度"},
            {"title": "生成综合图"},
            {"title": "总结最终结论"},
        ],
        task_contract={"required_outputs": [{"evidence_kind": "skill"}] * 6},
    ))

    assert [todo["task_type"] for todo in result["todos"]] == [
        "query", "anomaly", "forecast", "forecast", "visualization", "answer",
    ]
