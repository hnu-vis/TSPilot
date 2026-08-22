#!/usr/bin/env python3
"""Replay only visualization from persisted request states and capture screenshots."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.deps import get_llm
from core.visualization import VisualizationArtifactStore
from schemas.state import RequestStateModel
from scripts.e2e_echarts_suite import QUESTIONS, capture_screenshots
from tools.visualization import VisualizationInput, VisualizationTool


def _latest_upstream_state(log_root: Path, index: int) -> Path:
    candidates = sorted(
        log_root.glob(f"*e2e-echarts-{index:02d}-*/requests/*/state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        state = RequestStateModel.model_validate_json(path.read_text(encoding="utf-8"))
        if state.database_evidence_artifacts and any(
            insight.status == "verified" for insight in state.insight_set.insights
        ):
            return path
    raise FileNotFoundError(f"No persisted upstream state with verified Insights for question {index}")


def _prepare_state(path: Path) -> tuple[RequestStateModel, list[str]]:
    state = RequestStateModel.model_validate_json(path.read_text(encoding="utf-8"))
    state.visualizations = []
    state.observations = [item for item in state.observations if item.tool_name != "visualization"]
    refs = [
        f"insight:{insight.insight_id}"
        for insight in state.insight_set.insights
        if insight.status == "verified"
    ]
    return state, refs


async def replay(args) -> tuple[Path, list[dict]]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    store = VisualizationArtifactStore(output_dir / "store")
    tool = VisualizationTool(llm=get_llm(), artifact_store=store)
    indexes = [int(item) for item in args.indexes.split(",")] if args.indexes else list(range(1, 11))
    results: list[dict] = []
    print(f"OUTPUT_DIR={output_dir.resolve()}", flush=True)
    for index in indexes:
        question = QUESTIONS[index - 1]
        try:
            state_path = _latest_upstream_state(args.log_root, index)
            state, refs = _prepare_state(state_path)
            print(f"[{index:02d}/10] STATE {state_path}", flush=True)
            attempts: list[dict] = []
            result = None
            for attempt in range(1, args.attempts + 1):
                result = await tool.execute(
                    VisualizationInput(message=question, source_refs=refs),
                    request_state=state,
                )
                attempts.append({
                    "attempt": attempt,
                    "status": result.get("status"),
                    "unavailable_reason": result.get("unavailable_reason"),
                    "required_data_request": result.get("required_data_request"),
                })
                print(f"[{index:02d}/10] ATTEMPT {attempt} status={result.get('status')}", flush=True)
                if result.get("status") != "unavailable":
                    break
            artifacts = []
            if result and result.get("status") == "created":
                for chart_index, visualization_id in enumerate(result.get("visualization_ids") or [], start=1):
                    artifact = store.get(visualization_id)
                    if artifact is None:
                        continue
                    payload = artifact.model_dump(mode="json")
                    artifacts.append(payload)
                    (output_dir / f"q{index:02d}_chart{chart_index:02d}.json").write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
            entry = {
                "index": index,
                "question": question,
                "source_state": str(state_path),
                "source_refs": refs,
                "status": result.get("status") if result else "error",
                "attempts": attempts,
                "visualization_count": len(artifacts),
                "visualizations": artifacts,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - preserve the complete replay report
            entry = {
                "index": index,
                "question": question,
                "status": "error",
                "attempts": [],
                "visualization_count": 0,
                "visualizations": [],
                "error": str(exc),
            }
            print(f"[{index:02d}/10] ERROR {exc}", flush=True)
        results.append(entry)
        (output_dir / "report.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    await tool.close()
    return output_dir, results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("cache_data/conversation_logs"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/visualization_replay"))
    parser.add_argument("--frontend", default="http://127.0.0.1:5173")
    parser.add_argument("--indexes", help="Comma-separated 1-based question indexes")
    parser.add_argument("--attempts", type=int, default=2)
    args = parser.parse_args()
    output_dir, results = asyncio.run(replay(args))
    console_errors = capture_screenshots(args.frontend.rstrip("/"), output_dir, results)
    (output_dir / "report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    created = sum(item["visualization_count"] for item in results)
    errors = sum(item["status"] == "error" for item in results)
    print(
        f"SUMMARY questions={len(results)} charts={created} errors={errors} browser_errors={len(console_errors)}",
        flush=True,
    )
    return 1 if errors or created < len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
