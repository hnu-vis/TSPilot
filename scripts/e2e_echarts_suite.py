#!/usr/bin/env python3
"""Run real chat-to-artifact requests and screenshot every published ECharts V5 chart."""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


QUESTIONS = [
    "这一个月比特币美元价格哪一次跌得最急？找出 2 小时内跌幅最大的那段，给出起止时间。只统计价格在 1 万到 10 万美元之间的数据。",
    "1月比特币美元价格波动大吗？用每天的最高最低价差（日波幅）的平均来衡量。只统计价格在 1 万到 10 万美元之间的数据。",
    "从月内最低点算起，价格最多反弹了多少？低点和反弹高点分别在何时？（排除异常点）",
    "1 月里比特币美元价格涨得最猛的是哪一段？找出 2 小时内涨幅最大的时间窗口。只统计价格在 1 万到 10 万美元之间的数据。",
    "这个月有没有一段时间价格连续每天都创新高？持续了几天？只统计价格在 1 万到 10 万美元之间的数据。",
    "这个月哪一天价格波动最剧烈？当天最高最低差多少？只统计价格在 1 万到 10 万美元之间的数据。",
    "比特币美元价格上半月（1/4–1/19）和下半月（1/20–2/3）的平均价各是多少？下半月比上半月高多少？只统计价格在 1 万到 10 万美元之间的数据。",
    "从这个月的最低点算起，价格花了多久才反弹 20%？只统计价格在 1 万到 10 万美元之间的数据。",
    "月末价格是不是一直站在高位？找出价格持续稳定在 23000 以上的那段时间。只统计价格在 1 万到 10 万美元之间的数据。",
    "这个月比特币美元价格从月初到月末，整体是涨还是跌？净变化了百分之几？只统计价格在 1 万到 10 万美元之间的数据。",
]


def _request_json(url: str, *, payload: dict | None = None, timeout: int = 900) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:2000]}") from exc


def _safe_title(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return normalized[:80] or fallback


def run_requests(backend: str, output_dir: Path, indexes: set[int] | None = None) -> list[dict]:
    results: list[dict] = []
    for index, question in enumerate(QUESTIONS, start=1):
        if indexes and index not in indexes:
            continue
        print(f"[{index:02d}/10] REQUEST {question}", flush=True)
        conversation_id = f"e2e-echarts-{index:02d}-{uuid.uuid4().hex[:10]}"
        try:
            response = _request_json(
                f"{backend}/api/v1/chat",
                payload={
                    "message": question,
                    "conversation_id": conversation_id,
                    "stream": False,
                    "history": [],
                    "database_context": {
                        "database_id": "influxdb2-bitcoin-sample",
                        "database_type": "influxdb",
                        "display_name": "influxdb2-bitcoin-sample",
                    },
                },
            )
            answer = response.get("answer") if isinstance(response.get("answer"), dict) else {}
            descriptors = answer.get("visualizations") if isinstance(answer.get("visualizations"), list) else []
            artifacts = []
            for chart_index, descriptor in enumerate(descriptors, start=1):
                if not isinstance(descriptor, dict) or not descriptor.get("data_ref"):
                    continue
                artifact = _request_json(f"{backend}{descriptor['data_ref']}")
                artifacts.append(artifact)
                artifact_path = output_dir / f"q{index:02d}_chart{chart_index:02d}.json"
                artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
            result = {
                "index": index,
                "question": question,
                "conversation_id": response.get("conversation_id"),
                "request_id": response.get("request_id"),
                "status": response.get("status"),
                "response_kind": response.get("response_kind"),
                "used_tools": response.get("used_tools") or [],
                "answer_summary": answer.get("summary"),
                "visualization_count": len(artifacts),
                "visualizations": artifacts,
                "error": response.get("error"),
            }
            (output_dir / f"q{index:02d}_response.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[{index:02d}/10] RESULT status={result['status']} charts={len(artifacts)} "
                f"tools={','.join(result['used_tools'])}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - preserve suite progress
            result = {
                "index": index,
                "question": question,
                "conversation_id": conversation_id,
                "status": "request_error",
                "visualization_count": 0,
                "visualizations": [],
                "error": str(exc),
            }
            print(f"[{index:02d}/10] ERROR {exc}", flush=True)
        results.append(result)
        (output_dir / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def capture_screenshots(frontend: str, output_dir: Path, results: list[dict]) -> list[dict]:
    from playwright.sync_api import sync_playwright

    console_errors: list[dict] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.on("console", lambda message: console_errors.append({
            "type": message.type,
            "text": message.text,
        }) if message.type in {"error", "warning"} else None)
        page.on("pageerror", lambda error: console_errors.append({"type": "pageerror", "text": str(error)}))
        page.goto(f"{frontend}/visualization-audit", wait_until="networkidle")
        page.wait_for_function("typeof window.__TSPILOT_RENDER_VISUALIZATION__ === 'function'")
        for result in results:
            index = int(result["index"])
            screenshots = []
            for chart_index, artifact in enumerate(result.get("visualizations") or [], start=1):
                page.evaluate("artifact => window.__TSPILOT_RENDER_VISUALIZATION__(artifact)", artifact)
                page.locator("[data-visual-audit-ready='true']").wait_for()
                title = _safe_title(str(artifact.get("title") or "chart"), "chart")
                screenshot = output_dir / f"q{index:02d}_chart{chart_index:02d}_{title}.png"
                page.locator(".visualization-audit-stage").screenshot(path=str(screenshot))
                screenshots.append(str(screenshot))
                print(f"[{index:02d}/10] SCREENSHOT {screenshot}", flush=True)
            result["screenshots"] = screenshots
        browser.close()
    (output_dir / "browser_console.json").write_text(
        json.dumps(console_errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return console_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://127.0.0.1:5680")
    parser.add_argument("--frontend", default="http://127.0.0.1:5173")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/e2e_echarts_suite"))
    parser.add_argument("--indexes", help="Comma-separated 1-based question indexes; defaults to all questions")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIR={output_dir.resolve()}", flush=True)
    indexes = {int(item) for item in args.indexes.split(",")} if args.indexes else None
    results = run_requests(args.backend.rstrip("/"), output_dir, indexes)
    console_errors = capture_screenshots(args.frontend.rstrip("/"), output_dir, results)
    (output_dir / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    created = sum(int(item.get("visualization_count") or 0) for item in results)
    failed = sum(item.get("status") not in {"completed", "partial"} for item in results)
    print(f"SUMMARY questions={len(results)} charts={created} failed={failed} browser_errors={len(console_errors)}", flush=True)
    return 1 if failed or created < len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
