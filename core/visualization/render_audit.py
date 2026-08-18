"""Production ECharts render gate backed by Playwright and a multimodal LLM."""
from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from runtime.token_usage import record_llm_token_usage
from runtime.timeout_policy import load_timeout_policy


class RenderAuditDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "revise", "unavailable"]
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self):
        if self.decision == "approve" and self.issues:
            raise ValueError("approved render audit cannot contain issues")
        if self.decision != "approve" and not self.issues:
            raise ValueError("non-approved render audit requires actionable issues")
        return self


class PlaywrightEChartsRenderAuditor:
    """Render the real frontend chart at desktop/mobile sizes before publication."""

    def __init__(
        self,
        *,
        llm,
        audit_url: str,
        llm_timeout_seconds: float | None = None,
        navigation_timeout_seconds: float | None = None,
        render_timeout_seconds: float | None = None,
    ):
        policy = load_timeout_policy().tool("visualization")
        self._llm = llm
        self._audit_url = str(audit_url).rstrip("/") or "http://127.0.0.1:5173/visualization-audit"
        self._llm_timeout_seconds = float(
            llm_timeout_seconds
            if llm_timeout_seconds is not None
            else policy.stage_seconds("llm_call_seconds")
        )
        self._navigation_timeout_ms = float(
            navigation_timeout_seconds
            if navigation_timeout_seconds is not None
            else policy.stage_seconds("browser_navigation_seconds")
        ) * 1000
        self._render_timeout_ms = float(
            render_timeout_seconds
            if render_timeout_seconds is not None
            else policy.stage_seconds("browser_render_seconds")
        ) * 1000
        self._playwright = None
        self._browser = None
        self._browser_lock = asyncio.Lock()

    async def audit(self, *, visualizations, verification, request, request_state) -> dict:
        try:
            screenshots = await self._render(visualizations)
        except Exception as exc:
            return {
                "decision": "unavailable",
                "issues": [f"Production ECharts render gate was unavailable: {exc}"],
            }
        if not screenshots:
            return {
                "decision": "unavailable",
                "issues": ["Production ECharts render gate produced no screenshots."],
            }

        manifest = [
            {
                "visualization_id": item.visualization_id,
                "title": item.title,
                "summary": item.summary,
                "required_roles": item.required_roles,
                "layers": [
                    {
                        "role": layer.role,
                        "mark": layer.mark,
                        "encoding": layer.encoding,
                        "label": layer.label,
                    }
                    for layer in item.layers
                ],
            }
            for item in visualizations
        ]
        prompt = (
            "You are the final independent visual-communication gate for a data-analysis product. "
            "Review the actual ECharts screenshots at desktop and mobile widths together with the grounded manifest. "
            "Return exactly one JSON object matching {\"decision\":\"approve\"|\"revise\"|\"unavailable\","
            "\"issues\":[str]}. Approve only when a user can inspect the verification question without misleading scales, "
            "ambiguous actual/forecast roles, missing units or boundaries, illegible labels, clipped content, severe overlap, "
            "confusing dual axes, or a visual emphasis that contradicts the interpretation. Use revise for actionable chart-design "
            "problems. Use unavailable only when the screenshots cannot be meaningfully assessed. Do not suggest a generic fallback.\n"
            f"User question: {request.message}\n"
            f"Verification: {json.dumps(verification.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Grounded chart manifest: {json.dumps(manifest, ensure_ascii=False)}"
        )
        content = [{"type": "text", "text": prompt}]
        for screenshot in screenshots:
            encoded = base64.b64encode(screenshot).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"},
            })
        messages = [
            SystemMessage(content="Audit the rendered evidence, not the planner's intent. Output JSON only."),
            HumanMessage(content=content),
        ]
        started_at = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._llm.ainvoke(messages),
                timeout=self._llm_timeout_seconds,
            )
            text = _message_text(response)
            decision = RenderAuditDecision.model_validate(json.loads(_json_object(text)))
        except Exception as exc:
            return {
                "decision": "unavailable",
                "issues": [f"Multimodal render audit was unavailable: {exc}"],
            }
        finally:
            if "response" in locals():
                record_llm_token_usage(
                    request_state,
                    source="visualization.render_audit",
                    response=response,
                    messages=messages,
                    output_text=_message_text(response),
                    duration_ms=int((time.perf_counter() - started_at) * 1000),
                )
        return decision.model_dump(mode="json")

    async def _render(self, visualizations) -> list[bytes]:
        browser = await self._ensure_browser()
        screenshots: list[bytes] = []
        for visualization in visualizations:
            payload = visualization.model_dump(mode="json")
            for width, height in ((1100, 620), (390, 620)):
                page = await browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
                try:
                    await page.goto(
                        self._audit_url,
                        wait_until="networkidle",
                        timeout=self._navigation_timeout_ms,
                    )
                    await page.wait_for_function(
                        "typeof window.__TSPILOT_RENDER_VISUALIZATION__ === 'function'",
                        timeout=self._render_timeout_ms,
                    )
                    await page.evaluate(
                        "payload => window.__TSPILOT_RENDER_VISUALIZATION__(payload)",
                        payload,
                    )
                    stage = page.locator('[data-visual-audit-ready="true"]')
                    await stage.wait_for(state="visible", timeout=self._render_timeout_ms)
                    screenshots.append(await stage.screenshot(type="png"))
                finally:
                    await page.close()
        return screenshots

    async def _ensure_browser(self):
        if self._browser is not None and self._browser.is_connected():
            return self._browser
        async with self._browser_lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise RuntimeError(
                    "Python Playwright is not installed; install the project runtime dependencies and Chromium"
                ) from exc
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            return self._browser

    async def close(self) -> None:
        browser, self._browser = self._browser, None
        playwright, self._playwright = self._playwright, None
        if browser is not None:
            try:
                if browser.is_connected():
                    await browser.close()
            except Exception:
                # A cancelled request may already have closed the Playwright
                # transport. There is no live browser left to release then.
                pass
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass


def _message_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()


def _json_object(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("render-audit response did not contain a JSON object")
    return text[start:end + 1]
