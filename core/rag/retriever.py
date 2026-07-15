"""Local retrieval helpers for extension knowledge."""
from __future__ import annotations

from pathlib import Path


def retrieve_local_knowledge(root: Path, query: str, filters: dict | None = None) -> list[dict]:
    filters = filters or {}
    file_glob = str(filters.get("glob") or "*.md")
    limit = int(filters.get("file_limit", 200))
    terms = [term.strip().lower() for term in query.replace("_", " ").split() if len(term.strip()) > 1]
    results: list[dict] = []
    for path in sorted(root.rglob(file_glob))[:limit]:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue
        score = _score(content, terms, path.name)
        if score <= 0:
            continue
        results.append(
            {
                "score": score,
                "source_id": str(path.relative_to(root)),
                "title": path.name,
                "snippet": _snippet(content, terms),
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)[:3]


def _score(content: str, terms: list[str], filename: str) -> int:
    haystack = f"{filename}\n{content}".lower()
    return sum(haystack.count(term) for term in terms)


def _snippet(content: str, terms: list[str]) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for line in lines:
        normalized = line.lower()
        if any(term in normalized for term in terms):
            return line[:280]
    return (lines[0] if lines else "")[:280]
