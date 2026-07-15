# Query Translator Specification

## 1. Project Overview

**Project Name:** TSPilot
**Module Name:** QueryTranslator
**Type:** Core Module - Database (Python)
**File Path:** `core/database/query_translator.py`
**Core Functionality:** Translates or renders query intent into backend-specific query dialects.
**Target Users:** Query agent, guard agent.

---

## 2. Functionality Specification

### 2.1 Core Features

| Feature | Description |
|---------|-------------|
| Intent Rendering | Render logical query plans into backend-specific syntax |
| NL Fallback | Convert natural language to a query only when deterministic planning is insufficient |
| Dialect Translation | Translate between supported query dialects when needed |
| Query Validation | Validate generated backend queries |

### 2.2 Translation Types

| From | To | Use Case |
|------|-----|----------|
| Logical Query Plan | SQL | Relational execution |
| Logical Query Plan | Flux | InfluxDB execution |
| Logical Query Plan | PromQL | Prometheus execution |
| Natural Language | Backend query | Fallback path only |

### 2.3 Translator Interface

```python
class QueryTranslator:
    async def render_from_plan(
        self,
        plan: DatabaseQueryPlan,
        database: str,
        schema: DatabaseSchema,
    ) -> TranslationResult:
        """Render a backend query from a logical plan."""
        pass

    async def translate_nl_fallback(
        self,
        natural_language: str,
        database: str,
        schema: DatabaseSchema,
    ) -> TranslationResult:
        """Fallback path when no deterministic plan is available."""
        pass

    def validate_query(
        self,
        query: str,
        dialect: str,
    ) -> ValidationResult:
        """Validate query syntax and safety."""
        pass
```

### 2.4 Translation Result

```python
@dataclass
class TranslationResult:
    success: bool
    sql: str | None = None
    dialect: str
    explanation: str | None = None
    confidence: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    suggestions: list[str]
```

### 2.5 Prompt Template

```python
NL_TO_QUERY_PROMPT = """Convert the following request into the target backend query.

Database: {database_name}
Dialect: {dialect}
Schema: {schema_description}

Natural Language: {query}

Requirements:
- Preserve the requested time range exactly
- Do not add aggregations unless requested
- Respect the backend dialect
- Keep the query executable

Respond with:
Query: <your backend query>
Explanation: <brief explanation>
"""
```

---

## 3. Technical Specification

### 3.1 Renderer / Translation Integration

- Use compiler or renderer modules for deterministic plan rendering
- Use translator fallback only when deterministic rendering is insufficient

### 3.2 LLM Usage

- Use LLM fallback only after schema grounding and planning
- Never make translator prompts the only source of field mapping truth

---

## 4. Acceptance Criteria

| # | Criterion |
|---|-----------|
| 1 | Logical plans can render to multiple backend dialects |
| 2 | Time-range and aggregation semantics survive rendering |
| 3 | Validation catches wrong or unsafe generated queries |
| 4 | Explanations or warnings are provided |
| 5 | Translator fallback remains optional rather than mandatory |
