"""Shared capability policy for generated computation-only Python."""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from typing import Any, Mapping


class AnalysisPolicyError(ValueError):
    """Raised when generated analysis code violates the shared policy."""


SAFE_IMPORT_MODULE_NAMES = frozenset({
    "collections",
    "datetime",
    "math",
    "numpy",
    "pandas",
    "statistics",
})

_SAFE_BUILTIN_NAMES = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "hasattr", "int", "isinstance", "len", "list", "map", "max", "min",
    "pow", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
})

_BLOCKED_NAMES = frozenset({
    "__builtins__", "__import__", "breakpoint", "compile", "eval", "exec",
    "globals", "help", "input", "locals", "open", "quit", "exit", "vars",
})

_BLOCKED_NODE_TYPES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


@dataclass(frozen=True)
class ImportBinding:
    module: str
    alias: str
    attribute: str | None = None


@dataclass(frozen=True)
class PreparedAnalysisCode:
    code: str
    imports: tuple[ImportBinding, ...]
    loaded_names: frozenset[str]


def prepare_analysis_code(
    code: str,
    *,
    allowed_import_modules: set[str] | frozenset[str] = SAFE_IMPORT_MODULE_NAMES,
    require_result_assignment: bool = True,
) -> PreparedAnalysisCode:
    """Validate code and replace allow-listed imports with explicit bindings."""

    if not code or not code.strip():
        raise AnalysisPolicyError("analysis code cannot be empty")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise AnalysisPolicyError(f"analysis code is invalid Python: {exc}") from exc

    if require_result_assignment and not any(
        isinstance(node, (ast.Assign, ast.AnnAssign)) and _assigns_name(node, "result")
        for node in ast.walk(tree)
    ):
        raise AnalysisPolicyError("analysis code must assign a dict to result")

    bindings: list[ImportBinding] = []
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bindings.extend(_import_bindings(node, allowed_import_modules))
            continue
        if isinstance(node, ast.ImportFrom):
            bindings.extend(_import_from_bindings(node, allowed_import_modules))
            continue
        body.append(node)
    tree.body = body
    ast.fix_missing_locations(tree)
    _validate_sanitized_tree(tree)
    loaded_names = frozenset(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )
    return PreparedAnalysisCode(
        code=ast.unparse(tree),
        imports=tuple(bindings),
        loaded_names=loaded_names,
    )


def resolve_import_bindings(
    bindings: tuple[ImportBinding, ...],
    available_modules: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve validated symbolic imports from runtime-provided module objects."""

    resolved: dict[str, Any] = {}
    for binding in bindings:
        module = available_modules.get(binding.module)
        if module is None:
            raise AnalysisPolicyError(f"analysis code imports unavailable module: {binding.module}")
        value = module
        if binding.attribute is not None:
            if binding.attribute.startswith("_") or not hasattr(module, binding.attribute):
                raise AnalysisPolicyError(
                    f"analysis code imports unsupported name: {binding.module}.{binding.attribute}"
                )
            value = getattr(module, binding.attribute)
        resolved[binding.alias] = value
    return resolved


def safe_analysis_builtins() -> dict[str, Any]:
    return {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}


def _import_bindings(node: ast.Import, allowed: set[str] | frozenset[str]) -> list[ImportBinding]:
    output: list[ImportBinding] = []
    for item in node.names:
        if item.name not in allowed:
            raise AnalysisPolicyError(f"analysis code imports unsupported module: {item.name}")
        output.append(ImportBinding(module=item.name, alias=item.asname or item.name))
    return output


def _import_from_bindings(
    node: ast.ImportFrom,
    allowed: set[str] | frozenset[str],
) -> list[ImportBinding]:
    module = node.module or ""
    if node.level or module not in allowed:
        raise AnalysisPolicyError(f"analysis code imports unsupported module: {module}")
    output: list[ImportBinding] = []
    for item in node.names:
        if item.name == "*" or item.name.startswith("_"):
            raise AnalysisPolicyError(f"analysis code imports unsupported name: {module}.{item.name}")
        output.append(ImportBinding(module=module, attribute=item.name, alias=item.asname or item.name))
    return output


def _validate_sanitized_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise AnalysisPolicyError("analysis code imports must be top-level")
        if isinstance(node, _BLOCKED_NODE_TYPES):
            raise AnalysisPolicyError(f"analysis code contains blocked syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id in _BLOCKED_NAMES or node.id.startswith("__")):
            raise AnalysisPolicyError(f"analysis code uses blocked name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise AnalysisPolicyError(f"analysis code uses blocked attribute: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_NAMES or node.func.id.startswith("__"):
                raise AnalysisPolicyError(f"analysis code calls blocked function: {node.func.id}")


def _assigns_name(node: ast.Assign | ast.AnnAssign, name: str) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(isinstance(target, ast.Name) and target.id == name for target in targets)
