"""Grounded presentation-source inventory and template materialization."""

from core.visualization.materializer import (
    IncompatibleVisualDomainError,
    InvalidPresentationLineageError,
    PresentationCatalog,
)
from core.visualization.echarts import EChartsCompiler, EChartsValidationError, grounded_annotation_fields
from core.visualization.artifact_store import VisualizationArtifactStore

__all__ = [
    "InvalidPresentationLineageError",
    "IncompatibleVisualDomainError",
    "PresentationCatalog",
    "EChartsCompiler",
    "EChartsValidationError",
    "grounded_annotation_fields",
    "VisualizationArtifactStore",
]
