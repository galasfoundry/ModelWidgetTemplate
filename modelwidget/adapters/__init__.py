"""Import adapters so their ``@register`` decorators run."""

from modelwidget.adapters.image_classification import (
    TimmImageClassifier,
    TransformersImageClassifier,
)
from modelwidget.adapters.llama_cpp import LlamaCppTextGenerator
from modelwidget.adapters.text_classification import TransformersTextClassifier
from modelwidget.adapters.text_generation import TransformersTextGenerator

__all__ = [
    "LlamaCppTextGenerator",
    "TimmImageClassifier",
    "TransformersImageClassifier",
    "TransformersTextClassifier",
    "TransformersTextGenerator",
]
