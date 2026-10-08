"""Framework-specific capture primitives."""

from .pytorch_hooks import HookSession
from .selectors import ModuleSelector
from .tensor_utils import (
    compress_flat_values,
    compressed_attention,
    compressed_representation,
    extract_tensor,
    unit_matrix,
)

__all__ = [
    "HookSession",
    "ModuleSelector",
    "compress_flat_values",
    "compressed_attention",
    "compressed_representation",
    "extract_tensor",
    "unit_matrix",
]
