"""Module-selection policy for PyTorch capture."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

import torch

from vraven.core.config import VRavenConfig

_SEMANTIC_TYPES = (
    torch.nn.Linear,
    torch.nn.modules.conv._ConvNd,
    torch.nn.MultiheadAttention,
    torch.nn.Embedding,
    torch.nn.RNN,
    torch.nn.GRU,
    torch.nn.LSTM,
    torch.nn.TransformerEncoderLayer,
    torch.nn.TransformerDecoderLayer,
)


@dataclass(slots=True)
class ModuleSelector:
    config: VRavenConfig

    def _matches_pattern(self, name: str, patterns: tuple[str, ...]) -> bool:
        return any(fnmatch(name, pattern) or pattern in name for pattern in patterns)

    def _include(self, name: str, module: torch.nn.Module) -> bool:
        type_name = module.__class__.__name__
        if type_name in self.config.exclude_module_types:
            return False
        if self.config.exclude_name_patterns and self._matches_pattern(
            name, self.config.exclude_name_patterns
        ):
            return False
        if self.config.include_module_types and type_name not in self.config.include_module_types:
            return False
        if self.config.include_name_patterns and not self._matches_pattern(
            name, self.config.include_name_patterns
        ):
            return False
        if self.config.module_policy == "all":
            return True
        if self.config.module_policy == "semantic":
            return isinstance(module, _SEMANTIC_TYPES)
        return len(list(module.children())) == 0 or isinstance(module, torch.nn.MultiheadAttention)

    def select(self, model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
        selected: list[tuple[str, torch.nn.Module]] = []
        for name, module in model.named_modules():
            if not name:
                continue
            if self._include(name, module):
                selected.append((name, module))
            if len(selected) >= self.config.max_layers:
                break
        return selected
