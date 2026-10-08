"""Cusal Model Patcher for VRAVEN.

Allows dynamic, weight-free model intervention by attaching PyTorch forward hooks
to suppress shortcuts, fix failure modes, or enforce concept constraints at runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn


class PatchedModel(nn.Module):
    """Wrapper around a PyTorch model that enforces causal concept interventions at runtime."""

    def __init__(
        self,
        model: nn.Module,
        concept_interventions: Mapping[str, float | Callable[[torch.Tensor], torch.Tensor]],
    ) -> None:
        super().__init__()
        self.model = model
        self.interventions = concept_interventions
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._attach_hooks()

    def _attach_hooks(self) -> None:
        """Attach PyTorch forward hooks to targeted modules."""
        for name, module in self.model.named_modules():
            if name in self.interventions:
                target_val = self.interventions[name]
                
                def make_hook(target):
                    def hook(mod, inputs, output):
                        if isinstance(target, (int, float)):
                            return torch.full_like(output, float(target))
                        elif callable(target):
                            return target(output)
                        return output
                    return hook

                handle = module.register_forward_hook(make_hook(target_val))
                self._hooks.append(handle)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def remove_hooks(self) -> None:
        """Remove all active intervention hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def patch_model(
    model: nn.Module,
    interventions: Mapping[str, float | Callable[[torch.Tensor], torch.Tensor]],
) -> PatchedModel:
    """Create a causally patched PyTorch model with runtime intervention hooks."""
    return PatchedModel(model, interventions)
