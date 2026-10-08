"""Measured internal anatomy for Linear / MLP modules.

This is not a decorative diagram. It captures live activations, multiplies them
by the layer weight matrix, and reports the units that carry local energy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from vraven.core.config import VRavenConfig
from vraven.core.tree import call_model

from .contracts import AnalysisResult
from .interventions import InterventionEngine, evaluation_mode


def _linear_modules(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear)
    ]


def inspect_internal_anatomy(
    model: nn.Module,
    inputs: Any,
    *,
    target: int | None,
    layer_names: Sequence[str] | None = None,
    top_units: int = 12,
    config: VRavenConfig | None = None,
) -> AnalysisResult:
    """Printable / plottable anatomy of Linear layers on one live forward pass.

    For each selected ``nn.Linear`` layer with weight ``W`` and bias ``b``, and
    captured pre-layer input ``a``:

    * pre-activation output ``z = W a + b``
    * input contribution proxy ``c = a ⊙ ∂||z||²/∂a``
    * outgoing column energy ``||W[:, j]||₂`` and incoming row energy ``||W[j, :]||₂``

    Claims are local and measurement-backed. They are not synaptic “reasoning”.
    """

    if top_units < 1:
        raise ValueError("top_units must be positive")
    linears = _linear_modules(model)
    if not linears:
        raise ValueError("inspect_internal_anatomy requires at least one nn.Linear module")
    available = {name: module for name, module in linears}
    selected = list(layer_names) if layer_names is not None else [name for name, _ in linears]
    missing = [name for name in selected if name not in available]
    if missing:
        raise KeyError(f"Unknown Linear layers: {missing}")

    engine = InterventionEngine(model, inputs, target=target, config=config)
    acts: dict[str, torch.Tensor] = {}
    handles = []
    for name in selected:
        module = available[name]

        def _hook(_m, inp, _out, *, layer_name: str = name) -> None:
            if isinstance(inp[0], torch.Tensor):
                acts[layer_name] = inp[0].detach()

        handles.append(module.register_forward_hook(_hook))
    try:
        with evaluation_mode(model):
            call_model(model, inputs)
            score = float(engine.evaluate().trace.target_score)
    finally:
        for handle in handles:
            handle.remove()

    layer_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    print_blocks: list[str] = []

    for name in selected:
        module = available[name]
        activation = acts[name]
        if activation.ndim == 1:
            activation = activation.unsqueeze(0)
        a = activation.float().mean(dim=0)
        weight = module.weight.detach().float()
        bias = (
            module.bias.detach().float()
            if module.bias is not None
            else torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
        )
        z = weight @ a + bias
        a_var = a.detach().clone().requires_grad_(True)
        z_var = weight @ a_var + bias
        energy = (z_var**2).sum()
        grad_a = torch.autograd.grad(energy, a_var, retain_graph=False)[0]
        contribution = (a_var.detach() * grad_a.detach()).cpu().numpy()
        column_energy = torch.linalg.vector_norm(weight, dim=0).cpu().numpy()
        row_energy = torch.linalg.vector_norm(weight, dim=1).cpu().numpy()

        a_np = a.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        order_in = np.argsort(np.abs(contribution))[::-1][:top_units]
        order_out = np.argsort(np.abs(z_np))[::-1][:top_units]

        layer_rows.append(
            {
                "layer": name,
                "in_features": int(weight.shape[1]),
                "out_features": int(weight.shape[0]),
                "input_l2": float(np.linalg.norm(a_np)),
                "output_l2": float(np.linalg.norm(z_np)),
                "weight_frobenius": float(torch.linalg.matrix_norm(weight).cpu()),
                "top_input_unit": int(order_in[0]) if order_in.size else -1,
                "top_output_unit": int(order_out[0]) if order_out.size else -1,
                "max_abs_input_contribution": float(np.max(np.abs(contribution)))
                if contribution.size
                else 0.0,
            }
        )

        for rank, index in enumerate(order_in, start=1):
            unit_rows.append(
                {
                    "layer": name,
                    "side": "input",
                    "unit": int(index),
                    "rank": rank,
                    "activation": float(a_np[index]),
                    "contribution": float(contribution[index]),
                    "outgoing_weight_l2": float(column_energy[index]),
                }
            )
        for rank, index in enumerate(order_out, start=1):
            unit_rows.append(
                {
                    "layer": name,
                    "side": "output",
                    "unit": int(index),
                    "rank": rank,
                    "preactivation": float(z_np[index]),
                    "incoming_weight_l2": float(row_energy[index]),
                }
            )

        block = [
            f"LAYER {name}  Linear({weight.shape[1]} → {weight.shape[0]})",
            (
                f"  ||a||₂={np.linalg.norm(a_np):.4g}  ||z||₂={np.linalg.norm(z_np):.4g}"
                f"  ||W||_F={float(torch.linalg.matrix_norm(weight).cpu()):.4g}"
            ),
            "  top input units (activation × ∂energy/∂a):",
        ]
        for index in order_in[: min(5, order_in.size)]:
            block.append(
                f"    in[{int(index):4d}]  a={a_np[index]:+.4g}  "
                f"c={contribution[index]:+.4g}  ||W[:,j]||={column_energy[index]:.4g}"
            )
        block.append("  top output pre-activations z = W a + b:")
        for index in order_out[: min(5, order_out.size)]:
            block.append(
                f"    out[{int(index):4d}] z={z_np[index]:+.4g}  "
                f"||W[j,:]||={row_energy[index]:.4g}"
            )
        print_blocks.append("\n".join(block))

    text = "\n\n".join(print_blocks)
    return AnalysisResult(
        analysis="internal_anatomy",
        metrics={
            "layers": len(layer_rows),
            "target_score": score,
            "top_units": top_units,
        },
        tables={
            "layers": layer_rows,
            "units": unit_rows,
            "text": text,
        },
        metadata={
            "claim_type": "local measured linear anatomy",
            "formula": "z = W a + b; input contribution proxy c = a ⊙ ∂||z||²/∂a",
        },
        limitations=[
            "Anatomy is local to each Linear module and one forward sample (batch-mean).",
            "Input contribution uses energy of z, not the full-network target score gradient.",
            "This is not a claim that the network reasons symbolically.",
        ],
    )


def print_internal_anatomy(result: AnalysisResult) -> str:
    """Return the printable internal anatomy block from ``inspect_internal_anatomy``."""

    if result.analysis != "internal_anatomy":
        raise ValueError("print_internal_anatomy expects an internal_anatomy result")
    text = result.tables.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("internal_anatomy result has no text table")
    return text
