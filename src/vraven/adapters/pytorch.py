"""Native PyTorch adapter for VRAVEN."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import torch

from vraven.capture.pytorch_hooks import HookSession
from vraven.capture.tensor_utils import compress_flat_values
from vraven.core.config import VRavenConfig
from vraven.core.device import (
    floating_tensors,
    infer_backend,
    infer_model_device,
    iter_tensors,
    prepare_model_inputs,
    synchronise_device,
)
from vraven.core.errors import CapabilityError, DeviceMismatchError
from vraven.core.metadata import (
    model_parameter_count,
    model_structure_hash,
    model_trainable_parameter_count,
    runtime_metadata,
    tensor_input_fingerprint,
)
from vraven.core.output import PredictionComputation, compute_prediction, extract_primary_tensor
from vraven.core.trace import InputTrace, NeuralTrace, PredictionTrace


class PyTorchAdapter:
    """Instrument a live ``torch.nn.Module`` without changing its parameters."""

    framework_name = "pytorch"

    def __init__(
        self,
        model: torch.nn.Module,
        config: VRavenConfig,
        *,
        class_names: Mapping[int, str] | Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = infer_model_device(model)
        self._validate_requested_device(config.device)
        self.class_names = self._normalise_class_names(class_names)
        self._last_full_tensors: dict[str, torch.Tensor] = {}

    def _validate_requested_device(self, requested: str) -> None:
        if requested == "auto":
            return
        try:
            expected = torch.device(requested)
        except (TypeError, RuntimeError) as error:
            raise DeviceMismatchError(f"Invalid VRAVEN device declaration: {requested!r}") from error
        type_matches = expected.type == self.device.type
        index_matches = expected.index is None or expected.index == self.device.index
        if not (type_matches and index_matches):
            raise DeviceMismatchError(
                f"VRAVEN was configured for {expected}, but the live model is on {self.device}. "
                "VRAVEN does not move models. Move the model explicitly or use device='auto'."
            )

    def take_full_tensors(self) -> dict[str, torch.Tensor]:
        """Transfer ownership of explicitly retained tensors to one report."""

        tensors = self._last_full_tensors
        self._last_full_tensors = {}
        return tensors

    @staticmethod
    def _normalise_class_names(
        class_names: Mapping[int, str] | Sequence[str] | None,
    ) -> dict[int, str]:
        if class_names is None:
            return {}
        if isinstance(class_names, Mapping):
            return {int(key): str(value) for key, value in class_names.items()}
        return {index: str(value) for index, value in enumerate(class_names)}

    @staticmethod
    def _prediction_trace(computation: PredictionComputation) -> PredictionTrace:
        def values(tensor: torch.Tensor | None) -> list[float] | None:
            return tensor.detach().float().cpu().reshape(-1).tolist() if tensor is not None else None

        probabilities = (
            computation.probabilities.detach().float().cpu().tolist()
            if computation.probabilities is not None
            else None
        )
        predicted = (
            computation.predicted_classes.detach().cpu().reshape(-1).tolist()
            if computation.predicted_classes is not None
            else None
        )
        targets = (
            computation.target_indices.detach().cpu().reshape(-1).tolist()
            if computation.target_indices is not None
            else None
        )
        raw = computation.logits.detach().float().reshape(-1)
        if raw.numel() > 64:
            raw = compress_flat_values(raw, 64)
        return PredictionTrace(
            output_kind=computation.output_kind,
            logits_shape=tuple(computation.logits.shape),
            predicted_classes=[int(item) for item in predicted] if predicted is not None else None,
            target_indices=[int(item) for item in targets] if targets is not None else None,
            probabilities=probabilities,
            target_scores=computation.selected_scores.detach().float().cpu().reshape(-1).tolist(),
            confidence=values(computation.confidence),
            predictive_entropy=values(computation.entropy),
            margin=values(computation.margin),
            raw_output_sample=raw.cpu().tolist(),
        )

    def _input_trace(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        input_gradient_tensors: list[torch.Tensor],
        input_gradients: list[torch.Tensor | None],
    ) -> InputTrace:
        tensors = list(iter_tensors((args, kwargs)))
        trace = InputTrace(
            shapes=[tuple(tensor.shape) for tensor in tensors],
            dtypes=[str(tensor.dtype) for tensor in tensors],
            devices=[str(tensor.device) for tensor in tensors],
            batch_size=(
                int(tensors[0].shape[0]) if tensors and tensors[0].ndim > 0 else (1 if tensors else None)
            ),
            metadata={"tensor_count": len(tensors)},
        )
        first_pair = next(
            (
                (tensor, gradient)
                for tensor, gradient in zip(input_gradient_tensors, input_gradients, strict=True)
                if gradient is not None
            ),
            None,
        )
        if first_pair is None or not self.config.capture_input_attribution:
            return trace
        tensor, gradient = first_pair
        attribution = tensor.detach().float() * gradient.detach().float()
        # For batched tabular/image inputs, summarise across the batch so feature
        # importance reflects the explained set rather than only sample zero.
        if attribution.ndim >= 2 and attribution.shape[0] > 1:
            summary_attribution = attribution.mean(dim=0)
            summary_value = tensor.detach().float().mean(dim=0)
            reduction = "batch_mean"
        elif attribution.ndim > 0:
            summary_attribution = attribution[0]
            summary_value = tensor.detach().float()[0]
            reduction = "first_sample"
        else:
            summary_attribution = attribution
            summary_value = tensor.detach().float()
            reduction = "scalar"
        limit = max(self.config.max_units * 16, 256)
        compressed_attr = compress_flat_values(summary_attribution, limit).cpu()
        compressed_value = compress_flat_values(summary_value, limit).cpu()
        reduced = torch.stack((attribution.abs().mean(), attribution.abs().max())).cpu()
        trace.attribution_available = True
        trace.attribution_shape = tuple(summary_attribution.shape)
        trace.attribution_values = compressed_attr.tolist()
        trace.feature_values = compressed_value.tolist()
        trace.attribution_mean_abs = float(reduced[0])
        trace.attribution_max_abs = float(reduced[1])
        trace.metadata["attribution_semantics"] = "input_times_target_gradient"
        trace.metadata["attribution_sampling"] = "deterministic_even_spacing_when_truncated"
        trace.metadata["attribution_batch_reduction"] = reduction
        return trace

    def prepare_inputs(
        self,
        inputs: Any,
        *,
        require_grad: bool,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return prepare_model_inputs(
            inputs,
            device=self.device,
            move=self.config.auto_move_inputs,
            require_grad=require_grad,
        )

    def trace(
        self,
        inputs: Any,
        target: int | Sequence[int] | torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NeuralTrace:
        previous_training = self.model.training
        self._last_full_tensors = {}
        require_grad = self.config.capture_gradients
        args, kwargs = self.prepare_inputs(inputs, require_grad=require_grad)
        input_gradient_tensors = floating_tensors((args, kwargs)) if require_grad else []
        started = perf_counter()
        try:
            self.model.eval()
            context = torch.enable_grad() if require_grad else torch.inference_mode()
            with context, HookSession(self.model, self.config) as session:
                output = self.model(*args, **kwargs)
                logits = extract_primary_tensor(output)
                computation = compute_prediction(logits, target)
                layer_gradients: list[torch.Tensor | None] = []
                input_gradients: list[torch.Tensor | None] = []
                if require_grad:
                    retained = session.gradient_tensors()
                    differentiation_inputs = [*retained, *input_gradient_tensors]
                    if not differentiation_inputs or not computation.score.requires_grad:
                        raise CapabilityError(
                            "Gradient mode requires a differentiable model output and at least "
                            "one captured differentiable tensor."
                        )
                    gradients = torch.autograd.grad(
                        computation.score,
                        differentiation_inputs,
                        allow_unused=True,
                        retain_graph=False,
                        create_graph=False,
                    )
                    layer_gradients = list(gradients[: len(retained)])
                    input_gradients = list(gradients[len(retained) :])
                    session.attach_gradients(layer_gradients)
                synchronise_device(self.device)
                elapsed_ms = (perf_counter() - started) * 1000.0
                trace_metadata = {
                    "framework": self.framework_name,
                    "device": str(self.device),
                    "backend": infer_backend(self.device),
                    "model_class": self.model.__class__.__qualname__,
                    "model_module": self.model.__class__.__module__,
                    "parameter_count": model_parameter_count(self.model),
                    "trainable_parameter_count": model_trainable_parameter_count(self.model),
                    "model_structure_hash": model_structure_hash(self.model),
                    "execution_time_ms": elapsed_ms,
                    "selected_module_count": len(session.selected_modules),
                    "captured_layer_count": len(session.traces),
                    "capture_configuration": self.config.to_dict(),
                    "class_names": self.class_names,
                    "input_fingerprint_hash": tensor_input_fingerprint(
                        iter_tensors((args, kwargs))
                    ),
                    "input_hash_semantics": (
                        "sha256 over tensor shape, dtype, device type and deterministic "
                        "evenly spaced value samples"
                    ),
                    "target_score_reduction": "batch_mean",
                    "gradient_semantics": (
                        "target-specific first-order derivatives" if require_grad else None
                    ),
                    **runtime_metadata(),
                }
                if metadata:
                    trace_metadata["user_metadata"] = dict(metadata)
                trace = NeuralTrace(
                    layers=session.traces,
                    prediction=self._prediction_trace(computation),
                    inputs=self._input_trace(
                        args,
                        kwargs,
                        input_gradient_tensors,
                        input_gradients,
                    ),
                    target=(
                        target.detach().cpu().reshape(-1).tolist()
                        if isinstance(target, torch.Tensor)
                        else list(target)
                        if isinstance(target, Sequence) and not isinstance(target, (str, bytes))
                        else target
                    ),
                    mode=self.config.mode,
                    metadata=trace_metadata,
                )
                trace.validate()
                if self.config.capture_full_tensors:
                    self._last_full_tensors = dict(session.full_tensors)
                return trace
        finally:
            self.model.train(previous_training)
