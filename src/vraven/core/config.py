"""Configuration objects used by the VRAVEN orchestration layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ExplanationMode = Literal["fast", "gradient", "causal"]
ModulePolicy = Literal["leaf", "semantic", "all"]
CausalStrategy = Literal["zero", "mean", "scale"]


@dataclass(slots=True)
class VRavenConfig:
    """Validated configuration for one explanation run.

    ``fast`` performs a forward pass and immediately reduces captured tensors.
    ``gradient`` adds one target-specific reverse pass. ``causal`` performs the
    gradient trace and then a limited number of temporary layer interventions.
    """

    mode: ExplanationMode = "fast"
    device: str = "auto"
    auto_move_inputs: bool = True
    module_policy: ModulePolicy = "leaf"
    include_module_types: tuple[str, ...] = ()
    exclude_module_types: tuple[str, ...] = ("Dropout", "Identity")
    include_name_patterns: tuple[str, ...] = ()
    exclude_name_patterns: tuple[str, ...] = ()
    max_layers: int = 256
    max_units: int = 128
    representation_size: int = 64
    max_attention_tokens: int = 128
    histogram_bins: int = 32
    entropy_sample_limit: int = 262_144
    capture_entropy: bool = True
    capture_representations: bool = True
    capture_unit_profiles: bool = True
    capture_attention: bool = True
    capture_input_attribution: bool = True
    capture_full_tensors: bool = False
    capture_timing: bool = True
    causal_top_k: int = 3
    causal_strategy: CausalStrategy = "zero"
    causal_scale: float = 0.0
    epsilon: float = 1e-8
    random_seed: int = 17
    strict: bool = True
    # DarkSide / newer-API compatibility knobs (optional; unused by 0.2 capture).
    class_dim: int | None = None
    probability_transform: str | None = None
    target_reduction: str | None = None
    class_names: tuple[str, ...] | None = None
    target_label: str | None = None
    output_path: str | None = None

    @property
    def capture_gradients(self) -> bool:
        return self.mode in {"gradient", "causal"}

    def __post_init__(self) -> None:
        if self.mode not in {"fast", "gradient", "causal"}:
            raise ValueError("mode must be 'fast', 'gradient' or 'causal'")
        if self.module_policy not in {"leaf", "semantic", "all"}:
            raise ValueError("module_policy must be 'leaf', 'semantic' or 'all'")
        if self.causal_strategy not in {"zero", "mean", "scale"}:
            raise ValueError("Unsupported causal_strategy")
        positive_fields = {
            "max_layers": self.max_layers,
            "max_units": self.max_units,
            "representation_size": self.representation_size,
            "max_attention_tokens": self.max_attention_tokens,
            "histogram_bins": self.histogram_bins,
            "entropy_sample_limit": self.entropy_sample_limit,
            "causal_top_k": self.causal_top_k,
            "epsilon": self.epsilon,
        }
        invalid = [name for name, value in positive_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"The following values must be greater than zero: {', '.join(invalid)}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
