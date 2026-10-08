"""Automatic internal concept discovery and conservative semantic grounding."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils import _pytree as pytree

from .capture import CapturedModel
from .ir import IRNode, MechanismRole, NodeKind
from .runtime import FXValueRecorder, activation_matrix, infer_feature_axis


@dataclass(slots=True)
class ConceptDiscoveryConfig:
    method: str = "sparse_autoencoder"
    max_nodes: int = 6
    concepts_per_node: int = 6
    batch_size: int = 128
    min_samples: int = 16
    min_features: int = 2
    max_features: int = 4096
    sparse_epochs: int = 180
    sparse_learning_rate: float = 2e-3
    sparse_l1: float = 2e-3
    sparse_topk: int | None = 2
    sparse_coherence: float = 1e-3
    sparse_patience: int = 25
    pca_variance_floor: float = 1e-9
    seed: int = 17
    roles: tuple[MechanismRole, ...] = (
        MechanismRole.DETECT,
        MechanismRole.TRANSFORM,
        MechanismRole.GATE,
        MechanismRole.MERGE,
        MechanismRole.ACCUMULATE,
        MechanismRole.ATTEND,
        MechanismRole.STORE,
        MechanismRole.READOUT,
        MechanismRole.UNKNOWN,
    )


@dataclass(slots=True)
class GroundingEvidence:
    candidate_label: str
    association: float
    absolute_association: float
    permutation_p: float | None = None
    source: str = "input_association"
    validated: bool = False
    notes: str = "Association is not proof of semantic identity or causal use."


@dataclass(slots=True)
class Concept:
    id: str
    node_id: str
    node_name: str
    depth: float
    index: int
    feature_axis: int
    direction: Tensor
    mean_activation: float
    std_activation: float
    activation_sparsity: float
    score_variance: float
    top_samples: tuple[int, ...]
    preferred_class: int | None = None
    class_effect_size: float | None = None
    semantic_label: str | None = None
    semantic_confidence: float = 0.0
    grounding: list[GroundingEvidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return self.semantic_label or f"Unresolved internal concept {self.id}"

    @property
    def semantically_grounded(self) -> bool:
        if not (self.semantic_label and self.semantic_confidence > 0.0):
            return False
        # Association-derived labels make a concept easier to inspect, but they are
        # not counted as semantically grounded until independent evidence validates
        # the identity.  This prevents attractive labels from inflating the
        # explanation-coverage certificate.
        return self.metadata.get("semantic_status") != "association_only"

    @property
    def semantically_associated(self) -> bool:
        return bool(
            self.semantic_label
            and self.semantic_confidence > 0.0
            and self.metadata.get("semantic_status") == "association_only"
        )

    def to_dict(self, *, include_direction: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["direction"] = (
            self.direction.detach().cpu().tolist() if include_direction else None
        )
        payload["grounding"] = [asdict(item) for item in self.grounding]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Concept:
        return cls(
            id=str(payload["id"]),
            node_id=str(payload["node_id"]),
            node_name=str(payload.get("node_name", payload["node_id"])),
            depth=float(payload["depth"]),
            index=int(payload["index"]),
            feature_axis=int(payload["feature_axis"]),
            direction=torch.as_tensor(payload["direction"], dtype=torch.float32),
            mean_activation=float(payload["mean_activation"]),
            std_activation=float(payload["std_activation"]),
            activation_sparsity=float(payload["activation_sparsity"]),
            score_variance=float(payload["score_variance"]),
            top_samples=tuple(int(item) for item in payload.get("top_samples", ())),
            preferred_class=(
                int(payload["preferred_class"])
                if payload.get("preferred_class") is not None
                else None
            ),
            class_effect_size=(
                float(payload["class_effect_size"])
                if payload.get("class_effect_size") is not None
                else None
            ),
            semantic_label=payload.get("semantic_label"),
            semantic_confidence=float(payload.get("semantic_confidence", 0.0)),
            grounding=[GroundingEvidence(**item) for item in payload.get("grounding", ())],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ConceptSet:
    node_id: str
    node_name: str
    depth: float
    feature_axis: int
    concepts: list[Concept]
    scores: np.ndarray
    method: str
    reconstruction_error: float
    explained_variance: float
    feature_count: int
    sample_count: int
    activation_mean: np.ndarray | None = None
    activation_std: np.ndarray | None = None
    scoring_weight: np.ndarray | None = None
    scoring_bias: np.ndarray | None = None
    scoring_nonlinearity: str = "identity"
    scoring_topk: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def score_for(self, concept_id: str) -> np.ndarray:
        for concept in self.concepts:
            if concept.id == concept_id:
                return self.scores[:, concept.index]
        raise KeyError(concept_id)

    def score_activation(self, value: Tensor) -> Tensor:
        """Project an unseen activation into this set's fitted concept coordinates.

        The transformation must exactly match the transformation used while fitting the
        concept set.  Using a raw dot product for centred PCA coordinates, for example,
        silently shifts every local explanation and can make an executable programme
        disagree with the sample it is supposed to explain.
        """

        matrix = activation_matrix(value, self.feature_axis)
        if self.scoring_weight is None:
            weight = torch.stack(
                [concept.direction for concept in self.concepts], dim=0
            ).to(device=matrix.device, dtype=matrix.dtype)
        else:
            weight = torch.as_tensor(
                self.scoring_weight, device=matrix.device, dtype=matrix.dtype
            )
        bias = (
            torch.zeros(weight.shape[0], device=matrix.device, dtype=matrix.dtype)
            if self.scoring_bias is None
            else torch.as_tensor(
                self.scoring_bias, device=matrix.device, dtype=matrix.dtype
            )
        )
        scores = matrix @ weight.t() + bias
        if self.scoring_nonlinearity == "identity":
            return scores
        if self.scoring_nonlinearity == "relu_topk":
            scores = torch.relu(scores)
            if self.scoring_topk is not None and self.scoring_topk < scores.shape[-1]:
                values, indices = torch.topk(scores, self.scoring_topk, dim=-1)
                sparse = torch.zeros_like(scores)
                sparse.scatter_(-1, indices, values)
                scores = sparse
            return scores
        raise ValueError(
            f"Unsupported concept scoring nonlinearity: {self.scoring_nonlinearity!r}"
        )

    def score_component(self, value: Tensor, concept_index: int) -> Tensor:
        return self.score_activation(value)[:, int(concept_index)]

    def raw_projection_for_score(self, concept_index: int, score: float) -> float:
        """Translate a fitted concept score into a raw direction coefficient.

        Exact conversion is available for the linear PCA coordinate system.  Sparse
        autoencoder codes are non-linear and potentially top-k coupled, so treating a
        code value as a raw projection is not scientifically valid.
        """

        if self.scoring_nonlinearity != "identity":
            raise ValueError(
                "Raw concept-setting is currently exact only for linear concept "
                "coordinates such as PCA. Use erasure or a decoder-aware intervention "
                "for sparse-autoencoder concepts."
            )
        bias = 0.0
        if self.scoring_bias is not None:
            bias = float(np.asarray(self.scoring_bias)[int(concept_index)])
        return float(score) - bias


@dataclass(slots=True)
class ConceptAtlas:
    concept_sets: list[ConceptSet]
    sample_ids: tuple[str, ...]
    labels: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def concepts(self) -> list[Concept]:
        return [concept for concept_set in self.concept_sets for concept in concept_set.concepts]

    def concept_map(self) -> dict[str, Concept]:
        return {concept.id: concept for concept in self.concepts}

    def set_for_node(self, node_id: str) -> ConceptSet:
        for concept_set in self.concept_sets:
            if concept_set.node_id == node_id:
                return concept_set
        raise KeyError(node_id)

    def score_matrix(self, concept_ids: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
        concept_ids = list(concept_ids or [concept.id for concept in self.concepts])
        columns: list[np.ndarray] = []
        retained: list[str] = []
        for concept_id in concept_ids:
            concept = self.concept_map().get(concept_id)
            if concept is None:
                continue
            columns.append(self.set_for_node(concept.node_id).score_for(concept_id))
            retained.append(concept_id)
        if not columns:
            return np.empty((len(self.sample_ids), 0), dtype=float), []
        return np.column_stack(columns), retained

    def save(self, directory: str | Path) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        metadata = {
            "sample_ids": list(self.sample_ids),
            "labels": self.labels.tolist() if self.labels is not None else None,
            "metadata": self.metadata,
            "sets": [],
        }
        arrays: dict[str, np.ndarray] = {}
        for index, concept_set in enumerate(self.concept_sets):
            key = f"scores_{index}"
            arrays[key] = np.asarray(concept_set.scores, dtype=np.float32)
            metadata["sets"].append(
                {
                    "node_id": concept_set.node_id,
                    "node_name": concept_set.node_name,
                    "depth": concept_set.depth,
                    "feature_axis": concept_set.feature_axis,
                    "concepts": [concept.to_dict() for concept in concept_set.concepts],
                    "scores_key": key,
                    "method": concept_set.method,
                    "reconstruction_error": concept_set.reconstruction_error,
                    "explained_variance": concept_set.explained_variance,
                    "feature_count": concept_set.feature_count,
                    "sample_count": concept_set.sample_count,
                    "activation_mean": (
                        concept_set.activation_mean.tolist()
                        if concept_set.activation_mean is not None
                        else None
                    ),
                    "activation_std": (
                        concept_set.activation_std.tolist()
                        if concept_set.activation_std is not None
                        else None
                    ),
                    "scoring_weight": (
                        concept_set.scoring_weight.tolist()
                        if concept_set.scoring_weight is not None
                        else None
                    ),
                    "scoring_bias": (
                        concept_set.scoring_bias.tolist()
                        if concept_set.scoring_bias is not None
                        else None
                    ),
                    "scoring_nonlinearity": concept_set.scoring_nonlinearity,
                    "scoring_topk": concept_set.scoring_topk,
                    "metadata": concept_set.metadata,
                }
            )
        import json

        (destination / "concept_atlas.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        np.savez_compressed(destination / "concept_scores.npz", **arrays)
        return destination


class ConceptDiscoveryEngine:
    def __init__(self, config: ConceptDiscoveryConfig | None = None) -> None:
        self.config = config or ConceptDiscoveryConfig()

    def discover(
        self,
        captured: CapturedModel,
        reference_args: Sequence[Any],
        *,
        reference_kwargs: Mapping[str, Any] | None = None,
        labels: Sequence[int] | np.ndarray | Tensor | None = None,
        sample_ids: Sequence[str] | None = None,
        node_ids: Sequence[str] | None = None,
    ) -> ConceptAtlas:
        if captured.graph_module is None:
            raise ValueError(
                "Automatic concept discovery requires an export or FX graph capture. "
                f"Current route: {captured.ir.capture_route.value}."
            )
        labels_array = _labels_to_numpy(labels)
        sample_count = _batch_length((tuple(reference_args), dict(reference_kwargs or {})))
        if sample_count < self.config.min_samples:
            raise ValueError(
                f"Concept discovery requires at least {self.config.min_samples} samples; "
                f"received {sample_count}."
            )
        selected = list(node_ids or select_concept_nodes(captured, self.config))
        matrices, feature_axes = collect_activation_matrices(
            captured,
            reference_args,
            reference_kwargs=reference_kwargs,
            node_ids=selected,
            batch_size=self.config.batch_size,
        )
        topological = captured.ir.topological_index()
        denominator = max(1, len(captured.ir.nodes) - 1)
        concept_sets: list[ConceptSet] = []
        for node_id in selected:
            matrix = matrices.get(node_id)
            if matrix is None:
                continue
            if matrix.shape[0] < self.config.min_samples:
                continue
            if not (self.config.min_features <= matrix.shape[1] <= self.config.max_features):
                continue
            node = captured.ir.get_node(node_id)
            depth = topological.get(node_id, 0) / denominator
            concept_set = self._discover_node(
                matrix,
                node=node,
                depth=depth,
                feature_axis=feature_axes[node_id],
                labels=labels_array,
            )
            concept_sets.append(concept_set)
        ids = tuple(sample_ids or [f"sample_{index}" for index in range(sample_count)])
        if len(ids) != sample_count:
            raise ValueError("sample_ids length must match reference batch size")
        return ConceptAtlas(
            concept_sets=concept_sets,
            sample_ids=ids,
            labels=labels_array,
            metadata={
                "method": self.config.method,
                "selected_nodes": selected,
                "seed": self.config.seed,
                "claim": (
                    "Discovered components are candidate latent features. Semantic labels require "
                    "separate grounding evidence and causal use requires intervention testing."
                ),
            },
        )

    def _discover_node(
        self,
        matrix: np.ndarray,
        *,
        node: IRNode,
        depth: float,
        feature_axis: int,
        labels: np.ndarray | None,
    ) -> ConceptSet:
        finite = np.isfinite(matrix).all(axis=1)
        matrix = np.asarray(matrix[finite], dtype=np.float32)
        labels_local = labels[finite] if labels is not None else None
        if matrix.shape[0] < self.config.min_samples:
            raise ValueError(f"Insufficient finite samples at node {node.id}")
        concepts = min(self.config.concepts_per_node, matrix.shape[1], max(1, matrix.shape[0] - 1))
        method = self.config.method.lower()
        if method == "sparse_autoencoder":
            try:
                decomposition = _fit_sparse_autoencoder(matrix, concepts, self.config)
            except Exception as error:
                warnings.warn(
                    f"Sparse concept discovery failed for {node.id}: {error}. Falling back to PCA.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                decomposition = _fit_pca(matrix, concepts, self.config)
                method = "pca_fallback"
        elif method == "pca":
            decomposition = _fit_pca(matrix, concepts, self.config)
        else:
            raise ValueError("Concept discovery method must be 'sparse_autoencoder' or 'pca'")

        scores = decomposition.scores
        directions = decomposition.directions
        concept_records: list[Concept] = []
        for index in range(scores.shape[1]):
            score = scores[:, index]
            order = np.argsort(np.abs(score))[::-1]
            top_samples = tuple(int(item) for item in order[: min(12, len(order))])
            preferred_class = None
            effect_size = None
            if labels_local is not None and np.unique(labels_local).size == 2:
                classes = np.unique(labels_local)
                group_a = score[labels_local == classes[0]]
                group_b = score[labels_local == classes[1]]
                effect_size = _hedges_g(group_b, group_a)
                preferred_class = int(classes[1] if np.mean(group_b) >= np.mean(group_a) else classes[0])
            concept_id = _concept_id(node.id, index)
            threshold = max(1e-8, 0.05 * float(np.std(score)))
            concept_records.append(
                Concept(
                    id=concept_id,
                    node_id=node.id,
                    node_name=node.name,
                    depth=depth,
                    index=index,
                    feature_axis=feature_axis,
                    direction=torch.as_tensor(directions[index], dtype=torch.float32),
                    mean_activation=float(np.mean(score)),
                    std_activation=float(np.std(score)),
                    activation_sparsity=float(np.mean(np.abs(score) <= threshold)),
                    score_variance=float(np.var(score)),
                    top_samples=top_samples,
                    preferred_class=preferred_class,
                    class_effect_size=effect_size,
                    metadata={
                        "method": method,
                        "operator_role": node.role.value,
                        "operator_target": node.target,
                        "association_only": True,
                    },
                )
            )
        return ConceptSet(
            node_id=node.id,
            node_name=node.name,
            depth=depth,
            feature_axis=feature_axis,
            concepts=concept_records,
            scores=scores,
            method=method,
            reconstruction_error=decomposition.reconstruction_error,
            explained_variance=decomposition.explained_variance,
            feature_count=matrix.shape[1],
            sample_count=matrix.shape[0],
            activation_mean=matrix.mean(axis=0),
            activation_std=matrix.std(axis=0),
            scoring_weight=decomposition.scoring_weight,
            scoring_bias=decomposition.scoring_bias,
            scoring_nonlinearity=decomposition.scoring_nonlinearity,
            scoring_topk=decomposition.scoring_topk,
            metadata={"finite_samples": int(matrix.shape[0])},
        )


@dataclass(slots=True)
class _Decomposition:
    directions: np.ndarray
    scores: np.ndarray
    reconstruction_error: float
    explained_variance: float
    scoring_weight: np.ndarray
    scoring_bias: np.ndarray
    scoring_nonlinearity: str = "identity"
    scoring_topk: int | None = None


class _TopKSparseAutoencoder(nn.Module):
    def __init__(self, features: int, concepts: int, topk: int | None) -> None:
        super().__init__()
        self.encoder = nn.Linear(features, concepts)
        self.decoder = nn.Linear(concepts, features, bias=True)
        self.topk = min(topk, concepts) if topk is not None else None
        nn.init.orthogonal_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
            self.decoder.bias.zero_()

    def encode(self, value: Tensor) -> Tensor:
        code = torch.relu(self.encoder(value))
        if self.topk is not None and self.topk < code.shape[-1]:
            values, indices = torch.topk(code, self.topk, dim=-1)
            sparse = torch.zeros_like(code)
            sparse.scatter_(-1, indices, values)
            code = sparse
        return code

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        code = self.encode(value)
        return self.decoder(code), code


def _fit_sparse_autoencoder(
    matrix: np.ndarray,
    concepts: int,
    config: ConceptDiscoveryConfig,
) -> _Decomposition:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cpu")
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std().clamp_min(1e-6)
    centred = (x - mean) / scale
    model = _TopKSparseAutoencoder(matrix.shape[1], concepts, config.sparse_topk).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.sparse_learning_rate)
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    patience = 0
    for _epoch in range(config.sparse_epochs):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, code = model(centred)
        mse = torch.mean((reconstruction - centred) ** 2)
        l1 = torch.mean(torch.abs(code))
        directions = model.decoder.weight.t()
        normalised = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
        gram = normalised @ normalised.t()
        off_diagonal = gram - torch.eye(gram.shape[0], device=gram.device)
        coherence = torch.mean(off_diagonal**2)
        loss = mse + config.sparse_l1 * l1 + config.sparse_coherence * coherence
        loss.backward()
        optimizer.step()
        current = float(loss.item())
        if current < best_loss - 1e-7:
            best_loss = current
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.sparse_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        reconstruction, code = model(centred)
        directions = model.decoder.weight.t().detach()
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
        reconstruction_original = reconstruction * scale + mean
        mse_original = torch.mean((reconstruction_original - x) ** 2)
        total_variance = torch.mean((x - x.mean(dim=0, keepdim=True)) ** 2).clamp_min(1e-12)
        explained = 1.0 - float((mse_original / total_variance).item())
    # The global scalar scale does not rotate directions, so decoder directions remain
    # valid intervention axes in the original activation coordinates.
    return _Decomposition(
        directions=directions.cpu().numpy().astype(np.float32),
        scores=code.cpu().numpy().astype(np.float32),
        reconstruction_error=float(mse_original.item()),
        explained_variance=explained,
        scoring_weight=(
            model.encoder.weight.detach().cpu().numpy().astype(np.float32)
            / float(scale.item())
        ),
        scoring_bias=(
            model.encoder.bias.detach().cpu().numpy().astype(np.float32)
            - (
                mean.detach().cpu().numpy().reshape(-1)
                / float(scale.item())
            )
            @ model.encoder.weight.detach().cpu().numpy().astype(np.float32).T
        ).astype(np.float32),
        scoring_nonlinearity="relu_topk",
        scoring_topk=model.topk,
    )


def _fit_pca(
    matrix: np.ndarray,
    concepts: int,
    config: ConceptDiscoveryConfig,
) -> _Decomposition:
    mean = matrix.mean(axis=0, keepdims=True)
    centred = matrix - mean
    u, singular, vh = np.linalg.svd(centred, full_matrices=False)
    retained = min(concepts, vh.shape[0])
    directions = vh[:retained]
    scores = centred @ directions.T
    reconstruction = scores @ directions + mean
    mse = float(np.mean((matrix - reconstruction) ** 2))
    total = float(np.sum(singular**2))
    explained = float(np.sum(singular[:retained] ** 2) / total) if total > config.pca_variance_floor else 0.0
    return _Decomposition(
        directions=directions.astype(np.float32),
        scores=scores.astype(np.float32),
        reconstruction_error=mse,
        explained_variance=explained,
        scoring_weight=directions.astype(np.float32),
        scoring_bias=(-mean @ directions.T).reshape(-1).astype(np.float32),
        scoring_nonlinearity="identity",
        scoring_topk=None,
    )


def select_concept_nodes(
    captured: CapturedModel,
    config: ConceptDiscoveryConfig,
) -> list[str]:
    candidates: list[IRNode] = []
    for node in captured.ir.nodes:
        if node.kind not in {NodeKind.OPERATION, NodeKind.UNRESOLVED}:
            continue
        if node.role not in config.roles:
            continue
        if node.tensor is None or node.tensor.rank < 2:
            continue
        candidates.append(node)
    if len(candidates) <= config.max_nodes:
        return [node.id for node in candidates]
    # Select nodes evenly across topological depth rather than taking only early layers.
    positions = np.linspace(0, len(candidates) - 1, config.max_nodes)
    selected_indices = sorted({int(round(position)) for position in positions})
    return [candidates[index].id for index in selected_indices]


def collect_activation_matrices(
    captured: CapturedModel,
    reference_args: Sequence[Any],
    *,
    reference_kwargs: Mapping[str, Any] | None,
    node_ids: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if captured.graph_module is None:
        raise ValueError("Activation collection requires a graph capture")
    chunks: dict[str, list[np.ndarray]] = {node_id: [] for node_id in node_ids}
    axes: dict[str, int] = {}
    reference = (tuple(reference_args), dict(reference_kwargs or {}))
    total = _batch_length(reference)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        args_batch, kwargs_batch = _slice_reference(reference, start, end)
        recorder = FXValueRecorder(
            captured.graph_module,
            selected_nodes=node_ids,
            detach=True,
            move_to_cpu=True,
        )
        record = recorder.execute(args_batch, kwargs_batch)
        for node_id in node_ids:
            value = _first_tensor(record.values.get(node_id))
            if value is None or value.ndim < 2:
                continue
            if node_id not in axes:
                node = captured.ir.get_node(node_id)
                hint = str(node.metadata.get("feature_axis_hint", "auto"))
                axes[node_id] = infer_feature_axis(value, hint)
            matrix = activation_matrix(value, axes[node_id])
            chunks[node_id].append(matrix.numpy().astype(np.float32, copy=False))
    output = {
        node_id: np.concatenate(node_chunks, axis=0)
        for node_id, node_chunks in chunks.items()
        if node_chunks
    }
    return output, axes


def ground_concepts_from_input_groups(
    atlas: ConceptAtlas,
    inputs: np.ndarray | Tensor,
    groups: Mapping[str, Sequence[int]],
    *,
    permutations: int = 200,
    seed: int = 17,
    minimum_absolute_association: float = 0.25,
    promote_labels: bool = False,
    validated_labels: Sequence[str] | None = None,
    validation_alpha: float = 0.05,
) -> ConceptAtlas:
    """Attach conservative input-association evidence to discovered concepts.

    This function deliberately calls the result an association.  ``promote_labels`` is
    opt-in and still records that the label has not been independently validated.
    ``validated_labels`` is intended only for externally defined concepts whose semantic
    identity is known independently, such as planted benchmark concepts or a reviewed
    domain ontology.  It must not be used to upgrade a convenient post-hoc name.
    """

    x = inputs.detach().cpu().numpy() if isinstance(inputs, Tensor) else np.asarray(inputs)
    if x.ndim != 2:
        raise ValueError("Input grounding currently requires a two-dimensional feature matrix")
    if x.shape[0] != len(atlas.sample_ids):
        raise ValueError("Input rows must align with the ConceptAtlas samples")
    rng = np.random.default_rng(seed)
    validated = {str(item) for item in (validated_labels or ())}
    group_values: dict[str, np.ndarray] = {}
    for label, indices in groups.items():
        valid = [int(index) for index in indices if 0 <= int(index) < x.shape[1]]
        if not valid:
            continue
        group_values[str(label)] = np.nanmean(x[:, valid], axis=1)
    for concept_set in atlas.concept_sets:
        for concept in concept_set.concepts:
            score = concept_set.scores[:, concept.index]
            evidence: list[GroundingEvidence] = []
            for label, group_score in group_values.items():
                association = _pearson(score, group_score)
                if abs(association) < minimum_absolute_association:
                    continue
                null = []
                for _ in range(permutations):
                    null.append(abs(_pearson(score, rng.permutation(group_score))))
                p_value = float((1 + sum(item >= abs(association) for item in null)) / (1 + len(null)))
                evidence.append(
                    GroundingEvidence(
                        candidate_label=label,
                        association=association,
                        absolute_association=abs(association),
                        permutation_p=p_value,
                        validated=bool(label in validated and p_value <= validation_alpha),
                        source=(
                            "externally_validated_input_concept"
                            if label in validated
                            else "input_association"
                        ),
                    )
                )
            evidence.sort(key=lambda item: item.absolute_association, reverse=True)
            concept.grounding = evidence
            if promote_labels and evidence:
                best = evidence[0]
                if best.validated:
                    concept.semantic_label = best.candidate_label
                    concept.semantic_confidence = min(
                        0.99,
                        0.50 + 0.49 * best.absolute_association,
                    )
                    concept.metadata["semantic_status"] = "externally_validated"
                    concept.metadata["semantic_validation_source"] = best.source
                else:
                    concept.semantic_label = f"Associated with {best.candidate_label}"
                    concept.semantic_confidence = min(0.49, best.absolute_association)
                    concept.metadata["semantic_status"] = "association_only"
    return atlas


def _slice_reference(
    reference: tuple[tuple[Any, ...], dict[str, Any]],
    start: int,
    end: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args, kwargs = reference

    def slice_item(item: Any) -> Any:
        if isinstance(item, Tensor) and item.ndim > 0:
            return item[start:end]
        if isinstance(item, np.ndarray) and item.ndim > 0:
            return torch.as_tensor(item[start:end])
        return item

    return (
        tuple(pytree.tree_map(slice_item, args)),
        dict(pytree.tree_map(slice_item, kwargs)),
    )


def _batch_length(reference: Any) -> int:
    tensors, _ = pytree.tree_flatten(reference)
    lengths = [int(item.shape[0]) for item in tensors if isinstance(item, (Tensor, np.ndarray)) and item.ndim > 0]
    if not lengths:
        raise ValueError("Reference data must contain at least one batched tensor or array")
    if len(set(lengths)) != 1:
        raise ValueError(f"All reference tensors must share a batch dimension; received {lengths}")
    return lengths[0]


def _first_tensor(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    flat, _ = pytree.tree_flatten(value)
    return next((item for item in flat if isinstance(item, Tensor)), None)


def _labels_to_numpy(labels: Sequence[int] | np.ndarray | Tensor | None) -> np.ndarray | None:
    if labels is None:
        return None
    if isinstance(labels, Tensor):
        return labels.detach().cpu().numpy().reshape(-1)
    return np.asarray(labels).reshape(-1)


def _concept_id(node_id: str, index: int) -> str:
    digest = hashlib.sha1(node_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:5]
    return f"C{digest}_{index:02d}"


def _hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    pooled_numerator = (a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)
    pooled_denominator = a.size + b.size - 2
    if pooled_denominator <= 0:
        return 0.0
    pooled = math.sqrt(max(pooled_numerator / pooled_denominator, 0.0))
    if pooled <= 1e-12:
        return 0.0
    d = (np.mean(a) - np.mean(b)) / pooled
    correction = 1.0 - 3.0 / max(4.0 * (a.size + b.size) - 9.0, 1.0)
    return float(correction * d)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        return 0.0
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    return float(np.dot(a_centered, b_centered) / denominator) if denominator > 1e-12 else 0.0
