"""Shortcut reliance and brittle dominance diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from vraven import explain
from vraven.core.config import VRavenConfig
from vraven.core.output import select_target
from vraven.core.tree import call_model, extract_tensors

from .contracts import AnalysisResult, InputFeatureGroup
from .dataset import TraceDataset
from .interventions import evaluation_mode
from .math_utils import hedges_g


def _replace_flat_features(
    inputs: Any,
    group: InputFeatureGroup,
    *,
    baseline: str,
) -> Any:
    counter = -1

    def replace(value: Any) -> Any:
        nonlocal counter
        if isinstance(value, torch.Tensor):
            counter += 1
            if counter != group.tensor_index:
                return value
            if not value.is_floating_point():
                raise TypeError("Shortcut interventions require floating-point input tensors")
            result = value.clone()
            flattened = result.reshape(-1)
            indices = torch.as_tensor(group.indices, device=result.device, dtype=torch.long)
            if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= flattened.numel()):
                raise IndexError(f"Input group {group.identifier} exceeds flattened input size")
            if baseline == "zero":
                flattened[indices] = 0
            elif baseline == "mean":
                flattened[indices] = flattened.mean()
            elif baseline == "median":
                flattened[indices] = flattened.median()
            else:
                raise ValueError("baseline must be 'zero', 'mean' or 'median'")
            return result
        if isinstance(value, tuple):
            return tuple(replace(item) for item in value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, Mapping):
            return type(value)((key, replace(item)) for key, item in value.items())
        return value

    replaced = replace(inputs)
    if counter < group.tensor_index:
        raise IndexError(f"Input tensor index {group.tensor_index} was not found")
    return replaced


def detect_shortcut_reliance(
    model: torch.nn.Module,
    inputs: Any,
    *,
    target: int | None,
    feature_groups: Sequence[InputFeatureGroup],
    baseline: str = "zero",
    config: VRavenConfig | None = None,
) -> AnalysisResult:
    """Measure whether narrow input groups dominate a local prediction.

    The output is a *shortcut suspicion* audit. A large score drop after changing
    a semantically brittle group shows local dependence, but proving a shortcut
    also requires domain evidence that the group is non-causal or unstable.
    """

    if not feature_groups:
        raise ValueError("At least one feature group is required")
    if baseline not in {"zero", "mean", "median"}:
        raise ValueError("baseline must be 'zero', 'mean' or 'median'")
    effective = config or VRavenConfig(mode="gradient")
    report = explain(model, inputs, target=target, config=effective, mode="gradient")
    original_score = float(report.trace.prediction.target_score)
    original_prediction = report.trace.prediction.predicted_index
    attribution_lookup = {
        attribution.tensor_index: np.asarray(
            attribution.absolute_influence, dtype=np.float64
        )
        for attribution in report.trace.input_attributions
    }

    rows = []
    tensors = extract_tensors(inputs)
    with evaluation_mode(model):
        for group in feature_groups:
            altered = _replace_flat_features(inputs, group, baseline=baseline)
            with torch.inference_mode():
                output = call_model(model, altered)
                selection = select_target(
                    output,
                    target=target,
                    class_dim=effective.class_dim,
                    probability_transform=effective.probability_transform,
                    target_reduction=effective.target_reduction,
                    class_names=effective.class_names,
                    target_label=effective.target_label,
                    output_path=effective.output_path,
                )
            intervened_score = float(selection.trace.target_score)
            score_drop = original_score - intervened_score
            absolute_attribution = attribution_lookup.get(group.tensor_index, np.array([]))
            total_attribution = float(absolute_attribution.sum())
            valid_indices = [index for index in group.indices if index < absolute_attribution.size]
            attribution_mass = (
                float(absolute_attribution[valid_indices].sum()) if valid_indices else float("nan")
            )
            dominance = (
                attribution_mass / total_attribution
                if total_attribution > 1e-12 and valid_indices
                else float("nan")
            )
            if group.tensor_index >= len(tensors):
                raise IndexError(f"Input tensor index {group.tensor_index} was not found")
            input_fraction = len(group.indices) / max(tensors[group.tensor_index].numel(), 1)
            score_scale = max(abs(original_score), abs(intervened_score), 1e-12)
            normalised_drop = score_drop / score_scale
            suspicion = (
                abs(normalised_drop) / max(input_fraction, 1e-12)
                * (1.0 if not np.isfinite(dominance) else max(dominance, 1e-6))
            )
            rows.append(
                {
                    "group": group.identifier,
                    "tensor_index": group.tensor_index,
                    "indices": list(group.indices),
                    "input_fraction": input_fraction,
                    "attribution_mass": attribution_mass,
                    "attribution_dominance": dominance,
                    "intervened_score": intervened_score,
                    "score_drop": score_drop,
                    "relative_score_drop": normalised_drop,
                    "prediction_flipped": selection.trace.predicted_index != original_prediction,
                    "shortcut_suspicion_index": suspicion,
                }
            )

    strongest = max(rows, key=lambda row: abs(row["relative_score_drop"]))
    return AnalysisResult(
        analysis="shortcut_detection",
        metrics={
            "original_target_score": original_score,
            "original_prediction": original_prediction,
            "groups_tested": len(rows),
            "strongest_group": strongest["group"],
            "maximum_absolute_relative_drop": abs(strongest["relative_score_drop"]),
            "prediction_flips": sum(bool(row["prediction_flipped"]) for row in rows),
        },
        tables={"groups": rows},
        metadata={
            "claim_type": "local intervention plus first-order diagnostic",
            "baseline": baseline,
            "suspicion_index_is_calibrated": False,
        },
        warnings=[
            "Shortcut suspicion is not proof that a feature is spurious; domain and distribution-shift evidence are required."
        ],
        limitations=[
            "Input interventions can be out of distribution.",
            "Input-by-gradient mass is local and first order.",
            "The suspicion index ranks groups only within this declared audit and has no universal threshold.",
        ],
    )


def layerwise_shortcut_probe(
    dataset: TraceDataset,
    shortcut_labels: Sequence[Any],
    *,
    folds: int = 5,
    seed: int = 42,
) -> AnalysisResult:
    """Locate where a declared shortcut variable becomes linearly decodable."""

    raw_labels = np.asarray(shortcut_labels)
    if raw_labels.shape[0] != len(dataset.sample_ids):
        raise ValueError("shortcut_labels must align with the TraceDataset")
    classes, labels = np.unique(raw_labels.astype(str), return_inverse=True)
    if len(classes) < 2:
        raise ValueError("shortcut_labels must contain at least two classes")
    if folds < 2:
        raise ValueError("folds must be at least two")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("layerwise_shortcut_probe requires scikit-learn") from exc

    rows = []
    splits = min(folds, int(np.min(np.unique(labels, return_counts=True)[1])))
    if splits < 2:
        raise ValueError("Each shortcut class needs at least two samples")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    for layer, representation in dataset.representations.items():
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        scoring = "roc_auc" if len(np.unique(labels)) == 2 else "roc_auc_ovr_weighted"
        values = cross_val_score(estimator, representation, labels, cv=cv, scoring=scoring)
        groups = [representation[labels == label] for label in np.unique(labels)]
        maximum_effect = None
        if len(groups) == 2:
            effect = hedges_g(groups[0], groups[1])
            finite = np.abs(effect[np.isfinite(effect)])
            maximum_effect = float(finite.max()) if finite.size else None
        rows.append(
            {
                "layer": layer,
                "probe_auc_mean": float(values.mean()),
                "probe_auc_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "maximum_absolute_hedges_g": maximum_effect,
            }
        )
    transition = max(rows, key=lambda row: row["probe_auc_mean"])
    return AnalysisResult(
        analysis="layerwise_shortcut_probe",
        metrics={
            "layers": len(rows),
            "most_decodable_layer": transition["layer"],
            "maximum_probe_auc": transition["probe_auc_mean"],
        },
        tables={"layers": rows},
        metadata={
            "claim_type": "observational decodability",
            "cross_validation_folds": splits,
            "seed": seed,
        },
        limitations=[
            "Linear decodability does not prove that the model uses the shortcut variable.",
            "Causal dependence should be tested with interventions or balanced counterfactual data.",
        ],
    )


def analyse_prediction_depth(
    reference: TraceDataset,
    query: TraceDataset | None = None,
    *,
    neighbours: int = 5,
) -> AnalysisResult:
    """Estimate the earliest layer whose local prediction remains final-consistent.

    A k-nearest-neighbour probe is fitted on reference representations at each
    layer. For a separate query set, prediction depth is the earliest layer from
    which all later probe predictions agree with the model's final prediction.
    When ``query`` is omitted, leave-one-out neighbours are used on the reference
    set to avoid a sample selecting itself.
    """

    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise ImportError("analyse_prediction_depth requires scikit-learn") from exc
    evaluation = query or reference
    reference_label_text = reference.true_labels.astype(str)
    classes = np.unique(reference_label_text)
    label_to_index = {label: index for index, label in enumerate(classes.tolist())}
    train_labels = np.asarray([label_to_index[label] for label in reference_label_text], dtype=int)
    final_label_text = evaluation.predicted_labels.astype(str)
    unknown = sorted(set(final_label_text.tolist()).difference(label_to_index))
    if unknown:
        raise ValueError(
            "Evaluation predictions contain labels absent from reference true labels: "
            f"{unknown}"
        )
    final_labels = np.asarray([label_to_index[label] for label in final_label_text], dtype=int)
    layers = [layer for layer in reference.layers if layer in evaluation.representations]
    if not layers:
        raise ValueError("No shared representation layers were found")
    for layer in layers:
        if reference.representations[layer].shape[1] != evaluation.representations[layer].shape[1]:
            raise ValueError(f"Reference and query representation widths differ for {layer}")
    predictions = []
    accuracies = []
    for layer in layers:
        train_x = reference.representations[layer]
        test_x = evaluation.representations[layer]
        scaler = StandardScaler().fit(train_x)
        train_scaled = scaler.transform(train_x)
        test_scaled = scaler.transform(test_x)
        k = min(max(neighbours, 1), len(train_x) - (1 if query is None else 0))
        if k < 1:
            raise ValueError("Prediction depth requires at least two reference samples")
        if query is None:
            distances = np.linalg.norm(
                train_scaled[:, None, :] - train_scaled[None, :, :], axis=-1
            )
            np.fill_diagonal(distances, np.inf)
            neighbour_indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
            labels = train_labels[neighbour_indices]
            layer_prediction = np.asarray(
                [
                    int(np.argmax(np.bincount(row, minlength=len(classes))))
                    for row in labels
                ],
                dtype=int,
            )
        else:
            estimator = KNeighborsClassifier(n_neighbors=k, weights="distance")
            estimator.fit(train_scaled, train_labels)
            layer_prediction = estimator.predict(test_scaled).astype(int)
        predictions.append(layer_prediction)
        accuracies.append(float(np.mean(layer_prediction == final_labels)))
    prediction_matrix = np.vstack(predictions)
    depths = []
    for sample in range(prediction_matrix.shape[1]):
        final = final_labels[sample]
        depth = len(layers) - 1
        for index in range(len(layers)):
            if np.all(prediction_matrix[index:, sample] == final):
                depth = index
                break
        depths.append(depth)
    rows = [
        {
            "layer": layer,
            "depth_index": index,
            "probe_final_agreement": accuracies[index],
            "samples_first_stable_here": int(np.sum(np.asarray(depths) == index)),
        }
        for index, layer in enumerate(layers)
    ]
    return AnalysisResult(
        analysis="prediction_depth",
        metrics={
            "samples": len(evaluation.sample_ids),
            "layers": len(layers),
            "mean_depth_index": float(np.mean(depths)),
            "median_depth_index": float(np.median(depths)),
            "earliest_layer": layers[int(np.min(depths))],
            "latest_layer": layers[int(np.max(depths))],
        },
        tables={"layers": rows, "sample_depth_indices": depths},
        metadata={
            "claim_type": "probe-based decision localisation",
            "neighbours": neighbours,
            "evaluation": "leave-one-out" if query is None else "separate query cohort",
            "class_encoding": {str(label): int(index) for label, index in label_to_index.items()},
        },
        limitations=[
            "Probe predictions approximate accessible class information, not the model's exact internal computation.",
            "Early depth can indicate shortcut learning, but it can also reflect genuinely easy examples.",
        ],
    )
