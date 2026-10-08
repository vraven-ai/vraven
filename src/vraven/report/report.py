"""High-level report object returned by VRAVEN explainers."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from vraven.attribution.lrp import LRPResult
from vraven.causal.counterfactual import CounterfactualResult
from vraven.causal.perturbation import PerturbationResult
from vraven.core.errors import CapabilityError
from vraven.core.trace import NeuralTrace
from vraven.metrics.evidence_audit import ScientificEvidenceAudit, scientific_evidence_audit
from vraven.reasoning.fingerprint import reasoning_fingerprint
from vraven.report.export import html_to_pdf, resolve_report_destination
from vraven.visualisation.attention import attention_structure_figure
from vraven.visualisation.causal_pathway import causal_pathway_figure
from vraven.visualisation.community_graph import community_graph_figure
from vraven.visualisation.conflict_field import conflict_field_figure
from vraven.visualisation.consistency import AgreementMetric, consistency_matrix_figure
from vraven.visualisation.counterfactual_delta import counterfactual_delta_figure
from vraven.visualisation.decision_flow import decision_flow_figure
from vraven.visualisation.decision_sankey import decision_sankey_figure
from vraven.visualisation.energy import activation_energy_figure
from vraven.visualisation.evidence_trajectory import evidence_trajectory_figure
from vraven.visualisation.export import export_figure
from vraven.visualisation.feature_sequence_heatmap import feature_sequence_heatmap_figure
from vraven.visualisation.input_attribution import input_attribution_figure
from vraven.visualisation.latent_flight import latent_flight_figure
from vraven.visualisation.mechanism_concordance import mechanism_concordance_figure
from vraven.visualisation.neuron_impact import neuron_impact_figure
from vraven.visualisation.neuron_importance import neuron_importance_figure
from vraven.visualisation.palette import RenderMode
from vraven.visualisation.perturbation_stability import perturbation_stability_figure
from vraven.visualisation.reasoning_dna import reasoning_dna_figure
from vraven.visualisation.reasoning_graph import reasoning_graph_figure
from vraven.visualisation.relevance_flow import relevance_flow_figure
from vraven.visualisation.spectral_trace import spectral_trace_figure

# Friendly VRAVEN plot names → report methods.
PLOT_ALIASES: dict[str, str] = {
    "decision_flow": "decision_flow",
    "contribution": "decision_sankey",
    "contribution_over_time": "decision_sankey",
    "representation": "latent_flight",
    "representation_landscape": "latent_flight",
    "layer_relevance": "neuron_impact",
    "neuron_impact": "neuron_impact",
    "neuron_importance": "neuron_importance",
    "risk": "conflict_field",
    "attention": "attention_structure",
    "counterfactual": "counterfactual_delta",
    "perturbation": "perturbation_stability",
    "concordance": "mechanism_concordance",
    "mechanism_concordance": "mechanism_concordance",
    "feature_importance": "input_attribution",
    "feature_heatmap": "feature_sequence_heatmap",
    "relevance_flow": "relevance_flow",
    "lrp": "relevance_flow",
    "reasoning_dna": "reasoning_dna",
    "spectral_trace": "spectral_trace",
    "community": "community_graph",
}


@dataclass(slots=True)
class VRavenReport:
    trace: NeuralTrace
    full_tensors: dict[str, Any] = field(default_factory=dict, repr=False)
    lrp_result: LRPResult | None = field(default=None, repr=False)
    perturbation_result: PerturbationResult | None = field(default=None, repr=False)
    counterfactual_result: CounterfactualResult | None = field(default=None, repr=False)

    def summary(self) -> dict[str, object]:
        prediction = self.trace.prediction
        return {
            "trace_id": self.trace.trace_id,
            "mode": self.trace.mode,
            "layers_captured": len(self.trace.layers),
            "target": self.trace.target,
            "framework": self.trace.metadata.get("framework"),
            "device": self.trace.metadata.get("device"),
            "model_class": self.trace.metadata.get("model_class"),
            "parameter_count": self.trace.metadata.get("parameter_count"),
            "predicted_classes": prediction.predicted_classes if prediction else None,
            "confidence": prediction.confidence if prediction else None,
            "predictive_entropy": prediction.predictive_entropy if prediction else None,
            "causal_interventions": len(self.trace.causal_interventions),
        }

    def capabilities(self) -> dict[str, bool]:
        has_contrib = any(layer.unit_contributions is not None for layer in self.trace.layers)
        has_repr = any(
            layer.representation and len(layer.representation) >= 2 for layer in self.trace.layers
        )
        has_grad = any(layer.activation_gradient_signed is not None for layer in self.trace.layers)
        has_attr = bool(self.trace.inputs and self.trace.inputs.attribution_available)
        has_zero_ablation = any(
            item.strategy == "zero" for item in self.trace.causal_interventions
        )
        return {
            "activation_energy": bool(self.trace.layers),
            "decision_flow": has_grad,
            "decision_sankey": has_grad or has_attr,
            "spectral_trace": has_contrib,
            "neuron_impact": has_contrib,
            "neuron_importance": has_contrib,
            "attention_structure": any(layer.attention is not None for layer in self.trace.layers),
            "latent_flight": has_repr,
            "reasoning_dna": bool(self.trace.layers),
            "conflict_field": self.trace.prediction is not None,
            "evidence_trajectory": has_grad,
            "causal_pathway": bool(self.trace.causal_interventions),
            "mechanism_concordance": has_grad and has_zero_ablation,
            "input_attribution": has_attr,
            "feature_sequence_heatmap": has_attr,
            "reasoning_graph": bool(self.trace.layers),
            "community_graph": any(layer.unit_matrix for layer in self.trace.layers),
            "relevance_flow": self.lrp_result is not None,
            "perturbation_stability": self.perturbation_result is not None,
            "counterfactual_delta": self.counterfactual_result is not None,
        }

    def retained_tensor_names(self) -> tuple[str, ...]:
        """Names of full layer outputs retained by explicit opt-in capture."""

        return tuple(self.full_tensors)

    def full_tensor(self, layer_name: str) -> Any:
        try:
            return self.full_tensors[layer_name]
        except KeyError as error:
            if not self.full_tensors:
                raise CapabilityError(
                    "No full tensors are retained. Run with capture_full_tensors=True. "
                    "Full tensors are intentionally not stored in JSON traces."
                ) from error
            available = ", ".join(self.full_tensors)
            raise CapabilityError(
                f"No retained tensor is named {layer_name!r}. Available: {available}"
            ) from error

    def release_full_tensors(self) -> None:
        """Release explicitly retained tensors, including any GPU memory they reference."""

        self.full_tensors.clear()

    def reasoning_fingerprint(self, dimensions: int = 128) -> np.ndarray:
        return reasoning_fingerprint(self.trace, dimensions=dimensions)

    def scientific_audit(self) -> ScientificEvidenceAudit:
        """Return a transparent concentration and causal-concordance audit."""

        return scientific_evidence_audit(self.trace)

    def activation_energy(
        self,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return activation_energy_figure(self.trace, mode=mode, show_title=show_title)

    def decision_flow(
        self,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return decision_flow_figure(self.trace, mode=mode, show_title=show_title)

    def spectral_trace(
        self,
        *,
        mode: RenderMode = "publication_light",
        max_layers: int = 64,
        show_title: bool | None = None,
    ) -> go.Figure:
        return spectral_trace_figure(
            self.trace,
            mode=mode,
            max_layers=max_layers,
            show_title=show_title,
        )

    def neuron_impact(
        self,
        *,
        layer: str | int | None = None,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return neuron_impact_figure(
            self.trace,
            layer=layer,
            mode=mode,
            show_title=show_title,
        )

    def attention_structure(
        self,
        *,
        layer: str | int | None = None,
        head: int = 0,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return attention_structure_figure(
            self.trace,
            layer=layer,
            head=head,
            mode=mode,
            show_title=show_title,
        )

    def latent_flight(
        self,
        *,
        labels: list[Any] | tuple[Any, ...] | None = None,
        layer_names: list[str] | tuple[str, ...] | None = None,
        max_layers: int = 4,
        method: str = "pca",
        random_seed: int = 17,
        perplexity: float = 30.0,
        align: bool = True,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return latent_flight_figure(
            self.trace,
            labels=labels,
            layer_names=layer_names,
            max_layers=max_layers,
            method=method,  # type: ignore[arg-type]
            random_seed=random_seed,
            perplexity=perplexity,
            align=align,
            mode=mode,
            show_title=show_title,
        )

    representation_evolution = latent_flight
    representation_landscape = latent_flight

    def reasoning_dna(
        self,
        *,
        dimensions: int = 128,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return reasoning_dna_figure(
            self.trace,
            dimensions=dimensions,
            mode=mode,
            show_title=show_title,
        )

    def conflict_field(
        self,
        *,
        sample_index: int = 0,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return conflict_field_figure(
            self.trace,
            sample_index=sample_index,
            mode=mode,
            show_title=show_title,
        )

    def evidence_trajectory(
        self,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return evidence_trajectory_figure(self.trace, mode=mode, show_title=show_title)

    def causal_pathway(
        self,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return causal_pathway_figure(self.trace, mode=mode, show_title=show_title)

    def mechanism_concordance(
        self,
        *,
        mode: RenderMode = "publication_light",
        random_seed: int = 17,
        max_permutations: int = 4096,
        show_title: bool | None = None,
    ) -> go.Figure:
        return mechanism_concordance_figure(
            self.trace,
            mode=mode,
            random_seed=random_seed,
            max_permutations=max_permutations,
            show_title=show_title,
        )

    def input_attribution(
        self,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
        feature_names: list[str] | tuple[str, ...] | None = None,
        top_k: int | None = 25,
    ) -> go.Figure:
        return input_attribution_figure(
            self.trace,
            mode=mode,
            show_title=show_title,
            feature_names=feature_names,
            top_k=top_k,
        )

    def reasoning_graph(
        self,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return reasoning_graph_figure(self.trace, mode=mode, show_title=show_title)


    def community_graph(
        self,
        *,
        layer: str | int | None = None,
        n_clusters: int = 6,
        similarity_threshold: float = 0.20,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return community_graph_figure(
            self.trace,
            layer=layer,
            n_clusters=n_clusters,
            similarity_threshold=similarity_threshold,
            mode=mode,
            show_title=show_title,
        )

    def consistency_matrix(
        self,
        explanations: dict[str, list[float] | np.ndarray],
        *,
        metric: AgreementMetric = "rank_correlation",
        top_k: int | None = None,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return consistency_matrix_figure(
            explanations,
            metric=metric,
            top_k=top_k,
            mode=mode,
            show_title=show_title,
        )

    def counterfactual_delta(
        self,
        result: CounterfactualResult | None = None,
        *,
        feature_names: list[str] | tuple[str, ...] | None = None,
        class_names: list[str] | tuple[str, ...] | None = None,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        resolved = result if result is not None else self.counterfactual_result
        if resolved is None:
            raise CapabilityError(
                "Counterfactual figure requires a CounterfactualResult "
                "(pass result=... or attach report.counterfactual_result)"
            )
        return counterfactual_delta_figure(
            resolved,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            show_title=show_title,
        )

    def neuron_importance(
        self,
        *,
        layer: str | int | None = None,
        top_k: int = 12,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return neuron_importance_figure(
            self.trace,
            layer=layer,
            top_k=top_k,
            mode=mode,
            show_title=show_title,
        )

    def relevance_flow(
        self,
        result: LRPResult | None = None,
        *,
        max_layers: int = 6,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        resolved = result if result is not None else self.lrp_result
        if resolved is None:
            raise CapabilityError(
                "Relevance Flow requires an LRPResult (vraven.explain_lrp or report.lrp_result)"
            )
        return relevance_flow_figure(
            resolved,
            max_layers=max_layers,
            mode=mode,
            show_title=show_title,
        )

    def perturbation_stability(
        self,
        result: PerturbationResult | None = None,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        resolved = result if result is not None else self.perturbation_result
        if resolved is None:
            raise CapabilityError(
                "Perturbation stability requires a PerturbationResult "
                "(vraven.run_perturbation_stability or report.perturbation_result)"
            )
        return perturbation_stability_figure(resolved, mode=mode, show_title=show_title)

    def feature_sequence_heatmap(
        self,
        matrix: Any = None,
        *,
        feature_names: list[str] | tuple[str, ...] | None = None,
        sample_labels: list[str] | tuple[str, ...] | None = None,
        top_features: int = 16,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return feature_sequence_heatmap_figure(
            matrix,
            trace=self.trace if matrix is None else None,
            feature_names=feature_names,
            sample_labels=sample_labels,
            top_features=top_features,
            mode=mode,
            show_title=show_title,
        )

    def decision_sankey(
        self,
        *,
        feature_names: list[str] | tuple[str, ...] | None = None,
        class_names: list[str] | tuple[str, ...] | None = None,
        top_features: int = 7,
        depth_stages: int = 5,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ) -> go.Figure:
        return decision_sankey_figure(
            self.trace,
            feature_names=feature_names,
            class_names=class_names,
            top_features=top_features,
            depth_stages=depth_stages,
            mode=mode,
            show_title=show_title,
        )

    def figure(
        self,
        name: str,
        *,
        mode: RenderMode = "publication_light",
        **kwargs: Any,
    ) -> go.Figure:
        method = getattr(self, name, None)
        if method is None or not callable(method) or name.startswith("_"):
            raise ValueError(f"Unknown VRAVEN figure method: {name}")
        return method(mode=mode, **kwargs)

    def plot(
        self,
        name: str,
        *,
        mode: RenderMode = "interactive_dark",
        **kwargs: Any,
    ) -> go.Figure:
        """VRAVEN plot dispatcher.

        Examples::

            result.plot("decision_flow")
            result.plot("representation")
            result.plot("layer_relevance")
            result.plot("contribution")
            result.plot("counterfactual")
        """

        resolved = PLOT_ALIASES.get(name, name)
        return self.figure(resolved, mode=mode, **kwargs)

    def save_trace(self, path: str | Path) -> Path:
        return self.trace.save(path)

    def write_report(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        visuals: list[str] | tuple[str, ...] | None = None,
        mode: RenderMode = "interactive_dark",
        figure_kwargs: dict[str, dict[str, Any]] | None = None,
        theme_toggle: bool = True,
        include_scientific_audit: bool = True,
    ) -> Path:
        """Write exactly one report file in HTML, PDF or JSON format."""

        destination, resolved_format = resolve_report_destination(path, format)
        if resolved_format == "json":
            return self.save_trace(destination)
        if resolved_format == "html":
            return self.export_html(
                destination,
                visuals=visuals,
                mode=mode,
                figure_kwargs=figure_kwargs,
                theme_toggle=theme_toggle,
                include_scientific_audit=include_scientific_audit,
            )
        with tempfile.TemporaryDirectory(prefix="vraven-report-") as directory:
            html_path = Path(directory) / "report.html"
            self.export_html(
                html_path,
                visuals=visuals,
                mode="publication_light",
                figure_kwargs=figure_kwargs,
                theme_toggle=False,
                include_scientific_audit=include_scientific_audit,
            )
            return html_to_pdf(html_path, destination)

    @classmethod
    def load_trace(cls, path: str | Path) -> VRavenReport:
        return cls(NeuralTrace.load(path))

    def export(
        self,
        visual: str,
        path: str | Path,
        *,
        mode: RenderMode = "publication_light",
        **kwargs: Any,
    ) -> Path:
        figure_kwargs = kwargs.pop("figure_kwargs", {})
        figure = self.figure(visual, mode=mode, **figure_kwargs)
        return export_figure(figure, path, mode=mode, **kwargs)

    def export_html(
        self,
        path: str | Path,
        *,
        visuals: list[str] | tuple[str, ...] | None = None,
        mode: RenderMode = "interactive_dark",
        figure_kwargs: dict[str, dict[str, Any]] | None = None,
        theme_toggle: bool = True,
        include_scientific_audit: bool = True,
    ) -> Path:
        from vraven.report.html import export_html_report

        return export_html_report(
            self,
            path,
            visuals=visuals,
            mode=mode,
            figure_kwargs=figure_kwargs,
            theme_toggle=theme_toggle,
            include_scientific_audit=include_scientific_audit,
        )

    def show(
        self,
        *,
        visual: str | None = None,
        mode: RenderMode = "interactive_dark",
        **kwargs: Any,
    ) -> None:
        selected = visual
        if selected is None:
            available = self.capabilities()
            selected = next(
                (name for name in ("decision_flow", "reasoning_dna", "activation_energy") if available.get(name)),
                None,
            )
        if selected is None:
            raise CapabilityError("This trace does not contain data for a standard visualisation")
        self.figure(selected, mode=mode, **kwargs).show()
