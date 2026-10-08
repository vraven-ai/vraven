from __future__ import annotations

from vraven.core.trace import LayerTrace, NeuralTrace
from vraven.visualisation.decision_flow import decision_flow_figure
from vraven.visualisation.neuron_importance import neuron_importance_figure
from vraven.visualisation.palette import semantic_colour


def _gradient_trace() -> NeuralTrace:
    return NeuralTrace(
        layers=[
            LayerTrace(
                name="hidden",
                layer_type="Linear",
                execution_index=0,
                output_shape=(1, 2),
                device="cpu",
                dtype="torch.float32",
                activation_energy=5.0,
                activation_gradient_influence=3.0,
                activation_gradient_signed=-1.0,
                unit_activations=[0.5, 0.7],
                unit_contributions=[-2.0, 1.0],
            ),
            LayerTrace(
                name="head",
                layer_type="Linear",
                execution_index=1,
                output_shape=(1, 2),
                device="cpu",
                dtype="torch.float32",
                activation_energy=2.0,
                activation_gradient_influence=2.0,
                activation_gradient_signed=2.0,
                unit_activations=[0.2, 0.4],
                unit_contributions=[0.5, 1.5],
            ),
        ]
    )


def test_publication_key_results_do_not_change_with_render_mode() -> None:
    trace = _gradient_trace()
    light = decision_flow_figure(trace, mode="publication_light")
    dark = decision_flow_figure(trace, mode="interactive_dark")

    light_meta = light.layout.meta["scientific"]
    dark_meta = dark.layout.meta["scientific"]
    assert light_meta["key_results"] == dark_meta["key_results"]
    assert light_meta["metric_definitions"]
    assert light_meta["interpretation"]
    assert light_meta["evidence_tier"] == "observational"


def test_neuron_importance_uses_support_and_opposition_colours_by_sign() -> None:
    figure = neuron_importance_figure(
        _gradient_trace(), layer="hidden", mode="publication_light"
    )
    bars = next(trace for trace in figure.data if trace.type == "bar")
    signed_values = list(bars.customdata)
    colours = list(bars.marker.color)

    for signed, colour in zip(signed_values, colours, strict=True):
        expected = semantic_colour(
            "supporting" if float(signed) >= 0 else "opposing",
            "publication_light",
        )
        assert colour == expected

    results = figure.layout.meta["scientific"]["key_results"]
    assert results[0]["label"] == "Largest absolute contributor"
    assert results[0]["value"] == "Neuron 0"
