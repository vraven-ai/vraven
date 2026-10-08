# VRAVEN Public API Specification

## Package status

The repository provides a working PyTorch alpha. Install the latest published
release with `pip install vraven`, or install an editable checkout when
developing VRAVEN itself.

## Installation

```bash
python -m pip install -e .
python -m pip install -e ".[publication]"  # optional static export
python -m pip install -e ".[dev]"          # tests and quality tools
```

## Loading PyTorch model artefacts

The preferred integration is a live `torch.nn.Module`. VRAVEN also provides an explicit loader for trusted local artefacts. The safe default expects a state dictionary and an architecture factory:

```python
def build_model():
    return MyModel(...)

model = vraven.load_pytorch_model(
    "weights.pt",
    model_factory=build_model,
)
```

Nested checkpoints and distributed-training prefixes are supported:

```python
model = vraven.load_pytorch_model(
    "checkpoint.pt",
    model_factory=build_model,
    state_dict_key="state_dict",
    strip_prefix="module.",
    strict=True,
)
```

Legacy TorchScript artefacts can be loaded with `format="torchscript"`, but PyTorch
ScriptModules do not expose the forward hooks required by VRAVEN explanation capture.
Use a state dictionary and architecture factory for instrumented CLI workflows.
Loading a complete Python-pickled model is blocked unless `format="pickle"` and
`allow_unsafe_pickle=True` are both supplied. Pickled model files may execute
embedded code and must never be loaded from an untrusted source. A `.pt` suffix
does not identify which format is inside the file.

## Command-line explanation

The primary CLI entry point explains one model invocation and writes exactly one
report:

```bash
vraven explain weights.pt \
  --model-factory my_package.models:build_model \
  --factory-kwargs '{"input_dim":64,"num_classes":2}' \
  --input sample.npy \
  --target 1 \
  --mode causal \
  --module-policy leaf \
  --max-layers 128 \
  --max-units 64 \
  --causal-top-k 5 \
  --report-format html \
  --output explanation.html
```

Run `vraven explain -h` for all model-loading, capture, selection, causal and
report arguments. Run `vraven features` for a machine-readable capability
inventory and `vraven doctor` to show the active executable, imported package,
runtime backends and PDF browser availability.

The unreleased source CLI also exposes implemented analysis APIs directly:

* `vraven analyse` (`analyze` / `quick`) — beginner-facing routing to common
  tasks using quick, standard or thorough presets and automatic output names;
* `vraven bundle-init` / `vraven bundle` — create or validate a contained
  `vraven.json` model bundle, which can be passed directly to `vraven quick`;

* `vraven adversarial` — empirical sample attacks, batch robustness estimates,
  or matched-cohort robustness curves;
* `vraven counterfactual` — bounded continuous-input counterfactual search;
* `vraven stability` — repeated controlled-perturbation score stability;
* `vraven lrp` — epsilon-rule Layer-wise Relevance Propagation;
* `vraven certify` — formal local L∞ verification for supported FX operators.
* `vraven dataset` — independent per-sample traces and reasoning fingerprints;
* `vraven compare` — paired model comparison on an identical bounded cohort;
* `vraven evolution` — ordered checkpoint fingerprint and decision evolution;
* `vraven anatomy` — measured local Linear-layer anatomy from DarkSide.

All commands accept the same explicit saved-model loading contract. Empirical
attack non-discovery is not reported as a certificate, and unsupported formal
graphs retain an explicit unsupported verdict.

Interactive terminals receive compact human-readable result panels. Non-TTY
stdout remains JSON for pipelines and automated consumers. `vraven raven`
provides an optional static, blinking or dancing mascot without changing any
scientific result or artefact.

## Reverse feature reports

A decompilation result can write either the complete Decision Proof or one evidence-led standalone
feature report:

```python
result.write_report("decision_proof.html")
result.write_feature_report("causal_proof", "causal_proof.html")
```

Both methods accept `format="html"`, `format="pdf"` or `format="json"` and
write one file. The suffix may infer the format; if both are supplied, they must
agree. PDF export uses a locally installed Chrome, Chromium or Edge executable.

Standalone feature reports contain both official render modes, machine-traceable audit metrics,
their mathematical evidence tier and a visible critical limitation. Supported feature names are
`model_xray`, `concept_genome`, `causal_proof`, `programme_proof`, `contrastive_proof`,
`counterfactual_twin` and `dark_matter`.

## Single explanation

```python
import vraven

report = vraven.explain(
    model,
    inputs,
    target=1,
    mode="gradient",
)
```

Supported explanation modes:

```python
vraven.explain(model, inputs, mode="fast")
vraven.explain(model, inputs, target=1, mode="gradient")
vraven.explain(model, inputs, target=1, mode="causal")
```

`fast` captures reduced forward signals. `gradient` adds target-specific first-order derivatives. `causal` adds bounded temporary layer interventions after the gradient trace.

## Reusable explainer

```python
from vraven import Explainer, VRavenConfig

config = VRavenConfig(
    mode="gradient",
    device="auto",
    module_policy="leaf",
    capture_attention=True,
    capture_representations=True,
    capture_unit_profiles=True,
    max_units=128,
)

explainer = Explainer(
    model,
    config,
    class_names=["benign", "malware"],
)

report = explainer.explain(inputs, target=1)
```

A reusable `Explainer` avoids reconstructing adapter configuration for every call. Hooks remain scoped to each call and are removed before the report is returned.

## Configuration

`VRavenConfig` currently supports:

| Field | Purpose |
|---|---|
| `mode` | `fast`, `gradient` or `causal` |
| `device` | Reserved user declaration. The adapter follows the live model device |
| `auto_move_inputs` | Move input tensors to the model device when required |
| `module_policy` | `leaf`, `semantic` or `all` module selection |
| `include_module_types` | Restrict capture by module class name |
| `exclude_module_types` | Exclude module class names |
| `include_name_patterns` | Include module name patterns |
| `exclude_name_patterns` | Exclude module name patterns |
| `max_layers` | Maximum selected modules |
| `max_units` | Maximum reduced unit columns per layer |
| `representation_size` | Fixed width of compressed batch representations |
| `max_attention_tokens` | Maximum compressed query and key dimensions |
| `histogram_bins` | Activation-dispersion entropy histogram width |
| `entropy_sample_limit` | Maximum deterministic activation samples used for entropy |
| `capture_entropy` | Enable activation-dispersion entropy |
| `capture_representations` | Capture fixed-width batch representations |
| `capture_unit_profiles` | Capture reduced unit matrices |
| `capture_attention` | Capture accessible attention tensors |
| `capture_input_attribution` | Capture input-times-gradient in gradient modes |
| `capture_full_tensors` | Retain full outputs in the scoped hook session only. Off by default |
| `capture_timing` | Capture host-observed module dispatch timing |
| `causal_top_k` | Maximum layers intervened on in causal mode |
| `causal_strategy` | `zero`, `mean` or `scale` |
| `causal_scale` | Scale value used by the scale intervention |
| `epsilon` | Numerical zero threshold |
| `random_seed` | Controlled projection or future stochastic procedure seed |
| `strict` | Reserved strict capability behaviour |

## Report summary and capabilities

```python
report.summary()
report.capabilities()
report.reasoning_fingerprint(dimensions=128)
report.scientific_audit()
```

`scientific_audit()` returns separate evidence-concentration, signed-cancellation and matched
zero-ablation concordance measurements. See `docs/EVIDENCE_AUDIT.md` for definitions and limits.

`summary()` returns model, prediction, device and trace details. `capabilities()` states which figures are supported by the data actually captured.

## Render modes

Explanation mode and render mode are separate.

```python
light = report.decision_flow(mode="publication_light")
dark = report.decision_flow(mode="interactive_dark")
```

Both figures must contain the same scientific arrays and ordering. Background, contrast, annotation density and interaction may differ.

## Single-report figure methods

```python
report.activation_energy()
report.decision_flow()
report.decision_sankey(feature_names=None)
report.spectral_trace(max_layers=64)
report.neuron_impact(layer=None)
report.neuron_importance(layer=None, top_k=12)
report.attention_structure(layer=None, head=0)
report.latent_flight(labels=None, method="pca")  # or method="tsne"
report.reasoning_dna(dimensions=128)
report.conflict_field(sample_index=0)
report.evidence_trajectory()
report.causal_pathway()
report.mechanism_concordance()          # requires matched zero-ablation interventions
report.input_attribution(feature_names=None, top_k=25)
report.feature_sequence_heatmap(feature_names=None)
report.reasoning_graph()
report.community_graph(layer=None, n_clusters=6, similarity_threshold=0.20)
report.relevance_flow()                 # requires report.lrp_result
report.perturbation_stability()         # requires report.perturbation_result
report.counterfactual_delta()           # requires report.counterfactual_result
```

HTML exports include the Scientific Evidence Audit by default:

```python
report.export_html("report.html", include_scientific_audit=True)
```

For a single explicit output format use:

```python
report.write_report("report.html", format="html")
report.write_report("report.pdf", format="pdf")
report.write_report("trace.json", format="json")
```

Set `include_scientific_audit=False` only for a compact visual-only composition.

Additional analysis APIs:

```python
lrp = vraven.explain_lrp(model, inputs, target=1)
stability = vraven.run_perturbation_stability(model, inputs, target=1)
cf = vraven.generate_counterfactual(model, features, target_class=0)
```

Cross-explanation methods:

```python
report.consistency_matrix(
    {
        "method_a": vector_a,
        "method_b": vector_b,
    },
    metric="rank_correlation",
)

report.counterfactual_delta(cf, feature_names=feature_names)
```

Generic dispatch:

```python
figure = report.figure(
    "spectral_trace",
    mode="publication_light",
    max_layers=32,
)
```

## Explicit full-tensor capture

```python
report = vraven.explain(
    model,
    inputs,
    capture_full_tensors=True,
)

names = report.retained_tensor_names()
layer_output = report.full_tensor(names[0])
report.release_full_tensors()
```

This is an opt-in, ephemeral path. Tensors remain on their originating device, may retain substantial memory and are not serialised.

## Figure export

```python
report.export(
    "reasoning_dna",
    "reasoning_dna.svg",
    mode="publication_light",
    preset="double_column",
    dpi=300,
)
```

Supported file types:

* HTML
* JSON
* SVG
* PDF
* PNG
* JPEG
* WebP

Static formats require Kaleido and a compatible Chrome or Chromium installation. HTML does not.

Publication presets:

```text
single_column
 double_column
 square
 wide
```

## Local HTML report composition

```python
report.export_html(
    "report.html",
    visuals=[
        "decision_flow",
        "reasoning_dna",
        "spectral_trace",
        "conflict_field",
    ],
    mode="interactive_dark",
    theme_toggle=True,  # default: Dark/Light control; both figure themes embedded
)
```

The HTML report composes independent figures. With `theme_toggle=True` (default), each
figure is rendered in both `interactive_dark` and `publication_light`, and the report
header exposes a Dark / Light switch (preference stored in `localStorage`). It is not
required for viewing any core figure.

Every composed figure is preceded by a publication evidence panel containing its registered
scientific question, evidence tier, interpretation, quantitative findings where defined,
normalisation, visual mapping and claim boundary. The panel can copy a journal caption and export
the exact currently displayed figure data as JSON. See
[`PUBLICATION_VISUALISATIONS.md`](PUBLICATION_VISUALISATIONS.md).

## Trace persistence

```python
path = report.save_trace("trace.json")
loaded = vraven.VRavenReport.load_trace(path)
```

The JSON schema is implemented by `vraven.core.trace`. Full activation tensors are not persisted.

## Dataset analysis

```python
dataset_report = vraven.explain_dataset(
    model,
    data,
    targets=targets,
    mode="gradient",
    max_samples=128,
    fingerprint_dimensions=128,
)
```

Supported data forms:

1. A batched `torch.Tensor`.
2. A mapping or tuple of batched tensors sharing one first dimension.
3. An iterable of model inputs.
4. An iterable of `(input, scalar_class_label)` items.
5. An iterable of `vraven.DatasetItem` values.
6. A custom iterable with `input_getter`, `target_getter`, `label_getter` and `id_getter`.

Example with explicit dataset items:

```python
items = [
    vraven.DatasetItem(
        inputs=x,
        target=int(y),
        label=int(y),
        sample_id=f"test_{index}",
    )
    for index, (x, y) in enumerate(dataset)
]

report = vraven.explain_dataset(model, items, max_samples=100)
```

Dataset methods:

```python
dataset_report.summary()
dataset_report.fingerprints
dataset_report.similarity_matrix()
dataset_report.outlier_scores(neighbours=5)
dataset_report.reasoning_similarity()
dataset_report.reasoning_landscape(colour_by="prediction")
dataset_report.reasoning_outliers(neighbours=5)
dataset_report.save("dataset_report.json")
vraven.VRavenDatasetReport.load("dataset_report.json")
```

Each sample is explained independently. This provides sample-level fingerprints but is slower than one aggregate batch explanation.

## Matched model comparison

```python
comparison = vraven.compare_models(
    model_a,
    model_b,
    data,
    targets=targets,
    mode="fast",
    max_samples=64,
    left_name="baseline",
    right_name="candidate",
)

comparison.summary()
comparison.fingerprint_similarity
comparison.prediction_agreement
comparison.confidence_delta
comparison.alignment(mode="publication_light")
```

Both models receive the same materialised sample objects in the same order.

## Checkpoint evolution

```python
evolution = vraven.compare_checkpoints(
    [model_epoch_1, model_epoch_10, model_epoch_50],
    one_input,
    target=1,
    checkpoint_names=["epoch_1", "epoch_10", "epoch_50"],
    mode="gradient",
)

evolution.summary()
evolution.similarity_to_final
evolution.figure()
```

## Counterfactual generation

```python
result = vraven.generate_counterfactual(
    model,
    inputs,
    target_class=1,
    steps=200,
    learning_rate=0.05,
    l2_weight=0.01,
    clamp_min=0.0,
    clamp_max=1.0,
)
```

The built-in optimiser supports differentiable continuous tensors. It is not a valid domain-aware counterfactual for discrete bytes, tokens, executable files or constrained tabular features unless an appropriate representation and constraints are added.

## Adversarial evidence

```python
from vraven.decompiler import AdversarialAuditor

auditor = AdversarialAuditor(model, model_name="candidate")
sample_report = auditor.audit_sample(
    one_input,
    target=1,
    epsilon_grid=(0.01, 0.05, 0.1, 0.25),
)
cohort_report = auditor.audit_batch(
    inputs,
    labels,
    epsilon=0.1,
    protocol="vraven_ensemble_lite",
    sampling="stratified",
    max_samples=64,
)
profile = auditor.audit_robustness_curve(
    inputs,
    labels,
    epsilon_grid=(0.01, 0.05, 0.1, 0.25),
    sampling="stratified",
    max_samples=64,
    steps=40,
    pgd_restarts=5,
)
profile.save("robustness_profile.json")
profile.write_html("robustness_profile.html")

input_spec = vraven.InputDomainSpec(
    lower=0.0,
    upper=1.0,
    units="raw pixel intensity",
    preprocessing="divide uint8 values by 255; no model-side normalisation",
    semantic_domain="single-channel images",
    source="dataset protocol v2",
)
formal = vraven.IntervalBoundVerifier(model, model_name="candidate").certify(
    one_input,
    target=1,
    epsilon_grid=(0.0, 0.0001, 0.001, 0.01, 0.1),
    cohort_x=inputs,
    cohort_y=labels,
    input_spec=input_spec,
)
formal.save("formal_robustness_certificate.json")
formal.write_html("formal_robustness_certificate.html")

mechanism = vraven.FormalCausalMechanismVerifier(
    model, model_name="candidate"
).certify(
    one_input,
    target=1,
    mechanism=vraven.MechanismPropertySpec(
        node_id="head.2",
        max_relative_l2_drift=0.25,
        min_cosine=0.95,
        top_k=8,
    ),
    causal=vraven.CausalEffectPropertySpec(
        node_id="head.2",
        baseline=0.0,
        auto_top_k=4,
        minimum_clean_effect=1e-8,
        max_relative_l2_drift=0.50,
        min_cosine=0.80,
    ),
    epsilon_grid=(0.0, 0.00001, 0.0001, 0.001),
    input_spec=input_spec,
    calibration_x=coordinate_selection_inputs,
    calibration_sample_ids=coordinate_selection_ids,
    calibration_disjoint_from_evaluation=True,
    evaluation_sample_ids=evaluation_ids,
)
mechanism.save("formal_causal_mechanism_certificate.json")
mechanism.write_html("formal_causal_mechanism_certificate.html")
```

`cohort_report.robust_accuracy` uses every labelled sample as its denominator;
clean errors are non-robust. `conditional_robust_accuracy` uses only clean-correct
samples. Without labels, only `prediction_invariance` is meaningful. The
`vraven_ensemble_lite` protocol is a bounded VRAVEN pilot and does not claim parity
with reference AutoAttack. See [`ADVERSARIAL_EVIDENCE.md`](ADVERSARIAL_EVIDENCE.md).
Robustness curves retain the matched per-sample union of failures across epsilon,
report a paired-bootstrap AUC interval and a DKW simultaneous sampling band. PGD
retains the strongest candidate across every step and restart; diagnostics record
finite/zero-gradient rates and exact norm/domain checks.

`IntervalBoundVerifier` is a separate formal-evidence path. A `certified` result proves
the recorded target remains the strict argmax throughout the declared L∞ box for the
supported real-valued FX graph. `inconclusive` does not mean vulnerable, and
`unsupported` identifies the exact missing operator. See
[`FORMAL_ROBUSTNESS.md`](FORMAL_ROBUSTNESS.md).

`FormalCausalMechanismVerifier` constructs an additional intervention-defined
property. It separately bounds coordinate-aligned representation drift and the vector
of fixed-margin effects produced by executable internal coordinate interventions. Its
geometry, sign, ranking and joint radii are lower bounds; unequal certified radii do
not prove existence of a mechanism-changing input. See
[`FORMAL_CAUSAL_MECHANISM.md`](FORMAL_CAUSAL_MECHANISM.md).

Intervention effects use paired shared-input interval propagation: affine shared
terms and biases cancel exactly, supported monotone nonlinearities use sound secant-
slope bounds, and a terminal linear target-versus-competitor margin is fused before
interval evaluation. Cohort records include the exact mechanism conjunction
`R∩C`, joint conjunction `D∩R∩C`, and all four cells of the
decision-versus-mechanism contingency table. These counts do not rely on Fréchet
bounds.

The certificate also stores a per-sample zero-radius applicability ledger.
`n_representation_applicable` requires a non-zero clean representation norm and, when
declared, a uniquely separated top-k boundary. `n_causal_applicable` requires the
complete frozen coordinate basis and an absolute clean-effect lower bound of at least
`minimum_clean_effect` for every required coordinate. `n_mechanism_applicable` is
their exact conjunction. The `n_applicable_*` D/M cells are conditioned on this last
count and sum to it at every epsilon; their rates are `null` when it is zero.
Non-applicable samples carry explicit reasons and are applicability-inconclusive, not
negative mechanism-survival observations. The older unconditional cells remain
available as all-sample certificate-coverage counts.

Every selected causal coordinate must have a certified absolute clean intervention-
effect lower bound of at least `minimum_clean_effect`. If an auto-selection screen
cannot supply the requested cardinality, or a frozen coordinate has no eligible local
effect, causal-effect and joint survival are returned as explicitly `inconclusive`.
Decision and representation results remain independently interpretable; VRAVEN never
silently reduces the requested causal property to obtain a positive certificate.

`InputDomainSpec` distinguishes an intentionally unbounded mathematical domain from
missing input semantics. A declared specification records bounds, units,
preprocessing, semantic domain and source. When coordinates are not pre-registered,
publication experiments should provide a calibration split that is demonstrably
disjoint from evaluation. The resulting certificate includes a non-composite
`publication_readiness` ledger; failed gates do not invalidate a sound local proof,
but they block population-level wording.


## Command line

```bash
vraven --version
vraven demo --mode gradient --samples 48 --report-format html --output vraven_demo.html
vraven inspect trace.json
vraven render trace.json --visual reasoning_dna --output figure.svg
```

## Expected errors

VRAVEN raises explicit errors when:

* No adapter supports the supplied model.
* The target class is invalid.
* The output cannot be interpreted as a supported prediction tensor.
* Gradient mode receives a non-differentiable output.
* A requested figure lacks required trace data.
* A causal intervention cannot be performed safely.
* Dataset input or target lengths disagree.
* Projection inputs are insufficient or inconsistent.
* Static export dependencies are missing.
