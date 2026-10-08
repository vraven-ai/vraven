# VRAVEN Capability and Feature-to-Code Map

This document is the human-readable inventory of VRAVEN's executable capabilities. It maps
each scientific or engineering feature to its public entry point, primary implementation and
evidence boundary. Generated dark and publication-light HTML files are render variants, not
separate algorithms.

The authoritative full-suite runner is
[`scripts/run_vraven_on_models.py`](../scripts/run_vraven_on_models.py). The artefact-directory-only
workflow is [`scripts/run_vraven_artifact.py`](../scripts/run_vraven_artifact.py), and the top-level
Python API is [`src/vraven/__init__.py`](../src/vraven/__init__.py).

Evidence terminology used below:

- **observational**: derived from activations, gradients, attention, geometry or predictions;
- **interventional**: derived from an executed ablation, replacement or internal intervention;
- **empirical adversarial**: derived from bounded attack search and never a certificate;
- **formal**: a universal property proved over the declared set by a sound supported verifier;
- **infrastructure**: capture, persistence, reporting or reproducibility rather than a model claim.

## 1. End-to-end public workflows

| Capability | Public entry point | Primary code | Output or scope |
|---|---|---|---|
| Explain a live PyTorch model | `vraven.explain(...)` / `Explainer` | [`core/explainer.py`](../src/vraven/core/explainer.py) | `VRavenReport` with a reduced `NeuralTrace` |
| Explain a saved PyTorch model from the CLI | `vraven explain ...` | [`cli.py`](../src/vraven/cli.py) | Parameterised model loading, capture and exactly one HTML, PDF or JSON report |
| Run a beginner-friendly analysis | `vraven analyse ...` / `vraven quick ...` | [`cli.py`](../src/vraven/cli.py) | Routes a small option set and declared runtime preset to the full implemented command |
| Load a self-describing model folder | `vraven quick BUNDLE_DIR` / `vraven bundle ...` | [`bundle.py`](../src/vraven/bundle.py), [`cli.py`](../src/vraven/cli.py) | Validates contained model, input, optional labels, factory configuration and analysis defaults |
| Audit empirical adversarial behaviour from the CLI | `vraven adversarial ...` | [`cli.py`](../src/vraven/cli.py), [`decompiler/adversarial.py`](../src/vraven/decompiler/adversarial.py) | Sample attacks, fixed-ε batch estimates and labelled robustness curves; never a formal certificate |
| Search for an input counterfactual from the CLI | `vraven counterfactual ...` | [`cli.py`](../src/vraven/cli.py), [`causal/counterfactual.py`](../src/vraven/causal/counterfactual.py) | Bounded continuous-input candidate with explicit success and distance accounting |
| Measure perturbation stability from the CLI | `vraven stability ...` | [`cli.py`](../src/vraven/cli.py), [`causal/perturbation.py`](../src/vraven/causal/perturbation.py) | Repeated target-score response to declared perturbations |
| Run LRP from the CLI | `vraven lrp ...` | [`cli.py`](../src/vraven/cli.py), [`attribution/lrp.py`](../src/vraven/attribution/lrp.py) | Epsilon-rule relevance and supported-layer metadata |
| Request formal robustness from the CLI | `vraven certify ...` | [`cli.py`](../src/vraven/cli.py), [`decompiler/formal_robustness.py`](../src/vraven/decompiler/formal_robustness.py) | Formal local L∞ result or explicit unsupported/inconclusive verdict |
| Present terminal results and mascot | `vraven raven ...` / automatic TTY result cards | [`terminal.py`](../src/vraven/terminal.py) | Dependency-free display only; redirected analysis output remains machine-readable JSON |
| Build a dataset reasoning atlas from the CLI | `vraven dataset ...` | [`cli.py`](../src/vraven/cli.py), [`analysis/dataset.py`](../src/vraven/analysis/dataset.py) | Independent bounded per-sample traces, similarities and outlier evidence |
| Compare two models from the CLI | `vraven compare ...` | [`cli.py`](../src/vraven/cli.py), [`analysis/comparison.py`](../src/vraven/analysis/comparison.py) | Paired prediction agreement and fingerprint alignment on identical samples |
| Track checkpoint evolution from the CLI | `vraven evolution ...` | [`cli.py`](../src/vraven/cli.py), [`analysis/evolution.py`](../src/vraven/analysis/evolution.py) | Prediction changes and reasoning-fingerprint similarity to the final checkpoint |
| Inspect internal Linear anatomy from the CLI | `vraven anatomy ...` | [`cli.py`](../src/vraven/cli.py), [`darkside/anatomy.py`](../src/vraven/darkside/anatomy.py) | Local activation, weight-energy and contribution measurements; not symbolic reasoning |
| Fast activation explanation | `mode="fast"` | [`core/explainer.py`](../src/vraven/core/explainer.py), [`adapters/pytorch.py`](../src/vraven/adapters/pytorch.py) | Reduced activation, representation, unit and prediction evidence |
| Target-gradient explanation | `mode="gradient"` | [`adapters/pytorch.py`](../src/vraven/adapters/pytorch.py), [`capture/pytorch_hooks.py`](../src/vraven/capture/pytorch_hooks.py) | Target-specific gradients, input attribution and activation-gradient influence |
| Targeted causal explanation | `mode="causal"` | [`causal/validation.py`](../src/vraven/causal/validation.py), [`causal/interventions.py`](../src/vraven/causal/interventions.py) | Executed layer interventions with before/after target scores |
| Layer-wise relevance propagation | `vraven.explain_lrp(...)` | [`attribution/lrp.py`](../src/vraven/attribution/lrp.py), [`attribution/rules.py`](../src/vraven/attribution/rules.py) | LRP relevance conservation and rule metadata for supported layers |
| Input-space counterfactual search | `vraven.generate_counterfactual(...)` | [`causal/counterfactual.py`](../src/vraven/causal/counterfactual.py) | Candidate input change with success, score and distance accounting; domain validity requires caller constraints |
| Perturbation stability experiment | `vraven.run_perturbation_stability(...)` | [`causal/perturbation.py`](../src/vraven/causal/perturbation.py) | Target-score response to declared noise scales and repeated perturbations |
| Reverse/decompile a model | `vraven.reverse(...)` / `VRavenDecompiler` | [`decompiler/decompiler.py`](../src/vraven/decompiler/decompiler.py) | Mechanism IR, concepts, interventions, programme, certificate and Decision Proof |
| Analyse a labelled or unlabelled cohort | `vraven.explain_dataset(...)` | [`analysis/dataset.py`](../src/vraven/analysis/dataset.py) | Dataset atlas, fingerprints, similarity, outliers and landscape |
| Compare two models | `vraven.compare_models(...)` | [`analysis/comparison.py`](../src/vraven/analysis/comparison.py) | Matched prediction and fingerprint alignment |
| Compare checkpoints | `vraven.compare_checkpoints(...)` | [`analysis/evolution.py`](../src/vraven/analysis/evolution.py) | Internal evolution and similarity-to-final traces |
| Run advanced mechanism audits | `vraven.DarkSideSession` | [`darkside/session.py`](../src/vraven/darkside/session.py) | Registered DarkSide analyses with typed evidence contracts |
| Certify local decision robustness | `vraven.IntervalBoundVerifier` | [`decompiler/formal_robustness.py`](../src/vraven/decompiler/formal_robustness.py) | Formal L∞ decision certificate and cohort certified accuracy |
| Certify causal-mechanism survival | `vraven.FormalCausalMechanismVerifier` | [`decompiler/formal_mechanism.py`](../src/vraven/decompiler/formal_mechanism.py) | Separate decision, representation, intervention-effect and joint certificates |
| Run every applicable feature from one model bundle | `run_vraven_artifact.py ARTEFACT_DIR` | [`scripts/run_vraven_artifact.py`](../scripts/run_vraven_artifact.py) | Indexed local observatory and critical synthesis |

## 2. Capture, model loading and trace lifecycle

| Capability | Primary code | Evidence boundary or behaviour |
|---|---|---|
| PyTorch device-following execution | [`core/device.py`](../src/vraven/core/device.py), [`adapters/pytorch.py`](../src/vraven/adapters/pytorch.py) | Follows CPU, CUDA or MPS model placement; does not silently move the model |
| Safe state-dictionary loading | [`adapters/pytorch_loader.py`](../src/vraven/adapters/pytorch_loader.py) | `weights_only=True`; requires an architecture factory |
| TorchScript loading | [`adapters/pytorch_loader.py`](../src/vraven/adapters/pytorch_loader.py) | Explicit `format="torchscript"` path |
| Guarded pickle loading | [`adapters/pytorch_loader.py`](../src/vraven/adapters/pytorch_loader.py) | Disabled unless the caller explicitly enables unsafe trusted pickle loading |
| Adapter registration and resolution | [`adapters/base.py`](../src/vraven/adapters/base.py), [`adapters/registry.py`](../src/vraven/adapters/registry.py) | Framework capability dispatch |
| Activation and gradient hooks | [`capture/pytorch_hooks.py`](../src/vraven/capture/pytorch_hooks.py) | Scoped registration, cleanup and repeated-call safety |
| Layer selection | [`capture/selectors.py`](../src/vraven/capture/selectors.py) | Selects supported modules without claiming unsupported coverage |
| On-device tensor reduction | [`capture/tensor_utils.py`](../src/vraven/capture/tensor_utils.py) | Reduces representations and unit summaries before host transfer where possible |
| Prediction/output interpretation | [`core/output.py`](../src/vraven/core/output.py) | Scalar, binary and multiclass classification-shaped outputs |
| Nested input/output traversal | [`core/tree.py`](../src/vraven/core/tree.py) | Structured tensor-tree discovery and transformation |
| Neural trace schema | [`core/trace.py`](../src/vraven/core/trace.py) | Ordered layers, prediction, inputs, attribution, attention and causal interventions |
| Explicit ephemeral full-tensor capture | [`report/report.py`](../src/vraven/report/report.py) | Opt-in access and explicit release; full tensors are excluded from JSON persistence |
| Trace save/load | [`core/trace.py`](../src/vraven/core/trace.py), [`report/report.py`](../src/vraven/report/report.py) | JSON round trip with schema validation |
| Reproducibility metadata and bounded fingerprints | [`core/metadata.py`](../src/vraven/core/metadata.py) | Structure hash, runtime metadata and bounded input fingerprints |
| Typed configuration | [`core/config.py`](../src/vraven/core/config.py) | Capture limits, modes, gradient, causal and performance controls |
| Explicit capability errors | [`core/errors.py`](../src/vraven/core/errors.py) | Unsupported operations and device/model-loading failures remain visible |

## 3. Compiler-assisted Reverse decompiler

### 3.1 Capture and executable representation

| Capability | Public type or output | Primary code | Evidence boundary |
|---|---|---|---|
| Multi-route model capture | `ModelCaptureRouter` | [`decompiler/capture.py`](../src/vraven/decompiler/capture.py) | Tries `torch.export`, FX, dispatch and module hooks; retains failed attempts |
| Numerical and gradient equivalence checks | Capture certificate | [`decompiler/capture.py`](../src/vraven/decompiler/capture.py) | Validates captured execution within configured tolerances |
| Mechanism intermediate representation | `MechanismIR` | [`decompiler/ir.py`](../src/vraven/decompiler/ir.py) | Typed nodes, edges, tensor specs, constraints and unresolved operations |
| Operator semantic classification | `OperatorSemanticsRegistry` | [`decompiler/semantics.py`](../src/vraven/decompiler/semantics.py), [`operator_semantics.json`](../src/vraven/decompiler/data/operator_semantics.json) | Assigns conservative roles; unknown operations remain unknown |
| Runtime graph-value recording | `FXValueRecorder` | [`decompiler/runtime.py`](../src/vraven/decompiler/runtime.py) | Records declared executions and target margins |
| Model patching | `patch_model(...)` | [`decompiler/patching.py`](../src/vraven/decompiler/patching.py) | Controlled patched execution, not a permanent model mutation |
| Mechanism profiling | `MechanismProfiler` | [`decompiler/profiler.py`](../src/vraven/decompiler/profiler.py) | Node-level runtime and activation summaries |

### 3.2 Concepts, evolution and interventions

| Capability | Public type or output | Primary code | Evidence boundary |
|---|---|---|---|
| PCA internal-coordinate discovery | `ConceptDiscoveryEngine` | [`decompiler/concepts.py`](../src/vraven/decompiler/concepts.py) | Unsupervised latent directions; not human concepts by default |
| Sparse-autoencoder concept discovery | `ConceptDiscoveryEngine` | [`decompiler/concepts.py`](../src/vraven/decompiler/concepts.py) | Sparse latent codes with reconstruction and sparsity metadata |
| Exact scoring of unseen activations | `ConceptSet.score_activation(...)` | [`decompiler/concepts.py`](../src/vraven/decompiler/concepts.py) | Reuses the fitted centring, weights, bias and non-linearity |
| Conservative input-group association | `ground_concepts_from_input_groups(...)` | [`decompiler/concepts.py`](../src/vraven/decompiler/concepts.py) | Association-only labels do not count as validated semantics |
| Cross-layer concept evolution | `ConceptEvolutionBuilder` | [`decompiler/evolution.py`](../src/vraven/decompiler/evolution.py) | Optimal-transport lineage with retained mass and ambiguity diagnostics |
| Concept drift monitoring | `ConceptDriftMonitor` | [`decompiler/drift.py`](../src/vraven/decompiler/drift.py) | Comparative latent drift, not semantic identity drift without grounding |
| Graph-compiled interventions | `GraphInterventionCompiler` | [`decompiler/interventions.py`](../src/vraven/decompiler/interventions.py) | Executable erase/set/scale operations when the capture route supports rewriting |
| Causal concept graph | `CausalMechanismBuilder` | [`decompiler/causal_graph.py`](../src/vraven/decompiler/causal_graph.py) | Intervention effects and bootstrap evidence; not inferred from gradients alone |

### 3.3 Programme, contrast and counterfactual proof

| Capability | Public type or output | Primary code | Evidence boundary |
|---|---|---|---|
| Sparse executable mechanism programme | `ProgrammeCompiler` / `MechanismProgram` | [`decompiler/programme.py`](../src/vraven/decompiler/programme.py) | Fits observational and intervention constraints separately |
| Programme terms and interactions | `ProgrammeRule` | [`decompiler/programme.py`](../src/vraven/decompiler/programme.py) | Retains signed main and interaction contributions |
| Programme export | `to_pseudocode`, `to_python_code`, `to_c_code` | [`decompiler/programme.py`](../src/vraven/decompiler/programme.py) | Exported surrogate remains bounded by its fidelity certificate |
| Observational/interventional fidelity | `ProgrammeFidelity` | [`decompiler/programme.py`](../src/vraven/decompiler/programme.py) | R², MAE, decision agreement, intervention correlation and sign agreement remain separate |
| Contrastive mechanism explanation | `ContrastiveExplainer` | [`decompiler/contrastive.py`](../src/vraven/decompiler/contrastive.py) | Explains target-versus-alternative evidence within the extracted programme |
| Internal counterfactual search | `CounterfactualProofEngine` | [`decompiler/counterfactual.py`](../src/vraven/decompiler/counterfactual.py) | Separates programme crossing, model verification and domain validity |
| Analytic programme-boundary candidates | Counterfactual candidate ledger | [`decompiler/counterfactual.py`](../src/vraven/decompiler/counterfactual.py) | Exact within the declared programme term and empirical bounds |
| Multi-denominator mechanism certificate | `MechanismCertificate` | [`decompiler/certificate.py`](../src/vraven/decompiler/certificate.py) | Capture, semantic, intervention and fidelity dimensions are not collapsed |
| Per-feature mathematical audit | `ReverseFeatureAudit` | [`decompiler/feature_audit.py`](../src/vraven/decompiler/feature_audit.py) | Definitions, results and limitations per Reverse feature |
| Planted-mechanism benchmark | DecompileBench | [`decompiler/decompilebench.py`](../src/vraven/decompiler/decompilebench.py) | Development benchmark with known mechanism structure |

### 3.4 Reverse reports and signature views

| Feature | Capability shown | Algorithm | Visualisation |
|---|---|---|---|
| Raven Eye — Model X-ray | Capture route, graph, coverage and unresolved operations | [`capture.py`](../src/vraven/decompiler/capture.py) | [`model_xray.py`](../src/vraven/decompiler/visuals/model_xray.py) |
| Raven Flight — Concept Genome | Discovered coordinates and cross-layer lineage | [`concepts.py`](../src/vraven/decompiler/concepts.py), [`evolution.py`](../src/vraven/decompiler/evolution.py) | [`concept_genome.py`](../src/vraven/decompiler/visuals/concept_genome.py) |
| Raven Brain — Causal Proof | Executed intervention effects and causal graph | [`causal_graph.py`](../src/vraven/decompiler/causal_graph.py), [`interventions.py`](../src/vraven/decompiler/interventions.py) | [`causal_proof.py`](../src/vraven/decompiler/visuals/causal_proof.py) |
| Raven Code — Programme Proof | Executable sparse programme and fidelity | [`programme.py`](../src/vraven/decompiler/programme.py) | [`programme_proof.py`](../src/vraven/decompiler/visuals/programme_proof.py) |
| Raven Contrast | Supporting and opposing target-versus-alternative mechanisms | [`contrastive.py`](../src/vraven/decompiler/contrastive.py) | [`contrastive_proof.py`](../src/vraven/decompiler/visuals/contrastive_proof.py) |
| Raven Shadow — Counterfactual Twin | Candidate changes, verification and search accounting | [`counterfactual.py`](../src/vraven/decompiler/counterfactual.py) | [`counterfactual_twin.py`](../src/vraven/decompiler/visuals/counterfactual_twin.py) |
| Raven Dark Matter | Unresolved coverage, fidelity and claim boundaries | [`certificate.py`](../src/vraven/decompiler/certificate.py) | [`dark_matter.py`](../src/vraven/decompiler/visuals/dark_matter.py) |
| Decision Proof report | Integrated Reverse evidence and feature reports | `DecompilationResult.write_report(...)` | [`decompiler/report.py`](../src/vraven/decompiler/report.py) |

## 4. Adversarial mechanism forensics

The public entry point is `AdversarialAuditor` in
[`decompiler/adversarial.py`](../src/vraven/decompiler/adversarial.py). These attacks estimate
empirical failure; they are not formal robustness certificates.

| Capability | Primary code | Result semantics |
|---|---|---|
| FGSM | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Single-step L∞ white-box stress |
| BIM / I-FGSM | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Iterated projected sign-gradient stress |
| MI-FGSM | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Momentum iterative sign-gradient stress |
| PGD-L∞ | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Projected L∞ stress with configured steps and seed |
| PGD-L2 | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Projected L2 stress reported separately from L∞ |
| PGD-DLR | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | DLR-objective projected stress where logits support it |
| APGD-CE | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | VRAVEN-lite adaptive-step CE optimiser; not reference AutoAttack |
| APGD-DLR and targeted alternatives | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | VRAVEN-lite DLR optimiser with explicit target trials |
| FAB-L∞ | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | FAB-inspired lite boundary search |
| DeepFool | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Iterative local linearisation and empirical L2 flip radius |
| Flat-block black-box query search | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Query-limited flattened block search; not image Square Attack |
| Attack-union empirical robust accuracy | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Clean errors count as non-robust |
| Clean-conditional robustness and attack success rate | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Uses clean-correct samples as a separate denominator |
| Stratified sampling and exact sample manifest | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Records selected indices and per-class counts |
| Per-class and macro robust accuracy | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Exposes class imbalance and heterogeneous failure |
| Per-attack, unique and marginal yield | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Distinguishes redundant from uniquely discovered failures |
| Wilson confidence intervals | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Sampling uncertainty only; excludes attack incompleteness |
| Robust-accuracy curve and monotone attack-union envelope | [`adversarial_evidence.py`](../src/vraven/decompiler/adversarial_evidence.py) | Cumulative worst-case empirical curve over ε |
| Normalised robustness AUC and retention radius | [`adversarial_evidence.py`](../src/vraven/decompiler/adversarial_evidence.py) | Declared integration interval and grid |
| Attack-adequacy diagnostics | [`adversarial_evidence.py`](../src/vraven/decompiler/adversarial_evidence.py) | Can reveal attack failure but cannot prove attack completeness |
| Frozen-coordinate mechanism trace | [`adversarial_mechanism.py`](../src/vraven/decompiler/adversarial_mechanism.py) | Coordinate-aligned clean/adversarial activation comparison |
| Mechanism survival components | [`adversarial_mechanism.py`](../src/vraven/decompiler/adversarial_mechanism.py) | Cosine, retained energy, active-subspace and per-layer values remain available |
| Mechanism instability radius | [`adversarial_mechanism.py`](../src/vraven/decompiler/adversarial_mechanism.py) | Empirical pre-flip or flip-associated instability, not a certificate |
| Suppressed/activated units and input direction | [`adversarial.py`](../src/vraven/decompiler/adversarial.py) | Signed ΔM coordinates and clean input-gradient directions |
| Cross-attack agreement | [`adversarial_mechanism.py`](../src/vraven/decompiler/adversarial_mechanism.py) | Cosine agreement for aligned failure deltas |
| Deterministic failure families | [`adversarial_mechanism.py`](../src/vraven/decompiler/adversarial_mechanism.py) | Complete-link grouping of shared perturbation-mechanism patterns |
| Failure-mode spectrum | [`adversarial_mechanism.py`](../src/vraven/decompiler/adversarial_mechanism.py) | Singular-value energy and effective number of observed failure modes |

## 5. Formal verification and publication protocol

| Capability | Public type or output | Primary code | Formal scope |
|---|---|---|---|
| L∞ interval bound propagation | `IntervalBoundVerifier` | [`formal_robustness.py`](../src/vraven/decompiler/formal_robustness.py) | Supported `torch.fx` real-valued operator graph |
| Strict target-versus-all logit proof | `FormalVerificationPoint` | [`formal_robustness.py`](../src/vraven/decompiler/formal_robustness.py) | Certified only when every competing margin lower bound is positive |
| Certified radius bisection | `FormalRobustnessCertificate` | [`formal_robustness.py`](../src/vraven/decompiler/formal_robustness.py) | Lower bound; first inconclusive point is not a counterexample |
| Cohort and per-class certified accuracy | `CertifiedAccuracyPoint` | [`formal_robustness.py`](../src/vraven/decompiler/formal_robustness.py) | Uses all labelled samples and reports clean-conditional coverage separately |
| Bound-growth and operator ledger | Certificate metadata | [`formal_robustness.py`](../src/vraven/decompiler/formal_robustness.py) | Exposes relaxation bottlenecks and unsupported operations |
| Representation geometry survival | `MechanismPropertySpec` | [`formal_mechanism.py`](../src/vraven/decompiler/formal_mechanism.py) | Relative L2 drift and conservative cosine lower bounds |
| Representation top-k/order survival | `MechanismPropertySpec` | [`formal_mechanism.py`](../src/vraven/decompiler/formal_mechanism.py) | Coordinate-aligned ranking property |
| Executable intervention-effect-vector survival | `CausalEffectPropertySpec` | [`formal_mechanism.py`](../src/vraven/decompiler/formal_mechanism.py) | Fixed node, coordinates, baseline, target and competitor |
| Causal-effect sign survival | `CausalEffectSurvivalPoint` | [`formal_mechanism.py`](../src/vraven/decompiler/formal_mechanism.py) | Sign-definite effect intervals under the declared threshold |
| Joint decision–representation–causal certificate | `FormalCausalMechanismCertificate` | [`formal_mechanism.py`](../src/vraven/decompiler/formal_mechanism.py) | Conjunction over the same L∞ set |
| Independent calibration-split coordinate selection | `CoordinateSelectionProvenance` | [`formal_mechanism.py`](../src/vraven/decompiler/formal_mechanism.py) | Aggregate activation screen and aggregate executable intervention-effect ranking |
| Declared input semantics | `InputDomainSpec` | [`verification_protocol.py`](../src/vraven/decompiler/verification_protocol.py) | Bounds, units, preprocessing, semantic domain and source |
| Publication-readiness ledger | `PublicationReadinessLedger` | [`verification_protocol.py`](../src/vraven/decompiler/verification_protocol.py) | Separate protocol gates; deliberately not a trust score |

Definitions and limitations are documented in
[`FORMAL_ROBUSTNESS.md`](FORMAL_ROBUSTNESS.md),
[`FORMAL_CAUSAL_MECHANISM.md`](FORMAL_CAUSAL_MECHANISM.md) and
[`ADVERSARIAL_EVIDENCE.md`](ADVERSARIAL_EVIDENCE.md).

## 6. Mathematical metrics and reasoning objects

| Capability | Primary code | Interpretation |
|---|---|---|
| Activation statistics | [`metrics/basic.py`](../src/vraven/metrics/basic.py) | Mean, variance, energy, L1, extrema, sparsity, sign fractions and dispersion entropy |
| Gradient statistics | [`metrics/basic.py`](../src/vraven/metrics/basic.py) | Gradient norm and mean absolute magnitude |
| Robust scaling | [`metrics/basic.py`](../src/vraven/metrics/basic.py) | Median/IQR scaling with a safe fallback |
| Activation-gradient influence | [`metrics/influence.py`](../src/vraven/metrics/influence.py) | Local first-order target evidence, not causal evidence |
| Cumulative evidence and pathway concentration | [`metrics/influence.py`](../src/vraven/metrics/influence.py) | Layer-ordered evidence accumulation |
| Evidence concentration audit | [`metrics/evidence_audit.py`](../src/vraven/metrics/evidence_audit.py) | Effective layers, participation, HHI, Gini, cancellation and mass depth |
| Observational–causal concordance | [`metrics/evidence_audit.py`](../src/vraven/metrics/evidence_audit.py) | Gradient allocation versus matched zero-ablation effects with separate statistics |
| Agreement matrices | [`metrics/agreement.py`](../src/vraven/metrics/agreement.py) | Spearman, sign agreement and top-k overlap |
| Perturbation consistency | [`metrics/stability.py`](../src/vraven/metrics/stability.py) | Cosine stability and sign-flip behaviour |
| Conflict profile | [`metrics/uncertainty.py`](../src/vraven/metrics/uncertainty.py) | Internal opposition and intervention sensitivity, not calibrated predictive uncertainty |
| Deterministic Reasoning DNA | [`reasoning/fingerprint.py`](../src/vraven/reasoning/fingerprint.py) | Fixed-order reduced explanation fingerprint |
| Fingerprint similarity | [`reasoning/fingerprint.py`](../src/vraven/reasoning/fingerprint.py) | Similarity in the declared compressed trace space |
| Shared projections | [`reasoning/projection.py`](../src/vraven/reasoning/projection.py) | Shared PCA and aligned projection support |
| Activation communities | [`reasoning/communities.py`](../src/vraven/reasoning/communities.py) | Deterministic latent groups; not human concepts |
| Reasoning graph | [`reasoning/graph.py`](../src/vraven/reasoning/graph.py) | Execution-summary graph unless stronger edges are supplied |
| Pathway ranking and transition depth | [`reasoning/pathways.py`](../src/vraven/reasoning/pathways.py) | Ranked reduced evidence across execution order |

## 7. Core Proof visualisations

Every public figure is available through [`report/report.py`](../src/vraven/report/report.py),
supports `interactive_dark` and `publication_light`, and uses the same measured values in both
modes.

| Figure | Scientific capability | Implementation |
|---|---|---|
| Activation Energy | Localises reduced activation magnitude by layer | [`visualisation/energy.py`](../src/vraven/visualisation/energy.py) |
| Decision Flow | Shows supporting/opposing evidence progression | [`visualisation/decision_flow.py`](../src/vraven/visualisation/decision_flow.py) |
| Decision Sankey | Aggregates measured signed flow into feature–depth–output ribbons | [`visualisation/decision_sankey.py`](../src/vraven/visualisation/decision_sankey.py) |
| Spectral Trace | Shows signed layer-by-unit contribution structure | [`visualisation/spectral_trace.py`](../src/vraven/visualisation/spectral_trace.py) |
| Neuron Impact | Ranks signed compressed-unit influence | [`visualisation/neuron_impact.py`](../src/vraven/visualisation/neuron_impact.py) |
| Neuron Importance | Presents top supporting/opposing compressed units | [`visualisation/neuron_importance.py`](../src/vraven/visualisation/neuron_importance.py) |
| Attention Structure | Displays captured attention weights without relabelling them as causal importance | [`visualisation/attention.py`](../src/vraven/visualisation/attention.py) |
| Latent Flight | Shows aligned sample-representation structure across depth | [`visualisation/latent_flight.py`](../src/vraven/visualisation/latent_flight.py) |
| Reasoning DNA | Renders deterministic explanation fingerprints | [`visualisation/reasoning_dna.py`](../src/vraven/visualisation/reasoning_dna.py) |
| Conflict Field | Separates internal disagreement and explanation-risk dimensions | [`visualisation/conflict_field.py`](../src/vraven/visualisation/conflict_field.py) |
| Evidence Trajectory | Shows layer-ordered cumulative evidence | [`visualisation/evidence_trajectory.py`](../src/vraven/visualisation/evidence_trajectory.py) |
| Causal Pathway | Shows measured target-score changes under executed interventions | [`visualisation/causal_pathway.py`](../src/vraven/visualisation/causal_pathway.py) |
| Mechanism Concordance | Compares first-order influence with zero-ablation effects | [`visualisation/mechanism_concordance.py`](../src/vraven/visualisation/mechanism_concordance.py) |
| Input Attribution | Shows signed input-times-target-gradient evidence | [`visualisation/input_attribution.py`](../src/vraven/visualisation/input_attribution.py) |
| Feature Sequence Heatmap | Shows signed feature-by-sample/sequence attribution | [`visualisation/feature_sequence_heatmap.py`](../src/vraven/visualisation/feature_sequence_heatmap.py) |
| Reasoning Graph | Renders ordered execution-summary nodes and adjacency | [`visualisation/reasoning_graph.py`](../src/vraven/visualisation/reasoning_graph.py) |
| Activation Community Graph | Shows deterministic latent-community structure | [`visualisation/community_graph.py`](../src/vraven/visualisation/community_graph.py) |
| Relevance Flow | Shows layer-wise LRP relevance | [`visualisation/relevance_flow.py`](../src/vraven/visualisation/relevance_flow.py) |
| Perturbation Stability | Shows target-score behaviour across declared perturbations | [`visualisation/perturbation_stability.py`](../src/vraven/visualisation/perturbation_stability.py) |
| Counterfactual Delta | Shows original/counterfactual changes and verification state | [`visualisation/counterfactual_delta.py`](../src/vraven/visualisation/counterfactual_delta.py) |
| Consistency Matrix | Shows pairwise explanation-method or repeat agreement | [`visualisation/consistency.py`](../src/vraven/visualisation/consistency.py) |

Supporting attribution and causal algorithms reside in
[`attribution/lrp.py`](../src/vraven/attribution/lrp.py),
[`attribution/rules.py`](../src/vraven/attribution/rules.py),
[`causal/counterfactual.py`](../src/vraven/causal/counterfactual.py),
[`causal/interventions.py`](../src/vraven/causal/interventions.py),
[`causal/validation.py`](../src/vraven/causal/validation.py) and
[`causal/perturbation.py`](../src/vraven/causal/perturbation.py).

## 8. Atlas, Watch and Compare

| Capability | Algorithm | Visualisation |
|---|---|---|
| Dataset materialisation with stable IDs | [`analysis/dataset.py`](../src/vraven/analysis/dataset.py) | Report metadata |
| Dataset Reasoning Atlas | [`analysis/dataset.py`](../src/vraven/analysis/dataset.py) | [`visualisation/reasoning_dataset.py`](../src/vraven/visualisation/reasoning_dataset.py) |
| Reasoning similarity matrix | [`analysis/dataset.py`](../src/vraven/analysis/dataset.py) | [`visualisation/reasoning_dataset.py`](../src/vraven/visualisation/reasoning_dataset.py) |
| Nearest-neighbour reasoning outliers | [`analysis/dataset.py`](../src/vraven/analysis/dataset.py) | [`visualisation/reasoning_dataset.py`](../src/vraven/visualisation/reasoning_dataset.py) |
| Shared reasoning landscape | [`analysis/dataset.py`](../src/vraven/analysis/dataset.py) | [`visualisation/reasoning_dataset.py`](../src/vraven/visualisation/reasoning_dataset.py) |
| Checkpoint Watch | [`analysis/evolution.py`](../src/vraven/analysis/evolution.py) | [`visualisation/model_evolution.py`](../src/vraven/visualisation/model_evolution.py) |
| Matched Model Compare | [`analysis/comparison.py`](../src/vraven/analysis/comparison.py) | [`visualisation/model_comparison.py`](../src/vraven/visualisation/model_comparison.py) |

## 9. DarkSide advanced mechanism analyses

The canonical executable registry is [`darkside/registry.py`](../src/vraven/darkside/registry.py).
Each `AnalysisResult` retains metrics, tables, metadata, limitations and dark/publication figures.

| Feature | Capability | Primary code | Evidence type |
|---|---|---|---|
| Decision Circuitry | Finds an approximate sparse intervention-tested core | [`decision_circuitry.py`](../src/vraven/darkside/decision_circuitry.py) | Interventional |
| Internal Anatomy | Inspects live Linear-layer weight × activation structure | [`anatomy.py`](../src/vraven/darkside/anatomy.py) | Local observational |
| Circuit Wiring | Connects retained groups using measured weight × activation edge mass | [`wiring.py`](../src/vraven/darkside/wiring.py) | Measured coupling of intervention-selected groups |
| Circuit edge evidence cards | Adds contribution, intervention, stability and evidence-ladder records | [`edge_cards.py`](../src/vraven/darkside/edge_cards.py) | Mixed, explicitly typed |
| Shortcut Detection | Tests declared input groups for disproportionate decision reliance | [`shortcut.py`](../src/vraven/darkside/shortcut.py) | Mixed observational/interventional |
| Layer-wise Shortcut Probe | Measures decodability of declared groups across depth | [`shortcut.py`](../src/vraven/darkside/shortcut.py) | Probe-based observational |
| Prediction Depth | Finds the earliest stably class-decodable layer | [`shortcut.py`](../src/vraven/darkside/shortcut.py) | Probe-based observational |
| Causal Synergy | Detects cooperative, redundant and compensatory group effects | [`synergy.py`](../src/vraven/darkside/synergy.py) | Interventional |
| Activation Mediation | Tests whether clean internal state repairs a corrupted run | [`mediation.py`](../src/vraven/darkside/mediation.py) | Interventional |
| Circuit Multiplicity | Measures stable cores and alternative valid circuit masks | [`circuit_multiplicity.py`](../src/vraven/darkside/circuit_multiplicity.py) | Empirical mechanism multiplicity |
| Class-conditional Internals | Compares latent distributions between classes | [`class_conditional.py`](../src/vraven/darkside/class_conditional.py) | Cohort observational |
| Class-conditional Unit Field | Maps dense layer × unit class effect sizes | [`class_conditional.py`](../src/vraven/darkside/class_conditional.py) | Cohort observational |
| Failure Anatomy | Localises internal differences between errors and correct predictions | [`failure_anatomy.py`](../src/vraven/darkside/failure_anatomy.py) | Cohort observational |
| High-confidence OOD | Flags confident queries with atypical internal fingerprints | [`ood.py`](../src/vraven/darkside/ood.py) | Anomaly screening |
| Stability Geometry | Measures paired internal collapse under perturbation | [`stability_geometry.py`](../src/vraven/darkside/stability_geometry.py) | Paired observational |
| Adversarial Rerouting | Measures pathway redistribution after evasion | [`adversarial.py`](../src/vraven/darkside/adversarial.py) | Paired observational |
| Topological Shift | Compares neighbourhood and H0 persistence geometry | [`topology.py`](../src/vraven/darkside/topology.py) | Observational geometry |
| Explanation Uncertainty | Quantifies rank, sign and selection variability over repeats | [`explanation_uncertainty.py`](../src/vraven/darkside/explanation_uncertainty.py) | Empirical explanation uncertainty |
| Cross-method Disagreement | Exposes sign, rank and top-k disagreement | [`disagreement.py`](../src/vraven/darkside/disagreement.py) | Method uncertainty |
| Unused Capacity | Identifies dead, dormant-sensitive and active low-impact units | [`capacity.py`](../src/vraven/darkside/capacity.py) | Observational |
| Cost–Evidence Trade-off | Finds costly layers with little measured evidence | [`cost_evidence.py`](../src/vraven/darkside/cost_evidence.py) | Performance diagnostic |
| Representation Drift | Measures aligned geometric change across checkpoints/domains | [`representation_drift.py`](../src/vraven/darkside/representation_drift.py) | Comparative observational |
| Mechanism Transfer | Compares aligned intervention-effect maps across models/checkpoints | [`mechanism_transfer.py`](../src/vraven/darkside/mechanism_transfer.py) | Comparative interventional evidence |

Shared DarkSide infrastructure:

| Responsibility | Code |
|---|---|
| Session API | [`darkside/session.py`](../src/vraven/darkside/session.py) |
| Live intervention engine | [`darkside/interventions.py`](../src/vraven/darkside/interventions.py) |
| Cohort and confusion-set contract | [`darkside/dataset.py`](../src/vraven/darkside/dataset.py) |
| Typed result and feature contracts | [`darkside/contracts.py`](../src/vraven/darkside/contracts.py) |
| Result serialisation | [`darkside/serialisation.py`](../src/vraven/darkside/serialisation.py) |
| CKA, Procrustes, distribution distances, ranks, bootstrap and Pareto mathematics | [`darkside/math_utils.py`](../src/vraven/darkside/math_utils.py) |
| Dark/publication standalone figures | [`darkside/visualisation/figures.py`](../src/vraven/darkside/visualisation/figures.py) |
| Community-specific feature recommendations | [`darkside/registry.py`](../src/vraven/darkside/registry.py) |

## 10. Reporting, export and observatory generation

| Capability | Primary code | Behaviour |
|---|---|---|
| Scientific report object | [`report/report.py`](../src/vraven/report/report.py) | Capability detection, figures, trace persistence and export |
| Single-file report routing | [`report/export.py`](../src/vraven/report/export.py) | Validated HTML, PDF or JSON destination; local Chromium discovery for PDF |
| General self-contained Proof HTML | [`report/html.py`](../src/vraven/report/html.py) | Evidence audit, per-figure definitions, mappings, limitations, captions and exact data |
| Decision Proof and Reverse feature HTML | [`decompiler/report.py`](../src/vraven/decompiler/report.py) | Integrated and standalone Reverse evidence reports |
| Adversarial audit HTML/JSON | [`decompiler/adversarial.py`](../src/vraven/decompiler/adversarial.py) | Attack, mechanism and limitation ledger |
| Robustness-profile HTML/JSON | [`decompiler/adversarial_evidence.py`](../src/vraven/decompiler/adversarial_evidence.py) | Curves, AUC, intervals, class balance and attack yield |
| Formal certificate HTML/JSON | [`formal_robustness.py`](../src/vraven/decompiler/formal_robustness.py), [`formal_mechanism.py`](../src/vraven/decompiler/formal_mechanism.py) | Dual-theme proof ledgers and exact machine-readable values |
| Scientific cross-analysis synthesis | [`scripts/run_vraven_artifact.py`](../scripts/run_vraven_artifact.py) | Critical findings, claim boundaries and evidence links |
| Indexed observatory/gallery | [`scripts/build_vraven_gallery.py`](../scripts/build_vraven_gallery.py) | Searchable index across generated outputs |
| Plotly publication export | [`visualisation/export.py`](../src/vraven/visualisation/export.py) | SVG, PDF, PNG and journal-size presets where supported |
| Standalone HTML export | [`report/html.py`](../src/vraven/report/html.py) | Self-contained interactive Plotly reports with no remote runtime dependency |
| Scientific figure metadata contract | [`visualisation/base.py`](../src/vraven/visualisation/base.py) | Question, evidence tier, metrics, mappings, normalisation, limitations and exact data |
| Shared render themes | [`visualisation/theme.py`](../src/vraven/visualisation/theme.py) | Consistent dark and publication-light grammar |
| Semantic colour system | [`visualisation/palette.py`](../src/vraven/visualisation/palette.py) | Stable support, opposition, uncertainty and latent-community meanings |
| Label compression utilities | [`visualisation/labels.py`](../src/vraven/visualisation/labels.py) | Readable module, unit and feature labels |

## 11. CLI, validation and research-support tooling

| Capability | Code |
|---|---|
| `vraven explain`, `demo`, `capture`, `reverse`, `inspect`, `render`, `features`, `doctor` | [`cli.py`](../src/vraven/cli.py) |
| Complete model-family research runner | [`scripts/run_vraven_on_models.py`](../scripts/run_vraven_on_models.py) |
| Artefact-only research runner | [`scripts/run_vraven_artifact.py`](../scripts/run_vraven_artifact.py) |
| Gallery/index builder | [`scripts/build_vraven_gallery.py`](../scripts/build_vraven_gallery.py) |
| Code inventory generator | [`scripts/generate_code_inventory.py`](../scripts/generate_code_inventory.py) |
| Wheel installed-package smoke test | [`scripts/wheel_smoke.py`](../scripts/wheel_smoke.py) |
| Release/quality checks | [`scripts/quality_check.sh`](../scripts/quality_check.sh), [`scripts/release_validate.sh`](../scripts/release_validate.sh) |
| Conceptual architecture figures | [`figures/architecture.py`](../src/vraven/figures/architecture.py) |
| Methodology figures | [`figures/methodology.py`](../src/vraven/figures/methodology.py) |
| Validation/provenance figures | [`figures/validation.py`](../src/vraven/figures/validation.py) |
| Workflow and visual-contract figures | [`figures/integration.py`](../src/vraven/figures/integration.py) |
| Conceptual figure registry and renderer | [`figures/registry.py`](../src/vraven/figures/registry.py), [`figures/render.py`](../src/vraven/figures/render.py) |
| Conceptual figure primitives and visual grammar | [`figures/primitives.py`](../src/vraven/figures/primitives.py), [`figures/theme.py`](../src/vraven/figures/theme.py) |
| Synthetic tabular, CNN, UNet and Transformer probe architectures | [`scripts/probe_models.py`](../scripts/probe_models.py) |
| Synthetic probe training and reference-bundle generation | [`scripts/train_vraven_probe_models.py`](../scripts/train_vraven_probe_models.py), [`scripts/train_vraven_novel_probes.py`](../scripts/train_vraven_novel_probes.py) |
| Autoencoder anomaly-score adapter for research artefacts | [`scripts/ae_models.py`](../scripts/ae_models.py) |
| Adversarial-audit regeneration for existing outputs | [`scripts/regen_adversarial_audit.py`](../scripts/regen_adversarial_audit.py) |
| Circuit-wiring and edge-card regeneration | [`scripts/regen_circuit_wiring_cards.py`](../scripts/regen_circuit_wiring_cards.py) |
| Existing-report HTML layout repair and gallery rebuild | [`scripts/polish_vraven_html.py`](../scripts/polish_vraven_html.py) |
| Generated-output integrity validation | [`scripts/validate_outputs.py`](../scripts/validate_outputs.py) |
| Test-suite launcher | [`scripts/run_tests.sh`](../scripts/run_tests.sh) |
| Wheel build and installed-package validation | [`scripts/build_package.sh`](../scripts/build_package.sh) |
| Reproducible source release and archive checksums | [`scripts/create_release.py`](../scripts/create_release.py), [`scripts/create_archives.py`](../scripts/create_archives.py) |
| Safe DarkSide extension merge | [`scripts/merge_darkside.py`](../scripts/merge_darkside.py) |

The `src/vraven/figures` package produces conceptual methodology diagrams. They explain the
architecture and validation plan; they are not measurements from an analysed model.

## 12. Related documentation

- [`CODE_INVENTORY.md`](CODE_INVENTORY.md) — generated file-by-file source inventory.
- [`API_SPEC.md`](API_SPEC.md) — public Python and command-line contracts.
- [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) — implemented, partial and unavailable scope.
- [`VISUALISATION_CATALOGUE.md`](VISUALISATION_CATALOGUE.md) — scientific questions and visual contracts.
- [`VISUAL_SYSTEM.md`](VISUAL_SYSTEM.md) — mandatory visual language.
- [`PUBLICATION_VISUALISATIONS.md`](PUBLICATION_VISUALISATIONS.md) — journal figure and evidence-panel workflow.
- [`design_reference`](design_reference) — visual design references used to guide report presentation.
- [`REVERSE_FEATURE_AUDITS.md`](REVERSE_FEATURE_AUDITS.md) — mathematical definitions for Reverse audits.
- [`RESEARCH_METHOD.md`](RESEARCH_METHOD.md) — methodology, baselines and validation dimensions.

## 13. Explicit non-capabilities

To keep the inventory scientifically honest, VRAVEN does **not** currently claim:

- recovery of a unique or human-ground-truth reasoning algorithm;
- automatic human-semantic naming of latent coordinates;
- domain-valid counterfactuals without a supplied domain validator;
- formal certification for unsupported graph operators or arbitrary model families;
- equivalence between VRAVEN-lite adversarial optimisation and reference AutoAttack;
- that an inconclusive certificate proves vulnerability or mechanism change;
- that visual similarity, activation magnitude, gradients or attention alone establish causality;
- a calibrated universal explanation-confidence or trust score.
