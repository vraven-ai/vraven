# VRAVEN Code Inventory

This inventory is generated from the executable repository by `scripts/generate_code_inventory.py`. It helps contributors distinguish working modules from roadmap ideas. Every Python source file listed below contains executable code.

## Package summary

* Python modules: `158`
* Python source lines: `40734`
* Test modules: `56`
* Test cases: `200`
* Native framework adapter: PyTorch
* Official render modes: `interactive_dark` and `publication_light`
* Optional native crate: `rust_core`

## Source modules

| Module | Lines | Responsibility |
| --- | ---: | --- |
| `src/vraven/__init__.py` | 90 | VRAVEN: Visual Reasoning and Activation Visualisation for Explainable Networks. |
| `src/vraven/_version.py` | 3 | Single source of truth for the VRAVEN package version. |
| `src/vraven/adapters/__init__.py` | 13 | Model-framework adapters and explicit model-loading helpers. |
| `src/vraven/adapters/base.py` | 18 | Adapter protocol shared by model-framework integrations. |
| `src/vraven/adapters/pytorch.py` | 286 | Native PyTorch adapter for VRAVEN. |
| `src/vraven/adapters/pytorch_loader.py` | 131 | Explicit PyTorch model loading helpers with safe defaults. |
| `src/vraven/adapters/registry.py` | 27 | Adapter discovery for public VRAVEN APIs. |
| `src/vraven/analysis/__init__.py` | 16 | Dataset, model-comparison and checkpoint-evolution analyses. |
| `src/vraven/analysis/comparison.py` | 136 | Model-to-model comparison using the same materialised samples. |
| `src/vraven/analysis/dataset.py` | 505 | Dataset-level VRAVEN analysis built from independently reproducible traces. |
| `src/vraven/analysis/evolution.py` | 128 | Checkpoint evolution analysis for one fixed input. |
| `src/vraven/attribution/__init__.py` | 5 | Attribution backends beyond first-order gradients. |
| `src/vraven/attribution/lrp.py` | 283 | Layer-wise Relevance Propagation for supported PyTorch modules. |
| `src/vraven/attribution/rules.py` | 71 | LRP redistribution rules for supported PyTorch modules. |
| `src/vraven/bundle.py` | 139 | Validated self-describing model bundles for the public VRAVEN CLI. |
| `src/vraven/capture/__init__.py` | 21 | Framework-specific capture primitives. |
| `src/vraven/capture/pytorch_hooks.py` | 202 | Scoped PyTorch hooks that create reduced layer traces. |
| `src/vraven/capture/selectors.py` | 61 | Module-selection policy for PyTorch capture. |
| `src/vraven/capture/tensor_utils.py` | 155 | On-device tensor reduction utilities used by capture adapters. |
| `src/vraven/causal/__init__.py` | 17 | Targeted interventions, counterfactuals and perturbation probes. |
| `src/vraven/causal/counterfactual.py` | 154 | Constrained gradient counterfactuals for floating PyTorch inputs. |
| `src/vraven/causal/interventions.py` | 71 | Temporary, reversible PyTorch layer interventions. |
| `src/vraven/causal/perturbation.py` | 157 | Prediction-stability tests under controlled input perturbations. |
| `src/vraven/causal/validation.py` | 133 | Targeted intervention validation for candidate important layers. |
| `src/vraven/cli.py` | 1705 | VRAVEN command-line interface. |
| `src/vraven/core/__init__.py` | 38 | Core VRAVEN orchestration and trace types. |
| `src/vraven/core/config.py` | 83 | Configuration objects used by the VRAVEN orchestration layer. |
| `src/vraven/core/device.py` | 135 | Device and nested-input utilities. |
| `src/vraven/core/errors.py` | 35 | VRAVEN exception hierarchy. |
| `src/vraven/core/explainer.py` | 83 | Public orchestration API for VRAVEN explanations. |
| `src/vraven/core/metadata.py` | 86 | Reproducibility metadata helpers. |
| `src/vraven/core/output.py` | 227 | Model-output extraction and prediction summaries. |
| `src/vraven/core/trace.py` | 266 | Serializable trace schema for VRAVEN analyses. |
| `src/vraven/core/tree.py` | 67 | Nested model-input / output helpers used by DarkSide live interventions. |
| `src/vraven/darkside/__init__.py` | 73 | VRAVEN DarkSide: advanced internal mechanism audits for deep networks. |
| `src/vraven/darkside/adversarial.py` | 95 | Internal rerouting caused by adversarial, evasive or counterfactual inputs. |
| `src/vraven/darkside/anatomy.py` | 206 | Measured internal anatomy for Linear / MLP modules. |
| `src/vraven/darkside/capacity.py` | 101 | Unused capacity, dormant sensitivity and overparameterisation signatures. |
| `src/vraven/darkside/circuit_multiplicity.py` | 201 | Empirical multiplicity and stability of repeatedly discovered circuits. |
| `src/vraven/darkside/class_conditional.py` | 194 | Class-conditional internal representation and community comparison. |
| `src/vraven/darkside/contracts.py` | 150 | Common immutable contracts for VRAVEN DarkSide analyses. |
| `src/vraven/darkside/cost_evidence.py` | 95 | Computation cost versus measured evidence contribution. |
| `src/vraven/darkside/dataset.py` | 269 | Aligned cohort container for internal traces and representation studies. |
| `src/vraven/darkside/decision_circuitry.py` | 238 | Approximate sparse decision circuitry through constrained causal pruning. |
| `src/vraven/darkside/disagreement.py` | 124 | Cross-method explanation disagreement as a first-class diagnostic. |
| `src/vraven/darkside/edge_cards.py` | 740 | Precompute interactive CONNECTION cards for DarkSide circuit-wiring edges. |
| `src/vraven/darkside/explanation_uncertainty.py` | 90 | Uncertainty over repeated explanations, ranks and top-k membership. |
| `src/vraven/darkside/failure_anatomy.py` | 103 | Internal anatomy of false positives, false negatives and correct predictions. |
| `src/vraven/darkside/interventions.py` | 222 | Temporary unit, layer and activation interventions for live PyTorch models. |
| `src/vraven/darkside/math_utils.py` | 411 | Numerically guarded mathematical utilities for DarkSide analyses. |
| `src/vraven/darkside/mechanism_transfer.py` | 84 | Mechanism overlap and circuit distance across conditions or checkpoints. |
| `src/vraven/darkside/mediation.py` | 109 | Activation patching and bounded mediation-style recovery analysis. |
| `src/vraven/darkside/ood.py` | 140 | High-confidence internal mechanism anomaly detection. |
| `src/vraven/darkside/registry.py` | 160 | Feature catalogue and community-specific selection guidance. |
| `src/vraven/darkside/representation_drift.py` | 99 | Layerwise representation drift across checkpoints, periods or data domains. |
| `src/vraven/darkside/serialisation.py` | 25 | Persistence helpers for DarkSide result files. |
| `src/vraven/darkside/session.py` | 130 | Convenience session for live-model DarkSide analyses. |
| `src/vraven/darkside/shortcut.py` | 362 | Shortcut reliance and brittle dominance diagnostics. |
| `src/vraven/darkside/stability_geometry.py` | 88 | Geometry of internal-path stability under aligned perturbations. |
| `src/vraven/darkside/synergy.py` | 109 | Pairwise causal synergy, redundancy and compensation diagnostics. |
| `src/vraven/darkside/topology.py` | 130 | Topological signatures of internal representation geometry. |
| `src/vraven/darkside/visualisation/__init__.py` | 5 | Standalone visualisations for VRAVEN DarkSide result objects. |
| `src/vraven/darkside/visualisation/figures.py` | 814 | Scientifically mapped Plotly figures for every DarkSide analysis. |
| `src/vraven/darkside/wiring.py` | 191 | Measured circuit wiring from retained DarkSide cores + live Linear weights. |
| `src/vraven/decompiler/__init__.py` | 220 | Compiler-assisted causal decompilation for PyTorch models. |
| `src/vraven/decompiler/adversarial.py` | 2669 | VRAVEN Adversarial Mechanism Audit (stress + internal ΔM; not a certificate). |
| `src/vraven/decompiler/adversarial_evidence.py` | 667 | Statistical evidence and attack-adequacy diagnostics for adversarial audits. |
| `src/vraven/decompiler/adversarial_mechanism.py` | 555 | Layer-2 adversarial mechanism analysis (ΔM / survival / failure families). |
| `src/vraven/decompiler/capture.py` | 1282 | Multi-route PyTorch model capture for the VRAVEN causal decompiler. |
| `src/vraven/decompiler/causal_graph.py` | 575 | Intervention-validated causal mechanism graph over discovered internal concepts. |
| `src/vraven/decompiler/certificate.py` | 428 | Validity envelope and evidence certificate for VRAVEN decompilation results. |
| `src/vraven/decompiler/concepts.py` | 853 | Automatic internal concept discovery and conservative semantic grounding. |
| `src/vraven/decompiler/contrastive.py` | 256 | Deterministic contrastive explanations generated from an executable programme. |
| `src/vraven/decompiler/counterfactual.py` | 598 | Concept-level counterfactual search with behavioural and mechanistic verification. |
| `src/vraven/decompiler/data/__init__.py` | 1 | Packaged data files for the VRAVEN decompiler. |
| `src/vraven/decompiler/decompilebench.py` | 198 | Small PyTorch-native known-mechanism benchmark for VRAVEN development. |
| `src/vraven/decompiler/decompiler.py` | 548 | End-to-end compiler-assisted causal decompilation pipeline for PyTorch models. |
| `src/vraven/decompiler/drift.py` | 58 | Concept Drift and Out-of-Distribution (OOD) Monitor for VRAVEN. |
| `src/vraven/decompiler/evolution.py` | 306 | Cross-layer concept evolution using activation-profile transport. |
| `src/vraven/decompiler/feature_audit.py` | 417 | Mathematically explicit diagnostics for each Reverse feature family. |
| `src/vraven/decompiler/formal_mechanism.py` | 2000 | Formal survival certificates for internal and causal mechanism vectors. |
| `src/vraven/decompiler/formal_robustness.py` | 1868 | Sound interval certificates for local L-infinity classification robustness. |
| `src/vraven/decompiler/interventions.py` | 377 | Graph-compiled interventions and causal effect estimation. |
| `src/vraven/decompiler/ir.py` | 396 | Architecture-neutral mechanism intermediate representation for VRAVEN. |
| `src/vraven/decompiler/patching.py` | 63 | Cusal Model Patcher for VRAVEN. |
| `src/vraven/decompiler/profiler.py` | 211 | Execution and provenance profiling for the VRAVEN mechanism IR. |
| `src/vraven/decompiler/programme.py` | 723 | Compile discovered mechanisms into a compact executable causal programme. |
| `src/vraven/decompiler/report.py` | 510 | Focused HTML Decision Proof report for the VRAVEN causal decompiler. |
| `src/vraven/decompiler/runtime.py` | 314 | Runtime utilities shared by concept discovery, intervention and programme validation. |
| `src/vraven/decompiler/semantics.py` | 156 | Operator semantics registry used to lift PyTorch operations into VRAVEN's dialect. |
| `src/vraven/decompiler/verification_protocol.py` | 212 | Declared verification semantics and publication-evidence gates. |
| `src/vraven/decompiler/visuals/__init__.py` | 54 | Signature visualisations for the VRAVEN causal decompiler. |
| `src/vraven/decompiler/visuals/causal_proof.py` | 244 | Intervention-validated causal mechanism proof graph. |
| `src/vraven/decompiler/visuals/common.py` | 131 | Shared scientific visual helpers for VRAVEN decompiler figures. |
| `src/vraven/decompiler/visuals/concept_genome.py` | 216 | Cross-layer concept lineage visualisation. |
| `src/vraven/decompiler/visuals/contrastive_proof.py` | 266 | Contrastive causal programme view: why target rather than alternative. |
| `src/vraven/decompiler/visuals/counterfactual_twin.py` | 364 | Paired programme and concept view for counterfactual verification. |
| `src/vraven/decompiler/visuals/dark_matter.py` | 215 | Multi-denominator explanation-coverage iris. |
| `src/vraven/decompiler/visuals/model_xray.py` | 255 | Full-stack model X-ray for the captured VRAVEN mechanism IR. |
| `src/vraven/decompiler/visuals/programme_proof.py` | 275 | Executable programme proof tree. |
| `src/vraven/figures/__init__.py` | 6 | Reserved namespace for stable VRAVEN figure exports. |
| `src/vraven/figures/__main__.py` | 3 | Executable VRAVEN module. |
| `src/vraven/figures/architecture.py` | 311 | Executable VRAVEN module. |
| `src/vraven/figures/integration.py` | 130 | Executable VRAVEN module. |
| `src/vraven/figures/methodology.py` | 422 | Executable VRAVEN module. |
| `src/vraven/figures/primitives.py` | 460 | Executable VRAVEN module. |
| `src/vraven/figures/registry.py` | 156 | Executable VRAVEN module. |
| `src/vraven/figures/render.py` | 105 | Executable VRAVEN module. |
| `src/vraven/figures/theme.py` | 78 | Executable VRAVEN module. |
| `src/vraven/figures/validation.py` | 263 | Executable VRAVEN module. |
| `src/vraven/metrics/__init__.py` | 56 | Mathematically defined VRAVEN metrics. |
| `src/vraven/metrics/agreement.py` | 97 | Agreement metrics for comparing independent explanation vectors. |
| `src/vraven/metrics/basic.py` | 158 | Low-level reduced tensor statistics used by VRAVEN. |
| `src/vraven/metrics/evidence_audit.py` | 343 | Evidence concentration and observational--causal concordance audits. |
| `src/vraven/metrics/influence.py` | 80 | Target-specific first-order influence metrics. |
| `src/vraven/metrics/stability.py` | 39 | Explanation stability measurements. |
| `src/vraven/metrics/uncertainty.py` | 67 | Predictive and explanation uncertainty helpers. |
| `src/vraven/reasoning/__init__.py` | 38 | Reasoning objects built from reduced neural traces. |
| `src/vraven/reasoning/communities.py` | 91 | Deterministic discovery of latent activation communities. |
| `src/vraven/reasoning/fingerprint.py` | 116 | Deterministic reasoning fingerprints derived from reduced traces. |
| `src/vraven/reasoning/graph.py` | 44 | Execution-summary graphs for VRAVEN traces. |
| `src/vraven/reasoning/pathways.py` | 64 | Layer-level pathway summaries derived from target-specific influence. |
| `src/vraven/reasoning/projection.py` | 354 | Shared projection for layer-by-layer representation evolution. |
| `src/vraven/report/__init__.py` | 15 | VRAVEN report composition and export. |
| `src/vraven/report/export.py` | 134 | Single-file report export helpers. |
| `src/vraven/report/html.py` | 991 | Explicit composition of standalone figures into a self-contained HTML report. |
| `src/vraven/report/report.py` | 611 | High-level report object returned by VRAVEN explainers. |
| `src/vraven/terminal.py` | 157 | Dependency-free terminal presentation for the VRAVEN command-line interface. |
| `src/vraven/visualisation/__init__.py` | 78 | Standalone VRAVEN scientific visualisations. |
| `src/vraven/visualisation/attention.py` | 118 | Attention-weight structure visualisation. |
| `src/vraven/visualisation/base.py` | 139 | Scientific contracts and finalisation helpers for VRAVEN figures. |
| `src/vraven/visualisation/causal_pathway.py` | 123 | Intervention score changes for causally tested layers. |
| `src/vraven/visualisation/community_graph.py` | 274 | Latent activation-community graph for one captured layer. |
| `src/vraven/visualisation/conflict_field.py` | 146 | Separated predictive and explanation-risk dimensions. |
| `src/vraven/visualisation/consistency.py` | 111 | Multi-method explanation agreement matrix. |
| `src/vraven/visualisation/counterfactual_delta.py` | 215 | Counterfactual explanation UI: summary cards, change table and value diff. |
| `src/vraven/visualisation/decision_flow.py` | 158 | VRAVEN Decision Flow visualisation. |
| `src/vraven/visualisation/decision_sankey.py` | 448 | Contribution-over-time Sankey: features → model depth → class outputs. |
| `src/vraven/visualisation/energy.py` | 100 | Standalone activation-energy profile. |
| `src/vraven/visualisation/evidence_trajectory.py` | 240 | Cumulative evidence trajectory across model depth. |
| `src/vraven/visualisation/export.py` | 74 | Standalone figure export with journal-size presets. |
| `src/vraven/visualisation/feature_sequence_heatmap.py` | 150 | Feature × sample/sequence attribution heatmap. |
| `src/vraven/visualisation/input_attribution.py` | 248 | Signed input-space attribution view for continuous inputs. |
| `src/vraven/visualisation/labels.py` | 106 | Presentation helpers for long module and feature names. |
| `src/vraven/visualisation/latent_flight.py` | 243 | VRAVEN Latent Flight shared-projection / t-SNE small multiples. |
| `src/vraven/visualisation/mechanism_concordance.py` | 208 | Observational--causal layer concordance visualisation. |
| `src/vraven/visualisation/model_comparison.py` | 138 | Matched-sample comparison of two models. |
| `src/vraven/visualisation/model_evolution.py` | 174 | Checkpoint evolution of internal trace fingerprints and output confidence. |
| `src/vraven/visualisation/neuron_impact.py` | 162 | Signed unit-impact field for one selected layer. |
| `src/vraven/visualisation/neuron_importance.py` | 207 | Top contributing neuron table for one layer (mockup-style importance view). |
| `src/vraven/visualisation/palette.py` | 198 | Canonical semantic colour tokens for VRAVEN figures. |
| `src/vraven/visualisation/perturbation_stability.py` | 161 | Perturbation stability curves. |
| `src/vraven/visualisation/reasoning_dataset.py` | 349 | Dataset-level reasoning-fingerprint visualisations. |
| `src/vraven/visualisation/reasoning_dna.py` | 192 | VRAVEN Reasoning DNA matrix and compressed fingerprint ribbon. |
| `src/vraven/visualisation/reasoning_graph.py` | 155 | Execution-summary reasoning graph. |
| `src/vraven/visualisation/relevance_flow.py` | 174 | Layer-wise LRP relevance flow across supported modules. |
| `src/vraven/visualisation/spectral_trace.py` | 107 | Signed layer-by-unit spectral trace. |
| `src/vraven/visualisation/theme.py` | 363 | Two-mode visual theme shared by every standalone VRAVEN figure. |

## Tests

| Test module | Test cases |
| --- | --- |
| `tests/decompiler/test_capture_ir.py` | export capture is equivalent and serialisable |
| `tests/decompiler/test_certificate_semantics.py` | association label is not counted as validated semantics |
| `tests/decompiler/test_concepts_evolution.py` | concepts and evolution are deterministic, unseen activation scoring matches fitted reference coordinates |
| `tests/decompiler/test_decompilebench.py` | planted benchmark runs |
| `tests/decompiler/test_end_to_end.py` | end to end decision proof and visuals |
| `tests/decompiler/test_intervention_compiler.py` | graph intervention is explicit and changes output |
| `tests/decompiler/test_reverse_algorithms.py` | decisive depth excludes positive intercept only state, decisive depth can resolve evidence at depth zero, analytic candidate crosses exact multilinear programme boundary, analytic candidate rejects boundary outside empirical interval, wilson interval remains non degenerate at perfect accuracy |
| `tests/test_agreement_metrics.py` | agreement matrices are symmetric |
| `tests/test_attention_capture.py` | attention weights are captured when model exposes them |
| `tests/test_basic_metrics.py` | activation statistics |
| `tests/test_bundle.py` | model bundle runs without repeating model configuration, model bundle rejects paths outside directory, bundle init creates valid manifest |
| `tests/test_causal_mode.py` | causal mode runs temporary interventions and restores model |
| `tests/test_cli.py` | cli demo and inspect, cli demo writes one html report, cli explain model with public parameters, cli features and doctor are machine readable, cli raven has non animated fallback, cli explain parameter contract, cli analysis commands execute real algorithms, cli source backed cohort and model commands, cli beginner analyse workflow and alias |
| `tests/test_code_inventory.py` | every runtime subpackage contains real source, runtime source has no scaffold markers, inventory document matches runtime source tree |
| `tests/test_community_graph.py` | community graph is deterministic and dual mode |
| `tests/test_counterfactual.py` | continuous counterfactual can flip simple linear classifier |
| `tests/test_darkside_advanced.py` | method disagreement reports rank and sign structure, explanation uncertainty and transfer, high confidence internal anomaly, h0 persistence deaths match line spacing |
| `tests/test_darkside_all_visuals.py` | every analysis renders standalone dark and publication figures |
| `tests/test_darkside_anatomy_wiring.py` | internal anatomy prints and figures, circuit wiring uses measured edge mass, class conditional unit field heatmap |
| `tests/test_darkside_circuit_multiplicity.py` | circuit multiplicity finds stable core and alternatives, circuit multiplicity validates alignment and binary values, circuit multiplicity figure has both modes |
| `tests/test_darkside_cohorts.py` | class conditional and failure anatomy, drift stability and rerouting profiles, capacity cost topology and prediction depth, layerwise shortcut probe |
| `tests/test_darkside_dataset.py` | subset and confusion cohorts are aligned, dataset rejects misaligned matrix, from reports builds aligned dataset across fast and gradient traces |
| `tests/test_darkside_live.py` | sparse circuit retains dominant units, shortcut audit finds the dominant coordinate, pairwise synergy and activation mediation run, session wraps live analyses |
| `tests/test_darkside_math.py` | linear cka is invariant to orthogonal feature rotation, distribution distances are zero for identical sets, rank and sign agreement metrics, effective rank and hedges g are finite, neighbourhood and pareto frontier |
| `tests/test_darkside_registry_serialisation.py` | recommendations are focused by community, extension version is exposed, result round trip |
| `tests/test_darkside_safeguards.py` | intervention engine restores mixed training flags, group contracts reject invalid coordinates, trace dataset rejects duplicate sample ids, drift index is dimensionless and bounded, ood anomaly threshold is configurable |
| `tests/test_darkside_visuals.py` | both modes preserve scientific values, capacity and uncertainty figures render, unknown figure is rejected |
| `tests/test_dataset_analysis.py` | explain dataset fingerprints and round trip, dataset item preserves ids labels and targets, dataset outlier view has declared neighbourhood |
| `tests/test_decompiler_adversarial.py` | adversarial certificate alias, default suite covers literature attacks, fgsm flips at large epsilon not tiny, pgd and bim in suite, pgd restarts retain the strongest candidate, pgd restart budget must be positive, fab apgd and autoattack, mechanism survival and instability, mechanism instability uses the declared component, batch robust accuracy, vectorised batch pgd records constraints and restart union, linf projection is inward rounded for large float32 coordinates, overall robust accuracy counts clean errors as non robust, stratified selection covers ordered classes, cross entropy and dlr objectives have correct invariances, input gradient does not overwrite parameter gradients, activation probe freezes clean coordinate basis, robustness profile auc and monotone envelope, robustness curve uses per sample union across epsilon, report has no fake scores and uses succeeded, model parameters unchanged, concept direction attribution ranks planted direction, audit with concept directions end to end |
| `tests/test_evidence_audit.py` | evidence concentration has known uniform solution, observational causal concordance is scale invariant, scientific audit keeps dimensions separate, mechanism concordance is dual mode and scientifically labelled, html report contains audit ledger and can disable it |
| `tests/test_explainer.py` | vraven captures layers |
| `tests/test_figures.py` | every figure renders as svg, expected figure count |
| `tests/test_formal_mechanism.py` | joint certificate proves three distinct properties, representation and causal effect bounds contain all box corners, relational effect bounds are exact for planted affine readout, relational bounds cancel large shared output path, relational relu secant bound contains exact crossing interval, relational residual arithmetic contains dense box grid, auto selection uses intervention effect not activation alone, auto selection does not silently shrink requested causal basis, zero effect predeclared coordinate blocks causal certificate, calibration selected coordinate must have local clean effect, calibration split freezes coordinates and passes selection gate, frozen basis cohort reports each property separately, cohort records exact decision mechanism contingency, cohort applicability is frozen at zero and separates preconditions, invalid node and coordinate fail explicitly, public api model state and dual theme outputs |
| `tests/test_formal_robustness.py` | affine certified radius matches dual norm solution, interval bounds contain all box corners, functional residual arithmetic contains all box corners, shape pad and axis reindexing bounds contain all box corners, tokenwise linear bias uses last output axis, transformer frontend reaches explicit layernorm boundary, supported cnn operator ledger and zero radius, unsupported operator is visible not approximated, training dropout variant uses inference semantics and restores mode, verification preserves parameters and dtype, cohort certificate separates clean errors and inconclusive, certificate round trip outputs include both themes, declared unbounded input semantics are distinct from missing bounds |
| `tests/test_full_tensor_capture.py` | full tensor capture is explicit ephemeral and releasable, full tensors are not retained by default, explicit device declaration must match live model |
| `tests/test_gradient_capture.py` | gradient mode captures target specific signals without parameter grads, repeated capture removes hooks |
| `tests/test_html_theme_toggle.py` | export html includes theme toggle, export html can disable theme toggle |
| `tests/test_lrp_and_mockup_suite.py` | lrp linear relu conservation reasonable, lrp rejects unsupported only model, shared pca and layer tsne, dataset tsne and landscape highlight, neuron importance and sankey and perturbation, counterfactual ui figure |
| `tests/test_model_comparison.py` | identical models have identical reasoning fingerprints, checkpoint evolution uses declared order |
| `tests/test_model_evolution.py` | checkpoint evolution tracks similarity to final |
| `tests/test_model_loading.py` | state dict loading requires and uses architecture factory, nested state dict and prefix removal, pickle loading is explicitly blocked by default, torchscript loading, model loading rejects missing files and unknown formats, model loading rejects missing nested key, non strict state dict reports incompatibilities |
| `tests/test_optional_devices.py` | cuda smoke, mps smoke |
| `tests/test_public_api.py` | public api exports runtime entry points, adapter and core subpackages export their public types |
| `tests/test_public_hygiene.py` | public tree has no development instruction files, public text has no private environment markers, notebooks are clean source documents, public version metadata is consistent, package does not claim unverified pep561 compliance, generated reports have no remote runtime dependencies, examples do not disable restricted checkpoint loading |
| `tests/test_publication_evidence.py` | publication key results do not change with render mode, neuron importance uses support and opposition colours by sign |
| `tests/test_publication_model_suite.py` | publication models reconstruct exactly, suite has 26 configurations and 130 seeded checkpoints, wdbc preprocessing is train fitted and split disjoint, architecture payload builds for vector data, publication sequence model mps smoke |
| `tests/test_publication_presets.py` | publication preset pixel conversion |
| `tests/test_reasoning_and_visuals.py` | core visuals support identical data in both modes, latent flight uses one shared projection |
| `tests/test_release_script.py` | release selection excludes generated and build files, release runtime validation and version, release file list is limited to version controlled files, release requires a clean committed tree |
| `tests/test_report_export.py` | resolve report destination infers or adds extension, resolve report destination rejects mismatch, html to pdf requires local browser, html to pdf writes one valid pdf, html to pdf accepts valid file written before browser timeout |
| `tests/test_reproducibility_metadata.py` | input fingerprint is stable and changes with input |
| `tests/test_source_completeness.py` | source files are non empty and not placeholders |
| `tests/test_standalone_visualisation.py` | publication visual is white and standalone, dark visual uses same values |
| `tests/test_terminal.py` | terminal panel and dancing mascot, emit result remains json when not interactive |
| `tests/test_trace_serialisation.py` | trace round trip |
| `tests/test_visual_themes.py` | publication theme is white and has no glow, interactive theme uses vraven navy, semantic roles exist in both modes, vraven semantic meaning, signed scale matches vraven impact, vraven theme apply registers templates |

## Examples and benchmarks

* `examples/__init__.py`
* `examples/attention_demo.py`
* `examples/causal_demo.py`
* `examples/darkside_circuit_multiplicity.py`
* `examples/darkside_method_disagreement.py`
* `examples/darkside_quickstart.py`
* `examples/darkside_security_audit.py`
* `examples/darkside_training_audit.py`
* `examples/dataset_comparison.py`
* `examples/quickstart.py`
* `examples/visual_gallery.py`
* `examples/visual_modes.py`
* `benchmarks/benchmark_capture.py`
* `benchmarks/decompilebench.py`

## Build and repository support

* `SECURITY.md`
* `CITATION.cff`
* `Makefile`
* `.editorconfig`
* `scripts/build_package.sh`
* `scripts/generate_code_inventory.py`
* `scripts/quality_check.sh`
* `scripts/release_validate.sh`
* `scripts/run_vraven_artifact.py`
* `scripts/run_vraven_on_models.py`
* `scripts/run_tests.sh`
* `scripts/wheel_smoke.py`
* `scripts/create_release.py`
* `.github/workflows/release.yml`
* `.github/workflows/tests.yml`

## Documentation

* `docs/ADVERSARIAL_EVIDENCE.md`
* `docs/API_SPEC.md`
* `docs/ARCHITECTURE.md`
* `docs/CAPTIONS.md`
* `docs/CODE_INVENTORY.md`
* `docs/DEFINITION_OF_DONE.md`
* `docs/DEVELOPER_GUIDE.md`
* `docs/EDITING_GUIDE.md`
* `docs/EVIDENCE_AUDIT.md`
* `docs/FEATURES.md`
* `docs/FEATURE_CODE_MAP.md`
* `docs/FIGURE_INDEX.md`
* `docs/FORMAL_CAUSAL_MECHANISM.md`
* `docs/FORMAL_ROBUSTNESS.md`
* `docs/IMPLEMENTATION_STATUS.md`
* `docs/METHODOLOGY_NOTES.md`
* `docs/MODEL_BUNDLES.md`
* `docs/PUBLICATION_VISUALISATIONS.md`
* `docs/PYPI_RELEASE_GUIDE.md`
* `docs/QUALITY_CHECKLIST.md`
* `docs/RESEARCH_METHOD.md`
* `docs/REVERSE_FEATURE_AUDITS.md`
* `docs/ROADMAP.md`
* `docs/SECURITY_PRIVACY.md`
* `docs/SOURCE_BASIS.md`
* `docs/STYLE_GUIDE.md`
* `docs/VALIDATION_REPORT.md`
* `docs/VISUALISATION_CATALOGUE.md`
* `docs/VISUAL_SYSTEM.md`

## Rust core

`rust_core/src/lib.rs` contains parallel activation summary, cosine similarity and linear resampling primitives. It is intentionally not bound into the Python wheel until profiling and numerical parity tests justify the additional build complexity. The Python implementation remains authoritative in the current research alpha.

## Explicitly absent

The repository does not contain fake TensorFlow, ONNX, custom CUDA or VRAVEN Studio implementations. Those items remain in the roadmap. Contributors must add real code, tests and documentation before changing their status.
