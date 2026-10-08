# Changelog

## 0.10.1 - Unreleased

* Stop advertising PEP 561 compliance until the public API passes a dedicated static-type gate.
* Removed the remotely loaded commercial chart runtime; HTML reports now use
  the existing local Plotly backend throughout and remain offline after export.
* Removed remote font imports from generated galleries and report-polishing
  tools so generated research artefacts do not contact third-party font hosts.

* Removed private development instructions, stale internal validation logs and executed
  notebook outputs from the public source tree.
* Removed an obsolete conversational concept draft that mixed proposed future
  capabilities with the implemented public scope.
* Restricted source archives to version-controlled files and added public-tree
  hygiene checks for private paths and development-only instruction files.
* Standardised local quality commands, enforced Ruff in CI and added Python 3.13
  to the test matrix.
* Aligned the README, package metadata and command help with VRAVEN's
  evidence-aware explainability and causal model-auditing scope.

* Refreshed the PyPI project description from the current concise README so it
  no longer links to the repository-only editable Draw.io source.
* Constrained the documentation identity-banner wordmark so it cannot overlap
  adjacent copy at intermediate laptop and tablet widths.

## 0.10.0 - 2026-09-26

* Added a responsive GitHub Pages documentation site with short, problem-led
  guides for installation, model bundles, explanation, adversarial testing,
  counterfactual search, report selection and troubleshooting. The repository
  README now acts as a concise project entry point.
* Added executable CLI entry points for empirical adversarial auditing,
  counterfactual search, perturbation stability, Layer-wise Relevance Propagation
  and supported-operator formal L∞ certification.
* Expanded `vraven adversarial` with single-sample, fixed-epsilon batch and
  labelled robustness-curve scopes, including explicit attack budgets, sampling
  controls and JSON/HTML outputs where supported.
* Added a dependency-free terminal presentation layer with TTY-only result
  panels, JSON-safe redirected output and an optional still, blinking or dancing
  raven mascot through `vraven raven`.
* Exposed existing dataset-atlas, model-comparison, checkpoint-evolution and
  DarkSide internal-anatomy algorithms through the `dataset`, `compare`,
  `evolution` and `anatomy` CLI commands.
* Added the beginner-facing `vraven analyse` workflow, with `analyze` and
  `quick` aliases, task presets, automatic output names and short options for
  reaching common analyses without learning every expert parameter.
* Added validated self-describing model bundles using `vraven.json`, including
  `bundle-init`, `bundle` validation and direct `vraven quick BUNDLE_DIR`
  execution with command-line overrides.

## 0.9.0 - 2026-09-26

* Added the parameterised `vraven explain` command for saved PyTorch models,
  including capture selection, causal controls, target selection, device handling
  and explicit single-report output.
* Added `vraven features` for machine-readable capability discovery and expanded
  `vraven doctor` with active executable, package path, report formats and PDF
  browser diagnostics.
* Added exactly-one-report HTML, PDF or JSON export to explanation and Reverse
  workflows. PDF reports use a locally installed Chromium-family browser.
* Documented that state dictionaries require an architecture factory and that
  TorchScript modules cannot expose the hooks required for VRAVEN instrumentation.
* Promoted the package from `0.9.0a1` to the conventional pre-1.0 release `0.9.0`.

* Added a frozen zero-radius cohort applicability ledger for formal causal-mechanism
  studies. Representation applicability now separates undefined clean geometry and
  top-k ties from survival, causal applicability records complete threshold-eligible
  intervention bases, and the exact D/M table is additionally reported conditional on
  their conjunction. Non-applicable samples remain explicitly inconclusive rather than
  being counted as mechanism-property failures.
* Strengthened empirical PGD evaluation with deterministic multi-start best-path
  retention, batched full-cohort execution, finite/zero-gradient and constraint
  diagnostics, matched per-sample attack unions across epsilon and attack seeds,
  paired-bootstrap robustness-AUC intervals and simultaneous DKW sampling bands.
* Made causal-coordinate eligibility a certificate precondition: every frozen
  coordinate must have a strictly positive, thresholded lower bound on its clean
  intervention effect, and undersized auto-selected bases now produce explicit
  inconclusive causal/joint verdicts instead of silently certifying a smaller or
  partly zero-effect property.
* Added machine-readable input-domain semantics that distinguish intentional unbounded
  domains from omitted protocol information in both formal certificate families.
* Added disjoint calibration-split selection for causal intervention coordinates using
  aggregate activation screening and aggregate executable intervention effects, with frozen
  sample IDs, selection scores and leakage provenance.
* Added a non-composite formal-mechanism publication-readiness ledger covering input semantics,
  coordinate-selection independence, evaluation identity, independent verifier replay,
  complete separation testing and multi-model validation.

* Aligned the existing Proof figures, report shell, gallery, Reverse reports and formal evidence
  pages to the owner-supplied design-reference palette. Figure geometry and scientific values are
  unchanged; only colour tokens and previously hard-coded colour exceptions were updated.
* Added theme version 3.0 publication evidence panels to every composed Proof visualisation,
  including typed evidence tiers, metric definitions, deterministic key findings, exact plotted
  data download and copyable journal captions.
* Publication-light standalone figures now retain explanatory titles by default.
* Corrected support/opposition colour semantics in unit-contribution and Sankey views, renamed
  ambiguous feature-importance, layer-relevance and causal-pathway headings, and corrected the
  perturbation-stability axis so the registered score is not labelled as confidence.

* Added `FormalCausalMechanismVerifier`, which formally separates decision
  invariance, coordinate-aligned representation survival and survival of an executable
  internal intervention-effect vector. It reports independent geometry, sign,
  top-k and joint certified-radius lower bounds with complete property manifests and
  dual-mode JSON/HTML proof ledgers.
* Tightened formal intervention-effect enclosures with shared-input relational IBP:
  exact affine cancellation, sound secant-slope nonlinear bounds and fused terminal
  target-versus-competitor readouts. Cohort ledgers now retain the exact
  representation–causal conjunction and complete decision–mechanism contingency counts.
* Added sound graph-node bound capture and interval intervention transforms to the
  native verifier, plus focus-sample causal-coordinate screening by measured
  intervention effect rather than activation magnitude.

* Added `IntervalBoundVerifier`, a formal local L∞ robustness backend for supported
  PyTorch FX graphs. It emits strict competing-logit margin proofs, bisection-derived
  certified-radius lower bounds, cohort/per-class certified accuracy, layer bound-growth
  diagnostics, explicit unsupported verdicts and dual-mode JSON/HTML certificates.

* Corrected APGD-CE and DLR objectives, preserved model parameter gradients during
  attack differentiation and rejected structured outputs that cannot be interpreted
  safely as class logits.
* Replaced misleading AutoAttack and Square labels with implementation-accurate
  VRAVEN ensemble-lite and flat-block query-search terminology.
* Fixed mechanism comparisons to freeze clean-selected neuron coordinates, expose
  separate survival components and use deterministic complete-link failure families.
* Added overall and clean-conditional attacked robust accuracy, stratified sample
  manifests, per-class/macro results, raw/unique/marginal attack yield, robustness
  curves, normalised AUC and result-first dual-mode HTML.

* Reframed Reverse feature reports around primary scientific results, explicit interpretations
  and evidence-status labels. Limitations and definitions remain available in collapsible
  evidence-boundary sections instead of preceding the result.
* Added a result-first Decision Proof executive summary highlighting numerical equivalence,
  held-out decision agreement, programme fidelity, causal effects, retained transport mass and
  executable programme complexity.
* Added a Scientific Evidence Audit with entropy-based effective layer count, participation
  ratio, Herfindahl concentration, normalised Gini, signed cancellation and mass-crossing depth.
* Added scale-invariant observational–causal concordance for matched zero-output layer
  interventions, including rank, sign, cosine, L1 discrepancy and permutation significance.
* Added the dual-mode Mechanism Concordance figure and evidence-led HTML findings, hypothesis
  table, full layer ledger, mathematical assumptions and machine-readable audit values.
* Added an artefact-directory-only full-suite runner with cryptographic provenance, checkpoint
  reconstruction, cohort activation communities and an indexed HTML gallery.
* Expanded the full-suite integration to cover 19 applicable DarkSide analyses, including
  class-conditional effects, paired stability geometry, rerouting, topology, explanation
  uncertainty, checkpoint drift and aligned causal-mechanism transfer. Unsupported evidence
  contracts are now reported explicitly instead of being silently approximated.
* Added mathematically defined audits to every signature Reverse feature, including retained
  transport mass, lineage ambiguity, bootstrap interval exclusion, effective causal support,
  surrogate cancellation and evidence competition.
* Corrected contrastive decisive depth so a positive programme intercept cannot masquerade as
  internal evidence acquired at depth zero.
* Added exact one-coordinate programme-boundary candidates to bounded counterfactual search and
  separated boundary crossings, graph attempts, execution failures and verification-budget
  exhaustion.
* Added Wilson 95% intervals to finite-sample adversarial robust-accuracy estimates.

## 0.9.0a1 - Compiler-Assisted Causal Decompiler Research Alpha

### Added

* Multi-route PyTorch capture router prioritising `torch.export` and FX.
* Architecture-neutral VRAVEN Mechanism IR.
* Operator-semantics registry and typed mechanism dialect.
* Numerical-equivalence and autograd-path validation.
* Automatic PCA and sparse-autoencoder concept discovery.
* Conservative association-only concept grounding.
* Deterministic cross-layer concept evolution through optimal transport.
* Graph-compiled internal intervention engine.
* Causal concept graph with bootstrap intervals and derivative screening.
* Intervention-constrained executable programme extraction.
* Contrastive explanation and graph-verified internal counterfactual search.
* Separate validity, coverage and fidelity certificate.
* Seven signature decompiler visualisations in dark and publication modes.
* PyTorch-native planted-mechanism development benchmark.
* New `vraven capture`, `vraven reverse`, `vraven decompile-demo` and `vraven doctor` commands.
* Dedicated decompiler architecture, mathematics and validation documentation.

### Changed

* Programme concept selection now prioritises intervention-tested concepts rather than observational readout concepts.
* Programme fitting now includes explicit intervention-effect constraints.
* Association-derived labels no longer count as validated semantic coverage.
* Parameter coverage distinguishes parameters from buffers in exported graphs.

### Scientific boundary

This alpha does not claim universal model capture, unique mechanism recovery or state-of-the-art performance. All explanations are scoped to the captured execution paths, tested interventions and reported validity envelope.

## 0.2.0a1

* Replaced the template with a working PyTorch explanation engine.
* Added automatic CPU, CUDA and MPS device following through PyTorch.
* Added scoped activation, representation, unit-profile and attention capture.
* Added fast, gradient and targeted causal explanation modes.
* Added input-times-gradient attribution and first-order activation-gradient influence.
* Added deterministic trace serialisation, model structure metadata and bounded input fingerprints.
* Added causal layer intervention validation and continuous-input counterfactual generation.
* Implemented standalone VRAVEN figures for activation energy, decision flow, spectral trace, neuron impact, attention, latent flight, reasoning DNA, conflict, evidence trajectory, causal effects, input attribution, execution graphs and activation communities.
* Added method consistency, counterfactual, dataset similarity, model comparison and checkpoint evolution figures.
* Added dataset-level independent sample analysis, matched model comparison and checkpoint evolution APIs.
* Added local interactive HTML report composition and publication export presets.
* Added an optional Rust reference core for profiled CPU-side reductions.
* Added explicit, ephemeral full-tensor capture with release controls and no JSON persistence.
* Added strict device mismatch errors so VRAVEN never silently moves a live model.
* Added safe PyTorch model loading for state dictionaries, nested checkpoints and legacy TorchScript artefacts.
* Added complete public adapter and exception exports plus generated code inventory checks.
* Added full Apache License 2.0 text and project notice.
* Expanded tests across capture, gradients, attention, causal analysis, persistence, datasets, comparisons and both visual modes.

## 0.1.0a1

* Added mandatory `interactive_dark` and `publication_light` render modes.
* Added the VRAVEN semantic theme engine.
* Added standalone figure and scientific metadata contracts.
* Added publication size presets and optional Kaleido export support.
* Added a detailed visualisation catalogue.
* Added strict visualisation contribution and validation rules.
* Added design reference images with explicit non-copying guidance.

## 0.1.0a0

* Created the initial VRAVEN research architecture and documentation.
