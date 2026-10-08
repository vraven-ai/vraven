# VRAVEN Implementation Status

## Release

Version: `0.10.1`

Status: working PyTorch research alpha.

## Working public entry points

```python
vraven.explain
vraven.Explainer
vraven.VRavenConfig
vraven.VRavenReport
vraven.explain_dataset
vraven.VRavenDatasetReport
vraven.compare_models
vraven.VRavenModelComparison
vraven.compare_checkpoints
vraven.VRavenEvolutionReport
vraven.generate_counterfactual
vraven.load_pytorch_model
```

## Supported now

* PyTorch live models
* Safe state-dictionary loading with an explicit architecture factory and unsafe pickle blocked by default
* CPU, CUDA and MPS through PyTorch
* Scalar, binary and multiclass classification-shaped outputs
* Fast, gradient and bounded causal modes
* Independent sample dataset analysis
* Twenty-one core standalone scientific figure families
* Evidence-led publication report sections with explicit evidence tiers, measured key findings,
  claim boundaries, journal captions and exact plotted-data export
* Evidence concentration and observational–causal concordance audit
* Explicit ephemeral full-tensor capture with manual release
* One-report-per-invocation HTML, publication-light PDF or machine-readable JSON export
* JSON persistence
* Interactive HTML with a critical scientific evidence ledger and publication export
* Evidence-led standalone HTML reports for every signature Reverse feature
* Analytic programme-boundary counterfactual candidates and explicit verification accounting
* Publication-oriented adversarial evidence profiles with corrected CE/DLR objectives,
  stratified cohort manifests, per-class and macro estimates, Wilson intervals, attack-union
  attribution, matched per-sample epsilon/seed unions, multi-restart best-path retention,
  paired-bootstrap AUC intervals, DKW bands and frozen-coordinate mechanism comparisons
* Native formal L∞ robustness verification for supported PyTorch FX graphs using
  interval bound propagation, strict logit-margin proofs, certified-radius bisection,
  cohort certified accuracy and an explicit operator/proof ledger
* Formal internal representation and causal intervention-effect survival certificates
  with frozen coordinate bases, separately reported geometry/sign/ranking radii,
  executable do-interventions, minimum clean-effect eligibility and explicit
  inconclusive handling for incomplete coordinate bases, plus a conjunctive
  decision-mechanism proof ledger, shared-input relational effect bounds and exact
  cohort decision-versus-mechanism contingency accounting, plus per-sample
  zero-radius representation/causal applicability and an applicability-conditioned
  exact four-state survival table with explicit denominators and reasons
* Machine-readable input semantics, independent calibration-split coordinate selection
  and a non-composite publication-readiness ledger for formal mechanism studies
* Parameterised `vraven explain`, adversarial sample/batch/curve auditing,
  counterfactual search, perturbation stability, LRP, formal certification,
  dataset atlases, model comparison, checkpoint evolution, internal anatomy,
  capture, Reverse, demonstration, inspection and rendering commands
* Machine-readable `vraven features` inventory and path-aware `vraven doctor` diagnostics
* Dependency-free TTY result panels and optional still, blinking or dancing
  terminal raven, with JSON-preserving non-interactive output
* Beginner-facing `analyse` workflow with task presets, automatic output names,
  short flags and `analyze` / `quick` aliases
* Validated self-describing model folders with contained paths, safe
  state-dictionary reconstruction metadata and command-line overrides

## Partially supported

* CNNs: generic channel and layer summaries, but no complete spatial concept pipeline
* Transformers: generic modules and returned attention matrices, but no generation or residual-stream adapter
* TorchScript artefacts can be loaded through the Python loader, but ScriptModules cannot be
  instrumented by the forward hooks required for CLI explanation; use a state dictionary and model factory
* Autoencoders, GANs and GNNs: generic tracing may work, but no architecture-specific validated output semantics
* CUDA and MPS: code paths follow PyTorch, but hardware coverage is not exhaustive

## Not yet implemented

* TensorFlow or Keras adapter
* ONNX adapter
* Hugging Face generation adapter
* Segmentation and detection readout interfaces
* Rust Python extension
* Custom CUDA or ROCm kernels
* VRAVEN Studio
* Calibrated explanation-confidence score
* Human concept labelling
* Validated universal causal reconstruction

## Release archive rule

A release archive must contain the complete source tree, tests, examples, documentation, Rust reference crate, licence and build configuration. It must not contain caches, local HTML reports, temporary traces, `build/`, `dist/`, `.pytest_cache/`, `*.egg-info` or `__pycache__` directories.
