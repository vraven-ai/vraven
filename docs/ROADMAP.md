# VRAVEN Roadmap

## Completed executable foundation: 0.2 alpha

The repository currently contains working code for:

* PyTorch live-model adapter
* CPU, CUDA and MPS device following
* Scoped capture hooks and automatic cleanup
* Activation, representation, unit, attention and timing reduction
* Fast, gradient and limited causal modes
* Prediction, influence, uncertainty, stability and agreement metrics
* Deterministic reasoning fingerprints and shared PCA projections
* Temporary layer intervention validation
* Continuous-input counterfactual generation
* Twenty standalone scientific visualisations
* Dark interactive and white publication modes
* Trace and dataset persistence
* Dataset-level sample comparison
* Matched model comparison
* Checkpoint evolution
* CLI, examples, benchmark harness, tests and wheel build

## Next research phase: validation and benchmarking

1. Establish fixed benchmark models and datasets.
2. Compare runtime and memory against native inference.
3. Compare relevant signals against Integrated Gradients, DeepLIFT, SmoothGrad, SHAP, LIME and Grad-CAM where applicable.
4. Run deletion and insertion studies.
5. Quantify intervention fidelity and confounding.
6. Measure explanation stability under controlled task-valid perturbations.
7. Validate visual comprehension with qualified users where justified.
8. Publish the trace schema and metric versioning policy.

## Transformer specialisation

* Hugging Face adapter for classification and generation outputs
* Explicit attention-output capture configuration
* Residual-stream summaries
* Token-level evidence and uncertainty
* Head and MLP community analysis
* Long-context memory controls
* Generation-step traces

## Architecture-specific research

* CNN spatial regions and concept validation
* Autoencoder reconstruction-error pathways
* GAN latent generator and discriminator pathways
* GNN message and neighbourhood flow
* Temporal and recurrent hidden-state trajectories
* Multimodal cross-stream alignment

## Framework expansion

* TensorFlow and Keras adapter
* ONNX graph instrumentation
* Structured output readout interface
* Plugin SDK for validated metrics and figures

## Performance phase

Only after profiling identifies a bottleneck:

* Bind the Rust reference core through PyO3 and maturin
* Add Arrow or another compact batch trace format
* Evaluate fused CUDA reductions
* Evaluate ROCm paths
* Add controlled asynchronous report generation

## Product phase

* Public PyPI release
* Stable versioned trace schema
* Documentation site
* VRAVEN Studio as an optional local application
* Report redaction controls
* Signed release artefacts and supply-chain checks

## Scientific release gate

VRAVEN 1.0 should not be declared until:

* Core metrics have published definitions and validation results.
* Supported architecture claims are backed by tests and benchmarks.
* Observational and causal outputs are consistently separated.
* Runtime and memory overhead are reported transparently.
* Public APIs and trace formats are stable.
* Publication figures are checked at realistic journal dimensions.
