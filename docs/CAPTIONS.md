# Publication-ready captions

The captions below are deliberately separate from the figures because the user requested no heading inside the images.

## 01 System architecture

**End-to-end VRAVEN architecture.** Live PyTorch modules, trusted state dictionaries with a model factory and TorchScript artefacts enter through a model adapter. Scoped instrumentation observes selected modules, while large tensors are reduced on the model device into bounded activation, gradient, attention, timing and representation summaries. These summaries form an ordered `NeuralTrace`, from which pathways, reasoning fingerprints, latent communities and uncertainty analyses are derived. Dashed adapters indicate planned framework integrations rather than current native support.

## 02 Runtime dataflow

**VRAVEN runtime and memory dataflow.** Fast mode terminates after forward capture and on-device reduction; gradient mode adds target-score differentiation; causal mode adds bounded and reversible interventions. Full activations remain transient on the model device, whereas compact layer records are retained in the serialisable trace. The same trace can support multiple standalone scientific views without rerunning unrelated visual computations.

## 03 Package architecture

**Modular software architecture of VRAVEN.** A small Python API delegates to framework adapters, capture components, core trace and device services, numerical metrics, reasoning analyses and report or visualisation modules. Optional Rust and GPU kernels are treated as replaceable acceleration paths and must preserve numerical parity with the reference implementation.

## 04 Device execution

**Device-following execution strategy.** VRAVEN does not silently relocate the user model. Activation summaries and target-specific reductions are computed on the existing CPU, CUDA or Apple MPS device before bounded summaries are transferred to host memory. The ROCm lane is shown as a framework-dependent path requiring explicit validation. Full-tensor host copies remain opt-in and scoped.

## 05 Explanation modes

**Operational distinction between VRAVEN explanation modes.** Fast mode reports observational forward signals from one inference pass. Gradient mode adds one target-specific backward pass and activation-gradient interactions. Causal mode begins from the gradient trace and performs a bounded set of temporary interventions, producing intervention evidence rather than merely observational attribution.

## 06 Evidence reconstruction

**Layerwise internal evidence reconstruction.** VRAVEN retains activation energy, signed activation-gradient interaction and predictive uncertainty as separate channels. Robustly normalised layer summaries are used to identify supporting, opposing and mixed pathways. Edge width represents a defined contribution magnitude and must not be chosen decoratively.

## 07 Reasoning DNA

**Construction of the VRAVEN reasoning fingerprint.** An ordered layer-by-metric matrix is robustly scaled, clipped, flattened and deterministically resampled to a fixed dimension. The resulting fingerprint supports trace similarity, outlier detection, reasoning drift and comparison across models or checkpoints. Compression preserves a reproducible summary but does not retain every architectural detail.

## 08 Latent community discovery

**Latent activation community methodology.** Unit profiles across samples are standardised and compared using a declared similarity measure. A thresholded or weighted graph is partitioned into latent communities. These groups become human-interpretable concepts only after external examples, probes or controlled interventions provide supporting evidence.

## 09 Training monitoring

**Training-time VRAVEN instrumentation.** Scoped hooks capture selected internal summaries at defined intervals without replacing the training loop. Checkpoint fingerprints support longitudinal analysis of gradient flow, activation health, class geometry, reasoning stability, concept formation and drift across epochs. Any stored checkpoint can later be analysed through the same post-training trace path.

## 10 Architecture adapters

**Architecture-aware mapping to a common trace schema.** MLPs, convolutional models, transformers and autoencoders expose different native internal structures. Specialised adapters map these structures to common concepts such as ordered modules, activation summaries, target interactions, representation profiles, uncertainty sources and intervention points. Dashed models require additional specialised adapters.

## 11 Benchmark protocol

**Controlled evaluation protocol for VRAVEN and baseline explainers.** Each method receives the same model, ordered inputs, prediction target, hardware and precision. Method budgets are disclosed and matched where meaningful. Evaluation spans runtime, model calls, memory, fidelity, robustness, architecture coverage and task-oriented interpretation, with distributions and paired statistical analysis reported rather than single-run headline values.

## 12 Causal validation

**Bounded causal validation loop.** A pathway ranked by observational analysis is subjected to a declared intervention, followed by re-inference on the same input and model state. The signed target-score difference is compared with random or magnitude-matched controls and aggregated across samples, seeds or layers. The result supports the importance of the defined intervention but does not recover the model's complete causal mechanism.

## 13 Reliability calibration

**Multidimensional explanation reliability profile.** Prediction confidence, explanation stability, cross-method agreement, causal support and distribution support remain separately visible. A composite may be reported only when a calibration model has been trained and validated against a clearly defined explanation-success outcome.

## 14 Reasoning drift

**Reasoning drift analysis.** Independent traces from a reference and current window are represented in a shared fingerprint space. Distributional change is quantified with a declared distance or kernel statistic and tested using an appropriate null procedure. Class-conditional and layer-local analyses identify whether internal reasoning changes before overall predictive performance changes.

## 15 Reproducibility and provenance

**VRAVEN reproducibility bundle.** Model, input, environment, configuration and randomness metadata are attached to the ordered trace schema. The trace, figure metadata, release manifest, environment specification and regeneration script form a provenance package sufficient to reconstruct the numerical transformation and rendering path.

## 16 Visual contract

**Scientific visual contract.** Validated arrays are mapped through declared positional, signed, magnitude, colour, opacity and annotation encodings. Each figure answers one scientific question. Interactive dark and publication light modes preserve the same data, ordering, geometry and semantic meaning.

## 17 User workflow

**Local VRAVEN workflow.** Users may analyse an in-memory PyTorch model directly or reconstruct a trusted state dictionary through an explicit model factory, preprocessing path and output readout. Both routes produce the same report contract and standalone figure methods. Unsafe pickled model objects remain blocked by default.
