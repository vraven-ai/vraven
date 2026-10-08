# VRAVEN Architecture

## Objective

VRAVEN observes a live deep learning model with minimal interference, reduces intermediate tensors close to the model device, constructs a unified neural trace, derives documented analysis values and renders standalone scientific figures.

## Current executable architecture

```text
User model and input
        |
        v
PyTorch adapter
        |
        +--> device and nested-input preparation
        +--> scoped module selection
        +--> forward hooks
        +--> optional target gradient
        +--> optional temporary interventions
        |
        v
On-device trace reduction
        |
        +--> activation statistics
        +--> fixed-width representations
        +--> unit profiles
        +--> accessible attention matrices
        +--> input-times-gradient
        |
        v
Unified NeuralTrace
        |
        +--> metrics
        +--> pathways
        +--> fingerprints
        +--> communities
        +--> execution graphs
        |
        v
VRavenReport
        |
        +--> one standalone figure per question
        +--> interactive_dark
        +--> publication_light
        +--> JSON trace
        +--> optional HTML composition
```

Dataset and comparison paths reuse the same single-sample trace:

```text
Bounded dataset
        |
        v
Materialise identical ordered samples
        |
        +--> one independent trace per sample
        |
        v
Fingerprint matrix
        |
        +--> similarity and outlier analysis
        +--> shared PCA landscape
        +--> matched model comparison
        +--> checkpoint evolution
```

## Package boundaries

### `adapters`

Framework-specific execution and output handling. The current native adapter is PyTorch.

### `analysis`

Dataset materialisation, independent sample analysis, model comparison and checkpoint evolution.

### `capture`

Scoped hook management, module selection, tensor extraction, on-device reduction and attention discovery.

### `causal`

Temporary internal interventions, intervention validation and continuous-input counterfactual optimisation.

### `core`

Configuration, device handling, prediction interpretation, reproducibility metadata, orchestration, errors and the trace schema.

### `metrics`

Numerical calculations with explicit semantics. Rendering code must not live here.

### `reasoning`

Deterministic fingerprints, layer pathways, activation communities, shared representation projections and execution-summary graphs.

### `report`

High-level report APIs and explicit local HTML composition.

### `visualisation`

Mapping validated arrays to Plotly figures. This layer must not invent scientific values.

## Execution modes

### Fast

One inference pass under `torch.inference_mode()`. Captures reduced forward signals.

### Gradient

One inference pass with gradients enabled, followed by one `torch.autograd.grad` call over selected module outputs and floating inputs. Model parameter `.grad` fields are not populated.

The selected target score is averaged across the batch before differentiation.

### Causal

Runs the gradient trace, ranks candidate layers and performs a bounded number of temporary output interventions. Hooks are removed after each run and model training state is restored.

## Trace schema

`NeuralTrace` contains:

* ordered `LayerTrace` records
* `PredictionTrace`
* `InputTrace`
* target and mode
* runtime and reproducibility metadata
* optional `CausalInterventionTrace` records
* trace identifier, timestamp and schema version

Full intermediate tensors are not serialised.

## Model output support

The current prediction adapter supports:

* scalar regression output
* `[batch]` binary logits
* `[batch, 1]` binary logits
* `[batch, classes]` multiclass logits

Mappings, tuples and objects exposing `.logits` are traversed to locate the primary tensor. Segmentation, detection and structured outputs require future explicit readout adapters.

## Module capture

`ModuleSelector` supports leaf, semantic and all-module policies, class filters, name filters and maximum layer limits. Dropout and identity modules are excluded by default.

Each selected module is observed through a pre-hook for host timing and a post-hook for reduced trace capture. Handles are removed in `HookSession.__exit__`, including error paths.

## Device strategy

VRAVEN never silently moves the model. It infers the current model device from parameters or buffers. Nested input tensors may be moved to that device when configured.

Large tensors are reduced on their current device. Only compact scalars, vectors, matrices or sampled values are transferred to CPU for persistence and rendering.

## Reproducibility metadata

The adapter records:

* model class and module
* parameter counts
* model structure hash
* device and backend
* Python and PyTorch versions
* CUDA and MPS availability
* capture configuration
* execution time
* target score reduction
* bounded input fingerprint
* user metadata when supplied

The bounded input fingerprint uses tensor shape, dtype, device type and deterministic value samples. It is not a full secure hash of every byte.

## Reasoning fingerprint architecture

The fingerprint is derived from an ordered layer-by-metric matrix. Columns are robustly scaled using median and interquartile range, with a standard-deviation fallback. The matrix is clipped, flattened and deterministically resampled to a fixed width.

This enables comparison across traces but compresses architectural detail. It should be treated as an engineered internal trace representation.

## Visual architecture

```text
Validated arrays
        |
        v
VisualContract and VisualisationMetadata
        |
        v
Standalone Plotly renderer
        |
        +-----------------------------+
        |                             |
interactive_dark              publication_light
        |                             |
HTML interaction              white journal figure
```

`theme.py` controls presentation. `base.py` attaches scientific metadata and marks the figure as standalone. `export.py` provides publication dimensions and file output.

## Attribution and projection extensions

* `attribution/` provides epsilon-rule LRP for Linear, ReLU and Embedding modules.
* `reasoning/projection.py` provides shared PCA and optional sklearn t-SNE (extra `vraven[sklearn]`).
* `causal/perturbation.py` probes score stability under noise, masking and shuffle.

## Optional Rust core

`rust_core` contains tested reference implementations for activation summary, cosine similarity and deterministic resampling. It is not yet linked into the Python wheel. It may replace profiled CPU paths only after numerical parity and benchmark tests are added.

## Future adapters and kernels

TensorFlow, ONNX, structured outputs, Rust bindings and custom GPU kernels remain future work. They must not be represented as implemented merely because documentation describes them.
