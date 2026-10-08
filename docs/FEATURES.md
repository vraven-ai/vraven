# VRAVEN Feature Specification and Implementation Status

## Core purpose

VRAVEN captures and visualises measured internal decision signals in deep neural networks. It is not intended to claim that a neural network reasons like a person. Terms such as `reasoning fingerprint` are names for deterministic reduced trace representations.

## Implemented in the current PyTorch alpha

| Area | Implemented capability |
|---|---|
| Model integration | Live `torch.nn.Module` adapter |
| Devices | CPU, CUDA and MPS through the model's current PyTorch device |
| Modes | Fast, target-specific gradient and bounded causal intervention |
| Capture | Activations, outputs, compressed representations, unit matrices, timing and accessible attention |
| Input signal | Input-times-gradient for floating differentiable inputs |
| Metrics | Mean, variance, energy, mean energy, L1, sparsity, sign fractions and activation-dispersion entropy |
| Gradient metrics | Gradient norm, mean absolute gradient and signed or absolute activation-gradient interaction |
| Prediction metrics | Class, target score, confidence, normalised predictive entropy and margin where valid |
| Reasoning summaries | Layer influence, cumulative evidence, pathway concentration, transition estimate, fingerprint and execution graph |
| Communities | Deterministic activation-pattern clustering within one layer |
| Causal analysis | Temporary zero, mean or scale layer output interventions |
| Counterfactuals | Gradient-optimised continuous-input counterfactuals with summary UI |
| LRP | Epsilon-rule Layer-wise Relevance Propagation for Linear, ReLU and Embedding |
| Perturbation probes | Noise, feature-mask and shuffle stability curves |
| Projections | Shared PCA plus optional sklearn t-SNE (per-layer with Procrustes, dataset landscape) |
| Persistence | JSON neural traces and dataset reports |
| Dataset analysis | Independent sample traces, pairwise similarity, outlier score and PCA/t-SNE landscape |
| Comparison | Matched two-model trace comparison |
| Evolution | Ordered checkpoint comparison for one fixed input |
| Reporting | Standalone Plotly figures and optional local HTML composition |
| Evidence audit | Effective evidence depth, entropy, participation, Gini, cancellation, causal coverage and zero-ablation concordance |
| Export | HTML and JSON natively; SVG, PDF and raster through Kaleido |
| Rust | Optional reference crate for profiled CPU reductions. Not yet bound into the Python wheel |
| DarkSide extension | Isolated `vraven.darkside` namespace with 21 advanced internal audits (circuitry, wiring, Linear anatomy, unit fields, shortcuts, drift, disagreement, capacity, evasion, topology, …) |

## DarkSide extension (`vraven.darkside`)

Merged from `VRAVEN_DarkSide_Extensions` v1.0.0. Import:

```python
from vraven.darkside import DarkSideSession, TraceDataset, recommend_features
```

Docs: `docs/darkside/` (see especially `INTERNAL_ANATOMY.md`). Examples: `examples/darkside_*.py`. Tests: `tests/test_darkside_*.py`.  
Optional install: `pip install -e ".[analysis]"` or `".[darkside]"` (adds SciPy / sklearn; `darkside` also pulls ripser for H1).

Newer measured-internals APIs (beyond bars/scatters):

- `inspect_internal_anatomy` / `print_internal_anatomy` — live Linear `W·a` text + figure
- `build_circuit_wiring` — node/edge graph with edge mass `Σ|W_ji|·|a_i|` on an ε-core
- `map_class_conditional_unit_field` — dense layer×unit Hedges-*g* heatmap

## Implemented scientific visualisations

1. Activation Energy
2. Decision Flow
3. Contribution Flow (Sankey)
4. Spectral Trace
5. Neuron Impact Field
6. Neuron Importance (top units)
7. Attention Structure
8. Latent Flight / Representation Landscape (PCA or t-SNE)
9. Reasoning DNA
10. Conflict Field
11. Evidence Trajectory
12. Causal Pathway
13. Input Attribution / Feature Importance
14. Feature Attribution Heatmap
15. Reasoning Graph
16. Activation Community Graph
17. Consistency Matrix
18. Counterfactual Explanation
19. Perturbation Stability
20. Relevance Flow (LRP)
21. Reasoning Similarity Matrix
22. Reasoning Landscape (PCA or t-SNE)
23. Matched Model Comparison
24. Model Evolution
25. Dataset Reasoning Outliers
26. Mechanism Concordance

Every implemented figure follows the standalone contract and supports `interactive_dark` and `publication_light`. Composed HTML reports include a Dark / Light toggle by default (`theme_toggle=True`), embedding both themes. t-SNE requires `pip install 'vraven[sklearn]'`.

HTML reports also lead with a Scientific Evidence Audit: critical measured findings, an explicit
evidence-tier table, a complete layer evidence ledger, mathematical assumptions and
machine-readable audit values. The audit intentionally avoids a composite trust score.

## Trace statistics

For activation tensor `A_l` with `n` values, VRAVEN currently derives:

Activation energy:

`E_l = sum_i A_li^2`

Mean activation energy:

`Ebar_l = (1/n) sum_i A_li^2`

Mean:

`mu_l = (1/n) sum_i A_li`

Variance:

`sigma_l^2 = (1/n) sum_i (A_li - mu_l)^2`

Sparsity at threshold `epsilon`:

`S_l = count(|A_li| < epsilon) / n`

Target-specific first-order interaction for target score `s`:

`I_l = sum_i A_li * (partial s / partial A_li)`

Absolute local interaction:

`Iabs_l = sum_i |A_li * (partial s / partial A_li)|`

The target score is averaged across the batch before differentiation. These values are local first-order approximations and are not causal effects.

## Reasoning fingerprint

VRAVEN builds a layer-by-metric matrix from reduced trace values. Each metric column is robustly scaled using the median and interquartile range, with a standard-deviation fallback. The matrix is clipped to a documented range, flattened and deterministically resampled to the requested fixed width.

Intended uses:

* compare repeated explanations
* compare models
* compare checkpoints
* cluster independent sample traces
* identify unusual trace patterns
* study drift in later research

Limitations:

* Layer ordering and metric selection affect the fingerprint.
* Fixed-width resampling compresses architecture differences.
* Similarity does not prove semantic or causal equivalence.

## Predictive and conflict signals

VRAVEN distinguishes:

* predictive entropy derived from a valid class probability distribution
* classification margin uncertainty
* target-specific internal opposition ratio
* pathway concentration
* causal sensitivity when interventions were performed

These signals remain separate. The current package does not expose an invented combined explanation-confidence percentage.

## Dataset analysis

Dataset analysis runs independent explanations per sample. This avoids mixing unrelated samples in one aggregate fingerprint. It also increases runtime linearly with the number of samples, so `max_samples` is bounded by default.

Outputs include:

* fixed-width sample fingerprint matrix
* pairwise cosine similarity
* nearest-neighbour outlier scores
* shared deterministic PCA projection
* prediction and confidence summaries
* persisted per-sample traces

## Architecture-aware behaviour

### MLP and dense models

Fully supported by the native PyTorch adapter for the current output types.

### CNN and ResNet models

Convolution channels are treated as units and spatial dimensions are reduced. Current figures are channel and layer focused. Dedicated spatial region views remain future work.

### Basic Transformers

Linear, normalisation, embedding and attention modules are captured. Attention matrices are available only when the model returns them. Residual-stream and token-logit-lens specialisations remain future work.

### Autoencoders, GANs and GNNs

Generic PyTorch tracing can operate when outputs match supported shapes, but architecture-specific scientific views are not yet implemented.

### LLMs

Generic module capture may be technically possible for small models, but memory-aware residual, token and generation-specific support is not yet implemented or claimed.

## Later research and engineering phases

* Segmentation and structured-output readout adapters
* Spatial image attribution beyond sampled input-times-gradient
* Hugging Face generation and residual-stream adapter
* TensorFlow and ONNX adapters
* GNN message-flow specialisation
* GAN generator and discriminator views
* Autoencoder reconstruction-path analysis
* Drift distributions and calibrated alerting
* Validation against SHAP, LIME, Integrated Gradients, DeepLIFT, Grad-CAM and Captum baselines
* Rust Python bindings after profiling
* Custom CUDA or ROCm kernels after profiling
* VRAVEN Studio built from the same standalone figures
