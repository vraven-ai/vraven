# VRAVEN Decompiler Visual System

## Principle

Each figure represents a scientific object created by the decompiler. It is not decoration and it is not a generic dashboard tile.

Every signature visual supports:

```text
interactive_dark
publication_light
```

The underlying coordinates, values, ordering and labels remain identical across both modes.

## Signature figures

### Raven Eye: Full-Stack Model X-Ray

Shows the recovered execution graph, operator roles, tensor volume, gradient paths, intervention points, dynamic guards and unresolved operations.

### Raven Flight: Concept Genome

Shows deterministic cross-layer concept birth, persistence, splitting, merging and termination based on the optimal-transport alignment.

### Raven Brain: Bidirectional Causal Proof Graph

Shows intervention-supported concept edges together with derivative screening, uncertainty and evidence tier.

### Raven Code: Executable Programme Proof

Shows the extracted programme as rules, their cumulative target-margin contribution and their separate observational/interventional evidence.

### Raven Contrast: Why This Rather Than That?

Shows supporting and opposing programme evidence for the selected class relative to its strongest alternative.

### Raven Shadow: Counterfactual Twin

Shows original and graph-verified counterfactual concept states, programme change and original-model response.

### Raven Dark Matter: Explanation Coverage Iris

Shows operator semantics, intervention access, validated semantic coverage, association-only coverage, unresolved causal responsibility and independent fidelity dimensions.

## Visual grammar

| Meaning | Encoding |
|---|---|
| Supports selected class | Cyan to blue |
| Opposes selected class | Magenta to crimson |
| Alternative route | Teal |
| Synergy or interaction | Violet |
| Uncertainty | Amber |
| Association-only semantics | Purple |
| Semantically unresolved | Hatched or amber |
| Unsupported operation | Grey or black |
| Intervention validated | Solid line |
| Derivative support only | Dotted line |
| Replicated evidence | Double border |

## Publication requirements

Publication mode uses a pure white background, no glow, deterministic layout and vector-compatible Plotly output. The figure title can be suppressed by the caller where a journal requires captions outside the artwork.
