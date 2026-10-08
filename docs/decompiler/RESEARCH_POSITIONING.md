# Research Positioning

## What VRAVEN is not

VRAVEN is not a replacement for SHAP, LIME, Integrated Gradients or Grad-CAM. Those methods primarily address input attribution or task-specific localisation.

## What VRAVEN investigates

VRAVEN asks:

* What computation was recovered from the model?
* What candidate latent concepts exist?
* How do those concepts evolve through depth?
* Which concepts and edges survive direct interventions?
* Can the tested mechanism be compiled into an executable programme?
* Does the programme reproduce normal, intervened and counterfactual behaviour?
* Which parts remain unexplained?

## Adjacent research

The design is informed by, but does not duplicate:

* Neural circuit and neuron-group discovery
* Sparse feature decomposition
* Cross-layer feature alignment
* Attribution graphs and circuit tracing
* Causal abstraction
* Program extraction from restricted neural systems
* Known-circuit benchmarks

## Proposed scientific contribution

The central contribution is the integration of:

```text
compiler-assisted model capture
architecture-neutral Mechanism IR
graph-compiled interventions
intervention-constrained programme extraction
contrastive explanation
counterfactual verification
validity and dark-matter certificate
```

The contribution remains a hypothesis until evaluated across architectures, datasets, seeds and known-mechanism benchmarks.

## Defensible claim

> VRAVEN decompiles previously unseen PyTorch models across supported operators and observed execution paths, while producing an explicit certificate of captured, intervention-tested, semantically associated and unresolved computation.

## Claim that is not yet defensible

> VRAVEN can reverse every deep learning model or recover the unique true internal explanation.
