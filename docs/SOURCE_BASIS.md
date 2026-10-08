# Source basis

The diagrams were derived from the supplied VRAVEN framework concept and the current project documents describing:

* model adapters and safe loading
* scoped PyTorch instrumentation
* on-device trace reduction
* fast, gradient and causal modes
* the `NeuralTrace` schema
* reasoning pathways, fingerprints and communities
* scientific visualisation rules
* training and post-training analysis
* baseline evaluation against SHAP, LIME and related methods
* reproducibility and security requirements

The design preserves the project's central distinction:

```text
SHAP or LIME: which input features matter?
VRAVEN: how do internal neural signals evolve into a prediction?
```

The diagrams do not add empirical claims that are absent from the project. Planned framework adapters and optional acceleration paths are shown with dashed boundaries.

Standard baseline method names and validation concepts are supported by the canonical bibliography in [`docs/references/VRAVEN_REFERENCES.bib`](references/VRAVEN_REFERENCES.bib).
