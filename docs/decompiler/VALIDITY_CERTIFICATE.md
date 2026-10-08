# Mechanism Certificate and Validity Envelope

Every decompilation produces a certificate. Its purpose is to stop a visually persuasive explanation from exceeding the available evidence.

## Validity envelope

The envelope records:

* Model checkpoint class and framework version
* Capture route
* Observed input shapes
* Symbolic shape constraints
* Covered execution paths
* Unresolved operations
* Supported interventions
* Numerical-equivalence evidence
* Model evaluation state
* Restrictions

## Coverage dimensions

The certificate reports operator, parameter, gradient and intervention coverage separately. Semantic coverage is split into:

1. Validated semantic grounding
2. Association-only labels
3. Causally important but unresolved concepts

## Claim lattice

Claims are limited to evidence tiers:

```text
observed
associated
intervention-supported
replicated
domain-validated
```

Association-derived labels are deliberately written as `Associated with …` and remain excluded from validated semantic coverage.

## Dark matter

VRAVEN exposes:

* Unrecognised operators
* Intervention-inaccessible computation
* Causally important but semantically unresolved concepts
* Weak interventional fidelity
* Counterfactuals without domain validation

The system is permitted to conclude that a model is not sufficiently explained.
