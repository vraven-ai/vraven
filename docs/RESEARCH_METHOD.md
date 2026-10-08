# VRAVEN Research Method and Validation

## Research claim

VRAVEN should be positioned as a framework for reconstructing and visualising information flow, evidence accumulation and internal decision formation in deep neural networks.

It must not claim to recover human style reasoning or true causal mechanisms unless the experimental method supports those claims.

## Baselines

Evaluate against relevant methods such as:

- SHAP
- LIME
- Integrated Gradients
- DeepLIFT
- Grad CAM for suitable vision models
- SmoothGrad
- Captum implementations
- architecture specific baselines where appropriate

## Required evaluation dimensions

- runtime
- additional memory
- fidelity
- stability
- sensitivity
- deletion tests
- insertion tests
- causal ablation effect
- explanation consistency
- architecture coverage
- reproducibility
- human interpretability studies where justified

## Fidelity

If pathway `P` is ranked important, a controlled intervention can test:

`F = P(y | x) - P(y | x, not P)`

The intervention must be precisely defined. Large changes support importance but may still be confounded by network dependence.

## Observational–causal concordance

For zero-output layer interventions, VRAVEN compares the measured target-score drop with the
local activation–gradient approximation after separate signed L1 normalisation. Spearman rank
correlation, sign agreement, cosine similarity, normalised L1 discrepancy and permutation
significance remain separate. See `EVIDENCE_AUDIT.md` for the complete definition, assumptions
and validation plan.

## Deletion and insertion

Deletion: progressively remove ranked components and measure target output degradation.

Insertion: progressively restore ranked components and measure target output recovery.

## Explanation stability

For controlled perturbation `x' = x + epsilon`, compare reasoning fingerprints only when the perturbation is semantically or task appropriate.

Candidate form:

`S_E = similarity(R(x), R(x'))`

## Reasoning drift

Compare distributions of reasoning fingerprints between time periods or domains. Jensen Shannon divergence is one candidate measure where distribution estimation is appropriate.

## Reproducibility metadata

Every report should capture:

- model hash
- model class
- VRAVEN version
- framework version
- CUDA or MPS availability
- device
- input hash when possible
- random seed
- capture configuration
- explanation mode
- timestamp

## Reverse feature-level diagnostics

Signature Reverse figures expose independent diagnostics for capture coverage, retained
optimal-transport mass, lineage ambiguity, bootstrap interval exclusion, causal sign stability,
programme cancellation, contrastive evidence competition, counterfactual search accounting and
finite-sample Wilson intervals. Definitions and limitations are specified in
`REVERSE_FEATURE_AUDITS.md`.

## Performance research targets

Fast mode early target:

`T_vraven <= 1.5 * T_native`

Long term selected architecture target:

`T_vraven <= 1.2 * T_native`

Reduced trace memory target:

`M_overhead < 20 percent`

These values are hypotheses and engineering targets, not release guarantees.
