# Reverse Feature Audits

Every signature Reverse feature now carries an independent critical audit in its standalone
HTML report and Plotly metadata. The audits are not combined into a trust score.

## Result-first scientific communication

Every standalone feature report presents information in the following order:

1. the primary scientific result and its evidence status;
2. the scientific interpretation of that result;
3. the measurements that establish it;
4. the full visual analysis;
5. collapsible scope limitations and metric definitions.

This ordering prevents valid findings from being visually subordinated to their caveats. It does
not remove, soften or hide limitations: it separates the established result from the boundary of
what has not yet been established.

## Model X-ray

The unresolved-node fraction is

\[
U_{node}=\frac{|\{v:\text{operator semantics unresolved}\}|}{|V|}.
\]

Operator, gradient, parameter and intervention coverage retain their certificate denominators.
They describe the captured execution and do not guarantee unseen control-flow coverage.

## Concept genome

Each adjacent-layer optimal-transport plan has unit mass. The displayed retained-mass fraction is

\[
R_{OT}=\frac{\sum_{e\in E_{shown}}m_e}{L-1},
\]

where `m_e` is transport mass and `L-1` is the number of aligned layer pairs. Lineage ambiguity
uses the effective outgoing branch count `exp(H(p))`, with outgoing masses normalised to `p`.
These are aligned observational correspondences, not semantic identity or causality.

## Causal proof

The report separates node and edge bootstrap interval-exclusion rates:

\[
R_{CI}=\frac{|\{i:0\notin[\ell_i,u_i]\}|}{N_{tested}}.
\]

It also reports median sign stability, derivative–intervention sign agreement and the entropy-based
effective number of concepts carrying absolute output effect. Candidate preselection and internal
off-manifold interventions prevent interpreting the interval rate as family-wise error control.

## Programme proof

Training and held-out fidelity remain separate from intervention fidelity. Local signed
cancellation is

\[
C=1-\frac{|b+\sum_r c_r|}{|b|+\sum_r|c_r|}\in[0,1],
\]

where `b` is the intercept and `c_r` is the focus-sample contribution of rule `r`. High
observational `R²` does not compensate for weak intervention sign agreement.

## Contrastive proof

Supporting mass is `S=sum(max(c_r,0))` and opposing mass is
`O=sum(max(-c_r,0))`. Evidence cancellation is `1-|S-O|/(S+O)`. The decisive depth now excludes
the intercept-only state and is the earliest evidence-bearing depth after which the reconstructed
margin retains its final sign. This prevents a positive surrogate bias from being presented as
early internal evidence.

## Counterfactual proof

Candidate generation combines empirical score quantiles with an exact one-coordinate programme
boundary. With other concepts fixed, the current main-plus-pairwise programme is affine in concept
`z_j`; endpoint evaluations recover the slope and the root `z_j*` satisfying `m(z_j*)=0`. Roots are
retained only inside the empirical 1st–99th percentile interval.

The report separately counts programme evaluations, programme boundary crossings, graph
verification attempts, failed graph evaluations, successful graph executions and verification
budget exhaustion. A failed bounded search is never treated as proof of non-existence.

## Dark matter

Operator, intervention, semantic-responsibility and fidelity quantities retain different
denominators. The HTML explicitly prohibits adding or averaging them. Unresolved causal
responsibility is weighted by absolute tested output effect.

## Adversarial mechanism audit

The evaluator now reports overall attacked robust accuracy over every labelled
sample and clean-conditional survival as separate estimands. Deterministic stratified
sampling records exact sample IDs and class counts; per-class, macro and attack-union
results prevent an ordered one-class prefix from appearing representative.

Finite-sample robust accuracy now includes a nominal 95% Wilson score interval:

\[
\frac{\hat p+z^2/(2n)\pm z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n}.
\]

The interval quantifies sampling uncertainty only. It does not correct attack incompleteness,
threat-model mismatch or domain-invalid perturbations and is not a robustness certificate.

The robustness-profile report plots the cumulative-minimum attack-union curve over
epsilon and reports its interval-normalised trapezoidal AUC. Activation-probe
coordinates are frozen on the clean input before adversarial comparison. See
[`ADVERSARIAL_EVIDENCE.md`](ADVERSARIAL_EVIDENCE.md) for equations, protocol identity
and evidence boundaries.

## Formal local robustness

The native FX/IBP verifier proves strict target-logit dominance throughout a declared
L∞ input box. Its certified-radius lower bound and cohort certified-accuracy curve are
reported separately from empirical attack upper bounds. Layer-wise interval growth
localises relaxation bottlenecks, while unsupported operations remain explicit. See
[`FORMAL_ROBUSTNESS.md`](FORMAL_ROBUSTNESS.md).
