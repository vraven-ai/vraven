# Scientific Evidence Audit

## Purpose

The Scientific Evidence Audit turns a single VRAVEN trace into a transparent inferential
summary. It answers two bounded questions:

1. How concentrated is target-specific first-order evidence across captured model depth?
2. Where zero-output interventions exist, do they corroborate the direction and relative
   allocation suggested by the first-order layer evidence?

It does not produce a trust score, explanation-confidence percentage or claim of a unique
mechanism. Prediction, observational evidence and intervention evidence remain separate.

## Input domain

The concentration audit requires layer records containing
`activation_gradient_signed` and `activation_gradient_influence`. These are available in
`gradient` and `causal` modes. The concordance audit additionally requires
`CausalInterventionTrace` records with `strategy="zero"` and an unambiguous mapping from the
intervened module to one captured execution record.

All inputs are finite real layer summaries. Non-finite stored values are treated as numerical
zero so that a malformed value cannot silently dominate a report. Empty evidence produces
`None` estimates rather than invented zeros.

## Evidence distribution

For captured layer (l), absolute first-order evidence mass is

\[
q_l=\lVert a_l\odot\nabla_{a_l}s_t\rVert_1,
\qquad
p_l=\frac{q_l}{\sum_j q_j}.
\]

Here (s_t) is the selected target score. The vector (p) is non-negative and sums to one
when total evidence is non-zero. VRAVEN reports the following complementary quantities.

\[
H(p)=-\sum_{l:p_l>0}p_l\log p_l,
\qquad
H_{norm}=\frac{H(p)}{\log L}\in[0,1],
\qquad
N_{eff}=\exp(H(p))\in[1,L].
\]

Participation ratio and Herfindahl concentration are

\[
N_{PR}=\frac{1}{\sum_l p_l^2}\in[1,L],
\qquad
C_H=\sum_l p_l^2\in[1/L,1].
\]

VRAVEN also reports the finite-sample-normalised Gini coefficient in ([0,1]), plus the
earliest depth indices where cumulative evidence mass crosses 0.5 and 0.8.

Signed layer interaction and cancellation are

\[
u_l=\sum_i a_{li}\frac{\partial s_t}{\partial a_{li}},
\qquad
C_{cancel}=1-\frac{|\sum_l u_l|}{\sum_l|u_l|}\in[0,1].
\]

Cancellation measures opposing first-order signals. It is not predictive, epistemic or
aleatoric uncertainty.

## Observational–causal concordance

For a zero-output intervention at layer (l), the measured target-score drop is

\[
d_l=s_t(x)-s_t(do(a_l=0)).
\]

The local first-order approximation to that drop is (u_l). Because their absolute scales need
not agree, VRAVEN compares separately L1-normalised signed vectors:

\[
\tilde u=\frac{u}{\lVert u\rVert_1},
\qquad
\tilde d=\frac{d}{\lVert d\rVert_1}.
\]

Reported dimensions are Spearman rank correlation in ([-1,1]), active-element sign agreement
in ([0,1]), cosine similarity in ([-1,1]), and signed normalised L1 discrepancy

\[
D_{L1}=\frac12\lVert\tilde u-\tilde d\rVert_1\in[0,1].
\]

A two-sided permutation p-value is reported for the absolute Spearman statistic. Small matched
sets use exact enumeration when their full permutation space fits the configured budget. Larger
sets use deterministic Monte Carlo permutations with a recorded seed and add-one correction.
Fewer than three non-degenerate matches do not receive a p-value.

## Matching and assumptions

The calculation is restricted to zero-output interventions. Mean replacement and arbitrary
scale interventions answer different counterfactual questions and cannot be compared directly
with (a_l\nabla_{a_l}s_t).

One intervened module must map to exactly one captured execution. Reused modules with multiple
execution records are excluded instead of selecting an arbitrary occurrence. The permutation
test assumes matched layer effects are exchangeable under the null. Neural layers are ordered
and dependent, so the p-value is a descriptive calibration diagnostic, not definitive causal
discovery evidence.

Causal coverage is the number of uniquely matched zero-ablation effects divided by the number of
gradient-bearing layer executions. The raw zero-ablation test count is retained separately.

## Interpretation

Low effective layer count means the recorded absolute first-order mass is concentrated. It does
not prove that the remaining layers are removable. Positive concordance means tested zero-output
effects tend to agree with the local observational approximation. It does not prove that zero
output is in-distribution, that higher-order interactions are absent, or that the mechanism is
unique.

The HTML evidence tier states the strongest evidence type present:

* `descriptive_activation` — forward signal summaries only;
* `observational_first_order` — target-specific gradients, without intervention;
* `targeted_interventional` — at least one declared intervention is present.

These are evidence categories, not quality grades.

## Known limitations

* Current layer statistics are reduced over the supplied batch and do not provide per-sample
  confidence intervals.
* Causal mode tests a bounded top-k candidate set, not every layer or layer combination.
* Zero ablation can generate out-of-distribution internal states.
* First-order interaction omits curvature and cross-layer interaction terms.
* High correlation can coexist with poor absolute agreement, so discrepancy and sign agreement
  remain visible.
* Multiple-testing correction is not supplied across independently generated reports.

## Validation plan

Implemented numerical tests cover uniform evidence mass, closed-form entropy, effective count,
participation ratio, Gini, mass crossings, scale invariance, exact small-sample permutation
significance, dual-mode visual identity and HTML evidence-ledger output.

The next empirical validation stage should use planted-mechanism networks and report recovery of
known layer ordering, deletion/insertion performance, sensitivity to nonlinear saturation,
stability across seeds and calibration against randomly permuted or structurally null models.

The concentration calculation is linear in the number of captured layers. Concordance matching
is linear before the permutation diagnostic; that diagnostic is bounded by `max_permutations`
and runs only while composing the audit or concordance figure. It adds no model forward passes,
device transfers or capture-time memory.
