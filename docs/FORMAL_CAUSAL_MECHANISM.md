# Formal causal-mechanism survival

VRAVEN separates three local properties that are often conflated: output-decision
invariance, internal-representation survival and survival of a declared causal
intervention-effect vector. The implementation is
`FormalCausalMechanismVerifier` in
`src/vraven/decompiler/formal_mechanism.py`.

## Property contract

Let (g_{t,j}(x)=z_t(x)-z_j(x)) be the fixed target-versus-competitor logit margin,
let (m(x)) be the coordinate-aligned output of a declared internal graph node, and
let

\[
\mathcal B_\infty(x,\epsilon)=
\{x':\lVert x'-x\rVert_\infty\leq\epsilon\}\cap[a,b]^d
\]

when an input domain ([a,b]) is supplied.

For coordinate (c) and baseline (b_c), the executable intervention-effect
coordinate is

\[
e_c(x') = g_{t,j}(x')-
g_{t,j}\!\left(x';\operatorname{do}(m_c=b_c)\right).
\]

For selected coordinates (C), (e_C(x')) is the causal-effect vector. Coordinate
selection and the intervention baseline are part of the property specification; they
cannot be changed after viewing a radius result.

A selected coordinate is locally eligible for a causal-effect survival certificate
only when its clean intervention effect is bounded away from zero. For the declared
threshold \(\delta>0\), VRAVEN requires

\[
\underline{|e_c(x)|} =
\begin{cases}
\ell_c, & \ell_c>0,\\
-u_c, & u_c<0,\\
0, & 0\in[\ell_c,u_c]
\end{cases}
\geq\delta
\qquad\text{for every }c\in C.
\]

This is a lower-bound test, not a midpoint or floating-point non-zero test. If fewer
than the requested number of auto-selected coordinates meet it, VRAVEN retains the
eligible audit trail but marks causal-effect and joint survival `inconclusive`; it
does not silently certify a smaller property. A pre-declared or calibration-frozen
coordinate that has no eligible local effect likewise blocks the local causal and
joint verdict while leaving decision and representation verdicts separate.

For a population study, coordinates must either be pre-declared or selected on a
calibration split disjoint from every evaluated sample. VRAVEN's calibration rule
screens by mean absolute activation on the calibration split, ranks the screened
coordinates by their mean certified absolute intervention-effect lower bound, freezes
the resulting basis, and then certifies the evaluation cohort without reselection.
Sample identifiers, the selection statistic and split-disjointness declaration are
stored in the certificate. Local effect eligibility is checked per evaluation sample;
failing it does not trigger evaluation-adaptive reselection.

## Cohort applicability and denominators

Applicability is a zero-radius precondition, not an observed failure of a survival
property. VRAVEN resolves it exactly once for each evaluation sample after the node and
coordinate basis have been frozen. Let \(r\) be the requested causal-coordinate count.
The causal-effect applicability indicator is

\[
A_i^C = \mathbf 1\left\{|C|=r\ \land\
\underline{|e_c(x_i)|}\geq\delta\ \text{for every }c\in C\right\}.
\]

Representation applicability requires a non-zero certified clean norm, so relative
drift and direction are defined. If top-k stability is part of the declared property,
the clean top-k boundary must also be strictly separated; a tie at that boundary is
not silently broken. Writing \(A_i^R\) for this condition, full mechanism
applicability is

\[
A_i^M=A_i^R\land A_i^C.
\]

The JSON certificate retains one `cohort_applicability_records` entry per selected
sample, including its original cohort index, optional sample identifier, label,
zero-radius representation norm, top-k status, causal eligible/requested counts and
explicit reasons for an inconclusive applicability verdict. The aggregate cohort
curve repeats the fixed counts `n_representation_applicable`, `n_causal_applicable`
and `n_mechanism_applicable` at every epsilon so that accidental radius-dependent
reclassification is auditable.

For the conditional decision/mechanism table, define \(D_i(\epsilon)\) as a
clean-correct decision certificate and \(M_i(\epsilon)=R_i(\epsilon)\land
C_i(\epsilon)\). Its four exact cells are evaluated only where \(A_i^M=1\):

\[
\begin{aligned}
n_{DM\mid A} &= \sum_i A_i^M D_iM_i,&
n_{D\bar M\mid A} &= \sum_i A_i^M D_i(1-M_i),\\
n_{\bar D M\mid A} &= \sum_i A_i^M(1-D_i)M_i,&
n_{\bar D\bar M\mid A} &= \sum_i A_i^M(1-D_i)(1-M_i).
\end{aligned}
\]

These cells sum exactly to `n_mechanism_applicable`; conditional rates are `null`
when that denominator is zero. Decision/applicability intersections are retained for
representation, causal and conjunctive applicability. Overall and per-class reason
ledgers explain every excluded sample. Existing unconditional \(D/M\) counts remain
as certificate-coverage summaries over all selected samples, but an inapplicable
sample in those summaries must not be interpreted as a mechanism-property failure.

## Shared-input relational effect bounds

The original and intervened executions are evaluated on the same uncertain input;
they are not independent draws from the input box. VRAVEN therefore propagates an
additional relational interval

\[
\Delta v = v(x')-v\!\left(x';\operatorname{do}(m_c=b_c)\right).
\]

Before the intervention, \(\Delta v=0\) exactly. At the intervened coordinate,
\(\Delta m_c=m_c-b_c\), while untouched coordinates retain their incoming
difference. For an affine layer \(y=Wv+a\), shared bias and input terms cancel and

\[
\Delta y=W\Delta v.
\]

For a scalar monotone nonlinearity whose secant slopes lie in \([s_-,s_+]\), the
transformer uses the sound enclosure

\[
f(v)-f(\tilde v)=q(v-\tilde v),\qquad q\in[s_-,s_+],
\]

intersected with ordinary independent interval subtraction. ReLU uses \([0,1]\),
sigmoid \([0,1/4]\), tanh \([0,1]\), and LeakyReLU the interval between its two
slopes. Linear readouts fuse the fixed target-versus-competitor functional, so a
terminal layer uses \((W_t-W_j)\Delta v\) directly instead of separately bounding two
logit differences. For a supported ordinary operator without a specialised
relational rule, VRAVEN falls back to the wider independent subtraction enclosure;
this reduces tightness but does not turn an unsupported relational optimisation into
an unsound certificate.

## Certified vector survival

IBP produces a clean interval (M_0=[\ell_0,u_0]) and a perturbed interval
(M_\epsilon=[\ell_\epsilon,u_\epsilon]). A sound upper bound on displacement between
any clean and perturbed vectors represented by these boxes is

\[
d_\epsilon = \left\|
\max\left(
|u_\epsilon-\ell_0|,
|\ell_\epsilon-u_0|
\right)
\right\|_2.
\]

A lower bound on the clean vector norm is obtained coordinate-wise:

\[
r_0 = \left\|
\begin{cases}
\ell_{0,i}, & \ell_{0,i}>0,\\
-u_{0,i}, & u_{0,i}<0,\\
0, & 0\in[\ell_{0,i},u_{0,i}]
\end{cases}
\right\|_2.
\]

Therefore (d_\epsilon/r_0) upper-bounds relative L2 drift when (r_0>0). A
conservative angular lower bound is

\[
\underline{\cos}(M_0,M_\epsilon)=
\begin{cases}
\sqrt{1-(d_\epsilon/r_0)^2}, & d_\epsilon<r_0,\\
-1, & d_\epsilon\geq r_0.
\end{cases}
\]

This geometry is applied independently to (m(x')) and (e_C(x')). Sign survival
requires every configured fraction of clean sign-definite effect intervals to retain
that sign. Top-k survival requires the smallest selected lower bound to remain greater
than the largest unselected upper bound. Geometry, sign and ranking radii are reported
separately before any pre-registered conjunction.

## Causal interpretation

The term *causal* applies only to the effect of the explicitly executed internal
intervention. It does not imply that a selected unit corresponds to a human concept,
that the intervention is reachable from a natural input, or that the selected
coordinates form the unique mechanism. Auto-selection screens coordinates by clean
intervention-effect lower bound, not activation magnitude or effect midpoint, and is
recorded as focus-sample adaptive. `minimum_clean_effect` is an identifiability
precondition for the declared local causal-effect property; it is not evidence that a
unit represents a human concept or that the intervention lies on the data manifold.

## Verdict semantics

* `certified`: the complete declared inequality conjunction holds for every input in
  the perturbation set;
* `inconclusive`: the sound relaxation does not prove the property;
* `unsupported`: an operator, node, coordinate or property cannot be transformed
  soundly by the native backend.

An inconclusive point is not an existential counterexample. Consequently, a smaller
certified mechanism radius than decision radius does not by itself prove that a
mechanism-changing, label-preserving input exists. Such separation requires a verified
counterexample or a complete solver.

Insufficient effect-eligible coordinates and a frozen coordinate whose clean effect
interval contains zero are both explicit `inconclusive` cases. Neither may be rendered
as a zero-radius positive certificate.

## Outputs

`formal_causal_mechanism_certificate.json` and the dual-theme HTML report retain:

* the frozen node, reduction, coordinates, thresholds and intervention baseline;
* the fixed target and competitor;
* clean intervention effects for every selected coordinate;
* the minimum-effect threshold, requested and eligible coordinate counts, coordinate
  status and an explicit inconclusive reason;
* decision, representation-geometry, representation-ranking, causal-geometry,
  causal-sign and joint radius lower bounds;
* per-epsilon bound values and criterion checks;
* exact cohort counts for mechanism survival \(R\cap C\), joint survival
  \(D\cap R\cap C\), and the complete unconditional decision-versus-mechanism
  contingency table;
* per-sample zero-radius representation, causal-effect and conjunctive applicability,
  with overall/per-class counts, reasons and an exact four-cell D/M table conditioned
  on the conjunctively applicable denominator;
* bisection traces, assumptions and limitations.

## Limitations and validation plan

The current native backend uses relational IBP and can remain conservative after
nonlinear or branching operations despite exact cancellation through affine suffixes.
Internal interventions
may leave the activation manifold. When the node exceeds the configured screening
budget, auto-selection first retains the largest-magnitude clean activations and then
ranks that subset by intervention effect; it is not an exhaustive unit search.
The eligibility threshold is expressed in target-versus-competitor logit-margin units,
so it is model-scale dependent and must be pre-registered; cross-model studies should
also report threshold sensitivity rather than choosing it after seeing outcomes.
Coordinate selection on the focus sample introduces selection bias, so publication
experiments must use a separate selection split or pre-register coordinates before
test-set certification. The next verification tier is
an official alpha-beta-CROWN adapter with exact property manifests, followed by
complete branch-and-bound evaluation on tractable models and independently replayed
proof objects.

The eligibility test reuses the zero-radius intervention bounds already required for
the certificate and adds only an \(O(|C|)\) interval reduction. It does not add model
or intervention evaluations to the verification protocol.

The semantic input contract is also first-class. `InputDomainSpec` records mathematical
bounds (including an intentional unbounded side), units, preprocessing, semantic domain
and the source of that declaration. A publication-evidence ledger reports input-contract,
selection, sample-identity, independent-replay, complete-separation and multi-model
validation gates separately. It is deliberately not collapsed into a trust score.

## Invalidated development artefact

An earlier bundled CNN development artefact included a calibration-selected coordinate
whose focus-sample intervention effect was exactly zero. The former vector-level checks
could ignore that coordinate when computing sign coverage and still issue a positive
causal or joint verdict from the remaining coordinates. That result is invalid under
the property contract above and must not be cited. Regenerated reports mark such a
sample inconclusive unless every frozen coordinate meets `minimum_clean_effect`; valid
reference numbers will be documented only after the corrected protocol is rerun.

Validation requires planted affine/ReLU mechanisms with analytically or exhaustively
known effects, corner containment tests, unsupported-operator tests, parameter and
training-state preservation, multi-seed models, and comparison against a reference
verifier.

## Research relationship

The bound-propagation foundations are established by IBP, DeepZ and CROWN. VRAVEN's
research hypothesis is narrower: decision robustness, representation survival and the
survival of an executable internal intervention-effect vector should be treated as
different formal properties and analysed on a single aligned evidence ledger.

* Gowal et al., [*On the Effectiveness of Interval Bound Propagation for Training
  Verifiably Robust Models*](https://arxiv.org/abs/1810.12715).
* Singh et al., [*Fast and Effective Robustness
  Certification*](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html).
* Zhang et al., [*Efficient Neural Network Robustness Certification with General
  Activation Functions*](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html).
* Gross et al., [*Compact Proofs of Model Performance via Mechanistic
  Interpretability*](https://openreview.net/forum?id=2zWbzx50mH).
