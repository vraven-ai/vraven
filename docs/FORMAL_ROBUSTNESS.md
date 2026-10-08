# Formal local robustness certificates

VRAVEN maintains a strict boundary between empirical attack evidence and formal
verification. `IntervalBoundVerifier` proves a local classification property for a
supported PyTorch model; it never upgrades an inconclusive bound into a vulnerability
or a robustness claim.

## Verified property

For a reference input \(x\), target class \(t\), radius \(\epsilon\), and optional
input domain \([a,b]\), the admissible set is

\[
\mathcal X(x,\epsilon)=\{x' : \lVert x'-x\rVert_\infty\leq\epsilon\}
\cap[a,b]^d.
\]

The verifier propagates lower and upper tensors \((\ell_k,u_k)\) through every
supported operator. If the final logit bounds satisfy

\[
\underline z_t-\max_{j\ne t}\overline z_j>0,
\]

then \(\operatorname*{argmax}_k z_k(x')=t\) for every
\(x'\in\mathcal X(x,\epsilon)\). This is a local mathematical guarantee for the
real-valued network represented by the stored parameters.

For an affine map \(y=Wx+b\), with \(W^+=\max(W,0)\) and
\(W^-=\min(W,0)\), VRAVEN uses

\[
\ell_y=W^+\ell_x+W^-u_x+b,\qquad
u_y=W^+u_x+W^-\ell_x+b.
\]

Monotone activations and pooling apply the operator independently to both endpoints.
Batch normalisation in evaluation mode is converted to its signed affine transform.
Residual addition, subtraction and multiplication use interval arithmetic. A bisection
search returns a certified lower bound on the largest radius for which the IBP margin
remains strictly positive.

## Verdicts

* `certified`: every competing logit margin has a strictly positive lower bound;
* `inconclusive`: the relaxation crosses zero, so IBP did not prove the property;
* `unsupported`: at least one graph operation lacks a sound transformer.

Only `certified` is a positive proof. `Inconclusive` is compatible with both a robust
and a vulnerable model. Attack success supplies an empirical upper bound on the true
radius; formal certification supplies a lower bound. Reporting both exposes the
unresolved interval instead of hiding it.

## Supported graph operations

The native `vraven_ibp_fx_v1` backend traces the actual forward graph with
`torch.fx`. It supports:

* Linear and Conv1d/2d/3d;
* evaluation-mode BatchNorm1d/2d/3d;
* ReLU, non-negative-slope LeakyReLU, sigmoid and tanh;
* max, average and adaptive-average pooling;
* flatten, reshape, view, mean, indexing and concatenation-free sequential graphs;
* evaluation-mode dropout and identity;
* interval addition, subtraction and multiplication.

All other operators produce an explicit unsupported verdict. The operator ledger is
stored in every certificate.

## Certificate outputs

The JSON and dual-mode HTML reports contain:

* the exact property, norm, input domain, checkpoint name and verifier version;
* every tested epsilon and its worst competing-logit margin lower bound;
* a bisection-derived certified-radius lower bound;
* full-cohort and per-class certified accuracy;
* layer-wise interval growth and unstable-activation fractions at the first
  inconclusive epsilon;
* unsupported operations and limitations.

The bound-growth ledger identifies a verification bottleneck. A wide interval or
unstable ReLU is not automatically a causal model weakness.

## Numerical semantics

Parameters are lifted to float64 and affine endpoints receive conservative outward
padding based on operation count and magnitude. This mitigates verifier round-off. The
property nevertheless concerns the conventional real-valued abstraction of the stored
network, as in neural-network verification literature; it is not a bit-level proof of
every vendor-specific floating-point kernel.

## Reference runtime

On the bundled `image_pattern_cnn2d` checkpoint, a CPU run over all 64 declared
reference samples, 12 epsilon values and a 24-step focus-sample radius search took
3.19 seconds on an Apple M1 Max. This is an observed development measurement, not a
cross-platform performance guarantee. Formal verification is an opt-in audit path and
does not alter the fast explanation-mode latency target.

## Limitations and roadmap

IBP is incomplete and can become loose in deep networks. A failed proof is not a
counterexample. Tighter future backends should export the same certificate schema:

1. CROWN or alpha-CROWN optimised linear relaxations;
2. beta-CROWN branch-and-bound for complete verification;
3. domain-specific constraints for discrete or functionality-preserving inputs;
4. independent verifier replay and proof-object checking.

## Primary references

* Gowal et al., [*On the Effectiveness of Interval Bound Propagation for Training
  Verifiably Robust Models*](https://arxiv.org/abs/1810.12715), 2018/2019.
* Zhang et al., [*Efficient Neural Network Robustness Certification with General
  Activation Functions*](https://arxiv.org/abs/1811.00866), NeurIPS 2018 (CROWN).
* Wang et al., [*Beta-CROWN: Efficient Bound Propagation with Per-neuron Split
  Constraints for Neural Network Robustness Verification*](https://papers.nips.cc/paper/2021/hash/fac7fead96dafceaf80c1daffeae82a4-Abstract.html),
  NeurIPS 2021.
* Verified-Intelligence,
  [`alpha-beta-CROWN`](https://github.com/Verified-Intelligence/alpha-beta-CROWN),
  official complete-verifier implementation and VNN-COMP reference backend.
