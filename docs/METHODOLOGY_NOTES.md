# Methodology notes and mathematical definitions

## 1. Layer activation summaries

For activation tensor values `a_(l,i)` at layer `l`, a size-normalised energy is:

```text
E_l = (1 / n_l) * sum_i a_(l,i)^2
```

Normalisation by `n_l` is important when layers contain different numbers of elements. Additional summaries may include the mean, variance, robust quantiles and sparsity under a declared threshold.

## 2. Target-specific signed interaction

For target score `s_y`, a first-order interaction may be represented as:

```text
Q_l = (1 / n_l) * sum_i a_(l,i) * (partial s_y / partial a_(l,i))
```

The signed value distinguishes local support from opposition under the first-order approximation. The absolute value may be used for magnitude, but sign and magnitude should remain recoverable. This is observational sensitivity, not causal proof.

## 3. Predictive uncertainty

For a valid class probability distribution `p`, normalised predictive entropy is:

```text
H(p) = -sum_c p_c log(p_c) / log(C)
```

where `C` is the number of classes. Activation variance must not be labelled predictive uncertainty. Epistemic uncertainty requires a justified estimator such as an ensemble, posterior approximation or repeated stochastic model evaluation.

## 4. Evidence pathway graph

A pathway edge may combine separately normalised signals, but the combination must be declared. A candidate score is:

```text
I_l = alpha * A_l + beta * Q_l + gamma * Delta_l - delta * U_l
```

where each term is defined independently. The coefficients must be estimated, calibrated or subjected to sensitivity analysis. They must not be selected solely to create visually attractive flow widths.

## 5. Reasoning fingerprint

Let `M_x` be an ordered layer-by-metric matrix. The reference fingerprint path is:

```text
R(x) = resample(vec(clip(robust_scale(M_x))), d)
```

Robust scaling uses the median and interquartile range, with an explicitly documented fallback for zero-spread columns. Clipping limits extreme values. Deterministic resampling produces a fixed width `d` for comparisons across traces.

Cosine similarity is one possible comparison:

```text
sim(x_i, x_j) = dot(R_i, R_j) / (||R_i|| * ||R_j||)
```

The fingerprint is an engineered representation of the trace. It is not evidence that the model has a human-like reasoning process.

## 6. Latent communities

For unit profiles across `N` samples, a similarity matrix may be constructed as:

```text
C_ij = corr(A[:, i], A[:, j])
```

Alternative measures may be required for non-linear dependence or sparse activations. The similarity metric, standardisation, threshold, graph weighting, community algorithm and seed must be recorded.

A discovered community is initially a latent group. A semantic label requires external evidence, for example concept examples, a trained probe with held-out validation, or a controlled intervention.

## 7. Causal validation

For pathway `P` and declared intervention `I(P)`:

```text
Delta s_y(P) = s_y(x) - s_y(x; I(P))
```

The intervention may be zeroing, scaling, replacement or another defined transformation. Effects should be compared with matched controls and reported with uncertainty across independent observations. The result supports the effect of the intervention, not the recovery of the complete causal mechanism.

## 8. Deletion and insertion

Deletion progressively suppresses components ranked as important and measures target-score degradation. Insertion restores ranked components from a declared baseline and measures target-score recovery.

The area under the deletion or insertion curve is meaningful only when the baseline, component granularity and intervention are comparable across methods.

## 9. Explanation stability

For a task-valid perturbation `x'` that should preserve the underlying label:

```text
S_E = sim(R(x), R(x'))
```

Stability should be reported across a perturbation distribution rather than one arbitrary perturbation. High stability is not proof of fidelity, and low stability may reflect genuine model sensitivity.

## 10. Cross-method agreement

Agreement can be reported through:

* Spearman rank correlation
* sign agreement
* top-k overlap
* intersection over union for regions
* fingerprint similarity where representations are comparable

Agreement means that methods return similar outputs. It does not establish that either method is correct.

## 11. Reasoning drift

For reference and current fingerprint distributions, candidate measures include Jensen-Shannon divergence after justified density estimation, maximum mean discrepancy, energy distance and Wasserstein distance.

A permutation or bootstrap procedure should quantify uncertainty. Drift should also be evaluated conditionally by class, confidence or model outcome to avoid conflating label-distribution change with internal-representation change.

## 12. Representation landscapes

Independent t-SNE projections must not be presented as literal motion across layers. Use one shared projection, a transform-capable embedding fitted once, a deterministic linear projection or an explicit alignment procedure. Store the projection method, distance metric and seed.

## 13. Attention

Attention weights may be visualised as attention weights. They must not be relabelled as feature importance or causal importance without separate validation.

## 14. Performance comparison

A fair comparison with SHAP, LIME or other methods must control:

* model and checkpoint
* ordered inputs and targets
* device and numerical precision
* warm-up policy
* batch size
* method-specific sample, step or coalition budgets
* output score definition
* preprocessing and baseline
* synchronisation before timing on accelerators

Report distributions, confidence intervals and paired comparisons. Do not claim superiority from one synthetic benchmark.
