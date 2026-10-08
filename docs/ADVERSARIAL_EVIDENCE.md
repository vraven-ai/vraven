# Adversarial evidence protocol

VRAVEN separates attack discovery, sampling uncertainty and internal-representation
change. None of these outputs is a formal robustness certificate.

## Protocol identity

`vraven_ensemble_lite` is VRAVEN's bounded pilot ensemble: APGD-CE,
APGD-DLR, FAB-lite and a flat-block query search. It is intentionally not named
AutoAttack: the budgets and implementations do not claim parity with the reference
AutoAttack package. `audit_autoattack()` remains as a backwards-compatible method
name but records `protocol="vraven_ensemble_lite"`.

The flat-block query search changes contiguous flattened coordinates. It is not the
spatial Square Attack for images. The report uses the implementation-accurate name.

## Accuracy estimands

For labelled observations \((x_i,y_i)\), attacks \(\mathcal A\), and perturbation
budget \(\epsilon\), overall attacked robust accuracy is

\[
\widehat{RA}_{\mathcal A}(\epsilon)=\frac{1}{n}\sum_{i=1}^{n}
\mathbf 1[f(x_i)=y_i]\mathbf 1[\forall a\in\mathcal A,
f(a(x_i;y_i,\epsilon))=y_i].
\]

Clean errors therefore count as non-robust. Clean-conditional robustness is reported
separately:

\[
\widehat{CRA}_{\mathcal A}(\epsilon)=n_{\mathrm{robust}}/
n_{\mathrm{clean\ correct}}.
\]

If ground-truth labels are absent, VRAVEN reports prediction invariance instead of
accuracy. Attacks receive ground-truth labels when available.

The robustness profile reports a matched **per-sample** attack union over epsilon: if
an adversary is found for sample (i) at a smaller budget, that sample remains failed
at every larger budget. This survival construction is stronger than taking a
cumulative minimum of aggregate counts, which can lose failures when different
samples fail at different budgets. Its normalised area is computed by the trapezoidal
rule over the declared interval. Pointwise nominal 95% Wilson intervals, a paired
bootstrap interval for curve area and a simultaneous DKW band quantify sampling
uncertainty conditional on the fixed evaluator; none quantifies missed adversaries.
An attack-based robust-accuracy estimate is an empirical upper estimate of worst-case
robust accuracy because optimisation may fail to find an existing adversary.

PGD supports deterministic multi-start evaluation. Every random start is assessed,
the strongest objective point is retained across the complete path, and non-finite or
zero-gradient sample-steps plus norm/domain invariants are recorded. Publication
protocols may additionally union matched failures across attack seeds; a sample is
then counted robust only if no configured seed or restart found an adversary.

## Sampling and attribution

Publication runs should use the full declared cohort. When a limit is required,
deterministic stratified sampling records exact sample indices and class counts.
Reports with fewer than 100 observations are labelled `pilot`. Per-class and macro
robust accuracy accompany the aggregate value.

For every attack family VRAVEN reports:

* raw successes;
* unique successes found by no other family;
* marginal new successes in the declared ensemble order; and
* the cumulative attack union.

These quantities must not be interchanged.

## Mechanism comparison

Activation coordinates are selected once on the clean input and frozen for every
adversarial comparison. Re-selecting top activations after perturbation would compare
different neurons and invalidate the result.

The mechanism report retains coordinate-aligned cosine, retained activation energy,
top-active cosine and worst-layer cosine separately. Its legacy conservative survival
value is their minimum; it must not be described as cosine alone. Instability-radius
search accepts an explicit named component, and publication protocols must record
which component is primary. A low value is an observed representational instability,
not causal proof. Cross-attack vectors must share the same frozen basis, and failure
families use deterministic complete-link clustering.

## Mathematical objectives

APGD-CE uses cross-entropy. Untargeted DLR for multiclass logits is

\[
L_{DLR}=-\frac{z_y-\max_{k\ne y}z_k}{z_{(1)}-z_{(3)}}.
\]

Targeted DLR uses

\[
L_{DLR}^{t}=-\frac{z_y-z_t}{z_{(1)}-(z_{(3)}+z_{(4)})/2}.
\]

Binary logits do not have the required top-three/top-four order statistics, so VRAVEN
uses an explicitly documented margin fallback. Input gradients are computed with
`torch.autograd.grad`, preserving any existing parameter gradients.

## Evidence boundaries

* Mathematical norm constraints need not preserve malware functionality, tabular
  validity, token grammar or other domain invariants.
* Passing attack-adequacy checks is not evidence of security.
* Focus-sample mechanism traces are case studies; population claims require a
  declared cohort and uncertainty analysis.
* The smallest successful tested budget is not an exact or certified radius.
* Stochastic and unusual defences require adaptive attacks and expectation over
  transformation; the VRAVEN-lite suite does not provide these automatically.

## Primary references

* Croce and Hein, *Reliable Evaluation of Adversarial Robustness with an Ensemble
  of Diverse Parameter-free Attacks*, ICML 2020.
* Croce and Hein, *Minimally distorted Adversarial Examples with a Fast Adaptive
  Boundary Attack*, ICML 2020.
* Pintor et al., *Indicators of Attack Failure*, NeurIPS 2022.
* Zimmermann et al., *Increasing Confidence in Adversarial Robustness Evaluations*,
  NeurIPS 2022.
* Kornblith et al., *Similarity of Neural Network Representations Revisited*, ICML
  2019.
