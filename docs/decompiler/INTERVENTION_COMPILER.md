# Graph-Compiled Intervention Engine

## Why graph rewriting

Module hooks can miss functional operations, reused modules and compiler-normalised structure. VRAVEN therefore rewrites the captured FX or Export graph and inserts explicit `InterventionGate` modules after selected nodes.

## Supported interventions

```text
zero
replace
scale
add
erase_direction
set_concept
set_concept_mean
clamp
```

## Effect definition

For each intervention, VRAVEN records:

\[
\Delta=m_t(x)-m_t^I(x)
\]

with:

* Mean and median effect
* Bootstrap interval
* Sign stability
* Prediction-flip rate
* Baseline and intervened logits
* Hash of generated graph code

## Concept erasure

For concept direction \(d\) and coefficient \(\alpha=\langle h,d\rangle\):

\[
h'=h-s\alpha d
\]

where \(s\) is the intervention strength.

## Scientific limitation

An internal intervention may create a state outside the activation manifold observed during training. VRAVEN records a manifold-distance diagnostic and does not treat every intervention as naturally reachable.
