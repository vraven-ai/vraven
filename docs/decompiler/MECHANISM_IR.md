# VRAVEN Mechanism Intermediate Representation

## Design objective

The Mechanism IR is the boundary between framework-specific capture and framework-independent decompilation. PyTorch adapters produce the IR. Concept discovery, causal analysis, programme extraction and visualisation consume it.

## Core object

For model `M` and observed execution path `π(x)`, the IR is represented as:

\[
\mathcal{V}(M,x)=\left(G_F,G_S,G_C,\mu\right)
\]

where:

* \(G_F\) is the forward operator graph.
* \(G_S\) records parameters, buffers, shapes and state.
* \(G_C\) records control-flow and guard information available from capture.
* \(\mu\) records source, module, gradient, intervention and provenance metadata.

Backward-path availability is attached to nodes through an explicit autograd probe.

## Node contract

Every node records:

```text
id
name
kind
low-level operation
target
mechanism role
input node identifiers
tensor metadata
parameter count
differentiability
gradient availability
intervention accessibility
semantic recognition status
source location
unresolved reason
```

## Edge contract

Every edge records:

```text
source
target
tensor metadata
edge type
optional provenance
```

## Coverage

VRAVEN does not emit one combined coverage score. It reports:

\[
C_{op}=\frac{N_{recognised}}{N_{executed}}
\]

\[
C_{grad}=\frac{N_{gradient}}{N_{differentiable}}
\]

\[
C_{int}=\frac{N_{intervention}}{N_{candidate}}
\]

\[
C_{param}=\frac{P_{captured}}{P_{model}}
\]

The ratios have different denominators and must not be added together.

## Numerical equivalence

A captured graph is checked against the original eager model on the example inputs:

\[
\epsilon_{abs}=\max_i |M(x)_i-\hat{M}(x)_i|
\]

\[
\epsilon_{rel}=\max_i \frac{|M(x)_i-\hat{M}(x)_i|}{|M(x)_i|+\varepsilon}
\]

The result and tolerances are persisted in the certificate.

## Privacy

The IR stores tensor metadata, not full tensors. Raw activations are held only while an analysis is running unless the user explicitly exports them.
