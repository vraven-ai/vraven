# VRAVEN Compiler-Assisted Causal Decompiler Architecture

## Purpose

VRAVEN is designed to recover a compact, executable and experimentally testable high-level mechanism from a previously unseen trained PyTorch model. It does not claim to expose one uniquely true explanation. It recovers the smallest tested programme found by the configured search and reports the conditions under which that programme was evaluated.

The pipeline is:

```text
Trained PyTorch model
        ↓
Multi-route computation capture
        ↓
VRAVEN Mechanism IR
        ↓
Automatic internal concept discovery
        ↓
Cross-layer concept evolution
        ↓
Graph-compiled causal interventions
        ↓
Causal mechanism graph
        ↓
Executable decision programme
        ↓
Contrastive explanation
        ↓
Counterfactual proof
        ↓
Validity and coverage certificate
```

## Layer 1: Capture router

VRAVEN attempts capture routes in a declared order:

1. `torch.export`, preferred because it exposes a normalised ATen-level graph.
2. `torch.fx`, useful for Python-level symbolic graph capture.
3. Runtime dispatch tracing, limited to the executed operator path.
4. Module hooks, a conservative fallback that may omit functional operations.

Every failed route is retained in the certificate. A fallback route never inherits the coverage claims of a successful full-graph route.

## Layer 2: Mechanism IR

The architecture-neutral intermediate representation contains:

* Operator nodes
* Parameters and buffers
* Tensor-flow edges
* Shapes, strides, data types and devices
* Operator semantics
* Source and module provenance
* Dynamic-shape constraints
* Gradient availability
* Intervention accessibility
* Unsupported operations
* Capture attempts and numerical-equivalence evidence

The IR is deliberately separate from model activations. Persisted IR files contain structure and metadata, not raw private inputs by default.

## Layer 3: Operator semantics dialect

Low-level operations are lifted into typed roles:

```text
Input, Transform, Detect, Gate, Suppress, Normalise, Pool,
Merge, Route, Accumulate, Compare, Store, Retrieve, Attend,
Repeat, Readout, Shape, Random, Output and Unknown.
```

Operator semantics describe what an operation family does. They do not automatically identify what a learned latent feature means.

## Layer 4: Concept discovery

VRAVEN learns sparse or low-rank directions from selected internal representations. The current implementation supports:

* Deterministic PCA for rapid analysis
* A compact sparse autoencoder for more expressive decomposition

Each concept receives a technical identifier, activation distribution, top samples, class preference and optional semantic-association evidence. Association-derived labels are not counted as validated semantic grounding.

## Layer 5: Concept evolution

Concepts are aligned across model depth through a deterministic, regularised optimal-transport procedure. The result represents:

* Birth
* Persistence
* Strengthening or weakening
* Splitting
* Merging
* Suppression
* Reappearance
* Termination

Independent t-SNE embeddings are never treated as literal movement through a network.

## Layer 6: Causal mechanism graph

Candidate concepts and edges are screened structurally and through derivatives, then tested with explicit graph-rewritten interventions. The implementation records:

* Output-margin effect
* Downstream-concept effect
* Bootstrap confidence interval
* Sign stability
* Prediction-flip frequency
* Derivative support
* Intervention-manifold distance
* Evidence tier

Gradients are candidate-screening evidence, not causal proof.

## Layer 7: Executable programme

VRAVEN compiles selected concepts and interactions into a sparse continuous programme. The programme is trained jointly against:

1. Normal model margins
2. Intervention-effect constraints

This avoids a common failure where an observational surrogate predicts normal outputs perfectly but fails every internal intervention.

The programme reports observational, interventional and counterfactual fidelity separately.

## Layer 8: Counterfactual proof

A bounded concept-level search proposes changes in the programme, then verifies selected candidates by executing graph-compiled interventions in the original captured model. A counterfactual is not declared domain-valid unless an application-specific validator approves it.

## Layer 9: Mechanism certificate

The certificate reports independent dimensions rather than one opaque trust score:

* Operator coverage
* Parameter coverage
* Gradient coverage
* Intervention coverage
* Validated semantic coverage
* Association-only semantic coverage
* Causally weighted unresolved responsibility
* Observational fidelity
* Interventional fidelity
* Counterfactual fidelity
* Validity envelope
* Known limitations

## Current boundaries

The implementation is a research alpha. It supports PyTorch export and FX graphs most strongly. It does not guarantee capture of every custom operator, every dynamic Python branch, every model family or every naturally reachable counterfactual. Unsupported computation remains visible.
