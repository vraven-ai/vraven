# VRAVEN 0.9 Gap-Closure Implementation

## Research gap addressed

The implementation targets a gap between three existing classes of tooling:

1. **Input attribution**, which ranks input variables but does not reconstruct the model's internal mechanism.
2. **Mechanistic tooling**, which often depends on one architecture family, especially transformers, and usually exposes circuits rather than a portable executable explanation.
3. **PyTorch compiler tooling**, which recovers normalised computation but does not translate that computation into a human-level causal account.

VRAVEN 0.9 connects these layers:

```text
Previously unseen PyTorch model
    → compiler-level computation capture
    → architecture-neutral Mechanism IR
    → latent concept discovery
    → cross-layer concept evolution
    → graph-compiled causal interventions
    → intervention-constrained executable programme
    → contrastive explanation
    → counterfactual mechanism test
    → validity and dark-matter certificate
```

The release does not assert that the recovered programme is uniquely true. It records the smallest tested high-level programme found by the configured search and shows the evidence that supports or limits that programme.

## Gap-to-code map

| Research gap | Implemented component | Main source |
| --- | --- | --- |
| Hooks miss functional operations | Multi-route Export and FX capture | `src/vraven/decompiler/capture.py` |
| Framework objects are too low-level for common analysis | VRAVEN Mechanism IR | `src/vraven/decompiler/ir.py` |
| ATen operators lack explanatory roles | Operator semantics registry | `src/vraven/decompiler/semantics.py` and `data/operator_semantics.json` |
| Concept scores can be inconsistent outside the fitting cohort | Persisted concept scoring transforms | `src/vraven/decompiler/concepts.py` |
| Independent layer factors do not show evolution | Regularised optimal-transport alignment | `src/vraven/decompiler/evolution.py` |
| Gradients are local sensitivity, not causal evidence | Derivative screening plus compiled interventions | `src/vraven/decompiler/causal_graph.py` |
| Hook interventions are fragile and module-boundary limited | Explicit FX graph intervention compiler | `src/vraven/decompiler/interventions.py` |
| Observational surrogates can fit outputs but fail interventions | Joint observational and intervention-constrained programme fitting | `src/vraven/decompiler/programme.py` |
| Explanations rarely state why one class defeated another | Deterministic contrastive explanation | `src/vraven/decompiler/contrastive.py` |
| Counterfactuals may change output without changing the claimed mechanism | Programme-screened, original-model-verified concept intervention | `src/vraven/decompiler/counterfactual.py` |
| Explanation tools hide unsupported computation | Coverage certificate and Raven Dark Matter | `src/vraven/decompiler/certificate.py` |
| Visuals are commonly disconnected from evidence | Seven standalone scientific proof figures | `src/vraven/decompiler/visuals/` |
| Natural models have no known true circuit | Planted-mechanism development benchmark | `src/vraven/decompiler/decompilebench.py` |

## Scientific objects introduced by the implementation

### Mechanism IR

A typed graph records operators, tensor-flow edges, parameters, buffers, shapes, guards, source provenance, differentiation availability, intervention access and unresolved computation.

### Intervention-supported concept graph

Each concept node stores an output-margin intervention effect, interval, sign stability, prediction-flip rate, responsibility and off-manifold warning. Candidate edges combine structural reachability, cross-layer alignment, derivative screening and downstream intervention response.

### Executable mechanism programme

The programme is a sparse set of standardised main and interaction terms. Its parameters are fitted jointly to normal target margins and measured intervention effects. Observational and intervention fidelity are reported separately.

### Explanation validity envelope

The certificate states the captured route, observed shapes, dynamic constraints, execution paths, supported interventions, unresolved operations, model state and numerical equivalence. It does not issue one opaque trust score.

### Model dark matter

The report separately measures:

* computationally captured and recognised operations;
* intervention-accessible computation;
* externally validated semantics;
* association-only semantics;
* causally important but unresolved concepts;
* unsupported operations and uncovered paths.

## Current novelty boundary

The implementation uses established mathematical components, including PCA, sparse autoencoders, optimal transport, bootstrap intervals, regularised regression and graph interventions. Its candidate contribution is their compiler-assisted integration into an architecture-neutral causal decompilation workflow with an executable programme and explicit validity certificate.

The following are not claimed:

* universal reversal of every PyTorch model;
* uniquely identifiable human concepts;
* globally minimal circuits or programmes;
* domain-valid raw-input counterfactuals without a domain validator;
* state-of-the-art status across model families;
* superiority to SHAP or LIME on their input-attribution question.

## Next publication-critical work

1. Extend known-mechanism benchmarks across CNN, recurrent, transformer, graph and conditional architectures.
2. Add vectorised intervention execution using `torch.func.vmap` where graph structure permits it.
3. Add guarded path aggregation for data-dependent control flow.
4. Add architecture-invariant concept alignment across checkpoints and model families.
5. Evaluate programme recovery precision, causal-edge recovery, intervention fidelity and counterfactual fidelity against known mechanisms.
6. Run human studies comparing decision quality and time with attribution-only, circuit-only and VRAVEN Decision Proof outputs.
7. Validate across independent model seeds, datasets and accelerator backends.
