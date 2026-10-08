# Limitations

1. `torch.export` and FX cannot capture every Python construct or custom operator.
2. Fallback routes may omit functional operations.
3. Discovered concepts are not guaranteed to correspond to human concepts.
4. Association-derived labels are not causal or semantically validated.
5. Concept directions may be non-identifiable under equivalent basis transformations.
6. Internal ablations may create off-manifold states.
7. The causal graph is bounded by the tested concepts, edges, samples and baselines.
8. The programme is a sparse surrogate with explicit fidelity, not the unique model algorithm.
9. Counterfactual internal states may not be realisable as valid raw inputs.
10. The current implementation is strongest for deterministic PyTorch inference in evaluation mode.
11. Dynamic branches are certified only for observed and captured paths.
12. GPU, MPS and ROCm paths require hardware-specific validation.
13. Human-understanding claims require a user study.
14. State-of-the-art claims require matched external baselines and independent replication.
