# Scientific and engineering limitations

1. The extension does not claim to recover human-like reasoning or the complete causal mechanism of a network.
2. Sparse circuitry is approximate and conditional on the selected layers, grouping, baseline, sample and pruning order.
3. Circuit multiplicity is empirical. It is not a formal certificate over continuous inputs or dataset edits.
4. Zero, mean and patched activations can create states outside the model's learned activation manifold.
5. Activation patching can contain interaction effects. A large recovery is not necessarily a pure mediated effect.
6. Probes measure decodability. They do not prove that the model uses the decoded variable.
7. CKA, MMD, Wasserstein distance, principal angles and topology summarise different aspects of representation geometry. No single metric is complete.
8. Reasoning fingerprints are deterministic numerical summaries. They are not semantic explanations without external validation.
9. Unit-level summaries can miss distributed and higher-order mechanisms.
10. Pairwise synergy does not recover all higher-order interactions.
11. Timing measured through hooks includes instrumentation overhead. Accurate accelerator profiling requires device synchronisation and repeated measurements.
12. A model can be internally stable and still wrong. It can also be internally unusual and still correct.
13. High-confidence internal anomaly screening depends on reference coverage and is not an attack detector by itself.
14. Topological H1 analysis is optional and reports a lifetime-distribution summary rather than a complete persistence-diagram distance.
15. The extension provides analyses beyond input attribution, but universal superiority over SHAP, LIME, LRP or Integrated Gradients has not been established.
16. Every publication should report baselines, intervention budget, sample selection, omitted layers, seed, metric version and negative findings.
