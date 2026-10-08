# Publication Plan

## Proposed paper

**VRAVEN: Compiler-Assisted Causal Decompilation of PyTorch Models**

## Main contributions to evaluate

1. Mechanism IR and measurable capture certificate.
2. Graph-compiled intervention engine.
3. Intervention-constrained executable programme extraction.
4. Validity envelope and dark-matter accounting.
5. Cross-architecture known-mechanism benchmark.

## Core research questions

**RQ1:** Can VRAVEN recover numerically equivalent executed computation from previously unseen PyTorch models with measurable coverage?

**RQ2:** Do graph-compiled interventions reproduce declared model interventions more consistently than module-hook implementations?

**RQ3:** Does intervention-constrained programme extraction improve interventional fidelity compared with observational-only surrogates?

**RQ4:** Can the programme preserve observational, intervention and counterfactual behaviour within a declared validity envelope?

**RQ5:** Can VRAVEN recover known concepts, edges and programmes across MLP, CNN, residual, recurrent, transformer and graph models?

## Minimum experimental design

* At least five architecture families
* At least three independent training seeds
* Known-mechanism synthetic models
* Natural datasets in at least two domains
* Explicit dynamic-path and custom-operator cases
* Random and size-matched controls
* Multiple intervention baselines
* Confidence intervals and effect sizes
* Ablation of every central component
* Runtime and memory comparison
* Human expert evaluation for explanation usefulness

## Primary metrics

```text
operator coverage
parameter coverage
gradient coverage
intervention coverage
numerical equivalence
concept precision and recall
causal-edge precision and recall
programme observational R²
programme interventional correlation and sign agreement
counterfactual agreement
semantic coverage
unresolved causal responsibility
runtime and memory
```

## Baseline families

Compare only against methods addressing the same question:

* Input attribution: SHAP, Integrated Gradients, DeepLIFT and LIME
* Internal attribution: Captum layer/neuron methods
* Circuit recovery: ACDC, attribution patching and relevant circuit methods
* Representation comparison: CKA, Procrustes and MMD
* Surrogate programmes: rule extraction and programme-synthesis baselines
