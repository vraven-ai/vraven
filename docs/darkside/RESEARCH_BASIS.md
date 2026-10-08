# Research basis and design decisions

VRAVEN DarkSide extends the existing framework from feature attribution towards internal mechanism auditing. The original VRAVEN direction already treats information flow, evidence accumulation, internal disagreement, drift and causal validation as central aims. This extension implements a bounded set of methods without replacing the main project.

The sources below informed the design. Their inclusion does not mean VRAVEN reproduces every paper or guarantee. Each implemented method states its narrower claim.

## Circuit discovery, minimality and stability

Automated circuit discovery seeks a compact internal subgraph that preserves a selected behaviour. ACDC introduced automated graph pruning, while attribution patching and relevance-patching research explored faster approximations. IBCircuit used an information-bottleneck objective for holistic discovery. More recent work has moved towards formal robustness and stability guarantees and has shown that circuits can change across datasets, checkpoints and discovery conditions.

VRAVEN therefore separates four questions:

1. Can groups be removed while preserving this sample's prediction within ε?
2. Which retained groups become conditionally necessary after the accepted removals?
3. Do repeated discovery runs recover one empirical stable core or several alternative circuits?
4. Does the discovered effect pattern transfer to another checkpoint, domain or model?

`discover_decision_circuitry` answers the first two using verified interventions and greedy backward elimination. `analyse_circuit_multiplicity` answers the third from aligned repeated circuit masks. `analyse_mechanism_transfer` addresses the fourth using aligned effect vectors. None of these modules claims a formal certificate over a continuous input domain.

Key sources:

* Conmy et al. (2023), *Towards Automated Circuit Discovery for Mechanistic Interpretability*.
* Syed et al. (2023), *Attribution Patching Outperforms Automated Circuit Discovery*.
* Bian et al. (2025), *IBCircuit: Towards Holistic Circuit Discovery with Information Bottleneck*.
* Jafari et al. (2025), *RelP: Faithful and Efficient Circuit Discovery via Relevance Patching*.
* Hadad, Katz and Bassan (2026), *Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees*.
* Anani et al. (2026), *Certified Circuits: Stability Guarantees for Mechanistic Circuits*.
* Rai, Geva and Yao (2026), *Data-driven Circuit Discovery for Interpretability of Language Models*.
* Chen et al. (2026), *Navigating by Old Maps: The Pitfalls of Static Mechanistic Localization in LLM Post-Training*.

## Causal interventions, self-repair and interaction effects

Activation patching and mediation-style analyses require explicit intervention semantics. Causal abstraction provides a language for hard and soft interventions, path patching and mechanism transformations. Recent work on multiple mediators shows that an activation-patching estimate can contain interaction terms. Gradient-based localisation can also be affected by self-repair and redistribution.

VRAVEN reports clean-to-corrupt recovery, optional reverse damage and pairwise interaction separately. It does not silently label patch recovery as a pure natural indirect effect. Pairwise synergy is offered because greedy single-component ranking can miss groups whose effect depends on another component.

Key sources:

* Geiger et al. (2025), *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*.
* Edin et al. (2025), *Correcting Gradient-Based Circuit Localization via Interaction-Aware Backpropagation*.
* Vaidyanathan et al. (2026), *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching*.

## Shortcut learning and brittle dominance

Shortcut learning is a distribution-level claim. A large local attribution or ablation effect can identify dependence, but it does not prove that the cue is semantically spurious. Shortcut evidence is stronger when it combines domain knowledge, counterfactual or balanced data, worst-group behaviour and localisation across a representative cohort.

VRAVEN therefore separates local input-group intervention, first-order attribution dominance, layerwise decodability and prediction depth. The shortcut suspicion index is only a within-audit ranking device. It has no universal threshold.

Key sources:

* Geirhos et al. (2020), *Shortcut Learning in Deep Neural Networks*.
* Boland et al. (2024), *There Are No Shortcuts to Anywhere Worth Going*.
* Achara et al. (2025), *Localising Shortcut Learning in Pixel Space via Ordinal Scoring Correlations for Attribution Representations*.
* Teerapittayanon, McDanel and Kung (2017), *BranchyNet*.

## Representation drift, geometry and prognostic markers

Representation similarity is not one-dimensional. CKA can be invariant to some transformations, while Procrustes residual, MMD, sliced Wasserstein distance, neighbourhood overlap, principal angles and rank measurements expose different changes. Work on representational robustness and neural representation paths supports analysing a model as an ordered trajectory rather than only its final layer. Recent diagnostic work also shows why representation geometry can be valuable as a marker of hidden generalisation risk even when a full mechanism is not recovered.

VRAVEN uses exact sample alignment for paired comparisons and retains raw component measurements. Its drift and stability composites are transparent ranking indices, not probabilities.

Key sources:

* Kornblith et al. (2019), *Similarity of Neural Network Representations Revisited*.
* Nanda et al. (2022), *Measuring Representational Robustness of Neural Networks Through Shared Invariances*.
* Lange et al. (2023), *Deep Networks as Paths on the Manifold of Neural Representations*.
* Chou et al. (2026), *Diagnosing Generalization Failures from Representational Geometry Markers*.
* Chen et al. (2026), *Navigating by Old Maps*.

## Topology

Representation Topology Divergence motivates comparison of multi-scale point-cloud structure. VRAVEN implements exact H0 persistence death times through minimum spanning trees. H1 is optional and requires `ripser`. This is intentionally narrower than a complete topology-divergence implementation, but its data contract and distance are explicit.

Key source:

* Barannikov et al. (2022), *Representation Topology Divergence*.

## Explanation disagreement, multiplicity and uncertainty

SHAP, LIME, gradient methods, LRP and intervention effects can disagree because they target different quantities, use different baselines or rely on different approximations. Recent work on SHAP explanation multiplicity also shows that magnitude similarity can hide substantial churn in feature rank and top-feature identity. VRAVEN therefore records rank, sign and top-k disagreement instead of choosing the visually most convenient method. Repeated explanations add bootstrap intervals, sign probability, top-k selection probability and rank variation.

High agreement does not establish correctness. Disagreement can be the primary scientific finding.

Key sources:

* Krishna et al. (2022), *The Disagreement Problem in Explainable Machine Learning*.
* Hwang et al. (2026), *Explanation Multiplicity in SHAP: Characterization and Assessment*.

## Failure anatomy and high-confidence internal anomalies

A final confidence value can remain high even when the internal path is atypical or when a failure cohort follows a different geometry. VRAVEN separates outcome confidence from internal fingerprint distance and cohort separation. The OOD module uses shrinkage Mahalanobis distance and empirical reference percentiles as a review signal. It does not call an unusual fingerprint malicious or adversarial without external evidence.

The design is also influenced by the diagnostic-science view that task-relevant internal markers can reveal hidden vulnerability without claiming a complete mechanistic explanation.

## Why this goes beyond a feature-only view

SHAP and LIME remain useful baselines for input attribution. DarkSide adds analytical objects that their standard output does not represent:

* approximate causal sufficiency of internal groups
* empirical multiplicity of alternative circuits
* pairwise cooperation, compensation and redundancy
* clean-to-corrupt activation repair
* class and failure cohort geometry
* checkpoint and domain drift
* internal path collapse under perturbation
* capacity and runtime diagnostics
* adversarial or evasion rerouting
* topological change
* uncertainty over signs, ranks and top-k membership
* mechanism transfer
* high-confidence internal anomalies

This broader scope is a design advantage, not a universal quality claim. Each module still requires task-specific baselines, negative controls, representative data and negative-result reporting.
