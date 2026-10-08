# VRAVEN DarkSide feature catalogue

These modules inspect internal mechanisms that are not represented by a single SHAP or LIME feature vector. They should be selected according to the scientific question rather than run automatically as one large dashboard.

| Module | Scientific question | Required evidence | Typical cost | Recommended users |
|---|---|---|---|---|
| Decision circuitry | What is the smallest tested unit-group core that preserves this prediction within ε? | Live model and intervention budget | High | Mechanistic research, security audit |
| Shortcut detection | Does a narrow, brittle input group dominate the target score? | Declared input groups and domain knowledge | Medium | Malware, medical and bias audit |
| Prediction depth | At what layer does class-decodable information become stable? | Labelled reference traces | Low | Shortcut localisation, training |
| Class-conditional internals | Which latent units or communities flip between two classes? | Labelled cohort traces | Medium | Class audit, malware analysis |
| Representation drift | Where do checkpoint or domain representations reorganise? | Exactly aligned sample traces | Medium | Training observer, temporal drift |
| Failure anatomy | Where do false positives or false negatives diverge from correct predictions? | Labelled outcome cohorts | Medium | Model assurance, SOC analytics |
| Stability geometry | Which internal path collapses first under controlled perturbation? | Paired clean and perturbed traces | Medium | Robustness and evasion research |
| Method disagreement | Where do gradient, LRP, SHAP and causal methods rank or sign evidence differently? | Aligned explanation vectors | Low | XAI validation |
| Unused capacity | Which units are dead, dormant-sensitive or active but low impact? | Dataset-level unit statistics | Low | Model engineering |
| Adversarial rerouting | How does an evasion or counterfactual redirect the evidence path? | Paired clean and altered traces | Medium | Cybersecurity, adversarial ML |
| Cost–evidence trade-off | Which layers cost time but add little measured evidence? | Timed traces and influence summaries | Low | Inference optimisation |
| Circuit multiplicity | Do repeated valid searches recover one stable core or several alternative circuits? | Aligned binary circuit masks from repeated searches | Low | Mechanistic research, robustness audit |
| Causal synergy | Which groups cooperate, compensate or make each other redundant? | Pairwise interventions | Very high | Mechanistic research |
| Activation mediation | Where does clean state repair a corrupted execution? | Clean/corrupt pair and patchable layers | High | Causal localisation |
| Topological shift | Does the connectivity or loop structure of a representation change? | Aligned cohort representations | Medium to high | Geometry research |
| Explanation uncertainty | Which signs, ranks and top-k memberships are unstable across repeats? | Repeated aligned explanations | Medium | XAI assurance |
| Mechanism transfer | Does an intervention-effect circuit persist across models or checkpoints? | Aligned effect vectors | Medium | Model comparison |
| High-confidence OOD | Is a confident output produced through an atypical internal mechanism? | Reference and query fingerprints | Low | Security monitoring, OOD screening |

## Selection rule

Use intervention modules for causal questions, repeated-circuit analysis for non-uniqueness questions, and cohort geometry modules for distribution questions. Do not infer causality from probes, CKA, attention, activation magnitude or visual separation. Use `recommend_features("security")`, `recommend_features("training")`, `recommend_features("model_engineering")`, `recommend_features("research")` or `recommend_features("audit")` to obtain a focused plan.
