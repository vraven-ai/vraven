# Validation protocol against attribution baselines

VRAVEN DarkSide and SHAP or LIME do not always produce the same analytical object. Validation must therefore separate common-surface comparisons from additional internal-mechanism tasks.

## 1. Common input-attribution surface

Compare methods only when all outputs refer to the same input coordinates, target and baseline.

Recommended baselines:

```text
SHAP
LIME
Integrated Gradients
DeepLIFT
LRP
Input × gradient
Grad-CAM for suitable image models
```

Recommended measurements:

```text
Runtime and memory
Deletion and insertion AUC
Infidelity
Sensitivity or stability under controlled perturbations
Sign agreement
Rank correlation
Top-k overlap
Retraining or background-resampling multiplicity
```

Do not score an internal layer circuit as though it were a SHAP feature vector.

## 2. Internal-mechanism tasks

For DarkSide capabilities without a direct SHAP or LIME equivalent, use intervention and cohort-based validation.

| DarkSide analysis | Validation question |
|---|---|
| Decision circuitry | Does the retained core preserve the prediction and target score under the declared ablation? |
| Circuit multiplicity | Are stable components recovered across seeds, samples, baselines or resampled concept sets? |
| Shortcut reliance | Does the effect persist on balanced or counterfactual data, and does worst-group performance improve when the cue is removed? |
| Activation mediation | Does clean-state patching recover the corrupted score, and does reverse noising damage the clean score? |
| Causal synergy | Does the measured joint effect exceed or fall below the sum of individual effects? |
| Failure anatomy | Can the identified layer signature distinguish held-out failure cohorts without leaking labels? |
| Representation drift | Does the drift marker track known checkpoint, temporal or domain changes on an aligned audit panel? |
| Stability geometry | Does the first collapse point reproduce under repeated controlled perturbations? |
| Adversarial rerouting | Is rerouting larger for successful evasions than for matched benign transformations of similar input distance? |
| High-confidence anomaly | Does the reference percentile separate held-out unusual mechanisms without excessive false alarms? |

## 3. Negative controls

Every study should include relevant negative controls:

```text
Random unit groups matched by size
Random feature groups matched by cardinality
Label permutation
Sample-ID permutation for paired metrics, expected to break alignment
Random circuit masks with matched sparsity
Benign transformations matched to adversarial input distance
Alternative intervention baselines
Multiple discovery orders or seeds
```

## 4. Reporting requirements

Report:

```text
Model and checkpoint hash
Exact preprocessing
Target score definition
Candidate layers and grouping
Intervention baseline
Number of forward and backward passes
Sample and cohort selection
Random seeds
All metric versions
Omitted mass or omitted layers
Positive and negative findings
Hardware and software environment
```

## 5. Claim rule

A broader scope than SHAP or LIME is not the same as universal superiority. The defensible claim is:

> VRAVEN DarkSide exposes internal mechanism, cohort, interaction, drift and reliability questions that are not represented by a standard local feature-attribution vector.

Any stronger performance claim requires the matched experiments described above.
