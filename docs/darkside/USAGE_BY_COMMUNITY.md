# Usage by community

VRAVEN DarkSide is intentionally modular. A credible audit normally uses a small set of analyses selected for one question, rather than every method in one dashboard.

## Cybersecurity and malware detection

Start with:

1. Shortcut reliance for brittle static features, metadata, file-source indicators or one dominant LOLBIN flag.
2. Adversarial rerouting for paired original and evasive samples.
3. Failure anatomy for false positives and false negatives.
4. High-confidence internal anomaly screening for confident predictions with atypical fingerprints.
5. Representation drift when malware families or benign software change over time.

Use decision circuitry, circuit multiplicity, causal synergy and activation mediation only on a bounded set of high-risk samples because they require repeated interventions.

## Training and model development

Start with representation drift on a fixed evaluation panel at selected checkpoints. Add unused-capacity analysis to find dead or dormant-sensitive units, prediction depth to identify unusually early decisions, and cost–evidence analysis when inference efficiency matters. Use circuit multiplicity when the apparent internal mechanism changes across seeds, checkpoints or intervention baselines.

Do not compare checkpoints with different sample sets, different layer mappings or independently rescaled axes.

## Explainability and mechanistic research

Use cross-method disagreement before selecting a preferred attribution method. Add explanation uncertainty over retraining seeds, background resamples or controlled perturbations. For intervention claims, use decision circuitry, activation mediation and causal synergy with declared baselines and budgets. Use circuit multiplicity to report whether several alternative sufficient circuits were recovered.

Report unfavourable as well as favourable results. A method that disagrees with another method is not automatically wrong.

## Model engineering and efficiency

Use unused capacity, cost–evidence trade-off and stability geometry. Add mechanism transfer when comparing pruned, quantised or distilled versions. A high-cost layer with low local evidence is a profiling candidate, not proof that it can be deleted.

## Assurance and governance

Use failure anatomy, shortcut audits, explanation uncertainty and high-confidence internal anomaly screening. Preserve the model version, trace schema, sample identifiers, random seeds, grouping rule, baseline, intervention budget and omitted layers. A clean visual report must not hide weak sample size, missing cohorts or untested causal assumptions.

## Recommended API

```python
from vraven.darkside import recommend_features

for item in recommend_features("malware"):
    print(item.name, item.cost, item.claim_type)
```

Supported profiles are:

```text
security
malware
training
model_engineering
research
audit
```
