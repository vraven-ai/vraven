# VRAVEN Publication Evidence Visualisations

VRAVEN's publication visual system makes the inferential status of every figure visible. It is
designed to help an author assemble reviewable evidence; it does not guarantee acceptance by any
journal or turn an exploratory result into a confirmatory result.

## Section anatomy

Every figure in the composed Proof report exposes the same evidence hierarchy:

1. **Scientific question** — the single question registered by the visual contract.
2. **Evidence tier** — predictive, observational, comparative, behavioural, targeted
   interventional, observational–interventional or formal.
3. **Interpretation** — what the displayed estimate supports.
4. **Key measured results** — figure-specific values calculated from the same arrays used by the
   plotted marks.
5. **Methods and claim boundary** — metrics, symbolic definitions, visual encodings,
   normalisation and limitations.
6. **Reproducibility controls** — copyable journal caption and exact plotted-data JSON download.

The evidence panel does not compute a trust score. Missing estimates remain missing, and
unsupported views are omitted rather than populated with placeholders.

## Evidence tiers

| Tier | Permitted interpretation |
|---|---|
| Predictive | Describes the model output distribution or decision margin. |
| Observational | Describes activation, gradient, attribution, representation or structural association. |
| Comparative | Compares registered quantities in a common coordinate system. |
| Behavioural | Measures model responses to declared input perturbations or counterfactual search. |
| Targeted interventional | Measures a declared internal intervention such as zero-output ablation. |
| Observational–interventional | Directly cross-checks an observational proxy against intervention effects. |
| Formal | Proves a declared property over a quantified input set for a supported verifier graph. |

An evidence tier identifies the method, not result quality. For example, an intervention can be
poorly powered or off-manifold, and a formal result can be inconclusive because its bound is loose.

## Quantitative figure findings

The following core figures provide compact data-specific findings in addition to exact
plotted-data export:

| Figure | Findings surfaced in the report |
|---|---|
| Activation Energy | peak layer, peak share and captured depth |
| Decision Flow | dominant net layer and global supporting/opposing mass |
| Decision Sankey | largest local input contribution, named-feature mass and final signed routing |
| Spectral Trace | strongest supporting/opposing compressed units and matrix coverage |
| Neuron Impact | strongest supporting/opposing units and compressed-unit count |
| Neuron Importance | largest contributor, retained absolute mass and selected layer |
| Reasoning DNA | largest robust-scaled coordinate, trace shape and fingerprint width |
| Conflict Field | largest separate diagnostic, predictive uncertainty and available dimensions |
| Evidence Trajectory | registered transition, final net evidence and measured depth |
| Causal Pathway | largest and smallest intervention effects and intervention count |
| Mechanism Concordance | rank concordance, sign agreement and permutation result |
| Input Attribution | strongest position/feature and signed-mass summaries |
| Feature Sequence Heatmap | strongest feature, displayed shape and attribution provenance |
| Reasoning Graph | largest vertical signal, largest energy node and execution coverage |
| Community Graph | largest community, strongest retained similarity and graph coverage |
| Relevance Flow | conservation ratio, target score and rendered path coverage |
| Perturbation Stability | minimum stability for every registered perturbation family |
| Counterfactual Delta | target success and displayed L1/L2 change |
| Attention Structure | peak query-key weight, mean row maximum and selected head |
| Consistency Matrix | median/weakest off-diagonal agreement and method count |

These summaries are descriptive functions of the figure's registered values. They do not replace
cohort confidence intervals, hypothesis tests or preregistered endpoints when those are required.

## Colour and naming corrections

Theme version 3.1 uses the same scientific semantics in both render modes and applies the
project-owner design-reference palette without changing any figure geometry:

* blue and positive vertical position indicate supporting evidence;
* red and negative vertical position indicate opposing evidence;
* amber is reserved for uncertainty;
* teal indicates measured stability;
* purple identifies latent or comparative structure.

Local input-times-gradient and compressed-unit activation-gradient views are labelled as local
attribution or influence rather than global feature importance or layer relevance. A single-layer
ablation is labelled as an intervention effect rather than a complete causal pathway.

## Publication workflow

```python
report = vraven.explain(model, inputs, target=1, mode="causal")

# Standalone journal figure. Publication mode retains its own title by default.
figure = report.mechanism_concordance(mode="publication_light")
report.export(
    "mechanism_concordance",
    "mechanism_concordance.svg",
    mode="publication_light",
    preset="double_column",
)

# Evidence-led HTML containing both render modes, exact data and captions.
report.export_html("vraven_proof.html", theme_toggle=True)

```

SVG or PDF should be preferred for line, point and text figures. Dense heatmaps may be rasterised
at the venue-required resolution while preserving vector labels. Authors must report sample IDs,
cohort definition, target, metric version, normalisation, random seed and all selection rules.

## Validation and limitations

The report exporter is tested for paired render modes, white publication backgrounds, title
retention, scientific metadata, evidence panels and reproducibility controls. Figure-specific
tests protect critical data mappings and sign semantics.

The report cannot supply missing uncertainty. A single-example trace does not acquire a confidence
interval merely because it is presented clearly. Cohort inference, multiplicity control,
independent replication, baseline comparisons and domain-valid interventions remain the
responsibility of the registered experiment.
