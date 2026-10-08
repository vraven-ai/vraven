# VRAVEN Scientific Visual System

This document defines the mandatory visual language for VRAVEN. It is not a collection of optional styling suggestions. Every VRAVEN visualisation must follow this standard unless a documented scientific reason requires an exception.

VRAVEN visualisations must be visually distinctive, technically accurate and suitable for both interactive investigation and academic publication. A figure must never look impressive by inventing structure that is not present in the measured data.

## 1. Standalone first design

Each visualisation must answer one clear scientific question and must be usable as a standalone figure.

Examples include:

* How did evidence move through model depth?
* Where did the representation become class separable?
* Which latent communities supported or opposed the selected prediction?
* Where did uncertainty increase?
* How did the explanation change under a controlled counterfactual?

The default API must return one independent figure:

```python
figure = report.decision_flow(mode="publication_light")
figure.write_image("decision_flow.svg")
```

A coordinated figure may contain several tightly related panels when they are required to answer the same question. A large dashboard that combines unrelated analyses must be an optional composition layer and must not be the default output of any visual method.

## 2. Mandatory render modes

Every core visualisation must support the following two modes from the same underlying data and visual grammar.

### 2.1 `interactive_dark`

Purpose:

* Local exploration
* VRAVEN Studio
* Browser based reports
* Layer, head, concept and target selection
* Hover details, zooming, filtering and animation where scientifically useful

Visual character:

* Deep navy background
* Clear panel boundaries
* High contrast data marks
* Controlled glow only for selected or active pathways
* Smooth but restrained transitions
* Rich hover metadata
* No visual noise or decorative particles

Base colours:

| Role | Value |
|---|---|
| Canvas | `#030A12` |
| Panel | `#07111F` |
| Raised panel | `#0B1728` |
| Border | `#1C344D` |
| Grid | `rgba(28, 52, 77, 0.55)` |
| Primary text | `#F1F5F9` |
| Secondary text | `#94A3B8` |

### 2.2 `publication_light`

Purpose:

* Journal manuscripts
* Conference papers
* Theses
* Technical reports
* Vector export
* Printed review copies

Visual character:

* White background
* No glow
* No card style decoration
* Restrained saturation
* High data to ink ratio
* Clear axes and readable annotation
* Legible in colour and greyscale
* Suitable for single column and double column layouts

Base colours:

| Role | Value |
|---|---|
| Canvas | `#FFFFFF` |
| Panel | `#FFFFFF` |
| Subtle panel | `#F8FAFC` |
| Border | `#CBD5E1` |
| Grid | `#DCE3EC` |
| Primary text | `#111827` |
| Secondary text | `#475569` |

Interactive and publication modes must not use different scientific meanings. Only contrast, background, annotation density and interaction behaviour may change.

## 3. Semantic colour grammar

A colour must carry the same meaning across all VRAVEN figures.

| Meaning | Interactive dark | Publication light |
|---|---|---|
| Supporting evidence | `#168BFF` | `#0057B8` |
| Dominant supporting evidence | `#00D4FF` | `#0284C7` |
| Opposing evidence | `#FF2D55` | `#B42318` |
| Strong contradiction | `#E83E8C` | `#9D174D` |
| Uncertainty | `#FF8A00` | `#C2410C` |
| Stable reasoning | `#32D74B` | `#15803D` |
| Latent community | `#8B5CF6` | `#6D28D9` |
| Neutral or inactive | `#94A3B8` | `#64748B` |
| Unknown or unresolved | `#94A3B8` | `#475569` |
| Selected pathway | `#00D4FF` | `#0E7490` |

Colour alone must not encode critical meaning. Supporting and opposing signals must also differ through position, sign, direction, line style, marker shape or annotation.

## 4. Magnitude and sign

Magnitude may control:

* Width
* Area
* Opacity
* Height
* Saturation
* Line weight

The mapping must be stated in the figure metadata or caption.

When bubble markers are used, numerical magnitude must control marker area rather than marker radius.

Signed contribution views must use a diverging scale centred exactly on zero. Percentile clipping may be used to prevent extreme values from hiding structure, but the clipping rule must be disclosed.

## 5. Typography and layout

Use a modern sans serif family with broad system availability. Preferred order:

```text
Inter, IBM Plex Sans, Source Sans 3, Arial, sans-serif
```

Use a monospaced family only for byte offsets, token identifiers, addresses, hashes, opcodes and code fragments.

Publication targets:

* Figure title: 10 to 12 pt
* Axis and legend text: 8 to 10 pt
* Notes and metadata: not below 7 pt
* Major line width: approximately 0.8 to 1.2 pt
* Minor grid width: approximately 0.4 to 0.7 pt

Interactive mode may use slightly larger text but must remain compact and information dense.

## 6. VRAVEN visual identity

VRAVEN must not rely on default Plotly, Matplotlib or browser styling. All figures must pass through the VRAVEN theme engine.

The visual identity should be recognised through:

* Precise layered geometry
* Fine evidence strands rather than generic bars where flow is central
* Strong semantic contrast
* Compact numerical annotation
* Clear progression through model depth
* Consistent edge and node semantics
* Subtle raven inspired terminology without decorative bird imagery

Distinctive VRAVEN figure families include:

* **VRAVEN Decision Flow** for evidence movement across depth
* **VRAVEN Latent Flight** for aligned representation evolution
* **VRAVEN Spectral Trace** for signed layer and community activity
* **VRAVEN Reasoning DNA** for deterministic explanation fingerprints
* **VRAVEN Counterfactual Delta** for minimal changes and their effects
* **VRAVEN Conflict Field** for internal disagreement and uncertainty

These names describe scientific outputs. They must not be used to disguise weak or undefined measurements.

## 7. Scientific integrity safeguards

### 7.1 Representation landscapes

Do not compare independently fitted t-SNE projections across layers as though point movement were meaningful. Independent stochastic embeddings may rotate, reflect or distort without representing real model change.

Use one of the following:

* A shared projection fitted to concatenated representations
* A transform capable embedding fitted once and applied consistently
* A deterministic linear projection
* An explicitly documented alignment method such as orthogonal Procrustes after separate projection

The seed, projection method, distance metric and alignment method must be stored in report metadata.

### 7.2 Attention views

Attention weights may be labelled as attention weights. They must not be labelled as feature importance or causal importance unless a separate validation method supports that interpretation.

### 7.3 Confidence trajectories

Intermediate class confidence may only be shown when an explicit probe or valid intermediate classifier produces comparable probabilities. Raw hidden activations must not be silently converted into confidence.

### 7.4 Uncertainty

Predictive entropy requires a valid probability distribution. Activation variance is not predictive uncertainty. Epistemic, aleatoric, ensemble, Monte Carlo and explanation uncertainty must be named separately.

### 7.5 Flow diagrams

The width of every flow strand must correspond to a defined value. Decorative Bezier curves with arbitrary width are prohibited.

### 7.6 Causal claims

Gradient, attention and activation views are observational. Causal terminology is permitted only for intervention or ablation results.

## 8. Interaction standard

Interactive dark figures should support relevant subsets of:

* Hover values with metric name, units and definition
* Layer selection
* Target class selection
* Head or community selection
* Highlight and isolate
* Zoom and pan
* Lasso selection where point sets are present
* Toggle between absolute and signed contribution
* Toggle between raw and normalised values
* Animation through depth or time where order is real
* Export of the currently selected view

Animation must represent model depth, time, intervention sequence or another real ordering. Decorative looping animation is prohibited.

## 9. Publication export standard

Core figures must support:

* SVG
* PDF
* PNG
* HTML for interactive mode

TIFF may be added where required by a target venue.

Recommended presets:

| Preset | Width | Typical purpose |
|---|---:|---|
| `single_column` | 89 mm | One column paper figure |
| `double_column` | 183 mm | Full page width figure |
| `square` | 140 mm | Matrix or landscape figure |
| `wide` | 183 mm | Flow or temporal figure |

Raster exports should support 300, 600 and 1200 DPI. Vector export is preferred for lines, text and markers.

Every exported figure must carry enough metadata to reconstruct it, including metric version, mode, target, model identifier, input identifier, random seed and VRAVEN version.

Composed Proof reports must also expose, for every included figure, its evidence tier, scientific
question, interpretation, metric definitions, visual mappings, normalisation, limitations,
copyable journal caption and exact plotted-data export. Figure-specific key-result cards may only
repeat deterministic functions of the plotted arrays; they must not introduce an undocumented
score.

## 10. Figure composition rules

A VRAVEN figure should normally contain:

1. A precise title describing the question
2. A short subtitle or caption describing the metric
3. The main data region
4. A semantic legend
5. Optional method and normalisation note
6. Optional warning when interpretation is limited

Do not include large decorative headings inside journal figures. Do not place long paragraphs inside the plot area.

## 11. Quality gate

Before a visualisation is accepted, the implementation agent must confirm:

* The scientific question is explicit
* Every visual variable has a defined data mapping
* Both render modes are implemented
* The figure works independently of a dashboard
* The same semantic colours are used consistently
* Publication mode uses a white background
* Dark mode is interactive and readable
* Colour blind and greyscale interpretation has been considered
* No unsupported causal or confidence claim is made
* Randomness is controlled and recorded
* The visual exports without clipped labels
* Tests cover theme selection and critical data mappings

Detailed per figure contracts are defined in `docs/VISUALISATION_CATALOGUE.md`.
