# VRAVEN Visualisation Catalogue

This catalogue defines the intended scientific question, data requirements, visual encoding and failure conditions for each core VRAVEN visualisation. Implementation agents must follow these contracts rather than selecting a chart solely because it looks attractive.

Each method must return an independent figure. Optional report dashboards may compose several completed figures later.

## 1. VRAVEN Decision Flow

### Scientific question

How did supporting, opposing, uncertain and neutral evidence change across model depth before the final prediction?

### Required data

* Ordered model stages or layers
* Defined signed contribution or validated information flow value
* Target prediction
* Optional latent communities or feature groups

### Visual encoding

* Horizontal depth progression
* Strands or bands connect corresponding communities across stages
* Strand width represents absolute contribution magnitude
* Strand colour represents semantic role
* Vertical position separates supporting and opposing flow where possible
* Final output node shows the target and strongest alternative

### Interactive dark mode

* Hover reveals source, destination, signed value, normalisation and layer
* Click isolates a pathway
* Slider may move through depth if the model contains many layers
* Selected pathway may receive a restrained cyan emphasis

### Publication light mode

* White background
* Reduced number of visible pathways through principled top k selection
* No glow
* Direct labels for dominant pathways
* Caption states the omitted mass and top k rule

### Invalid uses

* Arbitrary line thickness
* Labelling activation magnitude as causal flow
* Mixing values from incomparable scales without normalisation

## 2. VRAVEN Latent Flight

### Scientific question

How do sample representations separate, merge or change structure through model depth?

### Required data

* Sample level representations from selected layers
* Stable sample identifiers
* Class or cohort labels where available
* A projection method and fixed random seed

### Visual encoding

* Small multiples with one layer per panel
* Common axes and coordinates when possible
* Same sample uses the same marker identity in all panels
* Optional centroid and confidence ellipse
* Arrows between panels indicate model order, not literal Euclidean motion unless coordinates are aligned

### Interactive dark mode

* Hover identifies the sample, class, prediction and local metrics
* Brushing a group highlights the same samples in every panel
* Layer selector may replace small multiples for deep models

### Publication light mode

* White panels with identical axis limits
* Minimal grid
* Class colours supported by marker shape
* Projection method, seed and alignment disclosed

### Scientific safeguard

t-SNE is supported when seed, perplexity and alignment metadata are disclosed. Prefer Procrustes-aligned per-layer t-SNE (`method="tsne", align=True`) or shared PCA for layer sequences. Do not interpret unaligned panel distances as literal Euclidean trajectories.

## 3. VRAVEN Spectral Trace

### Scientific question

Where are signed contributions concentrated across layers, tokens, positions, channels or latent communities?

### Required data

* Two dimensional indexed contribution matrix
* Defined sign and scale
* Axis labels and ordering

### Visual encoding

* Heatmap centred on zero
* Supporting and opposing evidence use a diverging scale
* Neutral values remain visually quiet
* Optional marginal summaries show total positive and negative mass

### Interactive dark mode

* Hover gives exact value and identifiers
* Range slider and layer filtering
* Toggle raw, normalised and percentile clipped values

### Publication light mode

* White background around the heatmap
* Clearly labelled colour bar with zero marked
* Vector text and rasterised heatmap when needed for file size
* Clipping rule disclosed

### Invalid uses

* Sequential colour map for signed values
* An unlabelled arbitrary normalisation
* Comparing matrices with different scales without a shared legend

## 4. VRAVEN Neuron Impact Field

### Scientific question

Which neurons, channels, heads or units show the strongest signed target specific influence within a selected layer?

### Required data

* Unit index
* Signed influence metric
* Optional activation, gradient, uncertainty or community membership

### Visual encoding

* Signed vertical position
* Unit index or grouped order on the horizontal axis
* Bubble area represents absolute influence
* Colour represents sign
* Shape may represent community or unit type

### Interactive dark mode

* Hover gives activation, gradient, influence and community
* Select and inspect top units
* Switch between neuron, channel and community aggregation

### Publication light mode

* Prefer a beeswarm or compact signed dot plot
* Top units may be labelled directly
* Avoid excessive overplotting through density aware jitter

### Invalid uses

* Bubble radius directly proportional to magnitude
* Ranking by activation while labelling the result as target contribution

## 5. VRAVEN Attention Structure

### Scientific question

What attention patterns were present, and how did they vary by head and layer?

### Required data

* Attention weight matrix
* Layer and head identifiers
* Token, patch or position labels
* Optional independently validated head relevance metric

### Visual encoding

* Matrix heatmap for selected head
* Optional separate head summary figure
* Diagonal, local and long range structure remain visible
* Importance bars only when a defined relevance metric exists

### Interactive dark mode

* Layer and head selector
* Hover displays query, key and weight
* Click a token to highlight incoming and outgoing attention

### Publication light mode

* One selected head or a carefully justified grid
* Clear token truncation policy
* Attention and relevance are labelled separately

### Invalid uses

* Calling raw attention weights explanation importance
* Combining heads without stating the aggregation

## 6. VRAVEN Consistency Matrix

### Scientific question

How consistently do explanation methods, repeated runs or perturbations rank and sign the same evidence?

### Required data

* Comparable explanation vectors
* Named methods or runs
* Defined agreement metric

### Visual encoding

* Symmetric matrix for pairwise agreement
* Diagonal fixed at the metric maximum where appropriate
* Separate visual or annotation for sign agreement
* Confidence intervals where repeated estimates exist

### Interactive dark mode

* Hover gives metric, sample count and uncertainty
* Toggle rank, sign, overlap and stability metrics

### Publication light mode

* Direct values in cells only when legible
* Colour scale selected for the actual range
* Method names fully expanded in the caption

### Invalid uses

* Averaging incomparable agreement metrics into one unexplained score
* Treating high agreement as proof of correctness

## 7. VRAVEN Reasoning DNA

### Scientific question

Can the internal explanation trajectory be represented as a deterministic fingerprint for comparison, clustering or drift analysis?

### Required data

* Fixed order trace vector
* Metric names and normalisation
* Deterministic encoding procedure

### Visual encoding

* Layer ordered barcode, circular ring or feather like radial structure
* Every segment maps to a defined metric and layer
* Segment length or intensity represents normalised magnitude
* Sign remains visible

### Interactive dark mode

* Hover opens the layer metric record
* Compare two fingerprints with overlay or difference view
* Search for nearest reasoning patterns

### Publication light mode

* Prefer a linear barcode or two ring comparison
* Keep the legend close to the figure
* State fingerprint version and dimensionality

### Invalid uses

* Random decorative pattern
* Changing segment order between samples
* Claiming semantic similarity without validation

## 8. VRAVEN Counterfactual Delta

### Scientific question

What minimal input or internal changes altered the prediction, and which changes had the greatest validated effect?

### Required data

* Original input or representation
* Counterfactual input or intervention
* Change set
* Original and resulting predictions
* Defined distance or cost

### Visual encoding

* Paired original and counterfactual view
* Changed regions are highlighted
* Difference magnitude and direction are explicit
* Prediction change is separated from risk interpretation

### Interactive dark mode

* Toggle semantic, raw, byte, token or feature difference
* Hover reveals exact modification and effect
* Step through accepted changes

### Publication light mode

* Compact paired view with changed regions numbered
* A small table may list the top changes
* No dashboard gauges

### Invalid uses

* Calling any random perturbation a counterfactual
* Hiding large or unrealistic changes
* Omitting the optimisation constraints

## 9. VRAVEN Conflict Field

### Scientific question

Where does internal evidence disagree, and how does that disagreement relate to predictive or explanation uncertainty?

### Required data

* Defined disagreement metric
* Layer, community, head or token index
* Optional uncertainty estimate with method name

### Visual encoding

* Layer by structure matrix or signed field
* Amber indicates uncertainty
* Red indicates opposing evidence, not uncertainty by default
* Stable low conflict regions use teal only when stability is measured

### Interactive dark mode

* Hover separates uncertainty, disagreement and contradiction
* Filters by metric type
* Links selected region to other figures

### Publication light mode

* Clear multi panel decomposition if several uncertainty types are shown
* Avoid circular gauges unless the gauge itself is under study
* Prefer calibrated curves, intervals or matrices

### Invalid uses

* Combining predictive uncertainty and explanation uncertainty without separation
* Labelling activation variance as uncertainty

## 10. VRAVEN Evidence Trajectory

### Scientific question

At what stage did the model begin to favour the final class, and where did the preference stabilise or reverse?

### Required data

* Comparable target scores from explicit intermediate probes or valid architecture outputs
* Layer order
* Calibration details where probabilities are shown

### Visual encoding

* Layer on the horizontal axis
* Target score or logit on the vertical axis
* Strongest alternative shown separately
* Confidence or uncertainty interval where available
* Decision transition point marked only after a formal rule is satisfied

### Interactive dark mode

* Hover reveals target and alternative scores
* Toggle probability, logit and margin
* Select a sample or cohort

### Publication light mode

* Direct label the target and top alternative
* Use a compact line figure
* Caption defines the transition and stability rule

### Invalid uses

* Projecting arbitrary hidden states into probabilities without a documented probe
* Calling an uncalibrated logit a probability

## 11. VRAVEN Community Graph

### Scientific question

Which latent communities interact strongly, and which communities dominate the selected decision pathway?

### Required data

* Community definitions
* Node attributes
* Defined edge weights
* Layer or stage membership

### Visual encoding

* Layered graph layout preferred over unconstrained force layout
* Node area represents significance
* Edge width represents defined interaction
* Colour represents role, not arbitrary cluster identity alone

### Interactive dark mode

* Expand and collapse communities
* Hover gives members, layer, score and uncertainty
* Isolate a path or target

### Publication light mode

* Prune by documented threshold
* Label only dominant communities
* Provide the omitted node and edge count

### Invalid uses

* Force directed layout presented as model topology
* Edge construction without a defined interaction metric

## 12. VRAVEN Model Evolution

### Scientific question

How did internal representation, stability or pathway concentration change across training checkpoints?

### Required data

* Comparable checkpoints
* Fixed evaluation samples
* Same capture and normalisation procedure

### Visual encoding

* Checkpoint order is explicit
* Small multiples or temporal line plots
* Shared scales across checkpoints
* Model performance shown separately from explanation behaviour

### Interactive dark mode

* Scrub through checkpoints
* Highlight one sample across training
* Compare two checkpoints directly

### Publication light mode

* Select a small, justified set of checkpoints
* State all checkpoint selection criteria
* Use common axes and normalisation

### Invalid uses

* Comparing different sample sets without disclosure
* Rescaling each checkpoint independently and implying absolute change

## 13. VRAVEN Activation Energy

### Scientific question

Where is reduced activation magnitude concentrated across captured layers?

### Required data

* Captured layer order.
* Mean activation energy or total activation energy.
* Layer names and types.

### Visual encoding

* Horizontal position represents execution order.
* Bar height represents the documented energy quantity.
* Colour identifies activation magnitude without implying target-specific importance.

### Scientific safeguard

Activation energy is a descriptive magnitude. It must not be labelled as causal importance or class-specific contribution.

## 14. VRAVEN Causal Pathway

### Scientific question

Which explicitly intervened layer outputs most changed the selected target score?

### Required data

* Baseline target score.
* Intervened target score.
* Intervention strategy.
* Layer name and score difference.

### Visual encoding

* Layer on the categorical axis.
* Signed score difference on the quantitative axis.
* Supporting colour for an intervention that reduces target support.
* Opposing colour for an intervention that increases target support.

### Scientific safeguard

A single output ablation can create out-of-distribution internal states. The visual reports intervention sensitivity and must not claim a complete causal mechanism.

## 15. VRAVEN Input Attribution

### Scientific question

Where is signed input-times-target-gradient concentrated in the first supported continuous input tensor?

### Required data

* Input values.
* Input gradient.
* Signed input-times-gradient product.
* Deterministic reduction rule when the input is truncated.

### Visual encoding

* Input position or compressed feature position on the horizontal axis.
* Signed attribution on the vertical axis.
* Diverging colour encodes sign.

### Scientific safeguard

The result is a local first-order signal. It does not replace domain-valid token, byte, image or structured-feature attribution.

## 16. VRAVEN Reasoning Graph

### Scientific question

How are captured layer summaries ordered, and where is reduced target-specific influence concentrated?

### Required data

* Captured execution order.
* Layer summary values.
* Optional signed target-specific influence.

### Visual encoding

* A deterministic layered or sequential layout.
* Nodes represent captured layer executions.
* Node size or colour represents a documented summary.
* Edges represent execution adjacency only unless another interaction metric is explicitly calculated.

### Scientific safeguard

The graph is an execution-summary graph. It must not be presented as exact neuron-to-neuron topology.

## 17. VRAVEN Reasoning Similarity Matrix and Landscape

### Scientific question

Which independently explained samples have similar reduced internal traces, and how are their fingerprints arranged in one shared low-dimensional space?

### Required data

* One fixed-width Reasoning DNA fingerprint per sample.
* Stable sample identifiers.
* Optional prediction or label groups.

### Visual encoding

* Similarity matrix cells represent cosine similarity.
* Landscape points represent samples in one deterministic PCA basis.
* Colour and marker shape represent a declared group.

### Scientific safeguard

Fingerprint similarity reflects the selected trace metrics and compression procedure. Two-dimensional proximity does not establish semantic identity, causal equivalence or a discovered human concept.

## 18. VRAVEN Matched Model Comparison

### Scientific question

For identical ordered samples, where do two models agree in output and align or diverge in reduced internal trace fingerprints?

### Required data

* Matched sample identifiers.
* Paired model fingerprints.
* Predicted classes.
* Prediction confidence where available.

### Visual encoding

* Fingerprint cosine similarity per sample.
* Marker colour and shape distinguish output agreement.
* A separate signed panel shows right-model confidence minus left-model confidence.

### Scientific safeguard

Fingerprint compression permits descriptive comparison across different layer counts. It does not prove that two different architectures perform equivalent internal computation.

## 19. VRAVEN Dataset Reasoning Outlier View

### Scientific question

Which bounded dataset samples follow reasoning fingerprints that are unusual relative to their nearest neighbours?

### Required data

* Per-sample fingerprints.
* Cosine similarities.
* A documented neighbourhood size.

### Visual encoding

* Outlier score is `1 - mean nearest-neighbour similarity`.
* Sample identifiers remain available in hover or labels.
* Cohort grouping is separate from the outlier score.

### Scientific safeguard

An unusual reasoning fingerprint is not automatically an adversarial sample, backdoor, drift event or error. It is a candidate for further investigation.

## 20. VRAVEN Mechanism Concordance

### Scientific question

Does local activation–gradient evidence predict the direction and relative magnitude of
measured zero-output ablation effects across unambiguously matched layers?

### Required data

* Signed layer activation–gradient interactions.
* Target-score drops from explicit `strategy="zero"` layer interventions.
* An unambiguous module-to-execution mapping.

### Visual encoding

* Horizontal position is the signed L1-normalised first-order interaction.
* Vertical position is the signed L1-normalised zero-ablation score drop.
* Each point is one matched layer.
* Shape and colour jointly distinguish sign agreement from contradiction.
* The identity diagonal represents perfect agreement in signed relative allocation.

### Interactive dark mode

Hover exposes the layer name, raw values, normalised values and sign agreement. Exact rank,
sign, cosine, discrepancy and permutation diagnostics remain available in figure metadata.

### Publication light mode

The same coordinates and labels appear on a white background with no glow. Sign disagreement
uses both a cross marker and opposing-evidence colour.

### Scientific safeguard

Separate L1 normalisation tests relative allocation rather than absolute calibration. Zero
ablation may be out-of-distribution, and agreement does not establish a unique causal mechanism.
Reused modules with ambiguous executions are excluded and disclosed.

## 21. Optional dashboard composition

VRAVEN Studio may combine several figures into an interactive dashboard. This is a secondary interface.

The dashboard must:

* Reuse completed standalone figures
* Preserve each figure's metric and legend
* Avoid shrinking figures below readable size
* Synchronise selection through explicit identifiers
* Never be the only way to export a figure

The large dark reference image in `docs/design_reference/` is a quality and density reference only. Agents must not reproduce its data, labels or exact layout. VRAVEN should use the same level of polish while producing original figures grounded in VRAVEN metrics.
