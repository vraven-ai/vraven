# Internal anatomy, wiring and unit fields

These three DarkSide analyses exist to show what the network is doing *inside*,
using measured tensors rather than decorative neural-net art.

## `inspect_internal_anatomy` / `print_internal_anatomy`

For each selected `nn.Linear` with weight `W`, bias `b` and captured input `a`:

```text
z = W a + b
c = a ⊙ ∂||z||²/∂a
```

The text dump lists top input units by `|c|` and top output units by `|z|`,
together with column/row weight energies. The figure is a dual bar chart of
those ranked units.

**Claim type:** local measured linear anatomy on one live sample (batch-mean).

## `build_circuit_wiring`

Requires a prior `discover_decision_circuitry` result. Nodes are retained
(optionally removed) unit groups. Between consecutive Linear layers in that
core, edge mass is:

```text
mass(G → H) = Σ_{i∈G, j∈H} |W_ji| · |a_i|
```

where `a` is the captured input activation to the downstream Linear module.

**Claim type:** measured weight×activation coupling of an intervention core.
Not a unique causal certificate.

## `map_class_conditional_unit_field`

Builds a dense layer×unit matrix of Hedges’ *g* between two labelled cohorts
(from `TraceDataset.unit_influences`). The figure is a signed heatmap.

**Claim type:** cohort-level observational unit field.

## Suggested notebook order

1. `discover_decision_circuitry` → sparse core
2. `build_circuit_wiring` → how that core is coupled by `W·a`
3. `inspect_internal_anatomy` → printable insides of each Linear block
4. `map_class_conditional_unit_field` → how unit fields differ by class
