# Operator Semantics Contract

An operator-semantics entry lifts a low-level PyTorch or custom operation into the VRAVEN Mechanism Dialect. It does not label the learned feature carried by the operation.

## Required fields

```json
{
  "patterns": ["conv1d", "convolution"],
  "role": "detect",
  "family": "convolution",
  "description": "Local receptive-field transformation",
  "intervention": true,
  "feature_axis": "channel"
}
```

## Permitted roles

```text
input, transform, detect, gate, suppress, normalise, pool, merge,
route, accumulate, compare, store, retrieve, attend, repeat, readout,
shape, random, output, unknown
```

## Custom-operation requirements

A strong custom-operation integration should also document:

* input and output tensor contracts;
* mutation and alias behaviour;
* fake-tensor or abstract behaviour;
* differentiation support;
* vectorisation support;
* valid intervention positions;
* replacement or erasure baseline;
* semantic limitations.

An unknown operation remains visible as unresolved computation. VRAVEN must not silently classify it using a vague substring match when the match is ambiguous.
