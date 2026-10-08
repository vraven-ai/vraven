# VRAVEN Developer Guide

## First commands

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

Run the local demonstration:

```bash
PYTHONPATH=src python -m vraven.cli demo --mode gradient --report-format html --output vraven_demo.html
```

## Development sequence

1. Read `docs/RESEARCH_METHOD.md`.
2. Read `docs/VISUAL_SYSTEM.md` before changing any visual.
3. Confirm whether the requested feature is implemented or planned in `docs/IMPLEMENTATION_STATUS.md`.
4. Add or change the trace schema before introducing a metric that requires new data.
5. Calculate metrics outside the visualisation layer.
6. Add both render modes.
7. Add mathematical tests and failure tests.
8. Run the complete suite.
9. Update the implementation status and changelog.

## Source responsibilities

### `adapters`

Framework model invocation, output extraction, device following and conversion into the common trace.

### `capture`

Scoped hooks, module selection and on device tensor reduction.

### `core`

Configuration, error types, output interpretation, trace schema, metadata and orchestration.

### `metrics`

Defined numerical measurements. Visual code must not create new metrics.

### `reasoning`

Fingerprints, pathways, projections, communities and graph objects built from trace fields.

### `causal`

Temporary interventions and constrained counterfactual routines.

### `visualisation`

Mapping validated values to marks, axes, colour and interaction. It does not decide what the values mean.

### `report`

Public figure methods, trace persistence and explicit HTML composition.

## Adding a visualisation

A new visual must define a `VisualContract` containing:

* A stable visual identifier.
* One research question.
* Required metrics.
* Data domain.
* Every visual encoding.
* Limitations.

The implementation must return one standalone Plotly figure and call `finalise_figure`. It must support both official modes with identical scientific arrays.

## Adding a metric

Document:

1. Symbolic definition.
2. Input domain.
3. Output range.
4. Interpretation.
5. Assumptions.
6. Failure cases.
7. Validation plan.

Then add unit tests using analytically known values.

## Build validation

```bash
rm -rf build dist src/vraven.egg-info
python -m pip wheel . --no-deps --no-build-isolation -w dist
python -m pip install --force-reinstall --no-deps dist/vraven-*.whl
```

## Creating a clean source archive

```bash
python scripts/create_release.py --output-dir release
```

The release script validates runtime source, excludes caches and generated artefacts, writes a deterministic ZIP file, embeds a per-file SHA-256 manifest and writes an external archive checksum.

## Release checklist

* All tests pass.
* No source file is empty.
* No placeholder code or invented data remains.
* Model parameters and training state are preserved.
* Both visual modes render the same measured values.
* Static export errors are clear when Kaleido or Chrome is missing.
* README, implementation status and changelog match the code.
