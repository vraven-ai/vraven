# VRAVEN Security and Privacy

VRAVEN is local first.

## Defaults

- no telemetry
- no model upload
- no remote inference
- no automatic external API calls
- no persistence of raw tensors unless requested

## Model safety

Treat every model bundle, checkpoint and architecture factory as code from its
publisher. Prefer a live model object or a trusted state dictionary with a
known architecture. Python-pickled modules are disabled by default because
deserialising them may execute code. Only enable trusted-pickle loading for a
file you control and have verified.

A bundle's factory import points to Python code used to construct the model.
Inspect that code before running `vraven analyse` on a bundle from another
person. VRAVEN does not sandbox model execution or architecture imports.

## Sensitive data

Standard traces and reports retain bounded fingerprints and analysis metadata,
not raw input tensors. They may still reveal predictions, class names, model
structure and derived evidence. Review a report before sharing it and avoid
placing identifying information in user-supplied metadata.

## Full tensor handling

`capture_full_tensors` is disabled by default. When enabled, detached outputs remain in the returned report on their originating device until `release_full_tensors()` is called or the report is destroyed. Full tensors are not included in JSON traces. Users must treat these tensors as potentially sensitive because they may reveal information about source inputs or model internals.
