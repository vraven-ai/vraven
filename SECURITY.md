# VRAVEN Security Policy

## Reporting a vulnerability

Do not publish a suspected vulnerability before the maintainers have had a reasonable opportunity to assess it. Provide the affected version, a minimal reproduction, expected impact and any relevant platform details through a [private GitHub security advisory](https://github.com/vraven-ai/vraven/security/advisories/new).

## Supported release

VRAVEN is currently a research alpha. Security fixes are applied to the latest development release only.

## Model and data safety

VRAVEN runs locally by default and does not upload models, traces or inputs.
Treat model files, architecture factories, full activations, gradients and
reports as sensitive artefacts. Do not load untrusted Python pickle model files
or execute an unreviewed model-bundle factory. Prefer a live model object or a
trusted state dictionary with a known architecture. VRAVEN does not sandbox
model execution.

Full tensor capture is disabled by default. When enabled, retained tensors may reveal model internals or information about the source input and may occupy GPU memory until explicitly released.

See `docs/SECURITY_PRIVACY.md` for implementation details.
