# VRAVEN model bundles

A model bundle keeps the model, representative inputs and reconstruction
configuration together. Once configured, the entire bundle can be analysed with:

```bash
vraven quick path/to/model_bundle
```

## Directory layout

```text
model_bundle/
├── model.pt
├── sample.npy
├── labels.npy       # optional
└── vraven.json
```

Create `vraven.json` after placing the model and input files in the directory:

```bash
vraven bundle-init model_bundle \
  --factory my_project.models:Classifier \
  --factory-kwargs '{"input_dim":12,"num_classes":3}' \
  --labels labels.npy \
  --class-names benign,suspicious,malicious
```

Validate the finished bundle:

```bash
vraven bundle model_bundle
```

## Manifest schema

```json
{
  "schema_version": "1.0",
  "model": {
    "path": "model.pt",
    "format": "state_dict",
    "factory": "my_project.models:Classifier",
    "factory_kwargs": {
      "input_dim": 12,
      "num_classes": 3
    },
    "device": "cpu"
  },
  "input": {
    "path": "sample.npy",
    "labels": "labels.npy"
  },
  "class_names": ["benign", "suspicious", "malicious"],
  "defaults": {
    "task": "explain",
    "preset": "quick",
    "format": "html",
    "target": 1,
    "output": "results/explanation.html"
  }
}
```

Command-line arguments override manifest defaults:

```bash
vraven quick model_bundle -t adversarial -p standard -o audit.html
```

## Safety and portability

- Bundle paths must be relative and must remain inside the bundle directory.
- State dictionaries require an importable `module:object` factory.
- Factory code must be installed in the active Python environment.
- Alternatively, the bundle may contain the referenced Python module; VRAVEN
  temporarily adds the bundle root to the import path while loading it.
- Pickled models remain blocked unless `--trusted-model` is supplied explicitly.
- Representative inputs must already include the preprocessing and shape expected
  by the model.
- A bundle records reconstruction information; it does not make unknown Python
  architecture code portable by itself.
