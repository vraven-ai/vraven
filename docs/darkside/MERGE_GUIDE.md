# Merge guide

This archive intentionally contains no copy of the main VRAVEN project.

## 1. Back up or commit the target workspace

Before merging, commit your current changes or create a separate copy of the repository.

## 2. Dry-run the merge

```bash
python /path/to/VRAVEN_DarkSide_Extensions/scripts/merge_darkside.py \
  /path/to/VRAVEN_DarkSide_Extensions \
  --target /path/to/your/VRAVEN \
  --dry-run
```

The command prints every source and destination without writing files.

## 3. Apply the merge

```bash
python /path/to/VRAVEN_DarkSide_Extensions/scripts/merge_darkside.py \
  /path/to/VRAVEN_DarkSide_Extensions \
  --target /path/to/your/VRAVEN
```

The script copies only:

```text
merge/src/vraven/darkside/   -> VRAVEN/src/vraven/darkside/
merge/tests/test_darkside*.py -> VRAVEN/tests/
merge/examples/darkside_*.py  -> VRAVEN/examples/
```

It does not edit existing VRAVEN modules.

## Existing DarkSide folder

The script refuses to overwrite existing files. After reviewing and backing up local work, an explicit overwrite is available:

```bash
python /path/to/VRAVEN_DarkSide_Extensions/scripts/merge_darkside.py \
  /path/to/VRAVEN_DarkSide_Extensions \
  --target /path/to/your/VRAVEN \
  --overwrite
```

Existing DarkSide files are first copied to:

```text
VRAVEN/.vraven_darkside_backups/<UTC timestamp>/
```

## Public import

No change to `src/vraven/__init__.py` is required:

```python
from vraven.darkside import DarkSideSession, TraceDataset
```

Optional top-level exports are described in `docs/OPTIONAL_TOP_LEVEL_EXPORTS.md`.

## Dependencies

The core uses NumPy, PyTorch and Plotly already expected by VRAVEN. SciPy and scikit-learn are used by the analysis extra. Optional H1 persistence requires `ripser`; exact H0 persistence is implemented directly.

## Validation

```bash
python -m pip install -e ".[analysis,publication]"
PYTHONPATH=src python -m pytest -q tests/test_darkside*.py
```
