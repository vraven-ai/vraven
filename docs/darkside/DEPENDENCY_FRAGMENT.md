# Optional dependency fragment

The extension reuses NumPy, PyTorch and Plotly from the existing VRAVEN package. Install the existing analysis and publication extras:

```bash
python -m pip install -e ".[analysis,publication]"
```

SciPy and scikit-learn support cohort distances, probes and shrinkage covariance. Most intervention and matrix analyses do not require a new dependency beyond the existing VRAVEN installation.

Optional H1 persistent homology:

```bash
python -m pip install ripser
```

A future main-project `pyproject.toml` may add:

```toml
[project.optional-dependencies]
darkside = [
  "scipy>=1.11",
  "scikit-learn>=1.4",
  "ripser>=0.6.8",
]
```

The extension package does not edit `pyproject.toml` automatically.
