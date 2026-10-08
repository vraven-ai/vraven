# Optional top-level exports

The extension works without editing `vraven/__init__.py`. To expose the API at the package root, add:

```python
from .darkside import DarkSideSession, TraceDataset, recommend_features
```

and append these names to `__all__`:

```python
"DarkSideSession",
"TraceDataset",
"recommend_features",
```
