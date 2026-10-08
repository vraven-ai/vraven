# Migration Note

This release replaces the earlier activation-dashboard result direction with the compiler-assisted causal-decompiler direction.

The active source archive contains only the new 0.9 validation outputs under `vraven_outputs/`. Earlier generated reports were removed from the active output tree to avoid confusion. The exact original uploaded workspace and original report remain preserved unchanged under `source_material/original_upload/` inside the source archive.

The main implementation namespace is:

```text
src/vraven/decompiler/
```

Contributors extending the decompiler should read:

```text
docs/decompiler/IMPLEMENTATION_GAP_CLOSURE.md
docs/decompiler/LIMITATIONS.md
docs/decompiler/MATHEMATICAL_FOUNDATIONS.md
```
