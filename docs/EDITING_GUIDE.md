# Editing guide

## Fastest route

1. Open an SVG from `exports/publication_light/.../svg/` in Inkscape or Illustrator.
2. Ungroup the objects if the editor imports the file as nested groups.
3. Edit labels, paths, boxes and colours.
4. Export a new SVG or PDF.

This is suitable for one-off manuscript adjustments.

## Reproducible route

Edit the Python source and regenerate every format.

### Palette

```text
src/vraven/figures/theme.py
```

### Typography, arrows, boxes, matrices and reusable shapes

```text
src/vraven/figures/primitives.py
```

### Architecture figures

```text
src/vraven/figures/architecture.py
```

### Methodology figures

```text
src/vraven/figures/methodology.py
```

### Validation figures

```text
src/vraven/figures/validation.py
```

### Visual system and user workflow

```text
src/vraven/figures/integration.py
```

## Example modification

Change a label:

```python
c.box_label(
    x,
    y,
    width,
    height,
    "new label",
    accent=theme.support,
)
```

Render only the modified figure:

```bash
python scripts/figures/render_one.py \
  --figure 01_system_architecture \
  --mode publication_light \
  --formats svg,png,pdf \
  --output exports
```

## Adding a new figure

1. Add a renderer function to the most suitable category module.
2. Register it in `src/vraven/figures/registry.py`.
3. Add a publication caption to `docs/CAPTIONS.md`.
4. Add the method and caveats to `docs/METHODOLOGY_NOTES.md`.
5. Render both modes.
6. Run the validation script.

## Maintaining the no-heading rule

Do not put `Figure 1`, a paper title or a long descriptive heading inside the image. Use component labels in the diagram and put the complete description in the manuscript caption.

## Colour changes

Change both modes together and preserve semantics. Do not use blue for supporting evidence in one figure and opposing evidence in another.

## Font substitution

The source uses `DejaVu Sans`. You can change `FONT` in `primitives.py`, but verify every PDF and PNG after the change. Do not distribute font files with this package.
