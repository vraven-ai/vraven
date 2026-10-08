# VRAVEN diagram style guide

## Publication light

* canvas: `#FFFFFF`
* primary text: `#111827`
* secondary text: `#475569`
* border: `#C8D2DE`
* panel tint: `#F8FAFC`
* no glow
* no decorative cards unrelated to structure
* restrained saturation
* vector export preferred

## Interactive dark

* canvas: `#050A12`
* panel: `#0A1320`
* raised panel: `#0E1A2A`
* border: `#223148`
* primary text: `#F7FAFC`
* secondary text: `#A9B7C9`
* subtle grid and controlled emphasis

## Semantic colours

| Meaning | Publication | Interactive |
|---|---|---|
| Supporting evidence | `#0057B8` | `#3B82F6` |
| Dominant support | `#0072CE` | `#38BDF8` |
| Opposing evidence | `#B42318` | `#FF4D6D` |
| Strong contradiction | `#8F1420` | `#FF1744` |
| Uncertainty | `#A15C00` | `#FFC857` |
| Stable reasoning | `#00796B` | `#2DD4BF` |
| Latent structure | `#6F42C1` | `#B084FF` |
| Neutral | `#6B7280` | `#7E8A9A` |
| Selected pathway | `#007C91` | `#22D3EE` |

Colour must not be the only encoding for a critical distinction. Use sign, position, direction, line style or labels as a second channel.

## Geometry

* arrows must connect to the intended boxes or nodes
* flow width must correspond to a defined magnitude
* bubble area, not radius, should encode quantitative magnitude
* dashed outlines indicate planned or conditional functions
* panel boundaries organise related operations, not dashboard decoration
* diagrams must remain interpretable at double-column paper width
