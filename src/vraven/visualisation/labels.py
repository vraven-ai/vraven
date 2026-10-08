"""Presentation helpers for long module and feature names.

These helpers change only labels shown on axes or markers. They never alter
measured scientific values.
"""

from __future__ import annotations

from collections.abc import Sequence


def shorten_layer_name(name: str, *, max_length: int = 28) -> str:
    """Compress dotted module paths for crowded categorical axes."""

    text = str(name).strip()
    if not text:
        return text
    aliases = (
        ("autoencoder.", "ae."),
        ("binary_embedding", "bin_emb"),
        (".encoder.", ".enc."),
        (".decoder.", ".dec."),
    )
    shortened = text
    for old, new in aliases:
        shortened = shortened.replace(old, new)
    if len(shortened) <= max_length:
        return shortened
    parts = [part for part in shortened.split(".") if part]
    if len(parts) >= 2:
        candidate = ".".join(parts[-2:])
        if len(candidate) <= max_length:
            return candidate
    if len(shortened) <= max_length:
        return shortened
    return shortened[: max_length - 1] + "…"


def shorten_layer_names(names: Sequence[str], *, max_length: int = 28) -> list[str]:
    return [shorten_layer_name(name, max_length=max_length) for name in names]


def readable_feature_name(name: str, *, max_length: int = 42) -> str:
    """Make snake_case feature names easier to read on bar charts."""

    text = str(name).replace("__", " / ").replace("_", " ").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def readable_unit_range(indices: Sequence[int], *, compact: bool = False) -> str:
    """Human label for a contiguous neuron/unit index bag.

    Inclusive ranges: ``[16, 17, 18, 19]`` → ``units 16–19`` (or ``16–19``).
    """

    values = sorted(int(v) for v in indices)
    if not values:
        return "units ?"
    if len(values) == 1:
        return f"unit {values[0]}" if not compact else str(values[0])
    lo, hi = values[0], values[-1]
    contiguous = values == list(range(lo, hi + 1))
    if contiguous:
        body = f"{lo}–{hi}"
    else:
        body = ",".join(str(v) for v in values[:6]) + ("…" if len(values) > 6 else "")
    if compact:
        return body
    return f"units {body}" if contiguous or len(values) > 1 else f"unit {body}"


def readable_unit_group_label(
    group_id: str,
    *,
    indices: Sequence[int] | None = None,
    compact: bool = True,
) -> str:
    """Turn ``layer#0[u16:20]`` / ``u16:20`` into a readable unit-range label."""

    text = str(group_id).strip()
    bracket = text
    if "[" in text and text.endswith("]"):
        bracket = text.rsplit("[", 1)[-1][:-1]
    if indices:
        return readable_unit_range(indices, compact=compact)
    # Legacy exclusive-end form: u16:20 means indices 16..19
    if bracket.startswith("u") and ":" in bracket:
        try:
            left, right = bracket[1:].split(":", 1)
            start, end_exclusive = int(left), int(right)
            if end_exclusive > start:
                return readable_unit_range(range(start, end_exclusive), compact=compact)
        except ValueError:
            return bracket
    # Newer form already stored as "units 16–19"
    if bracket.lower().startswith("units ") or bracket.lower().startswith("unit "):
        return bracket if not compact else bracket.replace("units ", "").replace("unit ", "")
    return bracket


def unit_group_source_label(indices: Sequence[int]) -> str:
    """Canonical label stored on UnitGroup (inclusive unit range)."""

    return readable_unit_range(indices, compact=False)
