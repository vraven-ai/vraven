"""VRAVEN exception hierarchy."""

from __future__ import annotations


class VRavenError(Exception):
    """Base class for all expected VRAVEN errors."""


class UnsupportedModelError(VRavenError):
    """Raised when no installed adapter can instrument the supplied model."""


class UnsupportedOutputError(VRavenError):
    """Raised when a model output cannot be converted into a supported prediction trace."""


class CapabilityError(VRavenError):
    """Raised when a requested analysis requires data that was not captured."""


class TraceValidationError(VRavenError):
    """Raised when a stored or generated trace violates the trace schema."""


class ModelLoadError(VRavenError):
    """Raised when an explicit model loading operation cannot be completed safely."""


class DeviceMismatchError(VRavenError):
    """Raised when a requested device conflicts with the live model device."""


class ExportError(VRavenError):
    """Raised when a figure or report cannot be exported safely."""
