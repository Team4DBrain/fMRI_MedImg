"""Public API for the fMRI temporal interpolation module."""

from .src.inference import FMRIInterpolator, interpolate_file

__all__ = ["FMRIInterpolator", "interpolate_file"]
