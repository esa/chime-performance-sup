"""Shared coordinate and image-validation operations."""

import numpy as np


def grid(coordinates, step=None, allow_coarsening=False):
    """Keep coordinates or generate regularly spaced samples within their extent."""
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("Each coordinate needs at least two samples.")

    spacing = np.diff(coordinates)
    if not np.isfinite(coordinates).all() or np.any(spacing <= 0):
        raise ValueError("Coordinates must be finite and strictly increasing.")

    if step is None:
        return coordinates

    if not np.isfinite(step) or step <= 0:
        raise ValueError("Grid spacing must be finite and positive.")

    if not allow_coarsening and step > spacing.min() * (1 + 1e-9):
        raise ValueError("Scene preparation must not coarsen the input grid.")

    count = int(np.floor((coordinates[-1] - coordinates[0]) / step + 1e-9)) + 1
    return np.linspace(coordinates[0], coordinates[0] + step * (count - 1), count)


def reflectance_image(data):
    """Validate the reflectance cube and arrange its axes as ACT, ALT, SPE."""
    reflectance = data.image().transpose("ACT", "ALT", "SPE")

    if reflectance.attrs.get("units") != "1":
        raise ValueError("BOA reflectance must have units='1'.")

    for dimension, unit in (("ACT", "m"), ("ALT", "m"), ("SPE", "nm")):
        if reflectance[dimension].attrs.get("units") != unit:
            raise ValueError(f"Coordinate {dimension} must have units={unit!r}.")

    return reflectance
