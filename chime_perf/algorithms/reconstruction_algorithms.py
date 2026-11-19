"""Nominal-band averaging and the inverse Lambertian atmospheric relation."""

import numpy as np
import xarray as xr
from eopl import image_t

from chime_perf.algorithms.algorithms import require_units


def average_atmosphere(data, response):
    """Average one saved atmospheric quantity over each nominal channel response."""
    quantity = data.image()
    weights = response.image()
    if quantity.ndim != 1 or quantity.size != weights.sizes["sample_SPE"]:
        raise ValueError("Atmospheric quantities must have one value per input wavelength.")
    if not np.array_equal(quantity[quantity.dims[0]].values, weights.sample_SPE.values):
        raise ValueError("Atmospheric wavelengths must match the response wavelengths.")

    values = np.empty(weights.sizes["SPE"])
    for channel, weight in enumerate(weights.values):
        support = weight > 0
        values[channel] = quantity.values[support] @ weight[support]

    averaged = xr.DataArray(
        values,
        dims="SPE",
        coords={"SPE": weights.SPE},
        name=quantity.name,
        attrs=quantity.attrs.copy(),
    )
    return image_t(averaged)


def toa_reflectance(data, solar_scale):
    """Divide measured radiance by the saved solar radiance scale C."""
    source = data.image()
    require_units(source, "W m-2 sr-1 um-1")
    scale = solar_scale.image()
    require_units(scale, "W m-2 sr-1 um-1")
    result = source / scale.where(np.isfinite(scale) & (scale > 0))
    result = source.copy(data=result.transpose(*source.dims).data)
    result.attrs["units"] = "1"
    return image_t(result)


def path_reflectance(data, solar_scale):
    """Express saved atmospheric path radiance in the same reflectance units."""
    source = data.image()
    require_units(source, "W m-2 sr-1 um-1")
    scale = solar_scale.image()
    require_units(scale, "W m-2 sr-1 um-1")
    result = source / scale.where(np.isfinite(scale) & (scale > 0))
    result = source.copy(data=result.data)
    result.attrs["units"] = "1"
    return image_t(result)


def remove_atmosphere(data, path, transmission):
    """Subtract path reflectance and divide by two-way transmission."""
    source = data.image()
    require_units(source, "1")
    require_units(path.image(), "1")
    t = transmission.image()
    require_units(t, "1")
    result = (source - path.image()) / t.where(np.isfinite(t) & (t > 0))
    result = source.copy(data=result.transpose(*source.dims).data)
    return image_t(result)


def surface_reflectance(data, spherical_albedo):
    """Remove repeated surface--atmosphere reflection: rho = y / (1 + S y)."""
    y = data.image()
    require_units(y, "1")
    require_units(spherical_albedo.image(), "1")
    denominator = 1 + spherical_albedo.image() * y
    result = y / denominator.where(np.isfinite(denominator) & (denominator > 0))
    result = y.copy(data=result.transpose(*y.dims).data)
    result.attrs["units"] = "1"
    # Negative noisy retrievals remain visible; do not clip them to physical bounds.
    return image_t(result)


# EOPAL operation contracts: CPU NumPy/Xarray implementations; inputs are read-only.
average_atmosphere.cuda_support = False
average_atmosphere.dask_support = False
average_atmosphere.input_access = ("read", "read")

toa_reflectance.cuda_support = False
toa_reflectance.dask_support = False
toa_reflectance.input_access = ("read", "read")

path_reflectance.cuda_support = False
path_reflectance.dask_support = False
path_reflectance.input_access = ("read", "read")

remove_atmosphere.cuda_support = False
remove_atmosphere.dask_support = False
remove_atmosphere.input_access = ("read", "read", "read")

surface_reflectance.cuda_support = False
surface_reflectance.dask_support = False
surface_reflectance.input_access = ("read", "read")
