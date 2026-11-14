"""Nominal-band averaging and the inverse Lambertian atmospheric relation."""

import numpy as np
import xarray as xr
from eopl import image_t


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
    scale = solar_scale.image()
    result = data.image() / scale.where(np.isfinite(scale) & (scale > 0))
    result.attrs = {**data.image().attrs, "units": "1"}
    return image_t(result.rename("toa_reflectance"))


def path_reflectance(data, solar_scale):
    """Express saved atmospheric path radiance in the same reflectance units."""
    scale = solar_scale.image()
    result = data.image() / scale.where(np.isfinite(scale) & (scale > 0))
    result.attrs["units"] = "1"
    return image_t(result.rename("path_reflectance"))


def remove_atmosphere(data, path, transmission):
    """Subtract path reflectance and divide by two-way transmission."""
    t = transmission.image()
    result = (data.image() - path.image()) / t.where(np.isfinite(t) & (t > 0))
    result.attrs = data.image().attrs.copy()
    return image_t(result.rename("surface_contribution"))


def surface_reflectance(data, spherical_albedo):
    """Remove repeated surface--atmosphere reflection: rho = y / (1 + S y)."""
    y = data.image()
    denominator = 1 + spherical_albedo.image() * y
    result = y / denominator.where(np.isfinite(denominator) & (denominator > 0))
    result.attrs = {**data.image().attrs, "units": "1"}
    # Negative noisy retrievals remain visible; do not clip them to physical bounds.
    return image_t(result.rename("reflectance"))


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
