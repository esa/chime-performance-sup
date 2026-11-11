"""Create a small synthetic reflectance scene and an explicitly synthetic library."""

from pathlib import Path

import numpy as np
import xarray as xr
from eopl.log import Log

output = Path("demo-input")
output.mkdir(exist_ok=True)

# Work over a short spectral interval so the live 6S demonstration stays small.
wavelength = np.arange(500.0, 601.0, 10.0)
library_wavelength = np.arange(495.0, 606.0, 0.5)
x = np.arange(0.0, 121.0, 30.0)
y = np.arange(0.0, 91.0, 30.0)


def spectra(wavelength_nm):
    """Two invented smooth materials; these are not measured mineral spectra."""
    bright = 0.35 - 0.10 * np.exp(-(((wavelength_nm - 550) / 15) ** 2))
    dark = 0.10 + 0.03 * (wavelength_nm - 500) / 100
    return np.stack([bright, dark], axis=-1)


# Smoothly vary the mixture across the map; NNLS can recover these known weights.
bright_fraction = 0.2 + 0.6 * x[:, None] / x[-1] + np.zeros((len(x), len(y)))
reflectance = (
    bright_fraction[..., None] * spectra(wavelength)[:, 0]
    + (1 - bright_fraction[..., None]) * spectra(wavelength)[:, 1]
)
scene = xr.DataArray(
    reflectance,
    dims=("ACT", "ALT", "SPE"),
    coords={"ACT": x, "ALT": y, "SPE": wavelength},
    name="reflectance",
    attrs={
        "units": "1",
        "description": "Synthetic two-material BOA reflectance",
        "geometry": "map",
    },
)
for name, unit in (("ACT", "m"), ("ALT", "m"), ("SPE", "nm")):
    scene[name].attrs["units"] = unit
scene.to_netcdf(output / "reflectance.nc")

# Input bands here are synthetic point samples; real input requires band-response averages.
library = xr.Dataset(
    {
        "measured": (("SPE", "material"), spectra(wavelength)),
        "reflectance": (("library_SPE", "material"), spectra(library_wavelength)),
    },
    coords={
        "SPE": wavelength,
        "library_SPE": library_wavelength,
        "material": ["synthetic_bright", "synthetic_dark"],
    },
    attrs={"description": "Synthetic point-sampled library; no real mineral abundances"},
)
library.SPE.attrs["units"] = "nm"
library.library_SPE.attrs["units"] = "nm"
library.to_netcdf(output / "library.nc")
Log().info(f"Created {output / 'reflectance.nc'} and {output / 'library.nc'}")
