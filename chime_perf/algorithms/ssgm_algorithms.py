"""Spatial interpolation, spectral interpolation and spectral-library reconstruction."""

import numpy as np
import xarray as xr
from eopl import image_t
from numba import njit
from scipy.interpolate import interp1d
from scipy.optimize import nnls

from .algorithms import grid, reflectance_image


def spatial_grid(coordinates, detector_spacing_m, oversampling="auto"):
    """Place oversampled samples at detector spacing divided by an integer factor."""
    coordinates = grid(coordinates)

    if not np.isfinite(detector_spacing_m) or detector_spacing_m <= 0:
        raise ValueError("Detector spacing must be finite and positive.")

    if oversampling == "auto":
        source_spacing_m = np.median(np.diff(coordinates))

        # Round to the nearest integer; an exact half chooses the larger oversampling factor.
        oversampling = max(1, int(np.floor(detector_spacing_m / source_spacing_m + 0.5)))

    if (
        isinstance(oversampling, (bool, np.bool_))
        or not isinstance(oversampling, (int, np.integer))
        or oversampling < 1
    ):
        raise ValueError("Spatial oversampling must be a positive integer or 'auto'.")

    step_m = detector_spacing_m / oversampling

    # Nearest-factor selection may slightly coarsen the scene (5.5 m -> 6 m).
    # The first input centre anchors both the oversampled grid and nominal detector centres.
    target = grid(coordinates, step_m, allow_coarsening=True)

    return target, int(oversampling)


def validate_spatial_sampling(data, detector_spacing_m):
    """Require regular ACT and ALT grids with integer detector-to-scene spacing ratios."""
    reflectance = reflectance_image(data)

    for dimension in ("ACT", "ALT"):
        coordinates = reflectance[dimension].values
        _, factor = spatial_grid(coordinates, detector_spacing_m)
        step_m = detector_spacing_m / factor

        if not np.allclose(np.diff(coordinates), step_m, rtol=1e-7, atol=1e-9):
            raise ValueError(
                f"{dimension} has non-integer or irregular spatial oversampling; "
                "enable scene.spatial_interpolation to resample it."
            )


def interpolate_spatial(data, detector_spacing_m=30.0, oversampling="auto"):
    """Interpolate ACT and ALT onto grids with integer spatial oversampling."""
    reflectance = reflectance_image(data)

    act, factor_act = spatial_grid(reflectance.ACT.values, detector_spacing_m, oversampling)
    alt, factor_alt = spatial_grid(reflectance.ALT.values, detector_spacing_m, oversampling)

    # Applying linear interpolation along ACT and ALT gives bilinear interpolation.
    # Existing missing samples remain NaN; no points outside the scene are requested.
    result = reflectance

    if not np.array_equal(act, reflectance.ACT.values):
        result = result.interp(ACT=act, method="linear")

    if not np.array_equal(alt, reflectance.ALT.values):
        result = result.interp(ALT=alt, method="linear")

    result.attrs = {
        **reflectance.attrs,
        "detector_spacing_m": float(detector_spacing_m),
        "spatial_oversampling_ACT": factor_act,
        "spatial_oversampling_ALT": factor_alt,
        "detector_origin_ACT_m": float(act[0]),
        "detector_origin_ALT_m": float(alt[0]),
    }

    return image_t(result)


# EOPL capabilities: these NumPy/SciPy operations read CPU images.
interpolate_spatial.cuda_support = False
interpolate_spatial.dask_support = False
interpolate_spatial.input_access = ("read",)


@njit(cache=True)
def interpolate_spectra(spectra, wavelength_nm, target_nm):
    """Evaluate the same straight-line spectral interpolant at every pixel."""
    result = np.empty((len(spectra), len(target_nm)))

    for pixel in range(len(spectra)):
        result[pixel] = np.interp(target_nm, wavelength_nm, spectra[pixel])

    return result


def interpolate_spectral(data, step_nm=None):
    """Interpolate each pixel spectrum onto the requested SPE grid."""
    reflectance = reflectance_image(data)

    source_nm = grid(reflectance.SPE.values)
    target_nm = grid(source_nm, step_nm)

    # Each row contains one pixel spectrum. Numba only compiles this small loop.
    spectra = reflectance.values.reshape(-1, len(source_nm))
    result = interpolate_spectra(spectra, source_nm, target_nm)

    image = xr.DataArray(
        result.reshape(len(reflectance.ACT), len(reflectance.ALT), len(target_nm)),
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": reflectance.ACT, "ALT": reflectance.ALT, "SPE": target_nm},
        name="reflectance",
        attrs={**reflectance.attrs, "spectral_method": "linear"},
    )
    image.SPE.attrs["units"] = "nm"

    return image_t(image)


# EOPL capabilities: these NumPy/SciPy operations read CPU images.
interpolate_spectral.cuda_support = False
interpolate_spectral.dask_support = False
interpolate_spectral.input_access = ("read",)


def fit_library(data, library):
    """Fit non-negative library weights independently at each spatial pixel."""
    reflectance = reflectance_image(data)
    measured = library["measured"].transpose("SPE", "material")

    # These library values must include the INPUT sensor's band-response averages.
    if not np.array_equal(measured.SPE.values, reflectance.SPE.values):
        raise ValueError("Library measured bands must match the input SPE coordinates.")

    if not np.isfinite(measured.values).all():
        raise ValueError("Measured-band library values must be finite.")

    weights = np.full(
        (len(reflectance.ACT), len(reflectance.ALT), len(measured.material)), np.nan
    )

    for act in range(len(reflectance.ACT)):
        for alt in range(len(reflectance.ALT)):
            spectrum = reflectance.values[act, alt, :]

            if np.isfinite(spectrum).all():
                weights[act, alt, :] = nnls(measured.values, spectrum)[0]

    image = xr.DataArray(
        weights,
        dims=("ACT", "ALT", "material"),
        coords={"ACT": reflectance.ACT, "ALT": reflectance.ALT, "material": measured.material},
        name="mixture_weights",
        attrs={**reflectance.attrs, "units": "1"},
    )

    return image_t(image)


# EOPL capabilities: these NumPy/SciPy operations read CPU images.
fit_library.cuda_support = False
fit_library.dask_support = False
fit_library.input_access = ("read",)


def reconstruct_library(data, library, target_nm):
    """Reconstruct rho(lambda) as the weighted sum of the library spectra."""
    weights = data.image()
    endmembers = library["reflectance"].transpose("library_SPE", "material")
    source_nm = grid(endmembers.library_SPE.values)

    if not np.array_equal(weights.material.values, endmembers.material.values):
        raise ValueError("Material order must match between fitted weights and library.")

    if not np.isfinite(endmembers.values).all():
        raise ValueError("Library spectra must be finite.")

    if target_nm[0] < source_nm[0] or target_nm[-1] > source_nm[-1]:
        raise ValueError("Library must cover the complete target SPE grid.")

    # Interpolate the library once, then apply rho_m = sum_r c_r e_rm to every pixel.
    spectra = interp1d(source_nm, endmembers.values, axis=0)(target_nm)
    reflectance = weights.values @ spectra.T

    image = xr.DataArray(
        reflectance,
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": weights.ACT, "ALT": weights.ALT, "SPE": target_nm},
        name="reflectance",
        attrs={**weights.attrs, "spectral_method": "NNLS"},
    )
    image.SPE.attrs["units"] = "nm"

    return image_t(image)


# EOPL capabilities: these NumPy/SciPy operations read CPU images.
reconstruct_library.cuda_support = False
reconstruct_library.dask_support = False
reconstruct_library.input_access = ("read",)
