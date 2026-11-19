"""Image operations for spectral response, spatial response and radiometric errors."""

import numpy as np
import xarray as xr
from eopl import image_t
from scipy.interpolate import RegularGridInterpolator

from chime_perf.algorithms.algorithms import require_units
from chime_perf.algorithms.instrument_algorithms import isrf_sigma


def spectral_response(centres, wavelength_nm, sigma_nm):
    """Normalise the Gaussian response, including each sample's wavelength width."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if wavelength_nm.size < 2 or np.any(np.diff(wavelength_nm) <= 0):
        raise ValueError("Input wavelengths must be strictly increasing.")

    centre = centres.image()
    require_units(centre, "nm")
    distance = wavelength_nm - centre.values[..., None]
    sigma_nm = np.asarray(sigma_nm)
    if np.any(sigma_nm <= 0) or not np.isfinite(sigma_nm).all():
        raise ValueError("Spectral response widths must be finite and positive.")

    # Four standard deviations retain the response tails to numerical accuracy.
    if np.any(centre.values - 4 * sigma_nm < wavelength_nm[0]) or np.any(
        centre.values + 4 * sigma_nm > wavelength_nm[-1]
    ):
        raise ValueError("Input wavelengths must cover each channel's +/-4 sigma response.")

    response = np.exp(-0.5 * (distance / sigma_nm[..., None]) ** 2)
    response *= np.abs(distance) <= 4 * sigma_nm[..., None]
    response *= np.gradient(wavelength_nm)
    total = response.sum(axis=-1, keepdims=True)
    if np.any(total == 0):
        raise ValueError(
            "Input spectral sampling is too sparse to resolve a channel response."
        )
    response /= total

    coords = {name: centre[name] for name in centre.dims}
    coords["sample_SPE"] = wavelength_nm
    weights = xr.DataArray(
        response,
        dims=(*centre.dims, "sample_SPE"),
        coords=coords,
        name=centre.name,
        attrs={"units": "1"},
    )
    return image_t(weights)


def sensor_coordinates(image, heading_deg):
    """Map a sensor grid to the source map, rotating about the scene centre."""
    angle = np.deg2rad(heading_deg)
    xc = float((image.ACT[0] + image.ACT[-1]) / 2)
    yc = float((image.ALT[0] + image.ALT[-1]) / 2)
    a, b = np.meshgrid(image.ACT.values - xc, image.ALT.values - yc, indexing="ij")

    x = xc + a * np.cos(angle) - b * np.sin(angle)
    y = yc + a * np.sin(angle) + b * np.cos(angle)
    return x, y


def column_coordinates(image, heading_deg, detector_spacing_m):
    """Find the detector ACT column covering each source-map pixel."""
    angle = np.deg2rad(heading_deg)
    xc = float((image.ACT[0] + image.ACT[-1]) / 2)
    yc = float((image.ALT[0] + image.ALT[-1]) / 2)
    x, y = np.meshgrid(image.ACT.values - xc, image.ALT.values - yc, indexing="ij")

    # Inverse of the rotation used when evaluating the sensor-grid image.
    sensor_act = xc + x * np.cos(angle) + y * np.sin(angle)
    column = np.floor((sensor_act - float(image.ACT[0])) / detector_spacing_m + 0.5)
    return column.astype(int)


def channel_centres(
    data, nominal_nm, spacing_m, heading_deg, sigma_error_nm, common_fraction, rng
):
    """Draw a fixed wavelength shift for each detector column and channel."""
    if not np.isfinite(sigma_error_nm) or sigma_error_nm < 0:
        raise ValueError("spectral_error_sigma_nm must be finite and non-negative.")
    if not np.isfinite(common_fraction) or not 0 <= common_fraction <= 1:
        raise ValueError("spectral_error_common_fraction must be between zero and one.")

    columns = column_coordinates(data.image(), heading_deg, spacing_m)
    indices = np.arange(columns.min(), columns.max() + 1)
    shifts = np.zeros((indices.size, len(nominal_nm)))

    if sigma_error_nm > 0:
        common = rng.normal()
        independent = rng.normal(size=shifts.shape)
        shifts = sigma_error_nm * (
            np.sqrt(common_fraction) * common + np.sqrt(1 - common_fraction) * independent
        )

    centres = xr.DataArray(
        nominal_nm + shifts,
        dims=("detector_column", "SPE"),
        coords={"detector_column": indices, "SPE": nominal_nm},
        name=data.image().name,
        attrs={"units": "nm"},
    )
    return image_t(centres)


def integrate_spectral(data, response, spacing_m, heading_deg):
    """Sum the input radiance with the response belonging to each detector column."""
    source = data.image().transpose("ACT", "ALT", "SPE")
    weights = response.image()
    if not np.array_equal(source.SPE.values, weights.sample_SPE.values):
        raise ValueError("Spectral response wavelengths must match the input image.")
    columns = column_coordinates(source, heading_deg, spacing_m)
    values = np.empty((*source.shape[:2], weights.sizes["SPE"]))

    # Only non-zero weights contribute; missing distant bands do not mask a channel.
    for column in weights.detector_column.values:
        pixels = columns == column
        for channel in range(weights.sizes["SPE"]):
            weight = weights.sel(detector_column=column).values[channel]
            support = weight > 0
            values[pixels, channel] = source.values[pixels][:, support] @ weight[support]

    result = xr.DataArray(
        values,
        dims=source.dims,
        coords={"ACT": source.ACT, "ALT": source.ALT, "SPE": weights.SPE},
        name=source.name,
        attrs=source.attrs.copy(),
    )
    result.SPE.attrs["units"] = "nm"
    return image_t(result)


def rotate_to_sensor(data, heading_deg):
    """Evaluate the map radiance at the rotated sensor-grid coordinates."""
    source = data.image()
    x, y = sensor_coordinates(source, heading_deg)

    if heading_deg == 0:
        result = source.copy(deep=True)
    else:
        interpolate = RegularGridInterpolator(
            (source.ACT.values, source.ALT.values),
            source.values,
            bounds_error=False,
            fill_value=np.nan,
        )
        result = source.copy(data=interpolate(np.stack((x, y), axis=-1)))

    # These coordinates retain the original map position of every output centre.
    result = result.assign_coords(map_ACT=(("ACT", "ALT"), x), map_ALT=(("ACT", "ALT"), y))
    result.map_ACT.attrs["units"] = "m"
    result.map_ALT.attrs["units"] = "m"
    result.attrs["heading_deg"] = float(heading_deg)
    result.attrs["geometry"] = "sensor"
    return image_t(result)


def convolve_downsample(data, kernel, factor_act, factor_alt, edge_mode="reflect"):
    """Evaluate spatial convolution only at the nominal centres that will be kept."""
    source = data.image()
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 2 or any(size % 2 != 1 for size in kernel.shape):
        raise ValueError("The spatial kernel must have odd ACT and ALT sizes.")
    if not np.isfinite(kernel).all() or not np.isclose(kernel.sum(), 1):
        raise ValueError("The spatial kernel must be finite and sum to one.")
    if any(not isinstance(s, (int, np.integer)) or s < 1 for s in (factor_act, factor_alt)):
        raise ValueError("Spatial oversampling factors must be positive integers.")
    if edge_mode not in ("reflect", "valid"):
        raise ValueError("edge_mode must be reflect or valid.")

    radius_act, radius_alt = np.array(kernel.shape) // 2
    act = np.arange(0, source.sizes["ACT"], factor_act)
    alt = np.arange(0, source.sizes["ALT"], factor_alt)

    if edge_mode == "valid":
        act = act[(act >= radius_act) & (act < source.sizes["ACT"] - radius_act)]
        alt = alt[(alt >= radius_alt) & (alt < source.sizes["ALT"] - radius_alt)]
    if not act.size or not alt.size:
        raise ValueError("The scene is too small for the spatial kernel's valid interior.")

    # Symmetric padding matches scipy.ndimage's half-sample 'reflect' boundary.
    padded = np.pad(
        source.values,
        ((radius_act, radius_act), (radius_alt, radius_alt), (0, 0)),
        mode="symmetric",
    )
    values = np.empty((act.size, alt.size, source.sizes["SPE"]))

    for i, u in enumerate(act):
        for j, v in enumerate(alt):
            patch = padded[u : u + kernel.shape[0], v : v + kernel.shape[1], :]
            # Reverse the kernel for convolution (rather than correlation).
            values[i, j] = np.einsum("abk,ab->k", patch, kernel[::-1, ::-1])

    result = source.isel(ACT=act, ALT=alt).copy(data=values)
    result.attrs["spatial_edge_mode"] = edge_mode
    result.attrs["source_spatial_oversampling_ACT"] = int(factor_act)
    result.attrs["source_spatial_oversampling_ALT"] = int(factor_alt)
    result.attrs["spatial_oversampling_ACT"] = 1
    result.attrs["spatial_oversampling_ALT"] = 1
    return image_t(result)


def relative_radiance(data, lref):
    """Express noise-free radiance as a fraction of the reference radiance."""
    source = data.image()
    require_units(source, "W m-2 sr-1 um-1")
    image = source.copy(data=(source / lref).data)
    image.attrs["units"] = "1"
    if np.any(image.values[np.isfinite(image)] < 0):
        raise ValueError("Noise-free radiance must be non-negative.")
    return image_t(image)


def temporal_snr(data, g, n0):
    """Calculate SNR = g r / sqrt(r + N0)."""
    r = data.image()
    require_units(r, "1")
    result = r.copy(data=(g * r / np.sqrt(r + n0)).data)
    result.attrs["units"] = "1"
    return image_t(result)


def temporal_sigma(data, lref, g, n0):
    """Evaluate noise amplitude without dividing zero radiance by zero SNR."""
    r = data.image()
    require_units(r, "1")
    result = r.copy(data=(lref / g * np.sqrt(r + n0)).data)
    result.attrs["units"] = "W m-2 sr-1 um-1"
    return image_t(result)


def draw_noise(data, rng, enabled=True):
    """Draw independent Gaussian errors in the supplied standard deviation's units."""
    sigma = data.image()
    values = rng.normal(size=sigma.shape) * sigma.values if enabled else np.zeros(sigma.shape)
    return image_t(sigma.copy(data=values))


def electron_signal(data, a, b):
    """Calculate the paper's equivalent electron signal n = a + b r."""
    r = data.image()
    require_units(r, "1")
    result = r.copy(data=(a + b * r).data)
    result.attrs["units"] = "electron"
    return image_t(result)


def absolute_offset(data, electrons, lref, a, b, coefficient, s0):
    """Calculate the additive ARA term, with the explicit zero-radiance limit."""
    signal = data.image()
    n = electrons.image()
    require_units(signal, "W m-2 sr-1 um-1")
    require_units(n, "electron")
    ratio = np.zeros(signal.shape)
    np.divide(signal.values, n.values, out=ratio, where=n.values != 0)

    # For a = 0, L / (b L / Lref) tends to Lref / b, including at L = 0.
    zero_limit = (signal.values == 0) & (np.asarray(a) == 0)
    ratio = np.where(zero_limit, np.asarray(lref / b), ratio)
    return image_t(signal.copy(data=coefficient * s0 * ratio))


def apply_absolute_error(data, noise, offset, gain):
    """Multiply only the signal by the ARA gain; retain the original noise draw."""
    source = data.image()
    values = gain * source + noise.image() + offset.image()
    return image_t(source.copy(data=values.transpose(*source.dims).data))


def striping_gains(data, sigma, rng, enabled=True):
    """Draw one RXRA gain per ACT column and channel, shared along ALT."""
    source = data.image()
    values = np.ones((source.sizes["ACT"], source.sizes["SPE"]))
    if enabled:
        values += rng.normal(size=values.shape) * sigma

    gains = xr.DataArray(
        values,
        dims=("ACT", "SPE"),
        coords={"ACT": source.ACT, "SPE": source.SPE},
        name=source.name,
        attrs={"units": "1"},
    )
    return image_t(gains)


def apply_striping(data, gains):
    """Apply the saved column gain to every ALT pixel."""
    source = data.image()
    values = (source * gains.image()).transpose(*source.dims)
    return image_t(source.copy(data=values.data))


def select_channels(wavelength_nm, parameters, fwhm_nm):
    """Keep nominal channels whose Gaussian response is covered by the input."""
    sigma = isrf_sigma(float(fwhm_nm))
    centres = parameters.wavelength_nm
    covered = (centres - 4 * sigma >= wavelength_nm[0]) & (
        centres + 4 * sigma <= wavelength_nm[-1]
    )
    if not covered.any():
        raise ValueError("The input spectrum does not cover any full CHIME channel response.")
    return np.flatnonzero(covered)


# EOPAL operation contracts: CPU NumPy implementations; inputs are read-only.
channel_centres.cuda_support = False
channel_centres.dask_support = False
channel_centres.input_access = ("read",)

spectral_response.cuda_support = False
spectral_response.dask_support = False
spectral_response.input_access = ("read",)

integrate_spectral.cuda_support = False
integrate_spectral.dask_support = False
integrate_spectral.input_access = ("read", "read")

rotate_to_sensor.cuda_support = False
rotate_to_sensor.dask_support = False
rotate_to_sensor.input_access = ("read",)

convolve_downsample.cuda_support = False
convolve_downsample.dask_support = False
convolve_downsample.input_access = ("read",)

relative_radiance.cuda_support = False
relative_radiance.dask_support = False
relative_radiance.input_access = ("read",)

temporal_snr.cuda_support = False
temporal_snr.dask_support = False
temporal_snr.input_access = ("read",)

temporal_sigma.cuda_support = False
temporal_sigma.dask_support = False
temporal_sigma.input_access = ("read",)

draw_noise.cuda_support = False
draw_noise.dask_support = False
draw_noise.input_access = ("read",)

electron_signal.cuda_support = False
electron_signal.dask_support = False
electron_signal.input_access = ("read",)

absolute_offset.cuda_support = False
absolute_offset.dask_support = False
absolute_offset.input_access = ("read", "read")

apply_absolute_error.cuda_support = False
apply_absolute_error.dask_support = False
apply_absolute_error.input_access = ("read", "read", "read")

striping_gains.cuda_support = False
striping_gains.dask_support = False
striping_gains.input_access = ("read",)

apply_striping.cuda_support = False
apply_striping.dask_support = False
apply_striping.input_access = ("read", "read")
