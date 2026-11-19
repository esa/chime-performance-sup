"""Instrument acquisition from oversampled TOA radiance to nominal detector radiance."""

from pathlib import Path

import numpy as np
import xarray as xr
from eopl import Pipeline, image_t

from chime_perf.algorithms.acquisition_algorithms import (
    absolute_offset,
    apply_absolute_error,
    apply_striping,
    channel_centres,
    convolve_downsample,
    draw_noise,
    electron_signal,
    integrate_spectral,
    relative_radiance,
    rotate_to_sensor,
    select_channels,
    spectral_response,
    striping_gains,
    temporal_sigma,
    temporal_snr,
)
from chime_perf.algorithms.instrument_algorithms import isrf_sigma, psf_kernel, psf_sigma_px
from chime_perf.algorithms.ssgm_algorithms import validate_spatial_sampling
from chime_perf.params import ChimeParams, boundary_settings


def instrument(
    input_path=None,
    output_root="outputs/instrument",
    config_path=None,
    atmosphere_path=None,
    data=None,
    save=True,
    parameters=None,
):
    """Simulate detector radiance and retain the atmosphere for reconstruction."""
    pl = Pipeline(output_root=output_root)

    # 1. INITIALIZATION AND CONFIGURATION
    if config_path is not None:
        pl.add_local_config(config_path)

    pl.set_run_options(**pl.local_config.get("run_options", {"precision": "float64"}))
    pl.set_image_type(pl.run_options.precision)
    report_params = pl.local_config.get("report")
    pl.generate_report(report_params is not None, report_params)
    pl.generate_breakpoints(pl.local_config.get("breakpoints", False) and save)

    # Read the image separately from auxiliary variables: its sampling will change.
    # The saved atmosphere keeps its original wavelengths throughout the chain.
    if data is None:
        if input_path is None:
            raise ValueError("Supply input_path or an in-memory image through data.")
        with xr.open_dataset(input_path) as source:
            data = image_t(source["radiance"].load())
        pl.manifest["input"] = {"file": str(Path(input_path).resolve())}

    coefficient_path = atmosphere_path if atmosphere_path is not None else input_path
    if coefficient_path is None:
        raise ValueError("Supply the saved propagation product through atmosphere_path.")
    with xr.open_dataset(coefficient_path) as source:
        atmosphere = source.load()

    coefficient_names = ("path_radiance", "spherical_albedo", "transmission", "solar_scale")
    if not all(name in atmosphere for name in coefficient_names):
        raise ValueError("The propagation product must contain the saved P, S, T and C.")

    source = data.image().transpose("ACT", "ALT", "SPE")
    if source.attrs.get("units") != "W m-2 sr-1 um-1":
        raise ValueError("Instrument input must be radiance in W m-2 sr-1 um-1.")
    for name, unit in (("ACT", "m"), ("ALT", "m"), ("SPE", "nm")):
        if source[name].attrs.get("units") != unit:
            raise ValueError(f"Coordinate {name} must have units={unit!r}.")
    if not np.isfinite(source.SPE).all() or np.any(np.diff(source.SPE) <= 0):
        raise ValueError("Input wavelengths must be finite and strictly increasing.")
    if not np.array_equal(source.SPE.values, atmosphere.SPE.values):
        raise ValueError("Saved atmosphere wavelengths must match the propagation image.")
    data = image_t(source)

    settings = pl.local_config.get("instrument", {})
    boundaries = boundary_settings(pl.local_config)
    if "edge_mode" in settings:
        raise ValueError("Replace instrument.edge_mode with boundaries.spatial: edge.")
    spacing_m = float(pl.local_config.get("scene", {}).get("detector_spacing_m", 30.0))
    validate_spatial_sampling(data, spacing_m)
    factors = [
        int(round(spacing_m / float(source[axis][1] - source[axis][0])))
        for axis in ("ACT", "ALT")
    ]
    heading = float(settings.get("heading_deg", 0.0))
    if not np.isfinite(heading):
        raise ValueError("heading_deg must be finite.")
    seed = settings.get("seed", 42)
    pl.rng = np.random.default_rng(seed)

    if parameters is None:
        parameters = ChimeParams.load()
    fwhm = float(settings.get("fwhm_nm", parameters.isrf_fwhm_nm))
    indices = select_channels(source.SPE.values, parameters)
    nominal_nm = parameters.wavelength_nm[indices]
    lref = parameters.lref[indices]
    g = parameters.snr_g[indices]
    n0 = parameters.snr_n0[indices]
    a = parameters.ara_a[indices]
    b = parameters.ara_b[indices]

    # 2. OPERATION PHASE
    # Step 1: Convert the spectral FWHM to Gaussian sigma, in nm.
    sigma = pl.run_function(isrf_sigma, args={"fwhm_nm": fwhm})

    # Step 2: True channel centres include a fixed spectral-knowledge error.
    centres = pl.run_operation(
        data=data,
        func=channel_centres,
        args={
            "nominal_nm": nominal_nm,
            "spacing_m": spacing_m,
            "heading_deg": heading,
            "sigma_error_nm": settings.get("spectral_error_sigma_nm", 0.0),
            "common_fraction": settings.get("spectral_error_common_fraction", 0.0),
        },
    )

    # Step 3: Gaussian response weights sum to one for every detector column.
    response = pl.run_operation(
        data=centres,
        func=spectral_response,
        args={
            "wavelength_nm": source.SPE.values,
            "sigma_nm": sigma,
            "support_sigma": boundaries["kernel_support_sigma"],
        },
    )

    # Step 4: Integrate the oversampled spectrum at each channel response.
    data = pl.run_operation(
        data=(data, response),
        func=integrate_spectral,
        args={"spacing_m": spacing_m, "heading_deg": heading},
    )
    data = pl.save_bkp(data, "spectral_response")

    # Step 5: Interpolate map radiance onto the sensor's rotated ACT--ALT grid.
    data = pl.run_operation(
        data=data,
        func=rotate_to_sensor,
        args={"heading_deg": heading, "edge_mode": boundaries["spatial"]},
    )

    # Step 6: Spatial Gaussian widths, in oversampled-grid pixels.
    sigma_act = pl.run_function(
        psf_sigma_px, args={"mtf_nyquist": parameters.mtf_act, "OF": factors[0]}
    )
    sigma_alt = pl.run_function(
        psf_sigma_px, args={"mtf_nyquist": parameters.mtf_alt, "OF": factors[1]}
    )

    # Step 7: Construct the normalised spatial response kernel.
    kernel = pl.run_function(
        psf_kernel,
        args={
            "sigma_act_px": sigma_act,
            "sigma_alt_px": sigma_alt,
            "support_sigma": boundaries["kernel_support_sigma"],
        },
    )

    # Steps 8--9: Convolve only at the nominal centres; do not blur the whole cube.
    signal = pl.run_operation(
        data=data,
        func=convolve_downsample,
        args={
            "kernel": kernel,
            "factor_act": factors[0],
            "factor_alt": factors[1],
            "edge_mode": boundaries["spatial"],
        },
    )
    signal = pl.save_bkp(signal, "spatial_response")

    # Step 10: Relative radiance, SNR, noise amplitude, then one saved noise draw.
    relative = pl.run_operation(data=signal, func=relative_radiance, args={"lref": lref})
    pl.run_operation(data=relative, func=temporal_snr, args={"g": g, "n0": n0})
    sigma_noise = pl.run_operation(
        data=relative, func=temporal_sigma, args={"lref": lref, "g": g, "n0": n0}
    )
    noise = pl.run_operation(
        data=sigma_noise,
        func=draw_noise,
        args={"enabled": settings.get("temporal_noise", True)},
    )

    # Step 11: Apply ARA gain and offset to noise-free signal; reuse the same noise.
    electrons = pl.run_operation(data=relative, func=electron_signal, args={"a": a, "b": b})
    offset = pl.run_operation(
        data=(signal, electrons),
        func=absolute_offset,
        args={
            "lref": lref,
            "a": a,
            "b": b,
            "s0": parameters.ara_s0,
            "coefficient": settings.get("ara_offset_coefficient", 0.0),
        },
    )
    gain = 1 + float(settings.get("ara_gain_coefficient", 0.0)) * parameters.ara_k
    data = pl.run_operation(
        data=(signal, noise, offset), func=apply_absolute_error, args={"gain": gain}
    )

    # Step 12: One RXRA gain per detector column and channel, constant along ALT.
    gains = pl.run_operation(
        data=data,
        func=striping_gains,
        args={"sigma": parameters.rxra_sigma[indices], "enabled": settings.get("rxra", True)},
    )
    data = pl.run_operation(data=(data, gains), func=apply_striping)
    data.image().attrs.update(
        parameter_version=parameters.version,
        random_seed=int(seed),
        nominal_fwhm_nm=fwhm,
        kernel_support_sigma=boundaries["kernel_support_sigma"],
        spectral_edge_mode=boundaries["spectral"],
        ara_gain=float(gain),
        ara_offset_coefficient=float(settings.get("ara_offset_coefficient", 0)),
    )

    # 3. FINAL PHASE
    if save:
        # Separate original atmospheric wavelengths from nominal detector channels.
        # Copy these values, rather than rerunning a solver or changing their sampling.
        for name in coefficient_names:
            quantity = atmosphere[name].rename({"SPE": "atmosphere_SPE"})
            pl.add_variable(name, quantity, pinned=True)

        width = xr.DataArray(
            np.full(nominal_nm.size, fwhm),
            dims="SPE",
            coords={"SPE": nominal_nm},
            attrs={"units": "nm"},
        )
        pl.add_variable("nominal_fwhm", width, pinned=True)
        pl.add_variable("true_channel_centre", centres.image(), pinned=True)
        pl.add_variable("temporal_noise", noise.image(), pinned=True)
        pl.add_variable("rxra_gain", gains.image(), pinned=True)
        pl.add_variable("valid", np.isfinite(data.image()), pinned=True)
        pl.manifest["atmosphere_source"] = str(Path(coefficient_path).resolve())
        pl.save_output(image_t(data.image(), name="radiance"), "instrument")
        pl.log.info(f"Detector radiance saved to {Path(output_root) / 'instrument.nc'}")

    return data
