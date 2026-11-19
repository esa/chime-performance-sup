"""Reconstruction of surface reflectance using the saved propagation atmosphere."""

from pathlib import Path

import numpy as np
import xarray as xr
from eopl import Pipeline, image_t

from chime_perf.algorithms.acquisition_algorithms import spectral_response
from chime_perf.algorithms.instrument_algorithms import isrf_sigma
from chime_perf.algorithms.reconstruction_algorithms import (
    average_atmosphere,
    path_reflectance,
    remove_atmosphere,
    surface_reflectance,
    toa_reflectance,
)
from chime_perf.params import boundary_settings


def reconstruction(
    input_path=None,
    output_root="outputs/reconstruction",
    config_path=None,
    atmosphere_path=None,
    data=None,
    save=True,
):
    """Retrieve BOA reflectance without invoking an atmospheric solver."""
    pl = Pipeline(output_root=output_root)

    # 1. INITIALIZATION AND CONFIGURATION
    if config_path is not None:
        pl.add_local_config(config_path)

    pl.set_run_options(**pl.local_config.get("run_options", {"precision": "float64"}))
    pl.set_image_type(pl.run_options.precision)
    report_params = pl.local_config.get("report")
    pl.generate_report(report_params is not None, report_params)
    pl.generate_breakpoints(pl.local_config.get("breakpoints", False) and save)

    boundaries = boundary_settings(pl.local_config)
    product = None
    if data is None:
        if input_path is None:
            raise ValueError("Supply input_path or an in-memory image through data.")
        with xr.open_dataset(input_path) as source:
            product = source.load()
        data = image_t(product["radiance"])
        pl.manifest["input"] = {"file": str(Path(input_path).resolve())}

    image = data.image().transpose("ACT", "ALT", "SPE")
    if image.attrs.get("units") != "W m-2 sr-1 um-1":
        raise ValueError("Reconstruction input must be radiance in W m-2 sr-1 um-1.")
    if image.SPE.attrs.get("units") != "nm":
        raise ValueError("SPE coordinates must have units='nm'.")
    data = image_t(image)

    # The archive is either embedded in the instrument product or supplied for
    # an in-memory handoff. YAML atmosphere settings never replace saved quantities.
    coefficient_path = atmosphere_path if atmosphere_path is not None else input_path
    if atmosphere_path is not None or product is None:
        if coefficient_path is None:
            raise ValueError("Supply the saved propagation product through atmosphere_path.")
        with xr.open_dataset(coefficient_path) as source:
            atmosphere = source.load()
    else:
        atmosphere = product

    names = ("path_radiance", "spherical_albedo", "transmission", "solar_scale")
    if not all(name in atmosphere for name in names):
        raise ValueError("Reconstruction requires the saved P, S, T and C quantities.")
    axis = "atmosphere_SPE" if "atmosphere_SPE" in atmosphere.dims else "SPE"
    original_nm = atmosphere[axis].values
    for name in names:
        if atmosphere[name].dims != (axis,):
            raise ValueError(f"Saved {name} must depend only on atmospheric wavelength.")
        unit = "1" if name in ("spherical_albedo", "transmission") else "W m-2 sr-1 um-1"
        if atmosphere[name].attrs.get("units") != unit:
            raise ValueError(f"Saved {name} must have units={unit!r}.")

    # Reuse the instrument's nominal response width, even if YAML has since changed.
    if product is not None and "nominal_fwhm" in product:
        if not np.array_equal(product.nominal_fwhm.SPE, image.SPE):
            raise ValueError("Saved response widths must match the measured channels.")
        fwhm = product.nominal_fwhm.values
    else:
        fwhm = image.attrs.get("nominal_fwhm_nm")
        if fwhm is None:
            raise ValueError("The instrument product must record its nominal response width.")

    support_sigma = image.attrs.get("kernel_support_sigma", boundaries["kernel_support_sigma"])

    # 2. OPERATION PHASE
    # Step 1: Average the saved C, T, S and P with the UNSHIFTED channel response.
    # This is a band approximation: averaging does not commute with the nonlinear
    # forward equation when atmosphere or surface spectra vary within a channel.
    nominal = image_t(image.SPE)
    sigma = pl.run_function(isrf_sigma, args={"fwhm_nm": np.asarray(fwhm)})
    response = pl.run_operation(
        data=nominal,
        func=spectral_response,
        args={
            "wavelength_nm": original_nm,
            "sigma_nm": sigma,
            "support_sigma": support_sigma,
        },
    )

    channels = {}
    for name in names:
        channels[name] = pl.run_operation(
            data=(image_t(atmosphere[name]), response), func=average_atmosphere
        )

    # Step 2: Convert measured radiance to TOA reflectance, rho* = L / C.
    toa = pl.run_operation(data=(data, channels["solar_scale"]), func=toa_reflectance)
    toa = pl.save_bkp(image_t(toa.image(), name="toa_reflectance"), "toa_reflectance")

    # Step 3: Atmospheric path reflectance, rho_atm = P / C.
    path = pl.run_operation(
        data=(channels["path_radiance"], channels["solar_scale"]), func=path_reflectance
    )

    # Step 4: Subtract the path term and remove transmission, y = (rho* - rho_atm) / T.
    contribution = pl.run_operation(
        data=(toa, path, channels["transmission"]), func=remove_atmosphere
    )
    contribution = pl.save_bkp(
        image_t(contribution.image(), name="surface_contribution"), "surface_contribution"
    )

    # Step 5: Remove repeated reflection, rho = y / (1 + S y).
    data = pl.run_operation(
        data=(contribution, channels["spherical_albedo"]), func=surface_reflectance
    )
    data.image().attrs["atmospheric_correction"] = "saved atmosphere; nominal band averages"

    # 3. FINAL PHASE
    if save:
        for name in names:
            original = atmosphere[name]
            if axis != "atmosphere_SPE":
                original = original.rename({axis: "atmosphere_SPE"})
            pl.add_variable(name, original, pinned=True)
            pl.add_variable(f"band_{name}", channels[name].image(), pinned=True)

        pl.add_variable("valid", np.isfinite(data.image()), pinned=True)
        pl.manifest["atmosphere_source"] = str(Path(coefficient_path).resolve())
        pl.save_output(image_t(data.image(), name="reflectance"), "reconstructed")
        pl.log.info(f"BOA reflectance saved to {Path(output_root) / 'reconstructed.nc'}")

    return data
