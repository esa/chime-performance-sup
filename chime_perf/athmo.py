"""Atmospheric propagation of a surface-reflectance scene to TOA radiance."""

from pathlib import Path

import numpy as np
import xarray as xr
from eopl import Pipeline, image_t

from chime_perf.algorithms.algorithms import reflectance_image
from chime_perf.algorithms.athmo_algorithms import (
    calculate_solar_scale,
    combine_transmission,
    prepare_atmosphere,
    propagate,
)


def athmo(
    input_path=None,
    output_root="outputs/athmo",
    config_path=None,
    atmosphere_path=None,
    sixs_path=None,
    data=None,
    save=True,
):
    """Apply the atmosphere; return the radiance image for the next module."""
    pl = Pipeline(output_root=output_root)

    # 1. INITIALIZATION AND CONFIGURATION
    if config_path is not None:
        pl.add_local_config(config_path)

    pl.set_run_options(**pl.local_config.get("run_options", {"precision": "float64"}))
    pl.set_image_type(pl.run_options.precision)

    report_params = pl.local_config.get("report")
    pl.generate_report(report_params is not None, report_params)
    pl.generate_breakpoints(pl.local_config.get("breakpoints", False) and save)

    if data is None:
        if input_path is None:
            raise ValueError("Supply input_path or an in-memory image through data.")
        data = pl.read_input(str(input_path), dask=False)

    reflectance = reflectance_image(data)

    if atmosphere_path is not None:
        with xr.open_dataset(atmosphere_path) as source:
            atmosphere = source.load()
    else:
        # The radiative-transfer solver supplies the inputs to the image equations.
        atmosphere = pl.run_function(
            prepare_atmosphere,
            args={
                "wavelength_nm": reflectance.SPE.values,
                "sixs_path": sixs_path,
                **pl.local_config.get("atmosphere", {}),
            },
        )

    # 2. OPERATION PHASE
    # Step 1: Combine gas absorption and the two scattering-transmission paths.
    # A saved solver product may already contain the combined T and C quantities.
    if "transmission" not in atmosphere:
        transmission = pl.run_operation(
            data=(
                image_t(atmosphere["gas_transmission"]),
                image_t(atmosphere["downward_transmission"]),
                image_t(atmosphere["upward_transmission"]),
            ),
            func=combine_transmission,
        )
        atmosphere["transmission"] = transmission.image()
        pl.save_bkp(image_t(transmission.image(), name="transmission"), "transmission")

    # Step 2: Convert solar irradiance to the radiance scale C.
    if "solar_scale" not in atmosphere:
        scale = pl.run_operation(
            data=image_t(atmosphere["solar_irradiance"]),
            func=calculate_solar_scale,
            args={"solar_zenith_deg": atmosphere.attrs["solar_zenith_deg"]},
        )
        atmosphere["solar_scale"] = scale.image()
        pl.save_bkp(image_t(scale.image(), name="solar_scale"), "solar_scale")

    # Step 3: L = P + C T rho / (1 - S rho), at every pixel and wavelength.
    data = pl.run_operation(
        data=data,
        func=propagate,
        args={"atmosphere": atmosphere},
    )

    # 3. FINAL PHASE
    if save:
        # Keep physical units and coordinates so another solver can supply the same product.
        for name in atmosphere.data_vars:
            pl.add_variable(name, atmosphere[name], pinned=True)

        valid = np.isfinite(data.image())
        valid.attrs = {"units": "1"}
        pl.add_variable("valid", valid, pinned=True)
        # NetCDF attributes may be NumPy scalars; YAML needs ordinary Python values.
        pl.manifest["atmosphere"] = {}
        for name, value in atmosphere.attrs.items():
            if isinstance(value, np.generic):
                value = value.item()
            pl.manifest["atmosphere"][name] = value

        Path(output_root).mkdir(parents=True, exist_ok=True)
        atmosphere.to_netcdf(Path(output_root) / "atmosphere.nc")
        pl.save_output(image_t(data.image(), name="radiance"), "toa")
        pl.log.info(f"TOA radiance saved to {Path(output_root) / 'toa.nc'}")

    return data
