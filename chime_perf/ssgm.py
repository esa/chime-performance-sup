"""Scientific scene generation from BOA reflectance and spectral libraries."""

from pathlib import Path

import xarray as xr
from eopl import Pipeline

from chime_perf.algorithms.algorithms import grid
from chime_perf.algorithms.ssgm_algorithms import (
    fit_library,
    interpolate_spatial,
    interpolate_spectral,
    reconstruct_library,
    validate_spatial_sampling,
)


def ssgm(
    input_path=None,
    output_root="outputs/ssgm",
    config_path=None,
    library_path=None,
    data=None,
    save=True,
):
    """Prepare a reflectance scene; return the image for the next module."""
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

    settings = pl.local_config.get("scene", {})
    spatial_interpolation = settings.get("spatial_interpolation", False)
    detector_spacing_m = settings.get("detector_spacing_m", 30.0)

    if not isinstance(spatial_interpolation, bool):
        raise ValueError("scene.spatial_interpolation must be true or false.")

    target_nm = grid(data.image().SPE.values, settings.get("spectral_step_nm"))

    library = None
    if library_path is not None:
        with xr.open_dataset(library_path) as source:
            library = source.load()

    # 2. OPERATION PHASE
    # Step 1: Spatial interpolation onto an integer-oversampling grid.
    # Align the scene once, before atmospheric propagation and instrument convolution.
    if spatial_interpolation:
        data = pl.run_operation(
            data=data,
            func=interpolate_spatial,
            args={
                "detector_spacing_m": detector_spacing_m,
                "oversampling": settings.get("spatial_oversampling", "auto"),
            },
        )
        data = pl.save_bkp(data, "spatial_interpolation")

    validate_spatial_sampling(data, detector_spacing_m)

    # Step 2a: Direct spectral interpolation.
    if library is None:
        # Direct spectral interpolation at every requested wavelength.
        data = pl.run_operation(
            data=data,
            func=interpolate_spectral,
            args={"step_nm": settings.get("spectral_step_nm")},
        )
    # Step 2b: Non-negative library fit at the input bands.
    else:
        # First find c >= 0 matching the measured input bands.
        weights = pl.run_operation(
            data=data,
            func=fit_library,
            args={"library": library},
        )
        weights = pl.save_bkp(weights, "mixture_weights")

        # Step 3: Reconstruct rho(lambda_m) = sum_r c_r e_r(lambda_m).
        data = pl.run_operation(
            data=weights,
            func=reconstruct_library,
            args={"library": library, "target_nm": target_nm},
        )

    # 3. FINAL PHASE
    if save:
        pl.save_output(data, "scene")
        pl.log.info(f"Scene saved to {Path(output_root) / 'scene.nc'}")

    return data
