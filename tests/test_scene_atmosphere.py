"""Analytical checks for scene preparation, atmosphere and module handoff."""

import os

import numpy as np
import pytest
import yaml

pytest.importorskip("numba")
xr = pytest.importorskip("xarray")
eopl = pytest.importorskip("eopl")

from chime_perf.algorithms.algorithms import grid  # noqa: E402
from chime_perf.algorithms.athmo_algorithms import (  # noqa: E402
    calculate_solar_scale,
    combine_transmission,
    prepare_atmosphere,
    propagate,
)
from chime_perf.algorithms.ssgm_algorithms import spatial_grid  # noqa: E402
from chime_perf.athmo import athmo  # noqa: E402
from chime_perf.ssgm import ssgm  # noqa: E402


@pytest.fixture
def reflectance():
    """A linear reflectance cube gives an exact reference for interpolation."""
    act = np.array([0.0, 30.0])
    alt = np.array([0.0, 30.0])
    spe = np.array([500.0, 510.0, 520.0])

    values = 0.1 + act[:, None, None] / 1000 + alt[None, :, None] / 2000
    values = values + (spe[None, None, :] - 500) / 10000

    image = xr.DataArray(
        values,
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": act, "ALT": alt, "SPE": spe},
        name="reflectance",
        attrs={"units": "1", "geometry": "map"},
    )
    for name, unit in (("ACT", "m"), ("ALT", "m"), ("SPE", "nm")):
        image[name].attrs["units"] = unit

    return eopl.image_t(image)


@pytest.fixture
def atmosphere(reflectance):
    """Fixed physical quantities make the forward equation easy to verify."""
    quantities = xr.Dataset(
        {
            "transmission": ("SPE", [0.8] * 3),
            "spherical_albedo": ("SPE", [0.1] * 3),
            "path_radiance": ("SPE", [2.0] * 3),
            "solar_scale": ("SPE", [100.0] * 3),
        },
        coords={"SPE": reflectance.image().SPE},
    )
    for name in quantities.data_vars:
        quantities[name].attrs["units"] = (
            "1" if name in ("transmission", "spherical_albedo") else "W m-2 sr-1 um-1"
        )

    return quantities


def test_scene_interpolation_preserves_linear_field(tmp_path, reflectance):
    """The SSGM evaluates a known linear field without writing intermediate products."""
    config = tmp_path / "config.yml"
    config.write_text(
        "scene:\n  spatial_interpolation: true\n  spatial_oversampling: 5\n"
        "  spectral_step_nm: 2\n"
    )
    output = tmp_path / "unused"

    result = ssgm(data=reflectance, config_path=config, output_root=output, save=False)
    fine = result.image()
    expected = 0.1 + fine.ACT / 1000 + fine.ALT / 2000 + (fine.SPE - 500) / 10000

    np.testing.assert_allclose(fine, expected.transpose(*fine.dims))
    assert fine.ACT.attrs["units"] == "m"
    assert fine.attrs["units"] == "1"
    assert not output.exists()


def test_scene_keeps_existing_grid(reflectance):
    """Keep known samples and missing pixels when existing coordinates are adequate."""
    reflectance.image().values[0, 0, :] = np.nan
    result = ssgm(data=reflectance, save=False)

    xr.testing.assert_allclose(result.image(), reflectance.image())


@pytest.mark.parametrize("values, step", [([0, 0], 1), ([0, 10], 0), ([0, 10], 20)])
def test_scene_rejects_invalid_or_coarser_grids(values, step):
    """Reject ambiguous coordinates and accidental loss of input resolution."""
    with pytest.raises(ValueError):
        grid(values, step)


def test_library_operations_are_independent_and_recover_mixture(tmp_path, reflectance):
    """The manifest separates spatial interpolation, NNLS fitting and reconstruction."""
    measured = np.array([[0.1, 0.4], [0.2, 0.3], [0.4, 0.1]])
    target = np.array([500, 505, 510, 515, 520])
    basis = np.array([[0.1, 0.4], [0.3, 0.35], [0.2, 0.3], [0.5, 0.2], [0.4, 0.1]])
    library = xr.Dataset(
        {
            "measured": (("SPE", "material"), measured),
            "reflectance": (("library_SPE", "material"), basis),
        },
        coords={
            "SPE": reflectance.image().SPE,
            "library_SPE": target,
            "material": ["one", "two"],
        },
    )
    library_path = tmp_path / "library.nc"
    library.to_netcdf(library_path)
    reflectance.image().values[:] = measured @ np.array([0.25, 0.75])

    config = tmp_path / "config.yml"
    config.write_text(
        "scene:\n  spatial_interpolation: true\n  spatial_oversampling: 1\n"
        "  spectral_step_nm: 5\nbreakpoints: true\n"
    )
    output = tmp_path / "scene"

    result = ssgm(
        data=reflectance, config_path=config, library_path=library_path, output_root=output
    )

    np.testing.assert_allclose(result.image()[0, 0], basis @ np.array([0.25, 0.75]))
    manifest = yaml.safe_load((output / "run_manifest.yaml").read_text())
    assert [operation["name"] for operation in manifest["operations"]] == [
        "interpolate_spatial",
        "fit_library",
        "reconstruct_library",
    ]
    assert (output / "breakpoints" / "mixture_weights.nc").is_file()


def test_transmission_and_solar_scale_are_separate_operations(reflectance):
    """Include gas and both scattering paths, and use degrees for solar zenith."""
    spe = reflectance.image().SPE
    gas = eopl.image_t(xr.DataArray([0.9] * 3, dims="SPE", coords={"SPE": spe}))
    downward = eopl.image_t(xr.DataArray([0.8] * 3, dims="SPE", coords={"SPE": spe}))
    upward = eopl.image_t(xr.DataArray([0.7] * 3, dims="SPE", coords={"SPE": spe}))
    irradiance = eopl.image_t(xr.DataArray([100.0] * 3, dims="SPE", coords={"SPE": spe}))

    transmission = combine_transmission(gas, downward, upward)
    scale = calculate_solar_scale(irradiance, solar_zenith_deg=60)

    np.testing.assert_allclose(transmission.image(), 0.9 * 0.8 * 0.7)
    np.testing.assert_allclose(scale.image(), 50 / np.pi)


def test_forward_atmosphere_and_ideal_inverse(reflectance, atmosphere):
    """The paper's inverse recovers reflectance before instrument effects."""
    radiance = propagate(reflectance, atmosphere).image()
    rho = reflectance.image()
    expected = 2 + 100 * 0.8 * rho / (1 - 0.1 * rho)

    np.testing.assert_allclose(radiance, expected)

    y = (radiance - 2) / (100 * 0.8)
    recovered = y / (1 + 0.1 * y)

    np.testing.assert_allclose(recovered, rho)


def test_missing_scene_pixels_remain_missing(reflectance, atmosphere):
    """Do not replace invalid scene pixels with an invented atmospheric signal."""
    reflectance.image().values[0, 0, :] = np.nan

    radiance = propagate(reflectance, atmosphere).image()

    assert np.isnan(radiance.values[0, 0]).all()
    assert np.isfinite(radiance.values[1, 1]).all()


def test_atmosphere_requires_matching_coordinates(reflectance, atmosphere):
    """A shifted coefficient grid must not silently discard spectral samples."""
    atmosphere = atmosphere.assign_coords(SPE=atmosphere.SPE + 1)
    atmosphere.SPE.attrs["units"] = "nm"

    with pytest.raises(ValueError):
        propagate(reflectance, atmosphere)


@pytest.mark.parametrize("disk_handoff", [False, True])
def test_all_matches_standalone_modules(tmp_path, reflectance, atmosphere, disk_handoff):
    """Memory and save/reload handoffs produce the same independently expected radiance."""
    from chime_perf.run import main

    input_path = tmp_path / "input.nc"
    reflectance.image().to_netcdf(input_path)
    atmosphere_path = tmp_path / "atmosphere.nc"
    # Reloaded NetCDF settings must stay readable in the YAML report manifest.
    atmosphere.attrs["solar_zenith_deg"] = np.float64(30)
    atmosphere.attrs["month"] = np.int64(6)
    atmosphere.to_netcdf(atmosphere_path)
    output = tmp_path / "all"

    arguments = [
        "--all",
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--atmosphere",
        str(atmosphere_path),
    ]
    if disk_handoff:
        arguments.append("--save-intermediates")

    main(arguments)

    with xr.open_dataset(output / "athmo" / "toa.nc") as saved:
        np.testing.assert_allclose(saved.radiance, propagate(reflectance, atmosphere).image())
        assert saved.transmission.attrs["units"] == "1"
        assert bool(saved.valid.all())

    assert (output / "ssgm").exists() == disk_handoff
    manifest = yaml.safe_load((output / "athmo" / "run_manifest.yaml").read_text())
    assert manifest["atmosphere"] == {"solar_zenith_deg": 30.0, "month": 6}


@pytest.mark.skipif(
    not os.environ.get("SIXS_PATH"), reason="Set SIXS_PATH for the solver check"
)
def test_6s_forward_matches_direct_solver(tmp_path, reflectance):
    """The composed atmospheric operations agree with direct 6S Lambertian radiance."""
    from Py6S import AeroProfile, AtmosProfile, Geometry, GroundReflectance, SixS, Wavelength

    image = reflectance.image().isel(SPE=[0]).assign_coords(SPE=[550.0])
    image.SPE.attrs["units"] = "nm"
    image.values[:] = 0.2
    output = tmp_path / "atmosphere"

    result = athmo(
        data=eopl.image_t(image), output_root=output, sixs_path=os.environ["SIXS_PATH"]
    )

    model = SixS(os.environ["SIXS_PATH"])
    model.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.USStandard1962)
    model.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    model.aot550 = 0.1
    model.geometry = Geometry.User()
    model.geometry.solar_z, model.geometry.solar_a = 30, 0
    model.geometry.view_z, model.geometry.view_a = 0, 0
    model.geometry.month, model.geometry.day = 6, 21
    model.altitudes.set_target_custom_altitude(0)
    model.altitudes.set_sensor_satellite_level()
    model.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.2)
    model.wavelength = Wavelength(0.55)
    model.run()

    # Physical quantities in the 6S output have limited printed precision.
    np.testing.assert_allclose(result.image(), model.outputs.apparent_radiance, rtol=0.003)
    manifest = yaml.safe_load((output / "run_manifest.yaml").read_text())
    assert [operation["name"] for operation in manifest["operations"]] == [
        "prepare_atmosphere",
        "combine_transmission",
        "calculate_solar_scale",
        "propagate",
    ]


@pytest.mark.skipif(
    not os.environ.get("SIXS_PATH"), reason="Set SIXS_PATH for the solver check"
)
def test_6s_quantities_cover_non_native_scene_wavelengths():
    """Interpolate solver quantities explicitly without non-finite results."""
    target = [549.5, 550.4]
    atmosphere = prepare_atmosphere(target, sixs_path=os.environ["SIXS_PATH"])

    np.testing.assert_array_equal(atmosphere.SPE, target)
    assert bool(np.isfinite(atmosphere.to_array()).all())
    assert atmosphere.attrs["solver_step_nm"] == 2.5


@pytest.mark.parametrize(
    "spacing_m, expected_factor", [(3.0, 10), (4.0, 8), (4.5, 7), (5.0, 6), (5.5, 5), (6.0, 5)]
)
def test_nearest_integer_spatial_grid_hits_detector_centres(spacing_m, expected_factor):
    """A shifted input origin still places every nominal centre on an exact fine sample."""
    coordinates = 101.25 + np.arange(31) * spacing_m

    fine, factor = spatial_grid(coordinates, detector_spacing_m=30)

    assert factor == expected_factor
    np.testing.assert_allclose(np.diff(fine), 30 / expected_factor)
    nominal = fine[::factor]
    np.testing.assert_allclose(nominal, coordinates[0] + 30 * np.arange(len(nominal)))
    assert fine[-1] <= coordinates[-1] + 1e-9


@pytest.mark.parametrize("factor", [5.5, 5.0, True, 0, -1])
def test_spatial_oversampling_requires_integer_factor(factor):
    """The requested oversampling is an integer, even when a float has an integer value."""
    with pytest.raises(ValueError, match="positive integer"):
        spatial_grid([0, 30, 60], 30, oversampling=factor)


def test_spatial_resampling_aligns_noninteger_input_without_changing_linear_field(
    tmp_path, reflectance
):
    """Nearest-factor interpolation changes a 5.5 m grid to 6 m with the same linear field."""
    act = 10 + np.arange(20) * 5.5
    alt = -20 + np.arange(18) * 5.5
    values = 0.2 + act[:, None, None] / 1000 + alt[None, :, None] / 2000
    values = values + np.arange(3)[None, None, :] / 1000
    image = xr.DataArray(
        values,
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": act, "ALT": alt, "SPE": reflectance.image().SPE},
        attrs={"units": "1"},
    )
    image.ACT.attrs["units"] = "m"
    image.ALT.attrs["units"] = "m"

    config = tmp_path / "sampling.yml"
    config.write_text(
        "scene:\n  spatial_interpolation: true\n"
        "  detector_spacing_m: 30\n  spatial_oversampling: auto\n"
    )

    result = ssgm(data=eopl.image_t(image), config_path=config, save=False).image()
    expected = 0.2 + result.ACT / 1000 + result.ALT / 2000
    expected = expected + xr.DataArray(
        np.arange(3) / 1000, dims="SPE", coords={"SPE": result.SPE}
    )

    np.testing.assert_allclose(result, expected.transpose(*result.dims))
    np.testing.assert_allclose(np.diff(result.ACT), 6)
    np.testing.assert_allclose(np.diff(result.ALT), 6)
    assert result.attrs["spatial_oversampling_ACT"] == 5
    assert result.attrs["detector_origin_ACT_m"] == 10


def test_noninteger_input_requires_explicit_spatial_interpolation(reflectance):
    """Disabling interpolation cannot silently pass a fractional grid to the instrument."""
    image = reflectance.image().assign_coords(ACT=[0, 5.5], ALT=[0, 5.5])
    image.ACT.attrs["units"] = "m"
    image.ALT.attrs["units"] = "m"

    with pytest.raises(ValueError, match="enable scene.spatial_interpolation"):
        ssgm(data=eopl.image_t(image), save=False)
