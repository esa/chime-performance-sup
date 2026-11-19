"""Analytical checks for atmospheric reconstruction and the complete processing chain."""

import numpy as np
import pytest
import yaml

xr = pytest.importorskip("xarray")
eopl = pytest.importorskip("eopl")
pytest.importorskip("numba")

from chime_perf.algorithms.reconstruction_algorithms import (  # noqa: E402
    path_reflectance,
    remove_atmosphere,
    surface_reflectance,
    toa_reflectance,
)
from chime_perf.reconstruction import reconstruction  # noqa: E402
from chime_perf.run import main  # noqa: E402


@pytest.fixture
def instrument_product():
    """A constant atmosphere gives an exactly invertible forward radiance."""
    reflectance = 0.2
    radiance = 2 + 100 * 0.8 * reflectance / (1 - 0.1 * reflectance)
    image = xr.DataArray(
        np.full((3, 4, 2), radiance),
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": 30 * np.arange(3), "ALT": 30 * np.arange(4), "SPE": [530, 570]},
        name="radiance",
        attrs={"units": "W m-2 sr-1 um-1", "nominal_fwhm_nm": 10.0},
    )
    for name, unit in (("ACT", "m"), ("ALT", "m"), ("SPE", "nm")):
        image[name].attrs["units"] = unit

    product = image.to_dataset()
    product["nominal_fwhm"] = xr.full_like(image.SPE, 10.0, dtype=float)
    product["nominal_fwhm"].attrs["units"] = "nm"
    product = product.assign_coords(atmosphere_SPE=np.arange(480.0, 621.0))
    product.atmosphere_SPE.attrs["units"] = "nm"
    for name, value in (
        ("path_radiance", 2.0),
        ("spherical_albedo", 0.1),
        ("transmission", 0.8),
        ("solar_scale", 100.0),
    ):
        product[name] = xr.full_like(product.atmosphere_SPE, value)
        product[name].attrs["units"] = (
            "1" if name in ("transmission", "spherical_albedo") else "W m-2 sr-1 um-1"
        )
    return product


def test_reconstruction_reuses_archive_without_solver(
    tmp_path, instrument_product, monkeypatch
):
    """A standalone instrument product is sufficient; no solver or new settings are used."""

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("Reconstruction must never rerun the atmospheric solver.")

    monkeypatch.setattr("chime_perf.athmo.prepare_atmosphere", unexpected_solver)
    monkeypatch.setattr(
        "chime_perf.algorithms.athmo_algorithms.prepare_atmosphere", unexpected_solver
    )
    input_path = tmp_path / "instrument.nc"
    instrument_product.to_netcdf(input_path)
    config = tmp_path / "changed.yml"
    # Changing these settings after acquisition must not change the reconstruction.
    config.write_text("atmosphere:\n  aot550: 99\ninstrument:\n  fwhm_nm: 99\n")

    result = reconstruction(
        input_path=input_path, output_root=tmp_path / "reconstruction", config_path=config
    )
    np.testing.assert_allclose(result.image(), 0.2)
    assert result.image().attrs["units"] == "1"

    with xr.open_dataset(tmp_path / "reconstruction" / "reconstructed.nc") as saved:
        xr.testing.assert_equal(saved.path_radiance, instrument_product.path_radiance)
        np.testing.assert_allclose(saved.band_transmission, 0.8)
        assert bool(saved.valid.all())

    manifest = yaml.safe_load((tmp_path / "reconstruction" / "run_manifest.yaml").read_text())
    assert [step["name"] for step in manifest["operations"]][-4:] == [
        "toa_reflectance",
        "path_reflectance",
        "remove_atmosphere",
        "surface_reflectance",
    ]


def test_spectral_atmosphere_uses_nominal_band_weights(tmp_path, instrument_product):
    """Varying coefficients are response-averaged, rather than sampled at channel centres."""
    product = instrument_product
    wave = product.atmosphere_SPE.values
    product["path_radiance"].values[:] = 2 + ((wave - 550) / 20) ** 2
    path = tmp_path / "instrument.nc"
    product.to_netcdf(path)
    result = reconstruction(input_path=path, save=False).image()

    centre = product.SPE.values
    weight = np.exp(-0.5 * ((wave[None, :] - centre[:, None]) / (10 / 2.354820045)) ** 2)
    weight *= abs(wave[None, :] - centre[:, None]) <= 4 * (10 / 2.354820045)
    weight /= weight.sum(axis=1, keepdims=True)
    averaged_path = weight @ product.path_radiance.values
    y = (product.radiance.values - averaged_path) / 100 / 0.8
    expected = y / (1 + 0.1 * y)

    np.testing.assert_allclose(result, expected, rtol=1e-9)


def test_invalid_correction_is_masked_and_negative_values_remain():
    """Zero transmission, invalid solar scale and singular inversion produce NaN."""

    def image(values, units="1", name="user_image"):
        return eopl.image_t(
            xr.DataArray(values, dims="SPE", name=name, attrs={"units": units})
        )

    signal = image([-1.0, 2.0, 3.0], "W m-2 sr-1 um-1", "user_signal")
    scale = image([1.0, 0.0, 1.0], "W m-2 sr-1 um-1", "user_scale")
    toa = toa_reflectance(signal, scale)
    path = path_reflectance(image([0.0, 0.0, 0.0], "W m-2 sr-1 um-1"), scale)
    y = remove_atmosphere(toa, path, image([1.0, 1.0, 0.0]))
    result = surface_reflectance(y, image([0.0, 0.0, 0.0])).image()
    np.testing.assert_allclose(result, [-1.0, np.nan, np.nan], equal_nan=True)
    singular = surface_reflectance(image([-1.0]), image([1.0])).image()
    assert np.isnan(singular[0])
    assert result.name == signal.image().name
    assert result.attrs["units"] == "1"
    assert signal.image().attrs["units"] == "W m-2 sr-1 um-1"


def test_missing_saved_atmosphere_is_rejected(tmp_path, instrument_product):
    """Missing coefficients cause a clear error instead of silently generating replacements."""
    path = tmp_path / "incomplete.nc"
    instrument_product.drop_vars("transmission").to_netcdf(path)
    with pytest.raises(ValueError, match="saved P, S, T and C"):
        reconstruction(input_path=path, save=False)


@pytest.mark.parametrize("errors_enabled", [False, True])
def test_full_chain_memory_and_disk_handoffs_agree(
    tmp_path, instrument_product, errors_enabled
):
    """All four modules recover a uniform surface, with identical memory and disk results."""
    names = ["path_radiance", "spherical_albedo", "transmission", "solar_scale"]
    atmosphere = instrument_product[names].rename({"atmosphere_SPE": "SPE"})
    atmosphere_path = tmp_path / "atmosphere.nc"
    atmosphere.to_netcdf(atmosphere_path)

    scene = xr.DataArray(
        np.full((11, 11, 141), 0.2),
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": 6 * np.arange(11), "ALT": 6 * np.arange(11), "SPE": atmosphere.SPE},
        name="reflectance",
        attrs={"units": "1"},
    )
    scene.ACT.attrs["units"] = "m"
    scene.ALT.attrs["units"] = "m"
    input_path = tmp_path / "scene.nc"
    scene.to_netcdf(input_path)
    config = tmp_path / "config.yml"
    settings = {
        "instrument": {
            "heading_deg": 30.0,
            "temporal_noise": errors_enabled,
            "rxra": errors_enabled,
            "spectral_error_sigma_nm": 0.2 if errors_enabled else 0,
            "ara_gain_coefficient": 1 if errors_enabled else 0,
            "ara_offset_coefficient": 1 if errors_enabled else 0,
        }
    }
    config.write_text(yaml.safe_dump(settings))

    for mode in ("memory", "disk"):
        arguments = [
            "--all",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / mode),
            "--config",
            str(config),
            "--atmosphere",
            str(atmosphere_path),
        ]
        if mode == "disk":
            arguments.append("--save-intermediates")
        main(arguments)

    with xr.open_dataset(tmp_path / "memory/reconstruction/reconstructed.nc") as memory:
        with xr.open_dataset(tmp_path / "disk/reconstruction/reconstructed.nc") as disk:
            xr.testing.assert_equal(memory.reflectance, disk.reflectance)
            assert bool(np.isfinite(memory.reflectance).all())
            if not errors_enabled:
                np.testing.assert_allclose(memory.reflectance, 0.2)
    assert (tmp_path / "memory/athmo/toa.nc").is_file()
    assert not (tmp_path / "memory/instrument/instrument.nc").exists()
    assert (tmp_path / "disk/instrument/instrument.nc").is_file()
