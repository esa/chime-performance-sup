"""Independent numerical checks for the instrument image operations."""

import numpy as np
import pytest
from scipy.ndimage import convolve

xr = pytest.importorskip("xarray")
eopl = pytest.importorskip("eopl")

from chime_perf.algorithms.acquisition_algorithms import (  # noqa: E402
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
    spectral_response,
    striping_gains,
    temporal_sigma,
    temporal_snr,
)
from chime_perf.instrument import instrument  # noqa: E402


@pytest.fixture
def radiance():
    """A linear spatial field and curved spectrum expose interpolation and averaging errors."""
    act = 6 * np.arange(11)
    alt = 6 * np.arange(11)
    wave = np.arange(480.0, 621.0)
    values = 20 + act[:, None, None] / 10 + alt[None, :, None] / 20
    values = values + ((wave[None, None, :] - 550) / 30) ** 2
    image = xr.DataArray(
        values,
        dims=("ACT", "ALT", "SPE"),
        coords={"ACT": act, "ALT": alt, "SPE": wave},
        name="user_signal",
        attrs={"units": "W m-2 sr-1 um-1"},
    )
    for name, unit in (("ACT", "m"), ("ALT", "m"), ("SPE", "nm")):
        image[name].attrs["units"] = unit
    return eopl.image_t(image)


@pytest.fixture
def propagation_product(tmp_path, radiance):
    """A propagation file carries both radiance and its original atmospheric coefficients."""
    product = radiance.image().rename("radiance").to_dataset()
    for name, value in (
        ("path_radiance", 2.0),
        ("spherical_albedo", 0.1),
        ("transmission", 0.8),
        ("solar_scale", 100.0),
    ):
        product[name] = xr.full_like(product.SPE, value)
        product[name].attrs["units"] = (
            "1" if name in ("transmission", "spherical_albedo") else "W m-2 sr-1 um-1"
        )
    path = tmp_path / "toa.nc"
    product.to_netcdf(path)
    return path


def test_spectral_integration_matches_weighted_sum(radiance):
    """The instrument computes a normalised band integral, without another band average."""
    centres = channel_centres(radiance, [530.0, 570.0], 30, 0, 0, 0, np.random.default_rng(1))
    response = spectral_response(centres, radiance.image().SPE.values, 4.0)
    assert centres.image().name == radiance.image().name
    assert centres.image().attrs["units"] == "nm"
    assert response.image().name == centres.image().name
    assert response.image().attrs["units"] == "1"
    response.change_array_name("user_response")
    result = integrate_spectral(radiance, response, 30, 0).image()
    assert result.name == radiance.image().name
    assert result.attrs["units"] == radiance.image().attrs["units"]

    wave = radiance.image().SPE.values
    weights = np.exp(-0.5 * ((wave[None, :] - np.array([530, 570])[:, None]) / 4) ** 2)
    weights *= abs(wave[None, :] - np.array([530, 570])[:, None]) <= 16
    weights /= weights.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result, radiance.image().values @ weights.T)


def test_fused_convolution_matches_scipy(radiance):
    """Edge extension and fused selection match SciPy, including the outermost centres."""
    # An asymmetric kernel catches accidental correlation instead of convolution.
    kernel = np.arange(1.0, 10).reshape(3, 3)
    kernel /= kernel.sum()
    result = convolve_downsample(radiance, kernel, 2, 3).image()
    expected = convolve(radiance.image().values, kernel[:, :, None], mode="nearest")
    act = np.arange(0, 11, 2)
    alt = np.arange(0, 11, 3)

    np.testing.assert_allclose(result, expected[act[:, None], alt[None, :], :])
    np.testing.assert_array_equal(result.ACT, radiance.image().ACT[act])
    assert result.attrs["spatial_edge_mode"] == "edge"


def test_rotation_preserves_linear_field_and_map_coordinates(radiance):
    """Rotation interpolates inside the map and repeats its nearest edge outside."""
    result = rotate_to_sensor(radiance, 30).image()
    lookup_act = result.map_ACT.clip(
        min=float(radiance.image().ACT[0]), max=float(radiance.image().ACT[-1])
    )
    lookup_alt = result.map_ALT.clip(
        min=float(radiance.image().ALT[0]), max=float(radiance.image().ALT[-1])
    )
    expected = 20 + lookup_act / 10 + lookup_alt / 20
    expected = expected + ((result.SPE - 550) / 30) ** 2
    np.testing.assert_allclose(result, expected.transpose(*result.dims))
    assert bool(np.isfinite(result).all())
    # The requested map coordinates remain available, even for edge-extended samples.
    assert float(result.map_ACT.min()) < float(radiance.image().ACT[0])
    assert result.attrs["geometry"] == "sensor"


def test_noise_ara_and_striping_reuse_draws(radiance):
    """ARA retains temporal noise, while RXRA reuses each gain along the whole column."""
    relative = relative_radiance(radiance, 100.0)
    sigma = temporal_sigma(relative, 100.0, 200.0, 0.2)
    noise = draw_noise(sigma, np.random.default_rng(2))
    assert relative.image().attrs["units"] == "1"
    assert sigma.image().attrs["units"] == "W m-2 sr-1 um-1"
    assert noise.image().attrs["units"] == "W m-2 sr-1 um-1"
    for image in (relative, sigma, noise):
        assert image.image().name == radiance.image().name
    noise.change_array_name("user_noise")
    reference = np.random.default_rng(2).normal(size=radiance.image().shape)
    np.testing.assert_allclose(noise.image(), reference * sigma.image())

    zero_offset = eopl.image_t(xr.zeros_like(radiance.image()))
    biased = apply_absolute_error(radiance, noise, zero_offset, 1.025)
    np.testing.assert_allclose(biased.image(), 1.025 * radiance.image() + noise.image())

    assert biased.image().name == radiance.image().name
    gains = striping_gains(radiance, np.full(141, 0.004), np.random.default_rng(3))
    assert gains.image().attrs["units"] == "1"
    assert gains.image().name == radiance.image().name
    gains.change_array_name("user_gain")
    striped = apply_striping(radiance, gains).image()
    expected = radiance.image() * gains.image()
    np.testing.assert_allclose(striped, expected.transpose(*striped.dims))
    assert not np.array_equal(gains.image()[0], gains.image()[1])
    assert striped.name == radiance.image().name


def test_zero_radiance_has_finite_noise_and_offset(radiance):
    """The zero-signal limits avoid 0/0 in noise and the additive ARA term."""
    signal = eopl.image_t(xr.zeros_like(radiance.image()))
    relative = relative_radiance(signal, 100.0)
    sigma = temporal_sigma(relative, 100.0, 200.0, 0.2)
    electrons = electron_signal(relative, 0.0, 50.0)
    snr = temporal_snr(relative, 200.0, 0.2)
    assert electrons.image().attrs["units"] == "electron"
    assert snr.image().attrs["units"] == "1"
    assert electrons.image().name == signal.image().name
    offset = absolute_offset(signal, electrons, 100.0, 0.0, 50.0, 1.0, 150.0)

    np.testing.assert_allclose(sigma.image(), 0.5 * np.sqrt(0.2))
    np.testing.assert_allclose(offset.image(), 300.0)


def test_shared_spectral_error_and_missing_coverage(radiance):
    """A fully shared error has one shift, and insufficient response coverage is rejected."""
    centres = channel_centres(
        radiance, [530.0, 570.0], 30, 30, 0.5, 1, np.random.default_rng(4)
    )
    shifts = centres.image().values - np.array([530, 570])
    np.testing.assert_allclose(shifts, np.full(shifts.shape, shifts[0, 0]))

    with pytest.raises(ValueError, match="cover"):
        spectral_response(centres, [529, 530, 531], 4.0)


def test_instrument_preserves_atmosphere_and_repeats_seed(tmp_path, propagation_product):
    """Independent instrument runs reproduce the seed and copy the original atmosphere."""
    config = tmp_path / "pipeline.yml"
    config.write_text("instrument:\n  seed: 17\n", encoding="utf-8")
    result = instrument(
        input_path=propagation_product, config_path=config, output_root=tmp_path / "instrument"
    )
    repeated = instrument(input_path=propagation_product, config_path=config, save=False)
    xr.testing.assert_equal(result.image(), repeated.image())
    np.testing.assert_allclose(np.diff(result.image().ACT), 30)
    np.testing.assert_allclose(np.diff(result.image().ALT), 30)

    with xr.open_dataset(tmp_path / "instrument" / "instrument.nc") as saved:
        with xr.open_dataset(propagation_product) as original:
            np.testing.assert_array_equal(saved.path_radiance, original.path_radiance)
            np.testing.assert_array_equal(saved.atmosphere_SPE, original.SPE)
        assert saved.radiance.sizes["SPE"] < saved.sizes["atmosphere_SPE"]
        assert saved.radiance.attrs["spatial_oversampling_ACT"] == 1
        assert saved.radiance.attrs["source_spatial_oversampling_ACT"] == 5
        assert saved.nominal_fwhm.dims == ("SPE",)
        assert saved.rxra_gain.dims == ("ACT", "SPE")
        assert saved.temporal_noise.shape == saved.radiance.shape


def test_generic_noise_and_gain_preserve_custom_units(radiance):
    """Gaussian draws and dimensionless multiplication also work on temperature data."""
    radiance.image().attrs["units"] = "K"
    original = radiance.image().copy(deep=True)
    noise = draw_noise(radiance, np.random.default_rng(1))
    gains = striping_gains(radiance, np.zeros(141), np.random.default_rng(2))
    gains.change_array_name("a_different_gain_name")
    result = apply_striping(radiance, gains).image()

    assert noise.image().attrs["units"] == "K"
    assert noise.image().name == original.name
    xr.testing.assert_identical(result, original)
    xr.testing.assert_identical(radiance.image(), original)


def test_physical_noise_model_rejects_wrong_units(radiance):
    """The radiance model rejects temperature, and SNR requires dimensionless input."""
    radiance.image().attrs["units"] = "K"
    with pytest.raises(ValueError, match="Expected units"):
        relative_radiance(radiance, 100.0)

    with pytest.raises(ValueError, match="Expected units"):
        temporal_snr(radiance, 200.0, 0.2)


def test_edge_padding_preserves_constant_small_scene(radiance):
    """Even a kernel larger than the image keeps every centre and a constant signal."""
    source = eopl.image_t(radiance.image().isel(ACT=slice(0, 2), ALT=slice(0, 2)))
    source.image().values[:] = 7
    rotated = rotate_to_sensor(source, 35)
    kernel = np.ones((9, 9)) / 81
    result = convolve_downsample(rotated, kernel, 1, 1).image()

    np.testing.assert_allclose(result, 7)
    assert result.shape == source.image().shape


def test_edge_padding_preserves_missing_input(radiance):
    """Real missing measurements remain missing; padding does not replace them."""
    radiance.image().values[0, 0, :] = np.nan
    result = rotate_to_sensor(radiance, 0).image()
    assert bool(np.isnan(result[0, 0]).all())


@pytest.mark.parametrize("mode", ["reflect", "valid"])
def test_spatial_operations_reject_other_boundary_modes(radiance, mode):
    """Only nearest-edge extension is supported by both spatial operations."""
    with pytest.raises(ValueError, match="Only edge padding"):
        rotate_to_sensor(radiance, 20, edge_mode=mode)
    with pytest.raises(ValueError, match="Only edge padding"):
        convolve_downsample(radiance, np.ones((3, 3)) / 9, 1, 1, edge_mode=mode)
