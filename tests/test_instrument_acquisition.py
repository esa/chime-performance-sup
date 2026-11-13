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
    integrate_spectral,
    relative_radiance,
    rotate_to_sensor,
    spectral_response,
    striping_gains,
    temporal_sigma,
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
        name="radiance",
        attrs={"units": "W m-2 sr-1 um-1"},
    )
    for name, unit in (("ACT", "m"), ("ALT", "m"), ("SPE", "nm")):
        image[name].attrs["units"] = unit
    return eopl.image_t(image)


@pytest.fixture
def propagation_product(tmp_path, radiance):
    """A propagation file carries both radiance and its original atmospheric coefficients."""
    product = radiance.image().to_dataset()
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
    result = integrate_spectral(radiance, response, 30, 0).image()

    wave = radiance.image().SPE.values
    weights = np.exp(-0.5 * ((wave[None, :] - np.array([530, 570])[:, None]) / 4) ** 2)
    weights *= abs(wave[None, :] - np.array([530, 570])[:, None]) <= 16
    weights /= weights.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result, radiance.image().values @ weights.T)


@pytest.mark.parametrize("edge_mode", ["reflect", "valid"])
def test_fused_convolution_matches_scipy(radiance, edge_mode):
    """Fusing convolution and selection preserves the kernel sign and sampling phase."""
    # An asymmetric kernel catches accidental correlation instead of convolution.
    kernel = np.arange(1.0, 10).reshape(3, 3)
    kernel /= kernel.sum()
    result = convolve_downsample(radiance, kernel, 2, 3, edge_mode).image()
    expected = convolve(radiance.image().values, kernel[:, :, None], mode="reflect")
    act = np.arange(0, 11, 2)
    alt = np.arange(0, 11, 3)
    if edge_mode == "valid":
        act = act[(act >= 1) & (act < 10)]
        alt = alt[(alt >= 1) & (alt < 10)]

    np.testing.assert_allclose(result, expected[act[:, None], alt[None, :], :])
    np.testing.assert_array_equal(result.ACT, radiance.image().ACT[act])


def test_rotation_preserves_linear_field_and_map_coordinates(radiance):
    """Bilinear rotation reproduces a linear field and masks positions outside the map."""
    result = rotate_to_sensor(radiance, 30).image()
    expected = 20 + result.map_ACT / 10 + result.map_ALT / 20
    expected = expected + ((result.SPE - 550) / 30) ** 2
    valid = np.isfinite(result)

    np.testing.assert_allclose(
        result.values[valid], expected.transpose(*result.dims).values[valid]
    )
    assert np.isnan(result).any()
    assert result.attrs["geometry"] == "sensor"


def test_noise_ara_and_striping_reuse_draws(radiance):
    """ARA retains temporal noise, while RXRA reuses each gain along the whole column."""
    relative = relative_radiance(radiance, 100.0)
    sigma = temporal_sigma(relative, 100.0, 200.0, 0.2)
    noise = draw_noise(sigma, np.random.default_rng(2))
    reference = np.random.default_rng(2).normal(size=radiance.image().shape)
    np.testing.assert_allclose(noise.image(), reference * sigma.image())

    zero_offset = eopl.image_t(xr.zeros_like(radiance.image()))
    biased = apply_absolute_error(radiance, noise, zero_offset, 1.025)
    np.testing.assert_allclose(biased.image(), 1.025 * radiance.image() + noise.image())

    gains = striping_gains(radiance, np.full(141, 0.004), np.random.default_rng(3))
    striped = apply_striping(radiance, gains).image()
    expected = radiance.image() * gains.image()
    np.testing.assert_allclose(striped, expected.transpose(*striped.dims))
    assert not np.array_equal(gains.image()[0], gains.image()[1])


def test_zero_radiance_has_finite_noise_and_offset(radiance):
    """The zero-signal limits avoid 0/0 in noise and the additive ARA term."""
    signal = eopl.image_t(xr.zeros_like(radiance.image()))
    relative = relative_radiance(signal, 100.0)
    sigma = temporal_sigma(relative, 100.0, 200.0, 0.2)
    electrons = eopl.image_t(xr.zeros_like(signal.image()))
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
