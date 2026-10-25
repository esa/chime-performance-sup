"""Radiometric reference values, broadcasting and column-wise striping."""

import numpy as np
import pytest

from chime_perf.radiometric import apply_rxra, ara, nedl, rxra_gain, snr


# Each input pair runs the same test with a different radiance and expected SNR.
@pytest.mark.parametrize("radiance, expected", [(0.0, 0.0), (10.0, 5.0), (60.0, 20.0)])
def test_snr_at_known_radiances(radiance, expected):
    """Check SNR against three values that can be calculated by hand."""

    result = snr(radiance, lref=10.0, g=10.0, n0=3.0)

    # Allow small differences from floating-point rounding.
    assert result == pytest.approx(expected)


def test_snr_and_nedl_broadcast_over_a_cube():
    """Check that channel parameters apply correctly across an image cube."""

    radiance = np.array([[[10.0, 20.0]], [[60.0, 120.0]]])

    result = snr(radiance, lref=np.array([10.0, 20.0]), g=np.array([10.0, 20.0]), n0=3.0)

    np.testing.assert_allclose(result, [[[5.0, 10.0]], [[20.0, 40.0]]])
    np.testing.assert_allclose(nedl(radiance, result), [[[2.0, 2.0]], [[3.0, 3.0]]])


@pytest.mark.parametrize("radiance, expected", [(10.0, 0.05), (20.0, 0.0375)])
def test_ara_combines_gain_and_offset_errors(radiance, expected):
    """Check that ARA includes both the gain bias and the offset contribution."""

    result = ara(radiance, lref=10.0, ara_a=0.0, ara_b=6000.0, k=0.025, s0=150.0)

    assert result == pytest.approx(expected)


def test_ara_without_offset_is_the_constant_gain_error():
    """Check that removing the offset leaves only the constant gain error."""

    result = ara(np.array([1.0, 10.0, 100.0]), 10.0, 0.0, 6000.0, 0.025, 0.0)

    np.testing.assert_allclose(result, 0.025)


def test_rxra_gains_are_reproducible_and_have_the_requested_spread():
    """Check that a fixed seed repeats the gains and gives the expected spread."""

    sigma = np.array([0.0, 0.004, 0.017])

    # Reusing a seed must reproduce the same simulated acquisition.
    gain = rxra_gain(20000, sigma, np.random.default_rng(42))
    repeated = rxra_gain(20000, sigma, np.random.default_rng(42))

    assert gain.shape == (20000, 3)

    np.testing.assert_array_equal(gain, repeated)
    np.testing.assert_array_equal(gain[:, 0], 1.0)

    # Check the distribution, with tolerances comfortably above sampling variation.
    np.testing.assert_allclose(gain.mean(axis=0), 1.0, atol=0.0005)
    np.testing.assert_allclose(gain[:, 1:].std(axis=0), sigma[1:], rtol=0.03)


def test_apply_rxra_uses_one_gain_per_column_and_channel():
    """Check that striping gains stay constant along track and preserve the inputs."""

    cube = np.full((2, 3, 2), 10.0)
    gain = np.array([[1.01, 0.99], [0.98, 1.02]])

    result = apply_rxra(cube, gain)

    # Check every along-track row; the same column gains must be reused.
    for row in range(3):
        np.testing.assert_allclose(result[:, row, :], [[10.1, 9.9], [9.8, 10.2]])

    # The operation must not change either input array.
    np.testing.assert_array_equal(cube, 10.0)
    np.testing.assert_array_equal(gain, [[1.01, 0.99], [0.98, 1.02]])


@pytest.mark.parametrize("columns", [0, -1])
def test_rxra_rejects_invalid_column_counts(columns):
    """Check that zero or negative column counts are rejected."""

    # The call must raise the stated error.
    with pytest.raises(ValueError, match="n_columns"):
        rxra_gain(columns, np.array([0.004]), np.random.default_rng(42))


def test_rxra_rejects_negative_sigma():
    """Check that a negative striping standard deviation is rejected."""

    with pytest.raises(ValueError, match="non-negative"):
        rxra_gain(2, np.array([-0.004]), np.random.default_rng(42))


@pytest.mark.parametrize("shape, gain_shape", [((2, 3), (2, 1)), ((2, 3, 4), (3, 4))])
def test_apply_rxra_rejects_incompatible_shapes(shape, gain_shape):
    """Check that cube and gain dimensions must match."""

    with pytest.raises(ValueError, match="must be"):
        apply_rxra(np.ones(shape), np.ones(gain_shape))
