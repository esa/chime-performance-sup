"""ISRF width, normalization, symmetry and spectral-channel broadcasting."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from chime_perf.spectral import isrf, isrf_sigma


def test_isrf_has_the_requested_full_width_at_half_maximum():
    """Check that the response reaches half its peak at the requested width."""

    fwhm = np.array([10.0, 12.0])
    centre = np.array([500.0, 600.0])

    sigma = isrf_sigma(fwhm)
    peak = isrf(centre, centre, sigma)
    left = isrf(centre - fwhm / 2, centre, sigma)
    right = isrf(centre + fwhm / 2, centre, sigma)

    np.testing.assert_allclose(left, peak / 2)
    np.testing.assert_allclose(right, peak / 2)


def test_isrf_is_normalized_for_each_channel():
    """Check that each response integrates to one and has the correct centre."""

    # Add a column axis to evaluate both channels on the same wavelength grid.
    wavelengths = np.linspace(450.0, 650.0, 20001)[:, np.newaxis]
    centres = np.array([500.0, 600.0])
    sigma = np.array([2.0, 4.0])

    response = isrf(wavelengths, centres, sigma)

    assert response.shape == (20001, 2)

    # Integrate along wavelength, keeping the two channels separate.
    np.testing.assert_allclose(trapezoid(response, wavelengths[:, 0], axis=0), 1.0)

    mean = trapezoid(response * wavelengths, wavelengths[:, 0], axis=0)

    np.testing.assert_allclose(mean, centres)


@pytest.mark.parametrize("width", [0.0, -1.0])
def test_isrf_sigma_rejects_nonpositive_widths(width):
    """Check that zero or negative spectral widths are rejected."""

    with pytest.raises(ValueError, match="fwhm_nm"):
        isrf_sigma(np.array([10.0, width]))


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_isrf_rejects_nonpositive_sigma(sigma):
    """Check that zero or negative spectral standard deviations are rejected."""

    with pytest.raises(ValueError, match="sigma_nm"):
        isrf(np.array([500.0]), np.array([500.0]), np.array([sigma]))
