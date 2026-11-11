"""PSF normalization, axis orientation and consistency with the target MTF."""

import numpy as np
import pytest

from chime_perf.spatial import psf_kernel, psf_sigma_px


def test_psf_sigma_scales_with_oversampling():
    """Check that spatial oversampling scales the PSF width in pixels."""

    mtf = np.exp(-(np.pi**2) / 2)

    assert psf_sigma_px(mtf, OF=1) == pytest.approx(1.0)
    assert psf_sigma_px(mtf, OF=5) == pytest.approx(5.0)


def test_psf_kernel_is_normalized_symmetric_and_centered():
    """Check that the kernel preserves total signal and is symmetric about its peak."""

    kernel = psf_kernel(1.0, 2.0)

    assert kernel.shape == (17, 17)
    assert kernel.sum() == pytest.approx(1.0)
    assert np.all(kernel >= 0)

    # Reverse each spatial axis to check symmetry around the centre.
    np.testing.assert_allclose(kernel, kernel[::-1, :])
    np.testing.assert_allclose(kernel, kernel[:, ::-1])

    assert kernel[8, 8] == kernel.max()


def test_psf_kernel_keeps_across_and_along_track_axes_in_order():
    """Check that the across-track and along-track widths are not swapped."""

    kernel = psf_kernel(1.0, 2.0, size=7)

    assert kernel.shape == (7, 7)
    assert kernel[4, 3] / kernel[3, 3] == pytest.approx(np.exp(-0.5))
    assert kernel[3, 4] / kernel[3, 3] == pytest.approx(np.exp(-0.125))


def test_psf_kernel_reproduces_the_requested_nyquist_mtf():
    """Check that the generated PSF gives the requested MTF in both directions."""

    oversampling = 5
    kernel = psf_kernel(psf_sigma_px(0.25, oversampling), psf_sigma_px(0.28, oversampling))

    # Express sample positions relative to the centre of the kernel.
    offsets = np.arange(kernel.shape[0]) - kernel.shape[0] // 2

    # Evaluate the Fourier response at the instrument Nyquist frequency.
    fourier_wave = np.exp(-2j * np.pi * offsets / (2 * oversampling))

    # Sum over the other axis to measure each direction separately.
    mtf_act = abs(np.sum(kernel.sum(axis=1) * fourier_wave))
    mtf_alt = abs(np.sum(kernel.sum(axis=0) * fourier_wave))

    assert mtf_act == pytest.approx(0.25, abs=0.001)
    assert mtf_alt == pytest.approx(0.28, abs=0.001)


@pytest.mark.parametrize("mtf", [-0.1, 0.0, 1.0, 1.1])
def test_psf_sigma_rejects_mtf_outside_its_defined_range(mtf):
    """Check that MTF values outside the model range are rejected."""

    with pytest.raises(ValueError, match="mtf_nyquist"):
        psf_sigma_px(mtf, OF=5)


@pytest.mark.parametrize("sigma_act, sigma_alt", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_psf_kernel_rejects_nonpositive_sigmas(sigma_act, sigma_alt):
    """Check that zero or negative spatial widths are rejected."""

    with pytest.raises(ValueError, match="sigmas"):
        psf_kernel(sigma_act, sigma_alt)


@pytest.mark.parametrize("size", [0, 2, -3])
def test_psf_kernel_rejects_invalid_sizes(size):
    """Check that the kernel size must be a positive odd number."""

    with pytest.raises(ValueError, match="positive odd"):
        psf_kernel(1.0, 1.0, size=size)
