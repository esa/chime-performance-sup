"""Radiometric, spectral and spatial instrument-performance equations."""

from __future__ import annotations

import numpy as np

# Radiometric performance


def snr(radiance: np.ndarray, lref: np.ndarray, g: np.ndarray, n0: np.ndarray) -> np.ndarray:
    """SNR vs at-sensor radiance L in W·sr⁻¹·m⁻²·µm⁻¹ (Eq. 1, §3.1)."""
    x = radiance / lref
    return x * g / np.sqrt(x + n0)


def nedl(radiance: np.ndarray, snr_value: np.ndarray) -> np.ndarray:
    """Noise-equivalent delta radiance, L / SNR, in W·sr⁻¹·m⁻²·µm⁻¹ (Eq. 2, §3.1)."""
    return radiance / snr_value


def ara(
    radiance: np.ndarray,
    lref: np.ndarray,
    ara_a: np.ndarray,
    ara_b: np.ndarray,
    k: float,
    s0: float,
) -> np.ndarray:
    """ARA for uniform scenes, as a fraction; k fractional, s0 in electrons (Eq. 4, §3.2).

    Linear sum, not RSS (§9.4); ara_a is kept though negligible (§9.5).
    """
    return k + s0 / (ara_a + ara_b * radiance / lref)


def rxra_gain(n_columns: int, sigma_eq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Equalization residual gain 1 + N(0, sigma^2), [I, K] (Eq. 7, §3.4).

    One draw per (column, channel): that structure is the striping (§9.2).
    """
    if n_columns < 1:
        raise ValueError(f"n_columns must be >= 1, got {n_columns}")
    sigma_eq = np.asarray(sigma_eq, dtype=float)
    if np.any(sigma_eq < 0.0):
        raise ValueError("sigma_eq must be non-negative")
    return 1.0 + rng.normal(0.0, 1.0, size=(n_columns, sigma_eq.size)) * sigma_eq


def apply_rxra(cube: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Apply the Eq. 7 gain [I, K] to a cube [I, J, K], broadcast along-track (§3.4)."""
    cube = np.asarray(cube, dtype=float)
    gain = np.asarray(gain, dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"cube must be [I, J, K], got shape {cube.shape}")
    if gain.shape != (cube.shape[0], cube.shape[2]):
        raise ValueError(
            f"gain must be [I, K] = {(cube.shape[0], cube.shape[2])}, got {gain.shape}"
        )
    return cube * gain[:, np.newaxis, :]


# Spectral response


def isrf_sigma(fwhm_nm: np.ndarray) -> np.ndarray:
    """ISRF standard deviation in nm, FWHM / (2*sqrt(2*ln 2)) (Eq. 8, §4.1)."""
    if np.any(np.asarray(fwhm_nm) <= 0.0):
        raise ValueError("fwhm_nm must be strictly positive")
    return fwhm_nm * 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def isrf(wavelength_nm: np.ndarray, centre_nm: np.ndarray, sigma_nm: np.ndarray) -> np.ndarray:
    """Gaussian ISRF in nm⁻¹, normalised to integrate to 1 (Eq. 9, §4.1)."""
    if np.any(np.asarray(sigma_nm) <= 0.0):
        raise ValueError("sigma_nm must be strictly positive")
    z = (wavelength_nm - centre_nm) / sigma_nm
    return np.exp(-0.5 * z**2) / (sigma_nm * np.sqrt(2.0 * np.pi))


# Spatial response


def psf_sigma_px(mtf_nyquist: float, OF: int) -> float:
    """PSF sigma in pixels, (p/pi)*sqrt(-2*ln(MTF)) (Eq. 10, §5)."""
    if not 0.0 < mtf_nyquist < 1.0:
        raise ValueError(f"mtf_nyquist must be in (0, 1), got {mtf_nyquist}")
    return OF * float(np.sqrt(-2.0 * np.log(mtf_nyquist)) / np.pi)


def psf_kernel(
    sigma_act_px: float,
    sigma_alt_px: float,
    size: int | None = None,
    support_sigma: float = 3.0,
) -> np.ndarray:
    """Separable 2-D Gaussian PSF kernel (ACT, ALT), normalised to 1 (Eq. 11, §5).

    Sigmas are in pixels of the grid it will be applied to: the oversampled grid
    ``size`` defaults to the smallest odd value covering +/-support_sigma sigma.
    """
    if sigma_act_px <= 0.0 or sigma_alt_px <= 0.0:
        raise ValueError("PSF sigmas must be strictly positive")

    if not np.isfinite(support_sigma) or support_sigma <= 0:
        raise ValueError("Kernel support must be finite and positive")
    if size is None:
        size = 2 * int(np.ceil(support_sigma * max(sigma_act_px, sigma_alt_px))) + 1
    if size < 1 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd number of samples, got {size}")

    offsets = np.arange(size, dtype=float) - (size - 1) / 2.0
    kernel = np.outer(
        np.exp(-0.5 * (offsets / sigma_act_px) ** 2),
        np.exp(-0.5 * (offsets / sigma_alt_px) ** 2),
    )
    return kernel / kernel.sum()
