"""Loaders for the versioned CHIME parameter files (§2, §3.2, §3.4).

The layer 1 -> 2 boundary, and the only module here that touches disk; everything it
returns is plain numpy. The grid values of §1 are asserted in tests/test_params.py rather
than hardcoded (CLAUDE.md non-negotiable #2), so what is validated here is structure.

Per-channel columns come from the CSVs; the scalar ARA, PSF, ISRF and RXRA-default
parameters come from chime_config.yaml. Excluded regions are not part of ChimeParams:
plotting metadata, never a gate (§9.8).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DATA_DIR = Path(__file__).resolve().parent / "data"

# Uniformity tolerance on the spectral grid. Structural, not a performance number.
_SPACING_RTOL = 1e-9
_STRICTLY_POSITIVE = ("Lref_W_sr_m2_um", "snr_g", "ara_b")

# yaml key -> field name. The file uses the paper's symbols; the fields follow the
# module's naming so ara_k/ara_s0 sit next to the per-channel ara_a/ara_b.
_CONFIG_KEYS = {
    "k": "ara_k",
    "S0": "ara_s0",
    "MTF_ACT": "mtf_act",
    "MTF_ALT": "mtf_alt",
    "ISRF_FWHM": "isrf_fwhm_nm",
    "RXRA_DEFAULT": "default_rxra_sigma",
}

# Keys where zero is physically meaningful; everything else must be strictly positive.
_ALLOW_ZERO = {"RXRA_DEFAULT"}


@dataclass(frozen=True)
class ChimeParams:
    """Per-channel and scalar instrument parameters (§2, §3.4).

    lref is W·sr⁻¹·m⁻²·µm⁻¹ at the §1 reference conditions; ara_a/ara_b are in electrons;
    rxra_sigma is the per-channel Eq. 7 sigma [K] and default_rxra_sigma the fraction it
    falls back to; mtf_act/mtf_alt are at Nyquist; isrf_fwhm_nm is in nm; version is the
    provenance string to echo into any output.
    """

    wavelength_nm: np.ndarray
    lref: np.ndarray
    snr_g: np.ndarray
    snr_n0: np.ndarray
    ara_a: np.ndarray
    ara_b: np.ndarray
    rxra_sigma: np.ndarray
    default_rxra_sigma: float
    ara_k: float
    ara_s0: float
    mtf_act: float
    mtf_alt: float
    isrf_fwhm_nm: float
    version: str

    def __post_init__(self) -> None:
        """Every per-channel array shares one grid length."""
        k = self.wavelength_nm.size
        for name in ("lref", "snr_g", "snr_n0", "ara_a", "ara_b", "rxra_sigma"):
            size = getattr(self, name).size
            if size != k:
                raise ValueError(f"{name} has {size} entries but the grid has {k}")

    @property
    def n_channels(self) -> int:
        """Number of spectral channels, K."""
        return int(self.wavelength_nm.size)

    @property
    def ssi_nm(self) -> float:
        """Spectral sampling interval in nm, derived from the shipped grid."""
        return float(np.mean(np.diff(self.wavelength_nm)))

    @classmethod
    def load(cls, data_dir: Path | str | None = None) -> ChimeParams:
        """Load the channel table, the RXRA exceptions and the scalar config (§2, §3.4).

        Channels absent from the exception table take RXRA_DEFAULT from the config file,
        as a fraction (0.004 = 0.4%), not a percentage.
        """
        directory = Path(data_dir or DATA_DIR)
        config = _load_config(directory / "chime_config.yml")
        t = _load_channels(directory / "chime_channel_params.csv")
        return cls(
            wavelength_nm=t["wavelength_nm"],
            lref=t["Lref_W_sr_m2_um"],
            snr_g=t["snr_g"],
            snr_n0=t["snr_N0"],
            ara_a=t["ara_a"],
            ara_b=t["ara_b"],
            rxra_sigma=_load_rxra_sigma(
                t["wavelength_nm"],
                config["default_rxra_sigma"],
                directory / "chime_rxra_exceptions.csv",
            ),
            **config,
            version=read_params_version(directory),
        )


def read_params_version(data_dir: Path | str | None = None) -> str:
    """Provenance string for the shipped parameter set, e.g. "CDR-2025" (§10)."""
    path = Path(data_dir or DATA_DIR) / "params_version.txt"
    with path.open(encoding="utf-8") as handle:
        return handle.readline().strip()


def load_excluded_regions(path: Path | str | None = None) -> list[tuple[float, float]]:
    """Wavelength bands excluded from compliance plots, in nm (§3.2).

    [IMPL CHOICE] bounds derived from Lref, not from the paper -- see README and the
    data file header.
    """
    csv_path = Path(path) if path is not None else DATA_DIR / "chime_excluded_regions.csv"
    table = _read_csv(csv_path, ["lower_nm", "upper_nm"])
    regions = [
        (float(lo), float(hi))
        for lo, hi in zip(table["lower_nm"], table["upper_nm"], strict=True)
    ]
    for lower, upper in regions:
        if upper < lower:
            raise ValueError(f"{csv_path}: region bounds out of order: {lower} > {upper}")
    return regions


def excluded_mask(wavelength_nm: np.ndarray, regions: list[tuple[float, float]]) -> np.ndarray:
    """Channels inside an excluded region, bounds inclusive (§3.2).

    Plotting metadata: never gate a computation with it (§9.8).
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    mask = np.zeros(wavelength_nm.shape, dtype=bool)
    for lower, upper in regions:
        mask |= (wavelength_nm >= lower) & (wavelength_nm <= upper)
    return mask


def _load_config(path: Path) -> dict[str, float]:
    """Read the scalar ARA, PSF, ISRF and RXRA-default parameters, keyed by field name.

    Every key in _CONFIG_KEYS must be present; extra keys raise rather than being
    silently ignored, so a typo in the file is caught at load.
    """
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of parameters")

    missing = sorted(set(_CONFIG_KEYS) - set(raw))
    unknown = sorted(set(raw) - set(_CONFIG_KEYS))
    if missing:
        raise ValueError(f"{path}: missing key(s) {missing}")
    if unknown:
        raise ValueError(f"{path}: unrecognised key(s) {unknown}")

    config = {}
    for key, field in _CONFIG_KEYS.items():
        value = float(raw[key])
        floor = 0.0 if key in _ALLOW_ZERO else np.nextafter(0.0, 1.0)
        if not np.isfinite(value) or value < floor:
            raise ValueError(f"{path}: {key} must be finite and positive, got {raw[key]}")
        config[field] = value

    for key in ("MTF_ACT", "MTF_ALT"):
        if config[_CONFIG_KEYS[key]] > 1.0:
            raise ValueError(f"{path}: {key} must be <= 1, got {raw[key]}")

    return config


def _load_channels(path: Path) -> dict[str, np.ndarray]:
    """Read the 250-channel table and validate its structure (§2).

    ara_a is not filtered, even at the ~1e-15 fit noise floor (§9.5).
    """
    table = _read_csv(
        path,
        ["wavelength_nm", "Lref_W_sr_m2_um", "snr_g", "snr_N0", "ara_a", "ara_b"],
    )

    spacing = np.diff(table["wavelength_nm"])
    if spacing.size == 0 or np.any(spacing <= 0.0):
        raise ValueError(f"{path}: need at least two strictly increasing channel centres")
    if not np.allclose(spacing, spacing[0], rtol=_SPACING_RTOL, atol=0.0):
        raise ValueError(
            f"{path}: channel spacing is not uniform "
            f"(min {spacing.min()} nm, max {spacing.max()} nm)"
        )

    for name, column in table.items():
        if not np.all(np.isfinite(column)):
            raise ValueError(f"{path}: column {name!r} contains non-finite values")
        if name in _STRICTLY_POSITIVE and np.any(column <= 0.0):
            raise ValueError(f"{path}: column {name!r} must be strictly positive")

    return table


def _load_rxra_sigma(centres: np.ndarray, default_sigma: float, path: Path) -> np.ndarray:
    """Per-channel Eq. 7 sigma [K] (§3.4).

    The table's nominal wavelengths are rounded and off-grid, so each maps to the nearest
    channel and collisions raise rather than letting file order decide (§9.3).
    """
    table = _read_csv(path, ["nominal_centre_nm", "rxra_percent"])
    sigma = np.full(centres.size, float(default_sigma))

    claimed: dict[int, float] = {}
    for nominal, percent in zip(
        table["nominal_centre_nm"], table["rxra_percent"], strict=True
    ):
        index = int(np.argmin(np.abs(centres - nominal)))
        if index in claimed:
            raise ValueError(
                f"{path}: nominal wavelengths {claimed[index]} nm and {nominal} nm both "
                f"map to channel {centres[index]} nm"
            )
        claimed[index] = float(nominal)
        sigma[index] = float(percent) / 100.0

    return sigma


def _read_csv(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """Read named float columns from a CSV, skipping ``#`` comment lines."""
    if not path.is_file():
        raise FileNotFoundError(f"parameter file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))

    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing = [name for name in columns if name not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")

    return {
        name: np.array([float(row[name]) for row in rows], dtype=float) for name in columns
    }