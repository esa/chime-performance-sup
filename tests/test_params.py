"""Parameter loading, units, channel alignment and invalid-file handling."""

from dataclasses import replace

import numpy as np
import pytest
import yaml

from chime_perf.params import (
    ChimeParams,
    excluded_mask,
    load_excluded_regions,
    read_params_version,
)


def test_load_shipped_chime_parameters():
    """Check the shipped channel grid, provenance and main instrument settings."""

    params = ChimeParams.load()

    assert params.n_channels == 250

    np.testing.assert_allclose(params.wavelength_nm[[0, -1]], [404.2, 2495.8])

    assert params.ssi_nm == pytest.approx(8.4)
    assert params.version == "CDR-2025"
    assert read_params_version() == params.version
    assert params.mtf_act == 0.25
    assert params.mtf_alt == 0.28
    assert params.isrf_fwhm_nm == 10.0
    assert len(load_excluded_regions()) == 3


def test_load_small_parameter_table(parameter_files):
    """Check loading, channel alignment and percentage conversion using three channels."""

    params = ChimeParams.load(parameter_files)

    assert params.n_channels == 3
    assert params.ssi_nm == 10.0
    assert params.version == "TEST-PARAMS"

    np.testing.assert_array_equal(params.lref, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(params.snr_g, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(params.snr_n0, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(params.ara_a, [1e-15, 1.0, 2.0])
    np.testing.assert_array_equal(params.ara_b, [100.0, 200.0, 300.0])

    # 509 nm maps to the nearest channel, 510 nm; 1.2% is stored as 0.012.
    np.testing.assert_allclose(params.rxra_sigma, [0.004, 0.012, 0.004])

    assert params.ara_k == 0.025
    assert params.ara_s0 == 150.0
    assert params.default_rxra_sigma == 0.004


def test_parameter_arrays_must_have_matching_lengths(parameter_files):
    """Check that every parameter array must match the channel count."""

    params = ChimeParams.load(parameter_files)

    with pytest.raises(ValueError, match="lref has 2 entries"):
        # Copy the settings, replacing only the reference-radiance array.
        replace(params, lref=np.ones(2))


def test_excluded_regions_include_their_boundaries(tmp_path):
    """Check that plotting exclusions include both wavelength boundaries."""

    path = tmp_path / "regions.csv"
    path.write_text("# Plotting exclusions only.\nlower_nm,upper_nm\n500,510\n520,530\n")
    regions = load_excluded_regions(path)

    assert regions == [(500.0, 510.0), (520.0, 530.0)]

    wavelengths = np.array([499.0, 500.0, 510.0, 515.0, 520.0, 530.0, 531.0])

    np.testing.assert_array_equal(
        excluded_mask(wavelengths, regions), [False, True, True, False, True, True, False]
    )

    assert not excluded_mask(wavelengths, []).any()


def test_excluded_regions_reject_reversed_bounds(tmp_path):
    """Check that an exclusion cannot end before it starts."""

    path = tmp_path / "regions.csv"
    path.write_text("lower_nm,upper_nm\n510,500\n")

    with pytest.raises(ValueError, match="bounds out of order"):
        load_excluded_regions(path)


@pytest.mark.parametrize(
    "filename",
    [
        "chime_config.yml",
        "chime_channel_params.csv",
        "chime_rxra_exceptions.csv",
        "params_version.txt",
    ],
)
def test_missing_parameter_files_are_reported(parameter_files, filename):
    """Check that a missing input file produces a clear file error."""

    (parameter_files / filename).unlink()

    with pytest.raises(FileNotFoundError):
        ChimeParams.load(parameter_files)


@pytest.mark.parametrize(
    "filename, contents, message",
    [
        ("chime_config.yml", "- not a mapping\n", "expected a mapping"),
        ("chime_config.yml", "k: 0.025\n", "missing key"),
        ("chime_channel_params.csv", "", "no data rows"),
        ("chime_channel_params.csv", "wrong_column\n500\n", "missing column"),
        (
            "chime_rxra_exceptions.csv",
            "nominal_centre_nm,rxra_percent\n509,1.2\n511,0.7\n",
            "both map",
        ),
    ],
)
def test_malformed_parameter_files_are_reported(parameter_files, filename, contents, message):
    """Check that incomplete or malformed parameter files are rejected."""

    (parameter_files / filename).write_text(contents)

    with pytest.raises(ValueError, match=message):
        ChimeParams.load(parameter_files)


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("unexpected", 1.0, "unrecognised key"),
        ("k", 0.0, "finite and positive"),
        ("S0", -1.0, "finite and positive"),
        ("ISRF_FWHM", float("nan"), "finite and positive"),
        ("MTF_ACT", 1.1, "must be <= 1"),
    ],
)
def test_invalid_configuration_values_are_reported(parameter_files, name, value, message):
    """Check that unknown settings and invalid parameter values are rejected."""

    path = parameter_files / "chime_config.yml"
    config = yaml.safe_load(path.read_text())
    config[name] = value
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match=message):
        ChimeParams.load(parameter_files)


def test_zero_rxra_is_a_valid_configuration(parameter_files):
    """Check that striping can be disabled while keeping channel exceptions."""

    path = parameter_files / "chime_config.yml"
    path.write_text(path.read_text().replace("RXRA_DEFAULT: 0.004", "RXRA_DEFAULT: 0"))
    params = ChimeParams.load(parameter_files)

    np.testing.assert_allclose(params.rxra_sigma, [0.0, 0.012, 0.0])


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("520,30", "525,30", "spacing is not uniform"),
        ("510,20", "500,20", "strictly increasing"),
        ("500,10,100", "500,0,100", "strictly positive"),
        ("500,10,100", "500,10,nan", "non-finite"),
    ],
)
def test_invalid_channel_values_are_reported(parameter_files, old, new, message):
    """Check that channel grids and radiometric values satisfy the loader rules."""

    path = parameter_files / "chime_channel_params.csv"
    path.write_text(path.read_text().replace(old, new))

    with pytest.raises(ValueError, match=message):
        ChimeParams.load(parameter_files)
