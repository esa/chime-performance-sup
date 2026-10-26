"""Small, readable parameter files shared by the loader tests."""

import pytest
import yaml


@pytest.fixture
def parameter_files(tmp_path):
    """Create temporary three-channel files for the parameter-loader tests."""

    # pytest gives each test its own temporary directory.
    config = {
        "k": 0.025,
        "S0": 150,
        "MTF_ACT": 0.25,
        "MTF_ALT": 0.28,
        "ISRF_FWHM": 10,
        "RXRA_DEFAULT": 0.004,
    }

    (tmp_path / "chime_config.yml").write_text(yaml.safe_dump(config))

    (tmp_path / "chime_channel_params.csv").write_text(
        "# Three synthetic channels, independent of the shipped CHIME table.\n"
        "wavelength_nm,Lref_W_sr_m2_um,snr_g,snr_N0,ara_a,ara_b\n"
        "500,10,100,1,1e-15,100\n"
        "510,20,200,2,1,200\n"
        "520,30,300,3,2,300\n"
    )

    (tmp_path / "chime_rxra_exceptions.csv").write_text(
        "nominal_centre_nm,rxra_percent\n509,1.2\n"
    )

    (tmp_path / "params_version.txt").write_text("TEST-PARAMS\nExtra provenance text.\n")

    return tmp_path
