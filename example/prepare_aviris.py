"""Download and prepare the AVIRIS-NG snow or agricultural reflectance scene."""

import argparse
import json
import re
from pathlib import Path

import netCDF4
import numpy as np
from eopl.log import Log

SNOW = "SNEX21_SSR_ang20210411t181022_SBB_rfl_v2z1a"
FIELDS = "bioscape_v02_30_16_rfl"
FIELDS_URL = "https://popo.jpl.nasa.gov/pub/bioscape/mosaics/ang/v02/rfl/"


def parse_arguments():
    """Choose a scene; downloaded data stays outside the repository by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=("snow", "fields"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "datasets" / "aviris_ng",
        help="Directory for raw downloads and prepared NetCDF inputs.",
    )
    parser.add_argument(
        "--sixs-path", help="Also prepare atmospheric coefficients using this 6S executable."
    )
    parser.add_argument(
        "--config", type=Path, help="YAML settings; defaults to the selected scene's example."
    )
    return parser.parse_args()


def header_integer(text, key):
    """Read an integer from the ENVI header."""
    return int(re.search(r"^" + re.escape(key) + r"\s*=\s*(\d+)", text, re.M)[1])


def download_snow(directory):
    """Reuse the local snow flight, or download it through Earthdata Login."""
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / (SNOW + ".hdr")
    image = header.with_suffix(".img")

    if not header.exists() or not image.exists():
        import earthaccess

        results = earthaccess.search_data(
            short_name="SNEX21_SSR", granule_name="*ang20210411t181022*", count=100
        )
        results = [item for item in results if item["umm"]["GranuleUR"] == SNOW]
        if len(results) != 1:
            raise ValueError("Could not identify the exact SnowEx reflectance product.")

        # The user enters credentials in the terminal; this script does not save them.
        auth = earthaccess.login(strategy="interactive", persist=False)
        if not auth.authenticated:
            raise RuntimeError("Earthdata login failed.")
        earthaccess.download(results, local_path=directory, threads=2)

    return header


def download_fields(directory):
    """Download a contiguous ALT range from the public agricultural mosaic."""
    directory.mkdir(parents=True, exist_ok=True)
    header = directory / (FIELDS + "_ALT600_1240.hdr")
    image = header.with_suffix(".img")
    if header.exists() and image.exists():
        return header

    import requests

    response = requests.get(FIELDS_URL + FIELDS + ".hdr", timeout=60)
    response.raise_for_status()
    text = response.text
    (directory / (FIELDS + "_original.hdr")).write_text(text)
    width = header_integer(text, "samples")
    bands = header_integer(text, "bands")
    if header_integer(text, "data type") != 4 or header_integer(text, "header offset") != 0:
        raise ValueError("Expected a float32 mosaic without a binary header offset.")

    first, last = 600, 1240
    row_bytes = width * bands * 4
    expected_bytes = (last - first) * row_bytes
    partial = image.with_suffix(".part")

    if not image.exists():
        completed = partial.stat().st_size if partial.exists() else 0
        if completed > expected_bytes:
            raise ValueError("The partial download is larger than the requested image.")

        if completed < expected_bytes:
            start = first * row_bytes + completed
            end = last * row_bytes - 1
            total = header_integer(text, "lines") * row_bytes
            Log().info(f"Downloading {expected_bytes / 1e9:.2f} GB; no login required.")

            with requests.get(
                FIELDS_URL + FIELDS,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}"
                ):
                    raise ValueError("The server did not return the requested byte range.")
                with partial.open("ab") as stream:
                    for block in response.iter_content(4 * 1024**2):
                        stream.write(block)

        if partial.stat().st_size != expected_bytes:
            raise ValueError("Incomplete download; rerun this command to resume.")
        partial.rename(image)

    # The downloaded rows remain an ordinary BIL image with their own map origin.
    text = re.sub(r"^lines\s*=.*$", f"lines = {last - first}", text, flags=re.M)
    match = re.search(r"^map info\s*=\s*\{([^}]+)\}", text, re.M)
    fields = [value.strip() for value in match[1].split(",")]
    fields[4] = str(float(fields[4]) - first * float(fields[6]))
    text = (
        text[: match.start()]
        + "map info = { "
        + ", ".join(fields)
        + " }"
        + text[match.end() :]
    )
    header.write_text(text)
    return header


def convert_scene(header, output, act, alt, source):
    """Copy the selected reflectance rectangle into NetCDF, preserving all bands."""
    text = header.read_text()
    width = header_integer(text, "samples")
    height = header_integer(text, "lines")
    bands = header_integer(text, "bands")
    if header_integer(text, "data type") != 4 or header_integer(text, "byte order") != 0:
        raise ValueError("Expected little-endian float32 reflectance.")
    if header_integer(text, "header offset") != 0 or not re.search(
        r"^interleave\s*=\s*bil", text, re.M
    ):
        raise ValueError("Expected BIL storage without a binary header offset.")
    image = header.with_suffix(".img")
    if image.stat().st_size != width * height * bands * 4:
        raise ValueError("Image byte count does not match the ENVI header.")
    if not (0 <= act.start < act.stop <= width and 0 <= alt.start < alt.stop <= height):
        raise ValueError("The requested example rectangle is outside the source image.")

    wavelength = np.array(
        [
            float(value)
            for value in re.search(r"^wavelength\s*=\s*\{([^}]+)", text, re.M)[1].split(",")
        ]
    )
    if len(wavelength) != bands or not np.all(np.diff(wavelength) > 0):
        raise ValueError("Wavelengths must be strictly increasing and match the band count.")
    map_info = re.search(r"^map info\s*=\s*\{([^}]+)", text, re.M)[1]
    fields = [value.strip() for value in map_info.split(",")]
    dx, dy = float(fields[5]), float(fields[6])
    raw = np.memmap(image, dtype="<f4", mode="r", shape=(height, bands, width))

    with netCDF4.Dataset(output, "w") as dataset:
        for name, values, unit in [
            ("ACT", np.arange(act.stop - act.start) * dx, "m"),
            ("ALT", np.arange(alt.stop - alt.start) * dy, "m"),
            ("SPE", wavelength, "nm"),
        ]:
            dataset.createDimension(name, len(values))
            coordinate = dataset.createVariable(name, "f8", (name,))
            coordinate[:] = values
            coordinate.units = unit

        reflectance = dataset.createVariable(
            "reflectance",
            "f4",
            ("ACT", "ALT", "SPE"),
            zlib=True,
            complevel=1,
            chunksizes=(64, 64, bands),
            fill_value=np.nan,
        )
        reflectance.units = "1"
        reflectance.geometry = "map"
        reflectance.source = source
        reflectance.source_header = header.name
        reflectance.source_map_info = map_info
        reflectance.source_window = json.dumps(
            {"ACT": [act.start, act.stop], "ALT": [alt.start, alt.stop]}
        )
        crs = re.search(r"^coordinate system string\s*=\s*\{(.*)\}", text, re.M)
        if crs is not None:
            reflectance.source_crs = crs[1]
        reflectance.preprocessing = (
            "Negative/nonfinite values masked; >1 retained; no band removal or gap filling"
        )

        # Keep the original raster axes locally; no new orthorectification is applied.
        for start in range(alt.start, alt.stop, 64):
            stop = min(start + 64, alt.stop)
            values = np.array(raw[start:stop, :, act], copy=True)
            values[~np.isfinite(values) | (values < 0)] = np.nan
            reflectance[:, start - alt.start : stop - alt.start, :] = values.transpose(2, 0, 1)

    Log().info(f"Prepared {output}")


def save_atmosphere(scene, output, config_path, sixs_path):
    """Prepare small float32 coefficient arrays once, before processing the image."""
    import xarray as xr
    import yaml

    from chime_perf.algorithms.algorithms import grid
    from chime_perf.algorithms.athmo_algorithms import prepare_atmosphere

    settings = yaml.safe_load(config_path.read_text())
    with xr.open_dataset(scene) as dataset:
        wavelength = grid(dataset.SPE.values, settings["scene"]["spectral_step_nm"])

    Log().info("Preparing atmospheric coefficients with 6S; this can take a few minutes.")
    atmosphere = prepare_atmosphere(wavelength, sixs_path=sixs_path, **settings["atmosphere"])
    atmosphere.astype(settings["run_options"]["precision"]).to_netcdf(output)
    Log().info(f"Prepared {output}")


def main():
    args = parse_arguments()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.scene == "snow":
        header = download_snow(args.data_dir / "full")
        convert_scene(
            header,
            args.data_dir / "snow.nc",
            slice(0, 608),
            slice(0, 779),
            "SnowEx21 AVIRIS-NG, Senator Beck Basin; doi:10.5067/ZAI3M64WWN5V",
        )
    else:
        header = download_fields(args.data_dir / "fields")
        convert_scene(
            header,
            args.data_dir / "fields.nc",
            slice(200, 712),
            slice(0, 640),
            "BioSCape AVIRIS-NG L3 mosaic, Wellington; original ALT [600,1240); "
            "doi:10.3334/ORNLDAAC/2427",
        )

    if args.sixs_path is not None:
        filename = args.scene + "_config.yml"
        config = args.config or Path(__file__).with_name(filename)
        save_atmosphere(
            args.data_dir / (args.scene + ".nc"),
            args.data_dir / (args.scene + "_atmosphere.nc"),
            config,
            args.sixs_path,
        )


if __name__ == "__main__":
    main()
