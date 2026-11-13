"""Command-line selection of CHIME processing modules."""

import argparse
from pathlib import Path

from chime_perf.athmo import athmo
from chime_perf.instrument import instrument
from chime_perf.ssgm import ssgm


def parse_arguments(argv=None):
    """Select a module and its input, output and configuration paths."""
    parser = argparse.ArgumentParser(description="Run CHIME processing modules")

    modules = parser.add_mutually_exclusive_group(required=True)
    modules.add_argument("--ssgm", action="store_true", help="Run scene generation")
    modules.add_argument(
        "--athmo", "--athm", action="store_true", help="Run atmospheric propagation"
    )
    modules.add_argument(
        "--instrument", action="store_true", help="Run instrument acquisition"
    )
    modules.add_argument("--all", action="store_true", help="Run modules in processing order")

    parser.add_argument(
        "--input", type=Path, required=True, help="Input module product NetCDF"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, help="YAML configuration")
    parser.add_argument("--library", type=Path, help="Spectral library NetCDF")
    parser.add_argument("--atmosphere", type=Path, help="Atmospheric quantities NetCDF")
    parser.add_argument("--sixs-path", help="6S executable; otherwise search PATH")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save and reload module products instead of passing images in memory",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """Dispatch the selected module or compose modules in processing order."""
    args = parse_arguments(argv)

    if args.ssgm:
        ssgm(
            input_path=args.input,
            output_root=args.output,
            config_path=args.config,
            library_path=args.library,
        )

    elif args.athmo:
        athmo(
            input_path=args.input,
            output_root=args.output,
            config_path=args.config,
            atmosphere_path=args.atmosphere,
            sixs_path=args.sixs_path,
        )

    elif args.instrument:
        instrument(
            input_path=args.input,
            output_root=args.output,
            config_path=args.config,
            atmosphere_path=args.atmosphere,
        )

    elif args.all:
        # In memory, each module receives the image returned by the previous one.
        scene = ssgm(
            input_path=args.input,
            output_root=args.output / "ssgm",
            config_path=args.config,
            library_path=args.library,
            save=args.save_intermediates,
        )

        # With disk handoff, the next module loads the saved product itself.
        if args.save_intermediates:
            scene = None

        radiance = athmo(
            input_path=args.output / "ssgm" / "scene.nc",
            output_root=args.output / "athmo",
            config_path=args.config,
            atmosphere_path=args.atmosphere,
            sixs_path=args.sixs_path,
            data=scene,
        )

        # Propagation always archives its coefficients, including with in-memory handoff.
        instrument(
            input_path=args.output / "athmo" / "toa.nc",
            output_root=args.output / "instrument",
            config_path=args.config,
            data=None if args.save_intermediates else radiance,
        )


if __name__ == "__main__":
    main()
