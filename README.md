# CHIME parametric performance model

This repo hosts a lightweight, shareable model of the instrument performance of **CHIME**, the Copernicus
Hyperspectral Imaging Mission for the Environment. It is a reference implementation of the
SPIE paper [*"CHIME System Performance at Critical Design Review: A parametric model for
end-to-end simulation analysis"*](https://spie.org/spie-sensors-imaging/presentation/CHIME-system-performance-at-critical-design-review--a-parametric/14279-7)
(Fernandez Nunez et al., ESA-ESTEC / OHB System AG / ESA-ESRIN).

This software is developed and maintained by the **CHIME Project** at the European Space Agency (ESA).

The model aims to support Sentinel User Preparation studies and similar initiatives that require knowledge of 
the CHIME performance beyond the system requirements and need to generate its own datasets

The model inputs oversampled TOA radiance and outpus CHIME like L1 data. As such it can be integrated in third party simulators that generate oversampled TOA radiance or be used within the proposed mini E2E reference implementation

It is **not** a replacement for dedicated E2E simulators like CHIME OPSI or CHEES
nor industrial verification tools, nor a rigorous per-pixel uncertainty propagation (CHIME ships that separately),
and not an orbit or direct-location framework. 


## Status

The package provides the radiometric, spectral and spatial performance functions,
CDR-2025 parameters, scene generation and atmospheric propagation.
Instrument acquisition and atmospheric correction remain to be implemented.

Roadmap:
- Oct/Nov 2026 First release
- Nov 2026 Iterations as part of SUP workshop in ESRIN
- Dec 2027 v1.0 including full e2e first implementation
- Updates and maintenance throughout Phase D
- End of 2028 Update of Instrument modelling after Instrument level TVAC Characterisation 

## Download and install

Download a source archive from the [GitHub releases](https://github.com/esa/chime-performance-sup/releases),
extract it and open a terminal in that directory. Install it locally:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows, activate with `.venv\Scripts\activate` instead. The release also provides
a universal Python wheel for local installation; it is the same file for all platforms.
The project is distributed through GitHub releases.

For development and tests, use `python -m pip install -e ".[dev]"` followed by
`python -m pytest`. To build the source archive and wheel yourself:

```bash
python -m pip install build
python -m build
```

The release workflow builds both files and tests the installed package on Linux,
Windows, Intel Mac and Apple Silicon with Python 3.12. The separate unit-test workflow
checks Python 3.10–3.12 on Linux. Publishing a GitHub release
tagged `v0.1.0` attaches the files once those checks pass. No registry publishing is configured.

## Use
The parameter files are included in `chime_perf/data/` and loaded with `ChimeParams.load()`.
`chime_excluded_regions.csv` supplies wavelength intervals for compliance-plot
annotations only. Its boundaries are estimates derived from low reference radiance,
not exact mission exclusion limits. The processing equations still evaluate those channels.
Use `ChimeParams.load(data_dir="path/to/parameters")` to supply a different parameter set.

Run `example/example.ipynb` for further details. A Jupyter environment is needed to
open the notebook interactively.

## Processing modules

Install the pipeline dependencies with `python -m pip install -e ".[pipeline]"`.
Global and local YAML configuration is loaded by EOPL (version 0.2.1 or later).

`chime_perf/run.py` parses arguments and selects modules. Each callable module
owns its EOPL Pipeline and schedules the individual mathematical operations:

- `chime_perf/ssgm.py`: spatial interpolation; spectral interpolation, or NNLS
  fitting followed by spectral reconstruction. Algorithms are in
  `chime_perf/algorithms/ssgm_algorithms.py`.
- `chime_perf/athmo.py`: transmission, solar radiance scaling and atmospheric
  propagation. Algorithms and the 6S adapter are in
  `chime_perf/algorithms/athmo_algorithms.py`.
- `chime_perf/instrument.py`: spectral integration, sensor rotation, fused spatial
  convolution and detector sampling, temporal noise, ARA and RXRA. Image operations
  are in `chime_perf/algorithms/acquisition_algorithms.py`.
- `chime_perf/reconstruction.py`: nominal-response averaging of saved atmospheric
  quantities, then inverse atmospheric correction to BOA reflectance. Algorithms
  are in `chime_perf/algorithms/reconstruction_algorithms.py`.

Algorithms preserve the first input image's name; module I/O chooses names for
saved products and checkpoints. Interpolation, convolution and dimensionless gains
preserve input units. Physical equations check their required units and assign
output units explicitly; no automatic unit conversion is performed.

Shared grid and image helpers are in `chime_perf/algorithms/algorithms.py`.
The radiometric, spectral and spatial performance equations are grouped in
`chime_perf/algorithms/instrument_algorithms.py`.

```bash
python example/setup_scene.py
chime-sup --ssgm --input demo-input/reflectance.nc --output outputs/scene --config example/pipeline.yml
chime-sup --athmo --input outputs/scene/scene.nc --output outputs/atmosphere --sixs-path /path/to/sixs
chime-sup --instrument --input outputs/atmosphere/toa.nc --output outputs/instrument --config example/pipeline.yml
chime-sup --reconstruction --input outputs/instrument/instrument.nc --output outputs/reconstruction --config example/pipeline.yml
```

Select `--all` to compose the modules in processing order. Images are passed
in memory. Propagation also saves `athmo/toa.nc` with its atmospheric coefficients;
reconstruction writes `reconstruction/reconstructed.nc`:

```bash
chime-sup --all --input demo-input/reflectance.nc --output outputs/chain --config example/pipeline.yml --sixs-path /path/to/sixs
```

Add `--save-intermediates` to save each module's product and have the next module
load it. Outputs then include `ssgm/scene.nc`, `athmo/toa.nc`,
`instrument/instrument.nc` and `reconstruction/reconstructed.nc`. Without this flag,
the SSGM and instrument products remain in memory, while the propagation file
is still saved to retain the original atmospheric coefficients.
Standalone modules save directly in `--output`. Each saved module has its own
EOPL manifest. `--athm` is an alias for `--athmo`.

The module functions can also be called directly:

```python
from chime_perf.ssgm import ssgm
from chime_perf.athmo import athmo
from chime_perf.instrument import instrument
from chime_perf.reconstruction import reconstruction

scene = ssgm(input_path="reflectance.nc", config_path="pipeline.yml", save=False)
radiance = athmo(data=scene, config_path="pipeline.yml", output_root="outputs/athmo")
detector = instrument(
    data=radiance, atmosphere_path="outputs/athmo/toa.nc", config_path="pipeline.yml",
    save=False,
)
reflectance = reconstruction(
    data=detector, atmosphere_path="outputs/athmo/toa.nc", config_path="pipeline.yml",
    output_root="outputs/reconstruction",
)
```

The input is a NetCDF BOA reflectance cube named `reflectance`. Dimensions are
`ACT`, `ALT`, `SPE`, with spatial coordinates in `m`, spectral coordinates in `nm`,
and reflectance units `1`. The image's `geometry` attribute records whether its
spatial grid is map or sensor geometry; axis names alone do not perform a rotation.
Missing samples remain NaN. Spatial interpolation is explicitly controlled by YAML:

```yaml
scene:
  spatial_interpolation: true
  detector_spacing_m: 30.0
  spatial_oversampling: auto  # Or a positive integer such as 5 for a 6 m grid.
```

`auto` chooses the nearest integer to detector spacing divided by input spacing,
separately for ACT and ALT (using the median input spacing). Ties select the larger
factor. For a 30 m detector, 5 m input keeps factor 6; 5.5 m input uses factor 5
and is linearly resampled to 6 m. Nearest-factor rounding can slightly coarsen data.
An explicit integer factor fixes the requested grid instead of inferring it.

The first input centre anchors the grid, with samples only inside the input extent.
Nominal centres at `origin + i * detector_spacing_m` therefore coincide with every
integer-factor oversampled sample. When resampling is enabled, the factor and origin are recorded in the image metadata.
Sensor-grid rotation remains an instrument operation; it must preserve this spacing
and centre convention before final detector sampling.

With `spatial_interpolation: false` (the default), the spatial operation is skipped.
Existing coordinates must already be regular with integer oversampling; other grids
raise an error asking to enable interpolation. This keeps fractional oversampling
out of the instrument interface. The example YAML explicitly selects factor 5 for
its synthetic scene.

Add `--library demo-input/library.nc` for the synthetic NNLS example.
A library contains `measured(SPE, material)` values averaged with the input
sensor's band responses and `reflectance(library_SPE, material)` spectra on
its oversampled wavelength grid. The supplied example uses synthetic point-sampled
bands over 500–600 nm, not the full CHIME spectrum.

Py6S requires a separately installed 6S executable; its
[installation guide](https://py6s.readthedocs.io/en/latest/installation.html)
describes the conda and source-build options. The 6SV1.1 adapter runs on a
2.5 nm grid and interpolates physical quantities to the scene wavelengths.
This does not resolve finer atmospheric features. Quantities and solver settings
are saved in `atmosphere.nc` beside the radiance product. Use
`--atmosphere path/to/atmosphere.nc` to reuse quantities or supply another
solver's compatible `SPE` arrays. The file may contain the gas/downward/upward
transmission and solar irradiance inputs, or already combined `transmission`
and `solar_scale`, together with path radiance and spherical albedo.

Instrument settings are shown in `example/pipeline.yml`. Nominal channel centres
inside the input wavelength range are simulated, with +/-3-sigma response support.
Response tails outside that range repeat the nearest available spectral value.
Spatial convolution is evaluated only at retained detector centres.
The shared `boundaries` settings are fixed to `spatial: edge`, `spectral: edge` and
`kernel_support_sigma: 3`. Rotation and spatial convolution repeat the nearest
boundary values outside the scene; no detector centres are cropped and spatial
boundaries introduce no NaNs. Genuine missing input values remain missing.
Convolution padding covers the complete kernel radius, and the detector sampling
origin and requested map coordinates are retained. Spectral padding extends response
tails only; it does not generate channels whose
nominal centres lie outside the input range. Reconstruction uses the same edge
extension when averaging the saved atmospheric coefficients. This is a simple
constant-extension approximation, not a reconstruction of unresolved spectral features.

Noise and RXRA are enabled independently; ARA gain and offset default to zero.
A fixed seed reproduces the saved noise, column gains and spectral shifts. Set
`temporal_noise: false` and `rxra: false`, keeping both ARA coefficients zero, for
an error-free radiometric case. Spectral shifts default to zero; their response
tails use the same edge extension.
The existing SSGM `--library` route can instead reconstruct scene spectra from
fitted material mixtures before atmospheric propagation, within the supplied
library coverage; library spectra are surface reflectance, not TOA radiance.

Reconstruction never calls 6S. Propagation saves its `path_radiance`,
`spherical_albedo`, `transmission` and `solar_scale` in `toa.nc`. A saved instrument
product copies these unchanged onto `atmosphere_SPE`, separately from the nominal
`SPE` detector channels. Reconstruction averages each quantity using the saved
nominal Gaussian response width, then applies the inverse atmospheric equations.
The output includes both original and band-averaged coefficients for inspection.
Changing the YAML atmosphere or response width after acquisition does not replace
these saved values. Invalid inverse denominators are masked; noisy negative
reflectance is retained. Band averaging is an approximation when atmospheric or
surface spectra vary within a channel; it is not an exact inversion of their
spectral integral.

The YAML also accepts `run_options` (EOPL precision and execution settings),
`breakpoints: true` to save operation checkpoints, and a `report` mapping with
EOPL report parameters. EOPL records operations, generates products and reports,
and handles module logging.

With `.[dev,pipeline]` installed, run `python -m pytest`. Set `SIXS_PATH` to the
executable to include direct 6S comparisons. Tests check individual equations,
operation order and agreement between memory and save/reload module handoffs.

## Citation and license

Please cite Fernandez Nunez et al., *CHIME System Performance at Critical Design Review:
A parametric model for end-to-end simulation analysis*, SPIE Sensors + Imaging,
[presentation 14279-7](https://spie.org/spie-sensors-imaging/presentation/CHIME-system-performance-at-critical-design-review--a-parametric/14279-7),
and state the software version used. The manuscript is included in `docs/`.

[CITATION.cff](CITATION.cff) contains the paper title, full author list and SPIE link.
The software is licensed under [Apache-2.0](LICENSE), all rights to the CHIME Project at European Space Agency.
