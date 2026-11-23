# 🛰️ CHIME parametric performance model
 
This repo hosts a lightweight, shareable model of the instrument performance of **CHIME**, the Copernicus
Hyperspectral Imaging Mission for the Environment. 🌈🌍 It is a reference implementation of the
SPIE paper [*"CHIME System Performance at Critical Design Review: A parametric model for
end-to-end simulation analysis"*](https://doi.org/10.1117/12.3119031)
(Fernandez Nunez et al., ESA-ESTEC / OHB System AG / ESA-ESRIN).
 
This software is developed and maintained by the **CHIME Project** at the European Space Agency (ESA).
 
The model aims to support Sentinel User Preparation studies and similar initiatives that require knowledge of
the CHIME performance beyond the system requirements and need to generate their own datasets.
 
The model takes oversampled TOA radiance as input and outputs CHIME-like L1 data. As such, it can be integrated into third-party simulators that generate oversampled TOA radiance, or be used within the proposed mini E2E reference implementation.
 
🚫 It is **not** a replacement for dedicated E2E simulators like CHIME OPSI or CHEES,
nor for industrial verification tools. It is also not a rigorous per-pixel uncertainty propagation (CHIME ships that separately),
nor an orbit or direct-location framework.
 
 
## 🚧 Status
 
Currently under development. The first release, v0.1.0, has left the launch pad 🧑‍🚀 and contains the radiometric,
spectral and spatial performance functions, CDR-2025 parameters and an example notebook.
 
Still to come: a full mini E2E reference implementation, including scene generation, atmospheric propagation and correction.
 
## 🚀 Roadmap
 
- **Oct/Nov 2026:** First release.
- **Nov 2026:** Iterations as part of the SUP workshop at ESRIN.
- **Dec 2027 — v1.0:** Includes the first full E2E implementation.
- 🔧 Updates and maintenance throughout Phase D.
- **End of 2028:** Update of instrument modelling after instrument-level TVAC characterisation. 🌡️
## 📦 Download and install
 
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
 
## 🔭 Use
 
The parameter files are included in `chime_perf/data/` and loaded with `ChimeParams.load()`.
Use `ChimeParams.load(data_dir="path/to/parameters")` to supply a different parameter set.
 
Run `example/example.ipynb` for further details. A Jupyter environment is needed to
open the notebook interactively.
 
## 📜 Citation and license
 
Please cite Fernandez Nunez et al., *CHIME System Performance at Critical Design Review:
A parametric model for end-to-end simulation analysis*, SPIE Sensors + Imaging,
[doi:10.1117/12.3119031](https://doi.org/10.1117/12.3119031),
and state the software version used. The manuscript is included in `docs/`.
 
[CITATION.cff](CITATION.cff) contains the paper title, full author list and DOI.
The software is licensed under [Apache-2.0](LICENSE), all rights to the CHIME Project at the European Space Agency.
 
