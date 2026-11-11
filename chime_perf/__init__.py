"""CHIME parametric performance model and processing modules.

Submodules are imported explicitly, so using one does not drag in the dependencies
of the others::

    from chime_perf.algorithms.instrument_algorithms import snr, nedl, ara

Parameters are CHIME System CDR status (2025), a conservative envelope, and will be
reissued after instrument TVAC characterisation.
"""

from __future__ import annotations

__version__ = "0.1.0"
