"""The example must run from a fresh namespace, without previous notebook state."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def test_example_notebook_runs_from_start_to_finish(monkeypatch):
    """Check that the notebook runs in order without relying on earlier execution."""

    path = Path(__file__).resolve().parents[1] / "example/example.ipynb"
    notebook = json.loads(path.read_text())
    monkeypatch.chdir(path.parent)

    # All cells share one namespace, as in a fresh notebook kernel.
    namespace = {}

    try:
        for number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                # Include the cell number in any error traceback.
                exec(compile(code, f"example.ipynb cell {number}", "exec"), namespace)

        # The 10% radiance case must exist and have larger error than the 50% case.
        assert np.all(namespace["ARA_error_Lref_01"] > namespace["ARA_error_Lref_05"])
        assert np.all(namespace["ARA_error_Lref_05"] > namespace["ARA_error_Lref"])
    finally:
        # Close figures even if a notebook cell fails.
        plt.close("all")
