"""Rewrite tests/reference/regression_reference.npz from the current code. Run this only after a change
that is meant to alter the snapper's output, and say so in the commit."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regression_cases import REFERENCE, run_all  # noqa: E402

if __name__ == "__main__":
    out = run_all()
    os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
    np.savez_compressed(REFERENCE, **out)
    for k, v in out.items():
        print(k, v.shape, v.dtype)
    print("wrote", REFERENCE)
