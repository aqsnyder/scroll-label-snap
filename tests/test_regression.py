"""The snapper's outputs on small fixed cases must match tests/reference/regression_reference.npz.

A behaviour-neutral change (an optimisation, a refactor) should pass unchanged. Floating-point
details may move a handful of samples by one 0.5-voxel step; more than that is a behaviour change,
which needs tests/make_regression_reference.py and a note in the commit.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regression_cases import REFERENCE, run_all  # noqa: E402


@pytest.fixture(scope="module")
def outputs():
    if not os.path.exists(REFERENCE):
        pytest.skip("no reference yet: run tests/make_regression_reference.py")
    return dict(np.load(REFERENCE)), run_all()


@pytest.mark.parametrize("key", ["label_shift", "mesh_dense_shift"])
def test_shifts_match_reference(outputs, key):
    ref, cur = outputs[0][key], outputs[1][key]
    assert ref.shape == cur.shape, f"{key}: {ref.shape} vs {cur.shape}"
    diff = np.abs(ref - cur)
    assert np.mean(diff > 1e-6) <= 0.002, f"{key}: {np.mean(diff > 1e-6):.2%} of samples moved"
    assert diff.max() <= 0.5 + 1e-6, f"{key}: a sample moved by {diff.max()} voxels"


def test_label_volume_matches_reference(outputs):
    ref, cur = outputs[0]["label_volume_sum"], outputs[1]["label_volume_sum"]
    assert abs(int(ref[0]) - int(cur[0])) <= max(10, int(0.001 * ref[0]))
    assert int(ref[1]) == int(cur[1])
    assert bool(outputs[1]["label_unchanged_2"][0])


def test_mesh_vertices_match_reference(outputs):
    ref, cur = outputs[0]["mesh_vertex_shift"], outputs[1]["mesh_vertex_shift"]
    assert np.allclose(ref, cur, atol=0.05)
