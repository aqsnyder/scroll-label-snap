"""The graph-cut solver must match brute force on small random problems."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scroll_label_snap.core import solve_shifts  # noqa: E402


def energy(D, i, j, w, K, lab):
    e = D[np.arange(len(D)), lab].sum() + w * np.abs(lab[i] - lab[j]).sum()
    if K is not None and np.any(np.abs(lab[i] - lab[j]) > K):
        return np.inf
    return e


def brute(D, i, j, w, K):
    N, L = D.shape
    best = np.inf
    for lab in itertools.product(range(L), repeat=N):
        best = min(best, energy(D, i, j, w, K, np.array(lab)))
    return best


def test_matches_brute_force():
    rng = np.random.default_rng(1)
    for trial in range(40):
        N, L = 6, 5
        D = rng.normal(size=(N, L))
        pairs = [(a, b) for a in range(N) for b in range(a + 1, N) if rng.random() < 0.4]
        if not pairs:
            pairs = [(0, 1)]
        i = np.array([a for a, _ in pairs])
        j = np.array([b for _, b in pairs])
        w = float(rng.uniform(0.0, 1.0))
        K = [None, 1, 2][trial % 3]
        lab = solve_shifts(D, i, j, w, K)
        got = energy(D, i, j, w, K, lab)
        want = brute(D, i, j, w, K)
        assert np.isclose(got, want), (trial, got, want)


if __name__ == "__main__":
    test_matches_brute_force()
    print("solver matches brute force on 40 random problems")
