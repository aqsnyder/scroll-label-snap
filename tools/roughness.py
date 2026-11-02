"""Edge roughness of label-1 bands: exposed voxel faces per band voxel (lower = smoother).

Usage: python tools/roughness.py LABEL_DIR [LABEL_DIR ...]
Pools over all .tif files in each folder (matched by name across folders).
"""
import os
import sys

import numpy as np
import tifffile


def faces(b):
    return sum(int(np.count_nonzero(np.diff(b.view(np.int8), axis=a))) for a in range(3))


def main(*dirs):
    names = sorted(set.intersection(*(set(f for f in os.listdir(d) if f.endswith(".tif")) for d in dirs)))
    for d in dirs:
        f = v = 0
        for name in names:
            b = tifffile.imread(os.path.join(d, name)) == 1
            f += faces(b)
            v += int(b.sum())
        print(f"{d}: {f / v:.3f} faces per band voxel ({len(names)} cubes, {v:,} band voxels)")


if __name__ == "__main__":
    main(*sys.argv[1:])
