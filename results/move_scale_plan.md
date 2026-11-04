# Diagnostic 2: which part of the snapper's moves costs the ink signal?

This was written and pushed before any of the meshes below was rendered. It follows the depth diagnostic ([`depth_sweep.md`](depth_sweep.md)). That diagnostic found that w052's ink signal doesn't depend on the surface's average depth, but is lowered by the snapper's varying moves, and by their mirror image too. Like that one, this is a diagnostic: it changes none of the earlier outcomes.

## What the moves look like

On w052's drop-in output, the snapper moves each vertex along its normal by 1.95 voxels on average. The moves have a standard deviation of 2.39 and almost no net offset (+0.05). Most of that is vertex to vertex: neighbouring vertices, 20 voxels apart, move with a correlation of only 0.47. A Gaussian of 2 vertices keeps 15% of the moves' variance, and a Gaussian of 4 vertices keeps 8%. So the snapped surface is bumpy at the scale of its vertex spacing.

## Meshes

Each mesh moves w052's original vertices along their normals by a filtered version of the snapper's move field (`tools/depth_sweep.py filter`):

| Name | Move | sd (voxels) | Neighbour r |
|---|---|---|---|
| lp1 | the large-scale part, Gaussian σ = 1 vertex (about 20 voxels) | 1.33 | 0.88 |
| lp2 | the large-scale part, σ = 2 vertices (about 40 voxels) | 0.93 | 0.97 |
| lp4 | the large-scale part, σ = 4 vertices (about 80 voxels) | 0.67 | 0.99 |
| hp2 | the rest after σ = 2: the moves' vertex-scale roughness only | 2.03 | 0.29 |
| rand | same mean, spread and spatial spectrum as the moves, random phases (seed 0) | 2.39 | 0.46 |

Each mesh is then processed like the ink test's meshes:
- it is rendered with the ink test's renderer (c5ffa62) and settings;
- it is run through `ink_9um` seed 42 with the same options and layer direction;
- it is compared with the original map (`tools/depth_sweep.py compare`), over pixels valid in every render, together with the snapped and mirrored maps;
- its face-step check is taken at 1 voxel, as in the ink test.

## Reading, fixed now

The measure is the median tile change in the 99th percentile against the original map, as before. A mesh **keeps** the signal if its median is −0.005 or higher, **loses** it if −0.015 or lower, and **partly** loses it in between. For reference, the original mesh moved 1–2 voxels gave 0.000 to −0.003, the snapped mesh −0.027 and the mirrored mesh −0.016.

- **Scale.**
  - If lp2 keeps the signal and hp2 loses it, the cost is in the moves' roughness at scales under about 40 voxels.
  - If lp2 loses it, the larger-scale moves cost too.
  - The smoothest lp mesh that loses it, or none, says at what scale the moves start to cost.
- **Is it the snapper's moves, or any roughness of that size?** Compare rand with the snapped mesh (−0.027) and the mirrored one (−0.016).
  - **rand within 0.005 of the mirrored mesh:** random bumps cost what the mirrored ones did, and the snapper's own moves cost more than chance.
  - **rand within 0.005 of the snapped mesh:** bumps of that size cost that much whatever their source.
  - **Anything else** is reported as it comes out.
- **If an lp mesh keeps the signal**, it is a candidate "smooth snap". Whether it reads better than the original would need a new blind review, which is not part of this diagnostic.
