"""The parts of the ink-test helpers that don't need the ink model or villa: reading the training
segments from a checkpoint's config, and the run log's output names."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import ink_infer  # noqa: E402
import ink_selfcheck as sc  # noqa: E402

CONFIG = {
    "description": "trained without w052",
    "datasets": [
        {"segments": ["pherc0139-w016", "pherc0139-w028", "pherc0139-w035"],
         "sampling_physical_segment_keys": {"pherc0139-w016": "0139:w016", "pherc0139-w028": "0139:w044",
                                            "pherc0139-w035": "0139:w035"}},
        {"segments": ["w035", "w044"], "sampling_physical_segment_keys": {"w035": "0139:w035", "w044": "0139:w044"}},
        {"segments": ["phercparis4-w05"]},
    ],
}


def test_training_segments_and_mentions():
    names, physical = sc.training_segments(CONFIG)
    assert names == ["pherc0139-w016", "pherc0139-w028", "pherc0139-w035", "phercparis4-w05", "w035", "w044"]
    assert physical == ["0139:w016", "0139:w035", "0139:w044"]
    ds = list(sc.walk_strings(CONFIG["datasets"]))
    assert sc.mentions(ds, "w052") == []
    assert sc.mentions(ds, "w044") == ["0139:w044", "w044"]
    assert sc.mentions(list(sc.walk_strings(CONFIG)), "w052") == ["trained without w052"]
    assert sc.mentions(["pherc0139-w052", "0139:W052", "w0521", "xw052", "w052"], "w052") == \
        ["0139:W052", "pherc0139-w052", "w052"]


def test_output_paths():
    assert ink_infer.output_paths(["in.zarr", "ck.pth", "out.tif", "--direction", "forward"]) == ["out.tif"]
    assert ink_infer.output_paths(["in.zarr", "ck.pth", "o/out.tif", "--direction", "both"]) == \
        ["o/out.tif", "o/out_reverse.tif"]
    assert ink_infer.output_paths(["in.zarr", "ck.pth", "out.tif"]) == ["out.tif"]
