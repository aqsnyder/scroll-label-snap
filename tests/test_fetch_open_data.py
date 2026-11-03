"""tools/fetch_open_data.py without the network: S3 listing pages, skipping finished files, and label
zarrs written out as TIFFs."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import fetch_open_data as fod  # noqa: E402

PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
{items}<IsTruncated>{trunc}</IsTruncated>{token}</ListBucketResult>"""


class Resp:
    def __init__(self, content, chunks=None):
        self.content = content.encode() if isinstance(content, str) else content
        self.chunks = chunks or [self.content]

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        return iter(self.chunks)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, stream=False, timeout=None):
        self.calls.append((url, params))
        if params is not None:
            if "continuation-token" not in params:
                items = "<Contents><Key>a/x.tif</Key><Size>3</Size></Contents>"
                return Resp(PAGE.format(items=items, trunc="true", token="<NextContinuationToken>t1</NextContinuationToken>"))
            items = "<Contents><Key>a/b/meta.json</Key><Size>2</Size></Contents>"
            return Resp(PAGE.format(items=items, trunc="false", token=""))
        return Resp(b"xyz"[: 3 if url.endswith("x.tif") else 2])


def test_listing_pages_and_download(tmp_path):
    s = Session()
    keys = fod.list_keys("a/", s)
    assert keys == [("a/x.tif", 3), ("a/b/meta.json", 2)]
    assert fod.download("a/x.tif", 3, str(tmp_path), s) is True
    assert (tmp_path / "a" / "x.tif").read_bytes() == b"xyz"
    assert fod.download("a/x.tif", 3, str(tmp_path), s) is False        # already there, right size


def test_labels_to_tif(tmp_path):
    zarr = pytest.importorskip("zarr")
    tifffile = pytest.importorskip("tifffile")
    d = tmp_path / "PHerc" / "ink-labels" / "v" / "inklabels.zarr"
    g = zarr.open_group(str(d), mode="w")
    lab = np.zeros((40, 30), np.uint8)
    lab[5:9, 3:20] = 255
    g.create_dataset("0", data=lab)
    g.create_dataset("1", data=lab[::2, ::2])
    out = fod.labels_to_tif(str(tmp_path))
    assert [o["shape"] for o in out] == [[40, 30], [20, 15]]
    assert (tifffile.imread(out[0]["tif"]) == lab).all()
