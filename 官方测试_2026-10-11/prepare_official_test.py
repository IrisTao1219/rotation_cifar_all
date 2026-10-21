"""Decode the checksum-pinned official test mirror and audit image overlap."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def digest(content, algorithm="sha256"):
    return hashlib.new(algorithm, content).hexdigest()


def image_keys(images):
    exact, orbit = [], []
    for image in images:
        hashes = [digest(np.rot90(image, k, axes=(-2, -1)).tobytes()) for k in range(4)]
        exact.append(hashes[0])
        orbit.append(min(hashes))
    return exact, orbit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    parquet_sha = digest(args.parquet.read_bytes())
    assert parquet_sha == "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"
    table = pq.read_table(args.parquet)
    images = np.stack([np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB")).transpose(2, 0, 1) for r in table["img"].to_pylist()])
    labels = np.asarray(table["label"].to_pylist(), dtype=np.int64)
    assert images.shape == (10000, 3, 32, 32) and images.dtype == np.uint8
    assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, 1000))
    folder = ROOT / "data"
    folder.mkdir(exist_ok=False)
    shutil.copy2(args.parquet, folder / "test.parquet")
    np.savez_compressed(folder / "official_test.npz", images=images, labels=labels, ids=np.arange(10000))
    test_exact, test_orbit = image_keys(images)
    overlap = {}
    all_overlap = set()
    for split in ["train", "dev", "confirm"]:
        with np.load(args.project / "data" / f"{split}.npz", allow_pickle=False) as source:
            exact, orbit = image_keys(source["images"])
        exact_set, orbit_set = set(exact), set(orbit)
        exact_ids = [i for i, h in enumerate(test_exact) if h in exact_set]
        orbit_ids = [i for i, h in enumerate(test_orbit) if h in orbit_set]
        overlap[split] = {"source_images": len(exact), "exact_overlap_test_ids": exact_ids,
                          "orbit_overlap_test_ids": orbit_ids}
        all_overlap.update(orbit_ids)
    overlap["any_training_source_orbit_overlap_test_ids"] = sorted(all_overlap)
    manifest = {"prepared_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "official CIFAR-10 test", "count": len(images),
                "class_counts": np.bincount(labels, minlength=10).tolist(),
                "official_dataset_page": "https://www.cs.toronto.edu/~kriz/cifar.html",
                "source_url": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/test-00000-of-00001.parquet",
                "revision": "0b2714987fa478483af9968de7c934580d0bb9a2",
                "checksum_reference": "Hugging Face fixed-revision LFS oid",
                "test_parquet_sha256": parquet_sha,
                "npz_sha256": digest((folder / "official_test.npz").read_bytes()),
                "test_internal_exact_duplicate_count": len(images) - len(set(test_exact)),
                "test_internal_orbit_duplicate_count": len(images) - len(set(test_orbit)),
                "overlap_audit": overlap, "overlap_policy": "full official set is primary; non-overlap analysis is secondary",
                "near_duplicates_checked": False}
    (ROOT / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"count": len(images), "test_parquet_sha256": parquet_sha,
                      "overlap_by_split": {s: len(overlap[s]["orbit_overlap_test_ids"]) for s in ["train", "dev", "confirm"]},
                      "total_overlap": len(all_overlap)}), flush=True)


if __name__ == "__main__":
    main()
