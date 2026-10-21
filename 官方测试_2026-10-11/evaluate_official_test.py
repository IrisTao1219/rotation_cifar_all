"""Evaluate the six frozen checkpoints once on the official CIFAR-10 test set."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(probability, labels):
    pred = probability.argmax(1)
    matrix = np.bincount(labels * 10 + pred, minlength=100).reshape(10, 10)
    tp = np.diag(matrix)
    f1 = 2 * tp / np.maximum(1, matrix.sum(0) + matrix.sum(1))
    recall = tp / np.maximum(1, matrix.sum(1))
    return {"accuracy": float((pred == labels).mean()),
            "macro_f1": float(f1.mean()), "per_class_f1": f1.tolist(),
            "per_class_recall": recall.tolist(), "confusion": matrix.tolist()}


def fuse(probability, choice):
    if choice == "mean":
        return probability.mean(0)
    if choice == "geometric":
        logp = np.log(probability.clip(1e-8)).mean(0)
        p = np.exp(logp - logp.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)
    raise ValueError(choice)


def comparison(candidate, reference, mask=None):
    delta = np.stack(candidate).astype(float) - np.stack(reference)
    if mask is not None:
        delta = delta[:, mask]
    rng = np.random.default_rng(20261011)
    samples = []
    for _ in range(5000):
        seeds = rng.integers(3, size=3)
        ids = rng.integers(delta.shape[1], size=delta.shape[1])
        samples.append(float(delta[seeds][:, ids].mean()))
    return {"image_count": delta.shape[1],
            "difference_by_seed": delta.mean(1).tolist(),
            "mean_difference": float(delta.mean()),
            "paired_image_and_seed_bootstrap_95_interval":
                np.percentile(samples, [2.5, 97.5]).tolist(),
            "bootstrap_draws": 5000,
            "bootstrap_units": "original images and paired training seeds"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    lock = json.loads((ROOT / "evaluation_lock.json").read_text())
    code_lock = json.loads((ROOT / "evaluation_code_lock.json").read_text())
    for filename, digest in code_lock["files_sha256"].items():
        assert sha256(ROOT / filename) == digest, filename
    for filename, digest in lock["source_files_sha256"].items():
        assert sha256(project / filename) == digest, filename
    for spec in lock["models"]:
        folder = project / "runs" / spec["run"]
        assert sha256(folder / "best.pt") == spec["best_pt_sha256"]
        assert sha256(folder / "config.json") == spec["config_sha256"]
        if "aggregation_selection_sha256" in spec:
            assert sha256(folder / "aggregation_selection.json") == spec["aggregation_selection_sha256"]
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    assert sha256(ROOT / "data/official_test.npz") == manifest["npz_sha256"]
    with np.load(ROOT / "data/official_test.npz", allow_pickle=False) as data:
        images, labels, ids = data["images"], data["labels"], data["ids"]
    assert images.shape == (10000, 3, 32, 32) and images.dtype == np.uint8
    assert np.array_equal(ids, np.arange(10000))
    assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, 1000))
    sys.path.insert(0, str(project))
    from experiment import LeNet, orbit

    # Check new metric/fusion plumbing against existing frozen confirmation output.
    previous = json.loads((project / "results/strong_control_confirmation.json").read_text())
    for row in previous["rows"]:
        run = project / "runs" / f"aug4_large_s{row['seed']}_e80"
        with np.load(run / "fixed_confirm_predictions.npz") as stored:
            result = metrics(fuse(stored["probabilities"], row["aggregation_selected_on_dev"]), stored["labels"])
        assert abs(result["accuracy"] - row["baseline_accuracy"]) < 1e-12

    output = ROOT / "predictions"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261011)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_per_process_memory_fraction(.2)
    x = torch.tensor(images, device="cuda", dtype=torch.float32) / 255
    mean = torch.tensor([.4914, .4822, .4465], device="cuda")[None, :, None, None]
    std = torch.tensor([.247, .2435, .2616], device="cuda")[None, :, None, None]
    x = (x - mean) / std
    rows = []
    correct = {k: [] for k in ["baseline_single_original", "baseline_mean", "baseline_selected", "feature_max"]}
    records = {k: [] for k in correct}
    for spec in lock["models"]:
        start = time.monotonic()
        checkpoint = torch.load(project / "runs" / spec["run"] / "best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["config"]["mode"] == spec["mode"]
        model = LeNet(spec["mode"]).cuda().eval()
        model.load_state_dict(checkpoint["model"])
        assert sum(p.numel() for p in model.parameters()) == spec["parameters"]
        blocks = []
        with torch.inference_mode():
            for batch in x.split(256):
                if model.invariant:
                    prob = torch.stack([model(batch.rot90(k, (-2, -1))).softmax(1) for k in range(4)])
                else:
                    prob = model(orbit(batch)).softmax(1).reshape(4, len(batch), 10)
                blocks.append(prob.cpu().numpy())
        p = np.concatenate(blocks, axis=1)
        assert np.isfinite(p).all() and np.allclose(p.sum(2), 1, atol=1e-5)
        methods = {"feature_max": p[0]} if model.invariant else {
            "baseline_single_original": p[0], "baseline_mean": fuse(p, "mean"),
            "baseline_selected": fuse(p, spec["fusion_selected_on_dev"])}
        row = {"run": spec["run"], "seed": spec["seed"], "parameters": spec["parameters"],
               "epoch_selected_on_dev": checkpoint["epoch"],
               "fusion_selected_on_dev": spec["fusion_selected_on_dev"],
               "angle_accuracy": (p.argmax(2) == labels[None]).mean(1).tolist(),
               "direction_disagreement": float((p.argmax(2) != p[0].argmax(1)).any(0).mean()),
               "probability_semantics": "final feature-aggregated predictions for four rotated inputs" if model.invariant else "single-view predictions at four angles",
               "methods": {}}
        if model.invariant:
            row["max_rotation_probability_deviation"] = float(np.max(np.abs(p - p[:1])))
            assert row["max_rotation_probability_deviation"] < 1e-6
        for method, probability in methods.items():
            value = metrics(probability, labels)
            row["methods"][method] = value
            records[method].append(value)
            correct[method].append(probability.argmax(1) == labels)
        np.savez_compressed(output / f"{spec['run']}.npz", probabilities=p, labels=labels, ids=ids)
        row["evaluation_seconds_including_rotation_checks"] = time.monotonic() - start
        (output / f"{spec['run']}.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(json.dumps({"run": row["run"], "accuracy": {k: v["accuracy"] for k, v in row["methods"].items()}}), flush=True)
        del model

    methods = {}
    for method, values in records.items():
        scores = [v["accuracy"] for v in values]
        methods[method] = {"accuracy_by_seed": scores, "mean_accuracy": float(np.mean(scores)),
                           "sd_accuracy": float(np.std(scores, ddof=1)),
                           "macro_f1_by_seed": [v["macro_f1"] for v in values],
                           "mean_macro_f1": float(np.mean([v["macro_f1"] for v in values])),
                           "mean_per_class_recall": np.mean([v["per_class_recall"] for v in values], axis=0).tolist()}
    result = {"dataset": "official CIFAR-10 test", "original_image_count": len(labels),
              "seeds": [1, 2, 3], "parameters_per_model": 256830,
              "training_images_per_model": 40000, "retrained": False,
              "methods": methods, "primary_comparison": comparison(correct["feature_max"], correct["baseline_selected"]),
              "probability_mean_comparison": comparison(correct["feature_max"], correct["baseline_mean"]),
              "single_original_view_comparison_descriptive": comparison(correct["feature_max"], correct["baseline_single_original"]),
              "runs": rows,
              "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda}}
    overlap = set(manifest["overlap_audit"]["any_training_source_orbit_overlap_test_ids"])
    if overlap:
        mask = np.array([i not in overlap for i in ids])
        result["nonoverlap_sensitivity_comparison"] = comparison(correct["feature_max"], correct["baseline_selected"], mask)
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": True, "methods": methods, "primary_comparison": result["primary_comparison"]}), flush=True)


if __name__ == "__main__":
    main()
