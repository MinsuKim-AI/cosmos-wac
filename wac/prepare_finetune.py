"""Prepare LIBERO-Spatial training rollouts and check them with Cosmos Policy."""

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import torch
from cosmos_policy.datasets.libero_dataset import LIBERODataset


def prepare(source, target):
    with (source / "index.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "train"]
    conditions = {row["condition"] for row in rows}
    expected = {
        (condition, task, init, 1)
        for condition in conditions for task in range(10) for init in range(5)
    }
    actual = {
        (row["condition"], int(row["task_id"]), int(row["init_state_idx"]), int(row["perturb_seed"]))
        for row in rows
    }
    if len(rows) != 100 or len(conditions) != 2 or actual != expected:
        raise ValueError("Expected 100 unique train episodes across both perturbation conditions")

    files = {}
    for row in rows:
        path = Path(row["hdf5"])
        if path.parts[:2] != (row["condition"], "train") or ".." in path.parts:
            raise ValueError(f"Invalid train HDF5 path: {path}")
        with h5py.File(source / path, "r") as episode:
            success = bool(int(row["success"]))
            if episode.attrs["task_description"] != row["task"] or bool(episode.attrs["success"]) != success:
                raise ValueError(f"HDF5 metadata does not match the index: {path}")
            image_keys = ("primary_images", "wrist_images")
            if "primary_images_jpeg" in episode:
                image_keys = ("primary_images_jpeg", "wrist_images_jpeg")
            steps = len(episode[image_keys[0]])
            if not steps or len(episode[image_keys[1]]) != steps:
                raise ValueError(f"Camera stream length mismatch: {path}")
            for field, width in (("actions", 7), ("proprio", 9)):
                if episode[field].shape != (steps, width):
                    raise ValueError(f"Invalid {field} array: {path}")
        files[target / path] = source / path

    for link, original in files.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(os.path.relpath(original, link.parent))
        if not link.is_symlink() or link.resolve() != original.resolve():
            raise ValueError(f"Unexpected file at {link}")
    if set(target.rglob("*.hdf5")) != files.keys():
        raise ValueError(f"Unexpected HDF5 files under {target}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("demos", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("demo_probability", type=float)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = prepare(args.source.resolve(), args.target.resolve())

    demos = args.demos.resolve()
    if len(list(demos.glob("libero_spatial_regen/*.hdf5"))) != 10 or len(list(demos.rglob("*.hdf5"))) != 10:
        raise ValueError(f"Expected the 10 official LIBERO-Spatial demonstration files under {demos}")
    dataset = LIBERODataset(
        data_dir=str(demos),
        rollout_data_dir=str(args.target.resolve()),
        t5_text_embeddings_path=str(args.embeddings.resolve()),
        chunk_size=16,
        demonstration_sampling_prob=args.demo_probability,
    )
    if dataset.rollout_num_episodes != len(rows) or not dataset.num_episodes:
        raise ValueError("Cosmos Policy did not discover the expected datasets")
    if not all((dataset.adjusted_demo_count, dataset.adjusted_success_rollout_count,
                dataset.adjusted_failure_rollout_count)):
        raise ValueError("Training needs demonstrations, successes, and failures")
    for index in (0, dataset.adjusted_demo_count,
                  dataset.adjusted_demo_count + dataset.adjusted_success_rollout_count):
        sample = dataset[index]
        if sample["video"].shape != (3, 33, 224, 224) or sample["actions"].shape != (16, 7):
            raise ValueError(f"Unexpected Cosmos Policy sample shapes at {index}")

    summary = {
        "rollout_episodes": dataset.rollout_num_episodes,
        "rollout_steps": dataset.rollout_num_steps,
        "demo_episodes": dataset.num_episodes,
        "demo_samples": dataset.adjusted_demo_count,
        "success_samples": dataset.adjusted_success_rollout_count,
        "failure_samples": dataset.adjusted_failure_rollout_count,
        "samples_per_epoch": len(dataset),
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
