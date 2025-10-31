#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge rollout debug dumps written by _save_debug_rollout_data.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Paths to input .pt files (e.g. data_0.pt data_1.pt).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the merged output file (e.g. data_union_0.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.input]
    output_path = Path(args.output)

    if len(input_paths) < 2:
        raise ValueError("At least two input files are required to perform a union.")

    merged_samples = []
    seen = set()
    base_rollout_id = None

    for path in input_paths:
        payload = torch.load(path, weights_only=False)
        if base_rollout_id is None:
            base_rollout_id = payload.get("rollout_id")
        samples = payload.get("samples", [])
        for sample in samples:
            sample_id = (sample.get("group_index"), sample.get("index"))
            if sample_id in seen:
                continue
            seen.add(sample_id)
            merged_samples.append(sample)

    output_payload = dict(rollout_id=base_rollout_id, samples=merged_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_path)

    print(
        f"Saved {len(merged_samples)} merged samples from {len(input_paths)} files to {output_path} "
        f"(rollout_id={base_rollout_id})."
    )


if __name__ == "__main__":
    main()
