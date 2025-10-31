#!/usr/bin/env python3
"""Dump element-wise differences between split and merged MTP loss dumps."""

import argparse
from pathlib import Path

import numpy as np
import torch


def load_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    return tensor.detach().double().squeeze(0)


def dump_diff(values: torch.Tensor, out_path: Path, header: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, values.numpy(), fmt="%.6f", header=header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("/root/slime/debug/1"),
        help="Directory containing mtp_loss_*.pt dumps.",
    )
    parser.add_argument(
        "--prefix",
        nargs="*",
        default=["mtp_loss", "mtp_loss_after_mask"],
        help="Prefixes to process (default: %(default)s).",
    )
    args = parser.parse_args()

    for prefix in args.prefix:
        tensors = {idx: load_tensor(args.debug_dir / f"{prefix}_{idx}.pt") for idx in (0, 1, 2)}

        diff_20 = tensors[2] - tensors[0]
        diff_21 = tensors[2] - tensors[1]
        diff_avg = tensors[2] - (tensors[0] + tensors[1]) / 2

        dump_diff(diff_20, args.debug_dir / f"{prefix}_diff_2_minus_0.txt", header=f"{prefix} (combined - split0)")
        dump_diff(diff_21, args.debug_dir / f"{prefix}_diff_2_minus_1.txt", header=f"{prefix} (combined - split1)")
        dump_diff(
            diff_avg, args.debug_dir / f"{prefix}_diff_2_minus_avg.txt", header=f"{prefix} (combined - average split)"
        )


if __name__ == "__main__":
    main()
