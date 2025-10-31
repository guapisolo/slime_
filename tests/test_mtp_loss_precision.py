import pathlib

import numpy
import pytest
import torch


def _load_tensor(path: pathlib.Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False).detach().float()


@pytest.mark.xfail(reason="Combined two-sequence MTP losses diverge from single-sequence runs.")
@pytest.mark.parametrize("prefix", ["mtp_loss", "mtp_loss_after_mask"])
def test_combined_mtp_loss_matches_split_runs(prefix: str) -> None:
    """Validate that the packed two-sample run retains per-token losses.

    The expectation is that running two sequences together should numerically
    reproduce the element-wise losses obtained from processing each sequence
    individually (up to floating-point error). We approximate this by averaging
    the split runs and comparing against the combined dump.
    """

    debug_dir = pathlib.Path("/root/slime/debug/1")
    single_runs = [_load_tensor(debug_dir / f"{prefix}_{idx}.pt") for idx in (0, 1)]
    merged_run = _load_tensor(debug_dir / f"{prefix}_2.pt")

    averaged = torch.stack(single_runs, dim=0).mean(dim=0)
    max_error = (merged_run - averaged).abs().max().item()

    assert torch.allclose(
        merged_run, averaged, atol=1e-5, rtol=0.0
    ), f"{prefix} mismatch; max abs error {max_error:.6f}"


if __name__ == "__main__":
    prefix = "mtp_loss"
    debug_dir = pathlib.Path("/root/slime/debug/1")
    # original_runs = [
    #     _load_tensor(debug_dir / f"{prefix}_{idx}.pt.backup").numpy() for idx in (0, 1)
    # ]
    # print("original_runs[0]", original_runs[0])
    # print("original_runs[1]", original_runs[1])

    deterministic_original_runs = [_load_tensor(debug_dir / f"{prefix}_{idx}_deter.pt").numpy() for idx in (0, 1)]
    print("deterministic_original_runs[0]", deterministic_original_runs[0])
    print("deterministic_original_runs[1]", deterministic_original_runs[1])

    deterministic_runs = [_load_tensor(debug_dir / f"{prefix}_{idx}.pt").numpy() for idx in (0, 1, 2)]
    print("deterministic_runs[0]", deterministic_runs[0])
    print("deterministic_runs[1]", deterministic_runs[1])
    print("deterministic_runs[2]", deterministic_runs[2])

    merge = numpy.concatenate((deterministic_runs[0], deterministic_runs[1]), axis=-1)
    print(f"len r0+r1: {merge.shape} r2: {deterministic_runs[2].shape}")

    # merged_run = _load_tensor(debug_dir / f"{prefix}_2.pt").numpy()

    # print("mtp_loss_0.pt", single_runs[0])
    # print("mtp_loss_1.pt", single_runs[1])
    # print("merged_run.pt", merged_run)
