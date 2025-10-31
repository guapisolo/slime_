import json
import pathlib

import numpy
import torch


def _load_tensor(path: pathlib.Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False).detach().float()


def _to_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() <= 32:
            return tensor.tolist()
        return {
            "type": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, (numpy.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _to_serializable(vars(value))
    return str(value)


if __name__ == "__main__":
    debug_dir = pathlib.Path("/root/slime/debug/1")
    # data = torch.load(
    #     open(debug_dir / "data_union_0.pt", "rb"),
    #     weights_only=False,
    # )
    for name in ["data_0.pt", "data_1.pt", "data_union_0.pt"]:
        data = torch.load(
            open(debug_dir / name, "rb"),
            weights_only=False,
        )
        structured = _to_serializable(data)
        # print(json.dumps(structured, indent=2, ensure_ascii=False))
        with open(f"/root/slime/tests/{name}.json", "w") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)
