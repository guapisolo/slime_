from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_EMPTY_VALUES = (None, [], {})

DATASET_RUNTIME_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "n_samples_per_eval_prompt": {
        "dataset_keys": ("n_samples_per_eval_prompt",),
        "default_keys": ("n_samples_per_eval_prompt",),
        "arg_attrs": ("n_samples_per_eval_prompt", "n_samples_per_prompt"),
    },
    "temperature": {
        "dataset_keys": ("temperature",),
        "default_keys": ("temperature",),
        "arg_attrs": ("eval_temperature", "rollout_temperature"),
    },
    "top_p": {
        "dataset_keys": ("top_p",),
        "default_keys": ("top_p",),
        "arg_attrs": ("eval_top_p", "rollout_top_p"),
    },
    "top_k": {
        "dataset_keys": ("top_k",),
        "default_keys": ("top_k",),
        "arg_attrs": ("eval_top_k", "rollout_top_k"),
    },
    "max_response_len": {
        "dataset_keys": ("max_response_len",),
        "default_keys": ("max_response_len",),
        "arg_attrs": ("eval_max_response_len", "rollout_max_response_len"),
    },
    "min_new_tokens": {
        "dataset_keys": ("min_new_tokens",),
        "default_keys": ("min_new_tokens",),
        "arg_attrs": ("eval_min_new_tokens",),
    },
}

DATASET_SAMPLE_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "prompt_key": {
        "dataset_keys": ("prompt_key",),
        "default_keys": ("prompt_key",),
        "arg_attrs": ("eval_input_key", "input_key"),
    },
    "label_key": {
        "dataset_keys": ("label_key",),
        "default_keys": ("label_key",),
        "arg_attrs": ("eval_label_key", "label_key"),
    },
    "tool_key": {
        "dataset_keys": ("tool_key",),
        "default_keys": ("tool_key",),
        "arg_attrs": ("eval_tool_key", "tool_key"),
    },
    "metadata_key": {
        "dataset_keys": ("metadata_key",),
        "default_keys": ("metadata_key",),
        "arg_attrs": ("metadata_key",),
    },
    "stop": {
        "dataset_keys": ("stop",),
        "default_keys": ("stop",),
        "arg_attrs": ("rollout_stop",),
    },
    "stop_token_ids": {
        "dataset_keys": ("stop_token_ids",),
        "default_keys": ("stop_token_ids",),
        "arg_attrs": ("rollout_stop_token_ids",),
    },
}


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _pick_from_mapping(data: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
    if not data:
        return None
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _ensure_metadata_overrides(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata_overrides must be a mapping.")
    return value


class EvalDatasetConfig(BaseModel):
    """Configuration for a single evaluation dataset."""

    name: str
    path: str
    rm_type: str | None = None

    # Dataset-specific overrides
    prompt_key: str | None = None
    label_key: str | None = None
    tool_key: str | None = None
    metadata_key: str | None = None

    n_samples_per_eval_prompt: int | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_response_len: int | None = None
    min_new_tokens: int | None = None

    stop: Sequence[str] | None = None
    stop_token_ids: Sequence[int] | None = None

    metadata_overrides: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @field_validator("metadata_overrides", mode="before")
    def _validate_metadata_overrides(cls, value: Any) -> dict[str, Any]:
        return _ensure_metadata_overrides(value)

    @property
    def cache_key(self) -> tuple[Any, ...]:
        """Return a tuple uniquely identifying dataset config for caching."""
        return (
            self.name,
            self.path,
            self.prompt_key,
            self.label_key,
            self.tool_key,
            self.metadata_key,
        )

    def inject_metadata(self, sample_metadata: Any) -> dict[str, Any]:
        """Return updated metadata merging overrides."""
        if not isinstance(sample_metadata, dict):
            metadata = {}
        else:
            metadata = dict(sample_metadata)

        if self.rm_type is not None:
            metadata["rm_type"] = self.rm_type

        for key, value in self.metadata_overrides.items():
            metadata[key] = value

        return metadata


def ensure_dataset_list(config: Any) -> list[dict[str, Any]]:
    """
    Normalize OmegaConf containers into a list of dicts.
    Accepts either a list or dictionary keyed by dataset name.
    """
    if config is None:
        return []

    if isinstance(config, dict):
        datasets = []
        for name, cfg in config.items():
            dataset = dict(cfg or {})
            dataset.setdefault("name", name)
            datasets.append(dataset)
        return datasets

    if isinstance(config, (list, tuple)):
        datasets = []
        for item in config:
            dataset = dict(item or {})
            if "name" not in dataset:
                raise ValueError("Each evaluation dataset entry must include a `name` field.")
            datasets.append(dataset)
        return datasets

    raise TypeError("eval.datasets must be either a list or a mapping.")


def _apply_dataset_field_overrides(
    args: Any, dataset_cfg: dict[str, Any], defaults: dict[str, Any], spec_names: dict[str, Any]
) -> None:
    for field_name, spec in spec_names.items():
        dataset_value = _pick_from_mapping(dataset_cfg, spec["dataset_keys"])
        default_value = _pick_from_mapping(defaults, spec["default_keys"])
        arg_values = [getattr(args, attr, None) for attr in spec["arg_attrs"]]
        resolved_value = _first_not_none(dataset_value, default_value, *arg_values)
        if resolved_value is not None:
            dataset_cfg[field_name] = resolved_value


def build_eval_dataset_configs(
    args: Any,
    raw_config: Iterable[dict[str, Any]],
    defaults: dict[str, Any],
) -> list[EvalDatasetConfig]:
    defaults = defaults or {}
    datasets: list[EvalDatasetConfig] = []
    for cfg in raw_config:
        cfg_dict = dict(cfg or {})
        combined_specs = {**DATASET_RUNTIME_SPECS, **DATASET_SAMPLE_SPECS}
        _apply_dataset_field_overrides(args, cfg_dict, defaults, combined_specs)
        dataset = EvalDatasetConfig(**cfg_dict)
        datasets.append(dataset)
    return datasets
