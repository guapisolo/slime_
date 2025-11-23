from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from slime.utils.types import Sample

_DEFAULT_LOG_DIR = Path("/root/arena/logs")
_LOGGER_CACHE: dict[tuple[bool, int, str], "EvalSampleLogger"] = {}


class EvalSampleLogger:
    """Optional helper to dump a few Arena-Hard eval samples for debugging."""

    def __init__(self, *, enabled: bool, limit: int, log_dir: Path) -> None:
        self.enabled = bool(enabled)
        self.limit = max(int(limit or 0), 0)
        self._count = 0
        self._lock = threading.Lock()
        self._log_path: Optional[Path] = None
        self._run_dir: Optional[Path] = None
        if self.enabled:
            base_dir = log_dir
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._run_dir = base_dir / f"arena-hard-eval-{timestamp}"

    def submit(self, sample: Sample, reward: float) -> None:
        if not self.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._log_sync(sample, reward)
        else:
            loop.create_task(asyncio.to_thread(self._log_sync, sample, reward))

    def _log_sync(self, sample: Sample, reward: float) -> None:
        if not self.enabled or self._count >= self.limit:
            return
        if sample is None:
            return

        with self._lock:
            if self._count >= self.limit:
                return
            if self._log_path is None:
                assert self._run_dir is not None
                self._run_dir.mkdir(parents=True, exist_ok=True)
                self._log_path = self._run_dir / "samples.log"

            metadata = getattr(sample, "metadata", {}) or {}
            uid = metadata.get("uid", "unknown")
            bench_name = metadata.get("bench_name", "unknown")
            prompt = sample.prompt if isinstance(sample.prompt, str) else repr(sample.prompt)
            response = sample.response if isinstance(sample.response, str) else repr(sample.response)
            reward_repr = repr(reward)

            entry = "\n".join(
                [
                    f"[sample {self._count + 1}] bench={bench_name} uid={uid} status={sample.status} reward={reward_repr}",
                    "Prompt:",
                    str(prompt),
                    "Response:",
                    str(response),
                    "-" * 60,
                ]
            )
            write_newline = self._log_path.exists() and self._log_path.stat().st_size > 0
            with self._log_path.open("a", encoding="utf-8") as fout:
                if write_newline:
                    fout.write("\n")
                fout.write(entry)
            self._count += 1


def _coerce_bool(value: Optional[object]) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_settings(
    *,
    enabled: Optional[object],
    limit: Optional[object],
    log_dir: Optional[str | Path],
) -> tuple[bool, int, Path]:
    enabled_val = _coerce_bool(enabled) if enabled is not None else False
    limit_val = int(limit) if limit is not None else 10
    log_dir_val = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR
    return enabled_val, limit_val, log_dir_val


def get_eval_sample_logger(
    args=None,
    *,
    enabled: Optional[bool] = None,
    limit: Optional[int] = None,
    log_dir: Optional[str | Path] = None,
) -> EvalSampleLogger:
    if args is not None:
        enabled = getattr(args, "arena_eval_log_samples", enabled)
        limit = getattr(args, "arena_eval_log_limit", limit)
        log_dir = getattr(args, "arena_eval_log_dir", log_dir)

    normalize_enabled, normalize_limit, normalize_dir = _normalize_settings(
        enabled=enabled,
        limit=limit,
        log_dir=log_dir,
    )

    cache_key = (normalize_enabled, normalize_limit, str(normalize_dir))
    logger = _LOGGER_CACHE.get(cache_key)
    if logger is None:
        if normalize_enabled:
            normalize_dir.mkdir(parents=True, exist_ok=True)
        logger = EvalSampleLogger(
            enabled=normalize_enabled,
            limit=normalize_limit,
            log_dir=normalize_dir,
        )
        _LOGGER_CACHE[cache_key] = logger
    return logger


__all__ = ["EvalSampleLogger", "get_eval_sample_logger"]
