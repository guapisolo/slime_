from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from slime.utils.types import Sample

_DEFAULT_LOG_DIR = Path("/root/arena/logs")
_ACTIVE_LOGGER: Optional["EvalSampleLogger"] = None
_ACTIVE_SETTINGS: Optional[tuple[bool, int, str]] = None


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
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._run_dir = log_dir / f"arena-hard-eval-{timestamp}"

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


def _resolve_settings(
    args=None,
    *,
    enabled: Optional[object] = None,
    limit: Optional[object] = None,
    log_dir: Optional[object] = None,
) -> tuple[bool, int, Path]:
    assert args is not None, "args is required"
    if args is not None:
        enabled = getattr(args, "arena_eval_log_samples", False)
        limit = getattr(args, "arena_eval_log_limit", 10)
        log_dir = getattr(args, "arena_eval_log_dir", _DEFAULT_LOG_DIR)

    return enabled, limit, log_dir


def get_eval_sample_logger(args=None, **overrides) -> EvalSampleLogger:
    global _ACTIVE_LOGGER, _ACTIVE_SETTINGS
    enabled, limit, log_dir = _resolve_settings(args, **overrides)
    settings_key = (enabled, limit, str(log_dir))
    if _ACTIVE_LOGGER is None or _ACTIVE_SETTINGS != settings_key:
        if enabled:
            log_dir.mkdir(parents=True, exist_ok=True)
        _ACTIVE_LOGGER = EvalSampleLogger(enabled=enabled, limit=limit, log_dir=log_dir)
        _ACTIVE_SETTINGS = settings_key
    return _ACTIVE_LOGGER


__all__ = ["EvalSampleLogger", "get_eval_sample_logger"]
