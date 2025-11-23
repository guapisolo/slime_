from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

TASK_ROOT = Path(__file__).resolve().parent
LOCAL_REPO = Path("/root/arena/arena-hard-auto")
LOCAL_REQUIREMENTS = TASK_ROOT / "requirements.txt"
CONFIG_PATH = LOCAL_REPO / "config" / "arena-hard-v2.0.yaml"
ENDPOINT_PATH = LOCAL_REPO / "config" / "api_config.yaml"


def _ensure_repo() -> Path:
    if LOCAL_REPO.exists():
        repo_path = LOCAL_REPO
    else:
        repo_path = LOCAL_REPO
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = [
            "git",
            "clone",
            "https://github.com/lmarena/arena-hard-auto.git",
            str(repo_path),
        ]
        subprocess.run(clone_cmd, check=True)

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    return repo_path


def _ensure_dependencies(repo_path: Path) -> None:
    if not LOCAL_REQUIREMENTS.exists():
        logger.debug("Arena-Hard requirements file missing at %s", LOCAL_REQUIREMENTS)
        return

    sentinel = repo_path / ".deps_installed"
    if sentinel.exists():
        return

    install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(LOCAL_REQUIREMENTS)]
    try:
        subprocess.run(install_cmd, check=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to install Arena-Hard dependencies automatically: %s", exc)
    else:
        sentinel.write_text("installed\n")


def load_arena_modules():
    repo_path = _ensure_repo()
    _ensure_dependencies(repo_path)
    completion = importlib.import_module("utils.completion")
    judge_utils = importlib.import_module("utils.judge_utils")
    return completion, judge_utils


__all__ = ["LOCAL_REPO", "CONFIG_PATH", "ENDPOINT_PATH", "load_arena_modules"]
