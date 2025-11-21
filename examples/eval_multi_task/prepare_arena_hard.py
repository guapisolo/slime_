"""Utility to fetch Arena-Hard-Auto data and convert it for slime eval datasets.

This script clones (or updates) the upstream Arena-Hard-Auto repository, reads the
question set for a specific benchmark (default: arena-hard-v2.0), and emits a JSONL
file that matches slime's eval dataset expectations (a top-level ``prompt`` field
plus any metadata).

Usage example:

```bash
python examples/eval_multi_task/prepare_arena_hard.py \
    --output-file examples/eval_multi_task/data/arena-hard-v2.0_eval.jsonl
```

The script is idempotent and will skip cloning if the repo directory already exists.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


DEFAULT_REPO_URL = "https://github.com/lmarena/arena-hard-auto.git"
DEFAULT_REPO_DIR = Path("/root/arena/arena-hard-auto")
DEFAULT_OUTPUT = Path("/root/arena/arena-hard-v2.0_eval.jsonl")


def ensure_repo(repo_dir: Path, repo_url: str, update: bool = False) -> None:
    """Clone or optionally update the Arena-Hard-Auto repository."""

    if repo_dir.exists():
        if update:
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)


def convert_questions(question_path: Path, bench_name: str) -> list[Dict[str, Any]]:
    """Load the upstream question file and convert rows to slime-friendly format."""

    outputs: list[Dict[str, Any]] = []
    with question_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            raw = json.loads(line)
            metadata = {
                "uid": raw.get("uid"),
                "category": raw.get("category"),
                "subcategory": raw.get("subcategory"),
                "bench_name": bench_name,
            }
            # Preserve any extra metadata block if present in future releases
            for key in ("source", "date", "extra_metadata"):
                if key in raw:
                    metadata[key] = raw[key]

            outputs.append(
                {
                    "prompt": raw["prompt"],
                    "metadata": metadata,
                }
            )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_REPO_DIR,
        help="Where the arena-hard-auto repo should live (default: /root/arena/arena-hard-auto).",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default=DEFAULT_REPO_URL,
        help="Git URL for arena-hard-auto.",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="arena-hard-v2.0",
        help="Benchmark folder name inside arena-hard-auto/data/.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSONL file for slime eval consumption (default: /root/arena/arena-hard-v2.0_eval.jsonl).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="If set, run 'git pull' inside an existing repository before converting.",
    )

    args = parser.parse_args()

    ensure_repo(args.source_dir, args.repo_url, update=args.update)

    question_path = args.source_dir / "data" / args.bench_name / "question.jsonl"
    if not question_path.exists():
        raise FileNotFoundError(f"Question file not found: {question_path}")

    rows = convert_questions(question_path, args.bench_name)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(rows)} prompts to {args.output_file} using source {question_path}",
    )


if __name__ == "__main__":
    main()
