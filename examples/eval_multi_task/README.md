# Multi-Task Evaluation Example

## Folder layout
- `multi_task.yaml` / `multi_task.sh`: top-level config + sample launch script.
- `tasks/<task_name>/`: per-benchmark helpers (data prep scripts, extra requirements, etc.). For example:
  - `tasks/arena_hard/requirements.txt` and `prepare_data.py`
  - `tasks/ifbench/requirements.txt`

## Configuring `multi_task.yaml`
- `eval.defaults` defines inference parameters shared by every dataset entry. Override them inside an individual dataset block if needed.
- `eval.datasets` enumerates the datasets to evaluate. Each entry should specify:
  - `name`: a short identifier that appears in logs and dashboards.
  - `path`: the path to the dataset JSONL file.
  - `rm_type`: which reward function to use for scoring.
  - `n_samples_per_eval_prompt`: how many candidate completions to generate per prompt.

## IFBench Notes
- When `ifbench` is used, `slime/rollout/rm_hub/ifbench.py` will automatically prepares the scoring environment, so no additional manual setup is required beyond providing the dataset path.

## Arena-Hard v2.0
1. **Download upstream repo & convert questions**
   ```bash
   python examples/eval_multi_task/tasks/arena_hard/prepare_data.py --update
   ```
   This clones `arena-hard-auto` into `/root/arena/arena-hard-auto` (or updates it) and writes `/root/arena/arena-hard-v2.0_eval.jsonl`, which is the dataset path already referenced in `multi_task.yaml`.
2. **Configure judge endpoints**: edit `/root/arena/arena-hard-auto/config/api_config.yaml` with your API keys (e.g., GPT-4.1 or Gemini). The reward integration reads this file directly, so no extra environment variables are needed beyond what the upstream repo requires.
3. **Run eval as usual**: launch `examples/eval_multi_task/multi_task.sh` (or your own script). When `rm_type: arena-hard` is detected, `slime/rollout/rm_hub/arena_hard.py` automatically installs the helper packages listed in `examples/eval_multi_task/tasks/arena_hard/requirements.txt`, reuses the upstream baseline answers, and calls the official judge in two directions to produce the Arena score.
