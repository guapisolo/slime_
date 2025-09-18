# Tau bench 
This example shows slime training in an agentic multi-turn tool use environment. 


## Environment Setup 
Use the `zhuzilin/slime:latest` image and initialize the environment required for Search-R1:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
# for tau bench 
cd /root/
git clone https://github.com/sierra-research/tau-bench
cd tau-bench
pip install -e . 
```

Use the following script to generate mock data for slime training. 

```bash
cd /root/slime/examples/tau-bench
python tau1_mock.py --local_dir /root/tau-bench/
```

Initialize the Qwen2.5-3B model:

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B \
    --save /root/Qwen2.5-3B_torch_dist
```

## Running the Script

You need to configure your litellm API in `generate_with_tau.py` for user simulation:

```python
TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "react",  # Select between ["tool-calling", "act", "react", "few-shot"]
    "user_model": "gemini-2.0-flash-lite",  # Replace if you want any other model
    "google_api_key": "YOUR_API_KEY",  # Replace with your actual API key for user sim
    "task_split": "train",  # Select between ["train", "test", "dev"]
    "proxy": None,  # Set to your proxy if needed
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
}
```

And run:

```bash
cd /root/slime
bash examples/tau-bench/run_qwen2.5_3B.sh
```
