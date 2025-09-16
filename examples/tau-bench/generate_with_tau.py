from typing import Any, Dict

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import agent_factory

from slime.utils.types import Sample

TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"], only tool-calling implemented for now
    "user_model": "gemini-2.0-flash-lite",  # Cheap Model for user simulator
    "google_api_key": "KEY_HERE",  # Replace with your actual API key for user sim
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail, ["test"] for airline
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
}
tau_config = RunConfig(**TAU_CONFIGS)


def res_to_sample(res) -> Sample:
    # Convert back to sample format with reward and metadata
    raise NotImplementedError
    return Sample()


async def generate(args: Dict[str, Any], sample: Sample, sampling_params: dict):
    # Generate a full environment trajectory with Tau-Bench
    assert not args.partial_rollout, f"Partial rollout is not supported for this function at the moment."
    env = get_env(
        tau_config.env,
        user_strategy=tau_config.user_strategy,
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=tau_config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tau_config,
    )
    # Samples are required to have prompt field. Instead of setting the actual sample, we set the index within the environment
    # for repeatability.
    task_index = int(sample.prompt)
    print(f"Running agent-environment interaction in task {sample.prompt}")
    res = await agent.asolve(env, args, sampling_params, task_index)
    return res_to_sample(res)
