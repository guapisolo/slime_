from typing import Any, Dict, List

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import RunConfig, SolveResult


class TrainableAgentMixin:
    async def asolve(self, env: Env, url: str, sampling_params: Dict[str, Any]) -> SolveResult:
        """
        Extend original Agent to support aync call interaction with LLM server.

        Given a URL and sampling parameters, this method should asynchronously
        send a request to the LLM server and return the sampling result.

        It book keeps the loss mask and records metadata for the request.
        """
        raise NotImplementedError


def agent_factory(tools_info: List[Dict[str, Any]], wiki, config: RunConfig) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent

        assert (
            config.few_shot_displays_path is not None
        ), "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")
