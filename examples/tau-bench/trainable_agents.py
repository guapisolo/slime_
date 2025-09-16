from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent, message_to_action
from tau_bench.types import RunConfig

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post


class InteractionResult(BaseModel):
    prompt: List[Dict[str, Any]]
    reward: float
    messages: List[Dict[str, Any]]
    info: Dict[str, Any]
    loss_mask: Optional[List[int]] = None
    tokens: Optional[int] = None

    class Status(str):
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    status: Status = Status.COMPLETED


class TrainableAgentMixin:
    async def asolve(
        self,
        rollout_args: Dict[str, Any],
        sampling_params: Dict[str, Any],
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        """
        Extend original Agent to support aync call interaction with LLM server.

        Given a URL and sampling parameters, this method should asynchronously
        send a request to the LLM server and return the sampling result.

        It book keeps the loss mask and records metadata for the request.
        """
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:{rollout_args.sglang_router_port}/generate"

        env = env.reset(task_index=task_index)
        obs, info = env.observation, env.info.model_dump()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.wiki}, {"role": "user", "content": obs}]
        prompt_text = state.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response, response_token_ids = "", []
        loss_masks = [0] * len(prompt_token_ids)

        # Overwrite prompt with correct task description.
        res = InteractionResult(prompt=prompt_text, reward=0, messages=[], info={})
        for _ in range(max_num_steps):
            payload = {
                "messages": messages,
                "tools": self.tools_info,
                "sampling_params": sampling_params,
                "tool_call_parser": "qwen25",  # or mistral, llama3 depending on your model
            }
            output = await post(url, payload, use_http2=rollout_args.use_http2)
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                res.status = InteractionResult.Status.ABORTED
                return res

            choice = output["choices"][0]
            next_message = choice.message.model_dump()

            # Handling the current assistant response
            cur_token_ids = state.tokenizer(next_message["content"], add_special_tokens=False)["input_ids"]
            response += next_message["content"]
            response_token_ids += cur_token_ids
            loss_masks += [1] * len(cur_token_ids)

            # Step in environment
            action = message_to_action(next_message)
            env_response = env.step(action)

            # Handling the environment response
            obs_tokens_ids = state.tokenizer(env_response.observation, add_special_tokens=False)["input_ids"]
            response += env_response.observation
            response_token_ids += obs_tokens_ids
            loss_masks += [0] * len(obs_tokens_ids)

            # Include the reward info
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                if len(next_message["tool_calls"]) > 1:
                    print("Multiple tool calls detected, only the first one will be executed.")
                tool_called = next_message["tool_calls"][0]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_called["id"],
                            "name": tool_called["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                res.status = InteractionResult.Status.COMPLETED
                break
        if not env_response.done:
            res.status = InteractionResult.Status.TRUNCATED
        res.reward = reward
        res.info = info
        res.messages = messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        return res


class TrainableToolCallingAgent(ToolCallingAgent, TrainableAgentMixin):
    pass


def agent_factory(tools_info: List[Dict[str, Any]], wiki, config: RunConfig) -> Agent:
    if config.agent_strategy == "tool-calling":
        return TrainableToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    else:
        raise NotImplementedError(f"Unsupported agent strategy: {config.agent_strategy}")
