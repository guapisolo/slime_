from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from typing import Any, Dict, Optional

from slime.utils.types import Sample

from .env_utils import CONFIG_PATH, ENDPOINT_PATH, LOCAL_REPO, load_arena_modules
from .logging_utils import get_eval_sample_logger

logger = logging.getLogger(__name__)

_DEFAULT_BENCH_NAME = "arena-hard-v2.0"

# Replace with your actual API key for user sim
GEMINI_API_KEY = "AIzaSyAoECOoKARlNp3k253JYDf4zybnmDyVFn8"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


class ArenaHardRuntime:
    _instance: Optional["ArenaHardRuntime"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        completion, judge_utils = load_arena_modules()

        self.make_config = completion.make_config
        self.get_endpoint = completion.get_endpoint
        self.load_questions = completion.load_questions
        self.load_model_answers = completion.load_model_answers
        self.registered_api_completion = completion.registered_api_completion
        self.JUDGE_SETTINGS = judge_utils.JUDGE_SETTINGS

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Arena-Hard config not found at {CONFIG_PATH}. Please ensure the repository is cloned."
            )
        if not ENDPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Arena-Hard API config missing at {ENDPOINT_PATH}. Please copy and fill your API credentials."
            )

        self.config = self.make_config(str(CONFIG_PATH))
        self.endpoints = self.make_config(str(ENDPOINT_PATH))
        self.prompt_template = self.config["prompt_template"]
        self.regex_patterns = [re.compile(pattern) for pattern in self.config["regex_patterns"]]
        self.judge_model = self.config["judge_model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]

        self._question_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._answer_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @classmethod
    def instance(cls) -> "ArenaHardRuntime":
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _load_questions_for_bench(self, bench_name: str) -> Dict[str, Dict[str, Any]]:
        if bench_name not in self._question_cache:
            question_path = LOCAL_REPO / "data" / bench_name / "question.jsonl"
            if not question_path.exists():
                raise FileNotFoundError(
                    f"Question file {question_path} not found. Please download arena-hard data first."
                )
            questions = self.load_questions(str(question_path))
            self._question_cache[bench_name] = {item["uid"]: item for item in questions}
        return self._question_cache[bench_name]

    def _load_answers_for_bench(self, bench_name: str) -> Dict[str, Dict[str, Any]]:
        if bench_name not in self._answer_cache:
            answer_dir = LOCAL_REPO / "data" / bench_name / "model_answer"
            if not answer_dir.exists():
                raise FileNotFoundError(
                    f"Baseline answers not found at {answer_dir}. Please copy official answers into this directory."
                )
            self._answer_cache[bench_name] = self.load_model_answers(str(answer_dir))
        return self._answer_cache[bench_name]

    @staticmethod
    def _extract_answer_text(answer_entry: Dict[str, Any]) -> str:
        messages = answer_entry.get("messages") or []
        if not messages:
            return ""
        content = messages[-1].get("content")
        if isinstance(content, dict):
            if "answer" in content:
                return str(content["answer"])
            if "content" in content:
                return str(content["content"])
        if isinstance(content, str):
            return content
        return str(content or "")

    def _build_messages(self, question: Dict[str, Any], answer_a: str, answer_b: str) -> list[Dict[str, str]]:
        system_prompt = self.JUDGE_SETTINGS[question["category"]]["system_prompt"]
        user_prompt = self.prompt_template.format(
            QUESTION=question["prompt"],
            ANSWER_A=answer_a,
            ANSWER_B=answer_b,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_judge(self, messages: list[Dict[str, str]]) -> Optional[str]:
        if self.judge_model not in self.endpoints:
            raise ValueError(f"Judge model {self.judge_model} is not defined in api_config.yaml")
        settings = self.endpoints[self.judge_model].copy()
        api_type = settings.get("api_type")
        if api_type not in self.registered_api_completion:
            raise ValueError(f"Unsupported api_type {api_type} for Arena-Hard judge")

        api_completion = self.registered_api_completion[api_type]
        kwargs = settings | {
            "api_dict": self.get_endpoint(settings.get("endpoints")),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        output = api_completion(**kwargs)
        if not output or "answer" not in output:
            return None
        return output["answer"]

    def _extract_score(self, judgment_text: str) -> Optional[str]:
        if not judgment_text:
            return None
        upper = judgment_text.upper()
        for pattern in self.regex_patterns:
            matches = pattern.findall(upper)
            matches = [match for match in matches if match]
            if matches:
                return matches[-1]
        return None

    @staticmethod
    def _score_to_verdict(score: str) -> Optional[str]:
        if not score:
            return None
        normalized = score.replace("[", "").replace("]", "").upper()
        if "A>>B" in normalized or "A>B" in normalized:
            return "A"
        if "B>>A" in normalized or "B>A" in normalized:
            return "B"
        if "A=B" in normalized:
            return "TIE"
        return None

    def _score_round(self, question: Dict[str, Any], answer_a: str, answer_b: str, model_is_b: bool) -> float:
        messages = self._build_messages(question, answer_a, answer_b)
        judgment = self._call_judge(messages)
        score = self._extract_score(judgment or "")
        verdict = self._score_to_verdict(score or "")
        if verdict == "TIE" or verdict is None:
            return 0.5
        if verdict == "B":
            return 1.0 if model_is_b else 0.0
        return 0.0 if model_is_b else 1.0

    def _score_sample_blocking(self, sample: Sample) -> float:
        metadata = sample.metadata or {}
        uid = metadata.get("uid")
        if not uid:
            logger.warning("Arena-Hard sample missing uid metadata: %s", metadata)
            return 0.0

        bench_name = metadata.get("bench_name") or self.config.get("bench_name") or _DEFAULT_BENCH_NAME
        question_map = self._load_questions_for_bench(bench_name)
        question = question_map.get(uid)
        if question is None:
            logger.warning("Question %s not found in bench %s", uid, bench_name)
            return 0.0

        answers_by_model = self._load_answers_for_bench(bench_name)
        category = question.get("category")
        if category not in self.JUDGE_SETTINGS:
            logger.warning("Unknown Arena-Hard category %s", category)
            return 0.0
        baseline_model = self.JUDGE_SETTINGS[category]["baseline"]
        baseline_answers = answers_by_model.get(baseline_model)
        if not baseline_answers or uid not in baseline_answers:
            logger.warning("Baseline %s missing uid %s", baseline_model, uid)
            return 0.0

        baseline_text = self._extract_answer_text(baseline_answers[uid])
        model_text = sample.response or ""

        total = 0.0
        rounds = 0
        total += self._score_round(question, baseline_text, model_text, model_is_b=True)
        rounds += 1
        total += self._score_round(question, model_text, baseline_text, model_is_b=False)
        rounds += 1
        if rounds == 0:
            return 0.0
        return total / rounds

    async def score_sample(self, sample: Sample) -> float:
        return await asyncio.to_thread(self._score_sample_blocking, sample)


async def compute_arena_reward(args, sample: Sample) -> float:
    runtime = ArenaHardRuntime.instance()
    reward = await runtime.score_sample(sample)
    logger_helper = get_eval_sample_logger(args)
    logger_helper.submit(sample, reward)
    return reward


__all__ = ["ArenaHardRuntime", "compute_arena_reward"]
