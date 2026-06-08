# -*- coding: utf-8 -*-
# @Desc    : Run baseline (base model) experiments across multiple env settings

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.humaneval import HumanEvalBenchmark
from scripts.async_llm import LLMsConfig, create_llm_instance
from scripts.operators import CustomCodeGenerate
from scripts.workflow import Workflow


# class MathCoTWorkflow(Workflow):
#     def __init__(self, name: str, llm_config, dataset: str):
#         self.name = name
#         self.dataset = dataset
#         self.llm = create_llm_instance(llm_config)
#         self.response = AnswerGenerate(self.llm)
#
#     async def __call__(self, problem: str):
#         solution = await self.response(input=problem)
#         return solution["answer"], self.llm.get_usage_summary()["total_cost"]


CODE_COT_INSTRUCTION = (
    "Think step by step to solve the programming task. "
    "Use the reasoning to design the algorithm and handle edge cases, "
    "but do not include the reasoning in the final response. "
)


class HumanEvalWorkflow(Workflow):
    def __init__(self, name: str, llm_config, dataset: str):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom_code_generate = CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str, question_id: str = ""):
        solution = await self.custom_code_generate(
            problem=problem,
            entry_point=entry_point,
            instruction=CODE_COT_INSTRUCTION,
        )
        return solution["response"], self.llm.get_usage_summary()["total_cost"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


async def run_baseline_for_model(
    model_name: str,
    humaneval_path: str,
    log_root: str,
    max_concurrent_tasks: int,
):
    models_config = LLMsConfig.default()
    llm_config = models_config.get(model_name)

    # MATH baseline is intentionally disabled for single-base-model code evaluation.
    # math_log_path = os.path.join(log_root, "MATH", model_name)
    # _ensure_dir(math_log_path)
    # math_benchmark = MATHBenchmark(name="MATH", file_path=math_path, log_path=math_log_path)
    # math_workflow = MathCoTWorkflow(name="math_baseline", llm_config=llm_config, dataset="MATH")
    # await math_benchmark.run_baseline(math_workflow, max_concurrent_tasks=max_concurrent_tasks)

    # HumanEval (code baseline with explicit CoT instruction)
    humaneval_log_path = os.path.join(log_root, "HumanEval", model_name)
    _ensure_dir(humaneval_log_path)
    humaneval_benchmark = HumanEvalBenchmark(
        name="HumanEval",
        file_path=humaneval_path,
        log_path=humaneval_log_path,
    )
    humaneval_workflow = HumanEvalWorkflow(name="humaneval_baseline", llm_config=llm_config, dataset="HumanEval")
    await humaneval_benchmark.run_baseline(humaneval_workflow, max_concurrent_tasks=max_concurrent_tasks)


async def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments across env settings")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated model names from config/config2.yaml",
    )
    # parser.add_argument(
    #     "--math_path",
    #     type=str,
    #     default="data/datasets/math_validate.jsonl",
    #     help="Path to MATH jsonl (e.g., MATH-500 file)",
    # )
    parser.add_argument(
        "--humaneval_path",
        type=str,
        default="data/datasets/humaneval_validate.jsonl",
        help="Path to HumanEval jsonl",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="experiments/baseline",
        help="Root folder for outputs",
    )
    parser.add_argument(
        "--max_concurrent_tasks",
        type=int,
        default=20,
        help="Max concurrent tasks for async evaluation",
    )
    args = parser.parse_args()

    model_names = _parse_list(args.models)
    _ensure_dir(args.log_root)

    for model_name in model_names:
        await run_baseline_for_model(
            model_name=model_name,
            humaneval_path=args.humaneval_path,
            log_root=args.log_root,
            max_concurrent_tasks=args.max_concurrent_tasks,
        )


if __name__ == "__main__":
    asyncio.run(main())
