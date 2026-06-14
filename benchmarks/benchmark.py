import asyncio
import json
import math
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from scripts.logs import logger
from scripts.utils.common import write_json_file


@dataclass
class RunningScoreStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def sample_variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def sample_std(self) -> float:
        return math.sqrt(self.sample_variance)


class BaseBenchmark(ABC):
    def __init__(self, name: str, file_path: str, log_path: str):
        self.name = name
        self.file_path = file_path
        self.log_path = log_path
        self._log_lock = threading.Lock()

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        t_cost = df["cost"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, a_cost, t_cost

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
        failure_type: str = "wrong_answer",
        error_message: str = "",
    ):
        log_data = {
            "failure_type": failure_type,
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
            "error_message": error_message,
        }
        self._append_log_entry(log_data)

    def log_failure(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        failure_type: str,
        error_message: str,
        extracted_output: Any = "",
        details: Any = None,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "failure_type": failure_type,
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
            "error_message": error_message,
        }
        if details is not None:
            log_data["details"] = details
        self._append_log_entry(log_data)

    def _append_log_entry(self, log_data: dict):
        log_data["logged_at"] = datetime.now().isoformat(timespec="seconds")
        log_file = Path(self.log_path) / "log.json"
        with self._log_lock:
            if log_file.exists():
                with log_file.open("r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                if isinstance(data, dict):
                    data = [data]
                elif not isinstance(data, list):
                    data = []
            else:
                data = []
            data.append(log_data)
            write_json_file(log_file, data, encoding="utf-8", indent=4)

    def log_validation_early_stop(self, details: Dict[str, Any]):
        log_data = {
            "failure_type": "validation_early_stop",
            "question": "__validation_early_stop__",
            "right_answer": "",
            "model_output": "",
            "extracted_output": "",
            "extract_answer_code": "None",
            "error_message": (
                "Empirical Bernstein UCB fell below the incumbent best workflow score; "
                "validation stopped early for this workflow node."
            ),
            "details": details,
        }
        self._append_log_entry(log_data)

    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], agent: Callable, max_concurrent_tasks: int = 50):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_evaluate(problem):
            async with semaphore:
                return await self.evaluate_problem(problem, agent)

        tasks = [sem_evaluate(problem) for problem in data]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

    async def evaluate_all_problems_with_eb_ucb(
        self,
        data: List[dict],
        agent: Callable,
        columns: List[str],
        incumbent_best_score: float,
        epsilon: float,
        max_concurrent_tasks: int = 50,
    ) -> Tuple[List[Tuple[Any, ...]], Optional[Dict[str, Any]]]:
        total_count = len(data)
        min_samples = max(10, math.ceil(0.1 * total_count))
        stats = RunningScoreStats()
        results: List[Tuple[Any, ...]] = []
        score_index = columns.index("score")

        if total_count < min_samples:
            return await self.evaluate_all_problems(data, agent, max_concurrent_tasks), None

        initial_data = data[:min_samples]
        initial_results = await self.evaluate_all_problems(initial_data, agent, max_concurrent_tasks)
        results.extend(initial_results)
        for result in initial_results:
            stats.update(float(result[score_index]))

        early_stop = self._build_eb_ucb_early_stop_details(
            stats=stats,
            epsilon=epsilon,
            incumbent_best_score=incumbent_best_score,
            total_count=total_count,
            min_samples=min_samples,
        )
        if early_stop is not None:
            return results, early_stop

        max_workers = max(1, min(max_concurrent_tasks, total_count))
        next_index = min_samples
        pending: Dict[asyncio.Task, int] = {}

        def schedule_one() -> None:
            nonlocal next_index
            if next_index >= total_count:
                return
            task = asyncio.create_task(self.evaluate_problem(data[next_index], agent))
            pending[task] = next_index
            next_index += 1

        for _ in range(min(max_workers, total_count - next_index)):
            schedule_one()

        while pending:
            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            early_stop = None
            for task in done:
                pending.pop(task, None)
                result = await task
                results.append(result)
                stats.update(float(result[score_index]))

                early_stop = self._build_eb_ucb_early_stop_details(
                    stats=stats,
                    epsilon=epsilon,
                    incumbent_best_score=incumbent_best_score,
                    total_count=total_count,
                    min_samples=min_samples,
                )

            if early_stop is not None:
                for pending_task in pending:
                    pending_task.cancel()
                await asyncio.gather(*pending.keys(), return_exceptions=True)
                return results, early_stop

            while len(pending) < max_workers and next_index < total_count:
                schedule_one()

        return results, None

    async def run_evaluation(
        self,
        agent: Callable,
        va_list: List[int],
        max_concurrent_tasks: int = 50,
        eb_ucb_early_stop: Optional[Dict[str, Any]] = None,
    ):
        data = await self.load_data(va_list)
        columns = self.get_result_columns()
        if self._should_run_eb_ucb_early_stop(eb_ucb_early_stop):
            results, early_stop_details = await self.evaluate_all_problems_with_eb_ucb(
                data=data,
                agent=agent,
                columns=columns,
                incumbent_best_score=float(eb_ucb_early_stop["incumbent_best_score"]),
                epsilon=float(eb_ucb_early_stop["epsilon"]),
                max_concurrent_tasks=max_concurrent_tasks,
            )
        else:
            results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
            early_stop_details = None

        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        if early_stop_details is not None:
            self.log_validation_early_stop(early_stop_details)
            logger.info(
                f"EB-UCB early stop on {self.name}: "
                f"n={early_stop_details['evaluated_samples']}/"
                f"{early_stop_details['validation_samples']}, "
                f"mean={early_stop_details['running_average']:.5f}, "
                f"std={early_stop_details['running_standard']:.5f}, "
                f"ucb={early_stop_details['empirical_bernstein_ucb']:.5f}, "
                f"incumbent={early_stop_details['incumbent_best_score']:.5f}"
            )
        return average_score, average_cost, total_cost

    @staticmethod
    def _should_run_eb_ucb_early_stop(config: Optional[Dict[str, Any]]) -> bool:
        if not config or not config.get("enabled", False):
            return False
        if config.get("incumbent_best_score") is None:
            return False
        epsilon = float(config.get("epsilon", 0.0))
        return 0.0 < epsilon < 1.0

    @staticmethod
    def _empirical_bernstein_ucb(stats: RunningScoreStats, epsilon: float) -> float:
        if stats.count < 2:
            return float("inf")
        log_term = math.log(2.0 / epsilon)
        return (
            stats.mean
            + math.sqrt(2.0 * stats.sample_variance * log_term / stats.count)
            + (7.0 * log_term) / (3.0 * (stats.count - 1))
        )

    def _build_eb_ucb_early_stop_details(
        self,
        stats: RunningScoreStats,
        epsilon: float,
        incumbent_best_score: float,
        total_count: int,
        min_samples: int,
    ) -> Optional[Dict[str, Any]]:
        if stats.count < min_samples:
            return None

        eb_ucb = self._empirical_bernstein_ucb(stats, epsilon)
        should_stop = eb_ucb < incumbent_best_score
        logger.info(
            f"EB-UCB check on {self.name}: "
            f"n={stats.count}/{total_count}, "
            f"threshold={min_samples}, "
            f"mean={stats.mean:.5f}, "
            f"std={stats.sample_std:.5f}, "
            f"ucb={eb_ucb:.5f}, "
            f"incumbent={incumbent_best_score:.5f}, "
            f"epsilon={epsilon}, "
            f"decision={'early_stop' if should_stop else 'continue'}"
        )

        if should_stop:
            return {
                "early_stopped": True,
                "epsilon": epsilon,
                "validation_samples": total_count,
                "min_samples_before_check": min_samples,
                "evaluated_samples": stats.count,
                "running_average": stats.mean,
                "running_standard": stats.sample_std,
                "sample_variance": stats.sample_variance,
                "empirical_bernstein_ucb": eb_ucb,
                "incumbent_best_score": incumbent_best_score,
            }
        return None
    

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 50):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        logger.info(f"Avg Cost:{average_cost:.5f}")
        return average_score, average_cost, total_cost
