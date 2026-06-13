import ast
import importlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Set

from scripts.prompts.optimize_prompt import (
    WORKFLOW_CUSTOM_USE,
    WORKFLOW_INPUT,
    WORKFLOW_OPTIMIZE_PROMPT,
    WORKFLOW_TEMPLATE,
)
from scripts.logs import logger


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def create_candidate_round_directory(self, graph_path: str, round_number: int, attempt: int) -> str:
        timestamp = time.time_ns()
        directory = os.path.join(graph_path, f"round_{round_number}_candidate_{attempt}_{timestamp}")
        os.makedirs(directory, exist_ok=False)
        return directory

    def load_graph(self, round_number: int, workflows_path: str, round_dir_name: Optional[str] = None):
        workflows_package = self._path_to_module(workflows_path)
        round_module = round_dir_name or f"round_{round_number}"
        graph_module_name = f"{workflows_package}.{round_module}.graph"

        try:
            self._clear_module_cache(f"{workflows_package}.{round_module}")
            importlib.invalidate_caches()
            graph_module = importlib.import_module(graph_module_name)
            graph_class = getattr(graph_module, "Workflow")
            return graph_class
        except Exception as e:
            logger.error(f"Error loading graph for {round_module}: {e}")
            raise

    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.error(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str]) -> str:
        path = f"{self.root_path}/workflows/template/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(id + 1, operator, path)
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            return f"{id}. {operator_name}: {desc}, with interface {interface})."

    def create_graph_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        operator_description: str,
        type: str,
        log_data: str,
    ) -> str:
        graph_input = WORKFLOW_INPUT.format(
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            operator_description=operator_description,
            type=type,
            log=log_data,
        )
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        return graph_input + WORKFLOW_CUSTOM_USE + graph_system

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                logger.error(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None

    def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str):
        directory_path = Path(directory)
        workflows_package = self._path_to_module(directory_path.parent)
        round_suffix = self._round_suffix_from_directory(directory_path.name, round_number)
        graph = WORKFLOW_TEMPLATE.format(
            graph=response["graph"],
            round=round_suffix,
            workflow_package=workflows_package,
        )

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(response["prompt"])

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")

    def validate_round_files(self, directory: str, llm_config=None, dataset=None):
        directory_path = Path(directory)
        prompt_path = directory_path / "prompt.py"
        graph_path = directory_path / "graph.py"

        prompt_content = prompt_path.read_text(encoding="utf-8")
        graph_content = graph_path.read_text(encoding="utf-8")

        try:
            prompt_tree = ast.parse(prompt_content, filename=str(prompt_path))
            graph_tree = ast.parse(graph_content, filename=str(graph_path))
        except SyntaxError as e:
            raise ValueError(f"Generated round has syntax error: {e}") from e

        defined_prompts = self._defined_prompt_names(prompt_tree)
        used_prompts = self._used_prompt_custom_names(graph_tree)
        missing_prompts = sorted(used_prompts - defined_prompts)
        if missing_prompts:
            raise ValueError(f"Generated prompt.py is missing prompt_custom fields: {missing_prompts}")

        workflows_path = str(directory_path.parent)
        graph_class = self.load_graph(
            round_number=0,
            workflows_path=workflows_path,
            round_dir_name=directory_path.name,
        )
        if llm_config is not None and dataset is not None:
            graph_class(name="validation", llm_config=llm_config, dataset=dataset)

    def promote_candidate_round(self, candidate_directory: str, graph_path: str, round_number: int, response: dict, dataset: str) -> str:
        final_directory = Path(graph_path) / f"round_{round_number}"
        if final_directory.exists():
            backup_directory = self._next_backup_path(final_directory)
            os.replace(final_directory, backup_directory)
            logger.warning(f"Existing candidate round moved aside: {final_directory} -> {backup_directory}")

        os.makedirs(final_directory, exist_ok=False)
        self.write_graph_files(str(final_directory), response, round_number, dataset)
        self._clear_round_cache(graph_path, round_number)
        return str(final_directory)

    @staticmethod
    def _defined_prompt_names(prompt_tree: ast.AST) -> Set[str]:
        defined = set()
        for node in ast.walk(prompt_tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
        return defined

    @staticmethod
    def _used_prompt_custom_names(graph_tree: ast.AST) -> Set[str]:
        used = set()
        for node in ast.walk(graph_tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "prompt_custom"
            ):
                used.add(node.attr)
        return used

    @staticmethod
    def _round_suffix_from_directory(directory_name: str, round_number: int) -> str:
        prefix = "round_"
        if directory_name.startswith(prefix):
            return directory_name[len(prefix):]
        return str(round_number)

    @staticmethod
    def _next_backup_path(path: Path) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base = path.with_name(f"{path.name}_replaced_{timestamp}")
        candidate = base
        index = 1
        while candidate.exists():
            candidate = path.with_name(f"{base.name}_{index}")
            index += 1
        return candidate

    def _clear_round_cache(self, graph_path: str, round_number: int):
        workflows_package = self._path_to_module(graph_path)
        self._clear_module_cache(f"{workflows_package}.round_{round_number}")

    @staticmethod
    def _clear_module_cache(module_prefix: str):
        for module_name in list(sys.modules):
            if module_name == module_prefix or module_name.startswith(f"{module_prefix}."):
                del sys.modules[module_name]

    @staticmethod
    def _path_to_module(path: str) -> str:
        path = Path(path)
        try:
            path = path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            pass
        return str(path).replace("\\", ".").replace("/", ".")
