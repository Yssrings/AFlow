from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.response = operator.AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.response(input=problem)
        answer = solution.get("answer", "")
        thought = solution.get("thought", "")
        final_answer = answer if "\\boxed" in answer else f"\\boxed{{{answer}}}"
        prediction = f"{thought}\n\nFinal answer: {final_answer}" if thought else final_answer
        return prediction, self.llm.get_usage_summary()["total_cost"]
