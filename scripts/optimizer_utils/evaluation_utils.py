from scripts.evaluator import Evaluator
from scripts.logs import logger


class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_initial_round(self, optimizer, graph_path, directory, validation_n, data):
        # Load graph with graph_utils from optimizer
        optimizer.graph = optimizer.graph_utils.load_graph(optimizer.round, graph_path)
        evaluator = Evaluator(eval_path=directory)

        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
                max_concurrent_tasks=optimizer.max_concurrent_tasks,
                eb_ucb_early_stop=None,
            )

            new_data = optimizer.data_utils.create_result_data(optimizer.round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)

        return data

    async def evaluate_graph(self, optimizer, directory, validation_n, data, initial=False):
        evaluator = Evaluator(eval_path=directory)
        sum_score = 0
        cur_round = optimizer.round if initial is True else optimizer.round + 1
        incumbent_best_score = None
        if not initial and optimizer.enable_eb_ucb_early_stop:
            incumbent_best_score = optimizer.data_utils.get_best_average_score(data, exclude_round=cur_round)
            if incumbent_best_score is not None:
                logger.info(
                    f"Clopper-Pearson early-stop incumbent best score before round {cur_round}: "
                    f"{incumbent_best_score:.5f}"
                )

        for i in range(validation_n):
            eb_ucb_early_stop = None
            if not initial and optimizer.enable_eb_ucb_early_stop:
                eb_ucb_early_stop = {
                    "enabled": True,
                    "epsilon": optimizer.eb_ucb_epsilon,
                    "incumbent_best_score": incumbent_best_score,
                }

            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
                max_concurrent_tasks=optimizer.max_concurrent_tasks,
                eb_ucb_early_stop=eb_ucb_early_stop,
            )

            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/workflows")
            optimizer.data_utils.save_results(result_path, data)

            sum_score += score

        return sum_score / validation_n

    async def evaluate_graph_test(self, optimizer, directory, is_test=True):
        evaluator = Evaluator(eval_path=directory)
        return await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            is_test=is_test,
            max_concurrent_tasks=optimizer.max_concurrent_tasks,
        )
