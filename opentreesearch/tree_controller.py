"""TreeSearch controller: OpenEvolve subclass using PUCT-based tree search."""

import json
import logging
import os
import signal
import sys
import time
import uuid

import yaml
from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.utils.format_utils import format_metrics_safe

from opentreesearch.tree_database import TreeNode, TreeProgramDatabase
from opentreesearch.tree_process_parallel import TreeParallelController

logger = logging.getLogger(__name__)

DEFAULT_PUCT_C = 1.0


def load_config(config_path: str | None = None):
    """Load config, extracting puct_exploration_constant before forwarding to openevolve."""
    puct_c = DEFAULT_PUCT_C
    if config_path is not None:
        with open(config_path) as f:
            raw = yaml.safe_load(os.path.expandvars(f.read())) or {}
        puct_c = float(raw.get("database", {}).pop("puct_exploration_constant", DEFAULT_PUCT_C))
        return Config.from_dict(raw), puct_c
    return Config(), puct_c


class TreeSearch(OpenEvolve):
    """OpenEvolve with PUCT tree search replacing island-based MAP-Elites.

    Swaps in TreeProgramDatabase, TreeParallelController, and TreeNode.
    LLM ensemble, prompt sampling, evaluation, and logging are inherited.
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config_path: str | None = None,
        config: Config | None = None,
        output_dir: str | None = None,
    ):
        if config is None:
            config, self.puct_c = load_config(config_path)
        else:
            self.puct_c = DEFAULT_PUCT_C
        super().__init__(
            initial_program_path, evaluation_file, config=config, output_dir=output_dir
        )

        self.database = TreeProgramDatabase(
            self.config.database,
            puct_exploration_constant=self.puct_c,
        )
        self.evaluator.database = self.database

        # Expose evaluator timeout to evaluation subprocesses so user-defined
        # evaluators can size their inner subprocess timeouts to match.
        os.environ["OPENEVOLVE_EVAL_TIMEOUT"] = str(int(self.config.evaluator.timeout))

    async def run(
        self,
        iterations: int | None = None,
        target_score: float | None = None,
        checkpoint_path: str | None = None,
    ) -> TreeNode | None:
        """Run evolution with TreeNode initial program and TreeParallelController."""
        max_iterations = iterations or self.config.max_iterations

        start_iteration = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            logger.info(f"Resuming from checkpoint at iteration {start_iteration}")
        else:
            start_iteration = self.database.last_iteration

        should_add_initial = start_iteration == 0 and not self.database.tree

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id
            )

            initial_node = TreeNode(
                id=initial_program_id,
                code=self.initial_program_code,
                iteration_found=start_iteration,
                parent_id=None,
                metrics=initial_metrics,
                language=self.config.language,
                depth=0,
                visits=1,
            )

            self.database.add(initial_node, root=True)

            if "combined_score" not in initial_metrics:
                numeric_metrics = [
                    v
                    for v in initial_metrics.values()
                    if isinstance(v, int | float) and not isinstance(v, bool)
                ]
                if numeric_metrics:
                    avg_score = sum(numeric_metrics) / len(numeric_metrics)
                    logger.warning(
                        "No 'combined_score' metric found. "
                        "Using average of numeric metrics (%.4f).",
                        avg_score,
                    )
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} "
                f"with {len(self.database.tree)} existing nodes)"
            )

        try:
            self.parallel_controller = TreeParallelController(
                self.config,
                self.evaluation_file,
                self.database,
                self.evolution_tracer,
                file_suffix=self.config.file_suffix,
            )

            def force_exit_handler(_signum, _frame):
                logger.info("Force exit requested - terminating immediately")
                sys.exit(0)

            def signal_handler(signum, _frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
                self.parallel_controller.request_shutdown()
                signal.signal(signal.SIGINT, force_exit_handler)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            self.parallel_controller.start()

            evolution_start = 1 if should_add_initial else start_iteration
            await self._run_evolution_with_checkpoints(
                evolution_start, max_iterations, target_score
            )

        finally:
            if self.parallel_controller:
                self.parallel_controller.stop()
                self.parallel_controller = None

            if self.evolution_tracer:
                self.evolution_tracer.close()
                logger.info("Evolution tracer closed")

        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        if best_program and "combined_score" in best_program.metrics:
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if (
                best_by_combined
                and best_by_combined.id != best_program.id
                and "combined_score" in best_by_combined.metrics
            ):
                if (
                    best_by_combined.metrics["combined_score"]
                    > best_program.metrics["combined_score"] + 0.02
                ):
                    logger.warning(
                        f"Found program with better combined_score: {best_by_combined.id}"
                    )
                    logger.warning(
                        f"Score difference: {best_program.metrics['combined_score']:.4f} vs "
                        f"{best_by_combined.metrics['combined_score']:.4f}"
                    )
                    best_program = best_by_combined

        if best_program:
            if (
                hasattr(self, "parallel_controller")
                and self.parallel_controller
                and self.parallel_controller.early_stopping_triggered
            ):
                logger.info(
                    f"Evolution complete via early stopping. Best program has metrics: "
                    f"{format_metrics_safe(best_program.metrics)}"
                )
            else:
                logger.info(
                    f"Evolution complete. Best program has metrics: "
                    f"{format_metrics_safe(best_program.metrics)}"
                )
            self._save_best_program(best_program)
            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            return None

    async def _run_evolution_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: float | None
    ) -> None:
        await self.parallel_controller.run_evolution(
            start_iteration, max_iterations, target_score, checkpoint_callback=self._save_checkpoint
        )

        if self.parallel_controller.shutdown_event.is_set():
            logger.info("Evolution stopped due to shutdown request")
            return
        elif self.parallel_controller.early_stopping_triggered:
            logger.info("Evolution stopped due to early stopping - saving final checkpoint")

        final_iteration = start_iteration + max_iterations - 1
        if final_iteration > 0 and final_iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint(final_iteration)

    def _save_checkpoint(self, iteration: int) -> None:
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        self.database.save(checkpoint_path, iteration)

        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        else:
            best_program = self.database.get_best_program()

        if best_program:
            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)

            best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(best_program_info_path, "w") as f:
                json.dump(
                    {
                        "id": best_program.id,
                        "depth": best_program.depth,
                        "iteration": best_program.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best_program.metrics,
                        "language": best_program.language,
                        "timestamp": best_program.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved best program at checkpoint {iteration} with metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _save_best_program(self, program: TreeNode | None = None) -> None:
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            json.dump(
                {
                    "id": program.id,
                    "depth": program.depth,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")
