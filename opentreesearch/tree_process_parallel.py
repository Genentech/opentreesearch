"""Parallel worker processes for tree search evolution."""

import asyncio
import logging
import re
import time
import uuid
from concurrent.futures import Future
from typing import Any

from openevolve.process_parallel import (
    ProcessParallelController,
    SerializableResult,
)
from openevolve.utils.metrics_utils import safe_numeric_average

from opentreesearch.tree_database import TreeNode

logger = logging.getLogger(__name__)

_EVOLVE_BLOCK_RE = re.compile(
    r"#\s*EVOLVE-BLOCK-START\s*\n(?P<body>.*?)\n\s*#\s*EVOLVE-BLOCK-END",
    re.DOTALL,
)


def _extract_evolve_block(code: str) -> str:
    """Return only the EVOLVE-BLOCK contents, or the full code if no block markers exist."""
    matches = _EVOLVE_BLOCK_RE.findall(code)
    if not matches:
        return code
    return "\n\n".join(m.strip("\n") for m in matches)


def _trim_inspirations(inspirations: list[TreeNode]) -> list[dict[str, Any]]:
    """Return inspiration dicts with the `code` field reduced to its EVOLVE-BLOCK only.

    Saves prompt tokens when programs include large fixed boilerplate outside the evolve block.
    """
    trimmed = []
    for node in inspirations:
        d = node.to_dict()
        d["code"] = _extract_evolve_block(d.get("code", ""))
        trimmed.append(d)
    return trimmed


def _worker_init(config_dict: dict, evaluation_file: str, parent_env: dict | None = None) -> None:
    """Reconstruct Config from dict and store as module globals."""
    import os

    if parent_env:
        os.environ.update(parent_env)

    global _worker_config, _worker_evaluation_file
    global _worker_evaluator, _worker_llm_ensemble, _worker_prompt_sampler

    config_dict = config_dict.copy()
    db = config_dict.get("database", {})
    if isinstance(db, dict):
        db = {k: v for k, v in db.items() if k not in ("puct_exploration_constant", "novelty_llm")}
        config_dict["database"] = db

    from openevolve.config import Config as _Config

    _worker_config = _Config.from_dict(config_dict)
    _worker_evaluation_file = evaluation_file
    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None


def _lazy_init_worker_components():
    """Create LLM ensemble, prompt sampler, and evaluator on first use."""
    global _worker_evaluator, _worker_llm_ensemble, _worker_prompt_sampler

    from openevolve.evaluator import Evaluator
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.prompt.sampler import PromptSampler

    if _worker_llm_ensemble is None:
        _worker_llm_ensemble = LLMEnsemble(_worker_config.llm.models)

    if _worker_prompt_sampler is None:
        _worker_prompt_sampler = PromptSampler(_worker_config.prompt)

    if _worker_evaluator is None:
        evaluator_prompt = PromptSampler(_worker_config.prompt)
        evaluator_prompt.set_templates("evaluator_system_message")
        _worker_evaluator = Evaluator(
            _worker_config.evaluator,
            _worker_evaluation_file,
            LLMEnsemble(_worker_config.llm.evaluator_models),
            evaluator_prompt,
            database=None,
            suffix=getattr(_worker_config, "file_suffix", ".py"),
        )


def _run_tree_search_iteration_worker(
    iteration: int, db_snapshot: dict[str, Any], parent_id: str, inspiration_ids: list[str]
) -> SerializableResult:
    """Run one LLM-generate-then-evaluate cycle, producing a child TreeNode."""
    try:
        _lazy_init_worker_components()

        tree = {pid: TreeNode(**prog_dict) for pid, prog_dict in db_snapshot["tree"].items()}
        parent = tree[parent_id]
        inspirations = [tree[pid] for pid in inspiration_ids if pid in tree]

        prompt = _worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            inspirations=_trim_inspirations(inspirations),
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
        )

        iteration_start = time.time()

        try:
            llm_response = asyncio.run(
                _worker_llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return SerializableResult(error=f"LLM generation failed: {str(e)}", iteration=iteration)

        if llm_response is None:
            return SerializableResult(error="LLM returned None response", iteration=iteration)

        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import apply_diff, extract_diffs, format_diff_summary

            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                return SerializableResult(
                    error="No valid diffs found in response", iteration=iteration
                )
            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            from openevolve.utils.code_utils import parse_full_rewrite

            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(
                    error="No valid code found in response", iteration=iteration
                )
            child_code = new_code
            changes_summary = "Full rewrite"

        if len(child_code) > _worker_config.max_code_length:
            return SerializableResult(
                error=f"Code exceeds max length "
                f"({len(child_code)} > {_worker_config.max_code_length})",
                iteration=iteration,
            )

        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))

        child_node = TreeNode(
            id=child_id,
            code=child_code,
            language=_worker_config.language,
            iteration_found=iteration,
            parent_id=parent.id,
            depth=parent.depth + 1,
            visits=1,
            metrics=child_metrics,
            metadata={"changes": changes_summary, "parent_metrics": parent.metrics},
        )

        return SerializableResult(
            child_program_dict=child_node.to_dict(),
            parent_id=parent.id,
            iteration_time=time.time() - iteration_start,
            prompt=prompt,
            llm_response=llm_response,
            iteration=iteration,
        )

    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult(error=str(e), iteration=iteration)


class TreeParallelController(ProcessParallelController):
    """Parallel evolution using PUCT tree search.

    Inherits config serialization and stop/shutdown from ProcessParallelController.
    Overrides start (to bind this module's worker), snapshot, submission,
    and the evolution loop (no islands, no migration).
    """

    def start(self) -> None:
        """Start process pool, binding this module's _worker_init."""
        import os
        from concurrent.futures import ProcessPoolExecutor

        config_dict = self._serialize_config(self.config)
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file, dict(os.environ)),
        )
        logger.info(f"Started process pool with {self.num_workers} processes")

    def _create_database_snapshot(self) -> dict[str, Any]:
        return {"tree": {pid: node.to_dict() for pid, node in self.database.tree.items()}}

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: float | None = None,
        checkpoint_callback=None,
    ):
        """Run tree search evolution loop."""
        if not self.executor:
            raise RuntimeError("Process pool not started")

        total_iterations = start_iteration + max_iterations
        logger.info(
            f"Starting tree search evolution: iterations {start_iteration}..{total_iterations - 1}"
        )

        # Pre-fill worker pool
        pending: dict[int, Future] = {}
        pending_parents: dict[int, str] = {}
        current_iteration = start_iteration
        for _ in range(self.num_workers):
            if current_iteration < total_iterations:
                submitted = self._submit_iteration(current_iteration)
                if submitted:
                    future, parent_id = submitted
                    pending[current_iteration] = future
                    pending_parents[current_iteration] = parent_id
                current_iteration += 1

        next_iteration = current_iteration
        completed = 0

        # Early stopping state
        early_stopping_enabled = self.config.early_stopping_patience is not None
        best_score = float("-inf")
        iterations_without_improvement = 0
        if early_stopping_enabled:
            logger.info(
                f"Early stopping: patience={self.config.early_stopping_patience}, "
                f"threshold={self.config.convergence_threshold}, "
                f"metric={self.config.early_stopping_metric}"
            )

        warned_about_combined_score = False

        while pending and completed < max_iterations and not self.shutdown_event.is_set():
            done_iter = next((it for it, f in pending.items() if f.done()), None)
            if done_iter is None:
                await asyncio.sleep(0.01)
                continue

            future = pending.pop(done_iter)
            parent_id = pending_parents.pop(done_iter, None)
            completed += 1

            if parent_id and self.num_workers > 1:
                self.database.undo_virtual_loss(parent_id)

            should_stop = False
            try:
                result = future.result(timeout=self.config.evaluator.timeout + 30)
            except Exception as e:
                logger.error(f"Iteration {done_iter} failed: {e}")
                result = None

            if result is not None and result.error:
                logger.warning(f"Iteration {done_iter} error: {result.error}")

            elif result is not None and result.child_program_dict:
                child = TreeNode(**result.child_program_dict)
                self.database.add(child, iteration=done_iter)

                # Log evolution trace
                if self.evolution_tracer and result.parent_id:
                    parent = self.database.get(result.parent_id)
                    if parent:
                        self.evolution_tracer.log_trace(
                            iteration=done_iter,
                            parent_program=parent,
                            child_program=child,
                            prompt=result.prompt,
                            llm_response=result.llm_response,
                            metadata={
                                "iteration_time": result.iteration_time,
                                "changes": child.metadata.get("changes", ""),
                            },
                        )

                # Log prompts
                if result.prompt:
                    template = (
                        "full_rewrite_user" if not self.config.diff_based_evolution else "diff_user"
                    )
                    self.database.log_prompt(
                        template_key=template,
                        program_id=child.id,
                        prompt=result.prompt,
                        responses=[result.llm_response] if result.llm_response else [],
                    )

                # Log metrics
                if child.metrics:
                    metrics_str = ", ".join(
                        f"{k}={v:.4f}" if isinstance(v, int | float) else f"{k}={v}"
                        for k, v in child.metrics.items()
                    )
                    logger.info(
                        f"Iteration {done_iter}: {child.id} "
                        f"(parent: {result.parent_id}) "
                        f"in {result.iteration_time:.2f}s | {metrics_str}"
                    )

                    if "combined_score" not in child.metrics and not warned_about_combined_score:
                        avg = safe_numeric_average(child.metrics)
                        logger.warning("No 'combined_score' metric. Using average (%.4f).", avg)
                        warned_about_combined_score = True

                if self.database.best_program_id == child.id:
                    logger.info(f"New best at iteration {done_iter}: {child.id}")

                # Checkpoint
                if done_iter > 0 and done_iter % self.config.checkpoint_interval == 0:
                    if checkpoint_callback:
                        checkpoint_callback(done_iter)

                # Target score
                if (
                    target_score is not None
                    and "combined_score" in child.metrics
                    and child.metrics["combined_score"] >= target_score
                ):
                    logger.info("Target score %s reached at iteration %d", target_score, done_iter)
                    should_stop = True

                # Early stopping
                if not should_stop and early_stopping_enabled and child.metrics:
                    metric = self.config.early_stopping_metric
                    score = child.metrics.get(metric)
                    if score is None:
                        score = safe_numeric_average(child.metrics)

                    if isinstance(score, int | float):
                        if score - best_score >= self.config.convergence_threshold:
                            best_score = score
                            iterations_without_improvement = 0
                        else:
                            iterations_without_improvement += 1

                        if iterations_without_improvement >= self.config.early_stopping_patience:
                            self.early_stopping_triggered = True
                            logger.info(
                                "Early stopping at iteration %d: "
                                "no improvement for %d iterations (best: %.4f)",
                                done_iter,
                                iterations_without_improvement,
                                best_score,
                            )
                            should_stop = True

            if should_stop:
                break

            # Submit next iteration
            if next_iteration < total_iterations and not self.shutdown_event.is_set():
                submitted = self._submit_iteration(next_iteration)
                if submitted:
                    future, parent_id = submitted
                    pending[next_iteration] = future
                    pending_parents[next_iteration] = parent_id
                    next_iteration += 1

        # Cleanup
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for f in pending.values():
                f.cancel()
        if self.num_workers > 1:
            for parent_id in pending_parents.values():
                self.database.undo_virtual_loss(parent_id)

        if self.early_stopping_triggered:
            logger.info("Evolution completed - early stopping")
        elif self.shutdown_event.is_set():
            logger.info("Evolution completed - shutdown requested")
        else:
            logger.info("Evolution completed - maximum iterations reached")

        return self.database.get_best_program()

    def _submit_iteration(self, iteration: int, island_id=None) -> tuple[Future, str] | None:
        """Select parent via PUCT and submit worker task.

        When num_workers > 1, applies virtual loss so consecutive calls select different parents.

        Returns (future, parent_id) so the caller can track and undo virtual loss.
        """
        try:
            parent, inspirations = self.database.ucb_expand(
                num_inspirations=self.config.prompt.num_top_programs
            )
            snapshot = self._create_database_snapshot()
            future = self.executor.submit(
                _run_tree_search_iteration_worker,
                iteration,
                snapshot,
                parent.id,
                [insp.id for insp in inspirations],
            )
            if self.num_workers > 1:
                self.database.apply_virtual_loss(parent.id)
            return future, parent.id
        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
