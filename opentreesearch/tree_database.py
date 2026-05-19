"""PUCT-based tree database."""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from openevolve.config import DatabaseConfig
from openevolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A node in the tree search, representing one evolved program."""

    id: str
    code: str
    iteration_found: int
    parent_id: str | None
    depth: int
    visits: int = 1

    language: str = "python"
    timestamp: float = field(default_factory=time.time)

    children_ids: list[str] = field(default_factory=list)

    # Performance metrics
    metrics: dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Prompts
    prompts: dict[str, Any] | None = None

    @property
    def generation(self) -> int:
        """Alias for depth, for compatibility with openevolve's EvolutionTracer."""
        return self.depth

    @generation.setter
    def generation(self, value: int) -> None:
        self.depth = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        """Create from dictionary representation."""
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading TreeNode: {filtered_out}")

        return cls(**filtered_data)


class TreeProgramDatabase:
    """
    Database for storing and sampling programs during evolution.

    Implements PUCT-based tree search as a replacement for
    openevolve's island-based MAP-Elites ProgramDatabase.
    """

    def __init__(self, config: DatabaseConfig, puct_exploration_constant: float = 1.0):
        self.config = config
        self.puct_exploration_constant = puct_exploration_constant

        # In-memory program storage
        self.tree: dict[str, TreeNode] = {}

        # Virtual loss: in-flight expansion counts per node
        self._in_flight: dict[str, int] = {}

        # Track the absolute best program separately
        self.best_program_id: str | None = None

        # Track the last iteration number (for resuming)
        self.last_iteration: int = 0

        # Load database from disk if path is provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

        # Prompt log
        self.prompts_by_program: dict[str, dict[str, dict[str, str]]] | None = None

        # Set random seed for reproducible sampling if specified
        if config.random_seed is not None:
            import random

            random.seed(config.random_seed)
            logger.debug(f"Database: Set random seed to {config.random_seed}")

        logger.info(f"Initialized tree search database with {len(self.tree)} nodes")

    @property
    def programs(self) -> dict[str, TreeNode]:
        """Alias for self.tree, for compatibility with OpenEvolve's controller."""
        return self.tree

    def log_island_status(self) -> None:
        """No-op. Tree search does not use islands."""

    def add(
        self,
        node: TreeNode,
        iteration: int | None = None,
        root: bool = False,
    ) -> str:
        """Add a node to the tree."""
        if iteration is not None:
            node.iteration_found = iteration
            self.last_iteration = max(self.last_iteration, iteration)

        self.tree[node.id] = node

        if not root:
            # If not root, then added node came from UCB expansion,
            # so update parent/child relationships and visit counts
            self.tree[node.parent_id].children_ids.append(node.id)

            # Backpropagate - update visit count of all ancestors
            ancestor_id = node.parent_id
            while ancestor_id in self.tree:
                self.tree[ancestor_id].visits += 1
                ancestor_id = self.tree[ancestor_id].parent_id

        self._update_best_program(node)

        if self.config.db_path:
            self._save_program(node)

        logger.debug(f"Added node {node.id} to tree")
        return node.id

    def ucb_expand(self, num_inspirations: int | None = None) -> tuple[TreeNode, list[TreeNode]]:
        """Choose a parent using PUCT and return top inspirations.

        PUCT(i) = RankScore(i) + c * P(i) * sqrt(total_visits) / (1 + visits_i)
        """
        nodes = list(self.tree.values())
        N = len(nodes)

        total_visits = sum(n.visits for n in nodes)
        scores = {n.id: get_fitness_score(n.metrics) for n in nodes}

        # Ascending-order ranks: lowest score -> rank 1, highest -> rank N
        # Stable tie-break on id to keep determinism across runs
        sorted_ids_asc = [n.id for n in sorted(nodes, key=lambda n: (scores[n.id], n.id))]
        ranks = {nid: i + 1 for i, nid in enumerate(sorted_ids_asc)}

        # Normalize to [0, 1] per spec
        if N > 1:
            denom = N - 1
            rank_scores = {nid: (ranks[nid] - 1) / denom for nid in ranks}
        else:
            rank_scores = {sorted_ids_asc[0]: 1.0}

        # Flat prior
        prior_P = 1.0 / N

        # PUCT score
        ucb_scores = {}
        c = self.puct_exploration_constant
        for n in nodes:
            v = max(0, n.visits) + self._in_flight.get(n.id, 0)
            ucb = rank_scores[n.id] + c * prior_P * (total_visits**0.5) / (1.0 + v)
            ucb_scores[n.id] = ucb

        # Select parent by max PUCT
        best_id = max(ucb_scores, key=ucb_scores.get)
        parent = self.tree[best_id]

        # Inspirations: top-K by task score (descending), excluding the selected parent
        k = num_inspirations if num_inspirations is not None else min(5, max(0, N - 1))
        inspirations = [
            n
            for n in sorted(nodes, key=lambda n: (scores[n.id], n.id), reverse=True)
            if n.id != best_id
        ][:k]

        return parent, inspirations

    def apply_virtual_loss(self, node_id: str) -> None:
        """Increment in-flight count so PUCT discourages re-selection."""
        self._in_flight[node_id] = self._in_flight.get(node_id, 0) + 1

    def undo_virtual_loss(self, node_id: str) -> None:
        """Decrement in-flight count when a worker finishes or fails."""
        count = self._in_flight.get(node_id, 0) - 1
        if count <= 0:
            self._in_flight.pop(node_id, None)
        else:
            self._in_flight[node_id] = count

    def get(self, node_id: str) -> TreeNode | None:
        """Return a node by ID, or None."""
        return self.tree.get(node_id)

    def get_best_program(self, metric: str | None = None) -> TreeNode | None:
        """Get the best program based on a metric."""
        if not self.tree:
            return None

        if metric is None and self.best_program_id:
            if self.best_program_id in self.tree:
                logger.debug(f"Using tracked best program: {self.best_program_id}")
                return self.tree[self.best_program_id]
            else:
                logger.warning(
                    "Tracked best program %s no longer exists, will recalculate",
                    self.best_program_id,
                )
                self.best_program_id = None

        if metric:
            sorted_programs = sorted(
                [p for p in self.tree.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by metric '{metric}': {sorted_programs[0].id}")
        else:
            sorted_programs = sorted(
                self.tree.values(),
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by fitness score: {sorted_programs[0].id}")

        if sorted_programs and (
            self.best_program_id is None or sorted_programs[0].id != self.best_program_id
        ):
            old_id = self.best_program_id
            self.best_program_id = sorted_programs[0].id
            logger.info(f"Updated best program tracking from {old_id} to {self.best_program_id}")

            if (
                old_id
                and old_id in self.tree
                and "combined_score" in self.tree[old_id].metrics
                and "combined_score" in self.tree[self.best_program_id].metrics
            ):
                old_score = self.tree[old_id].metrics["combined_score"]
                new_score = self.tree[self.best_program_id].metrics["combined_score"]
                diff = new_score - old_score
                logger.info(
                    "Score change: %.4f -> %.4f (%+.4f)",
                    old_score,
                    new_score,
                    diff,
                )

        return sorted_programs[0] if sorted_programs else None

    def save(self, path: str | None = None, iteration: int = 0) -> None:
        """Save the database to disk."""
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        os.makedirs(save_path, exist_ok=True)

        for node in self.tree.values():
            prompts = None
            if (
                self.config.log_prompts
                and self.prompts_by_program
                and node.id in self.prompts_by_program
            ):
                prompts = self.prompts_by_program[node.id]
            self._save_program(node, save_path, prompts=prompts)

        metadata = {
            "best_program_id": self.best_program_id,
            "last_iteration": iteration or self.last_iteration,
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saved tree search database with {len(self.tree)} nodes to {save_path}")

    def load(self, path: str) -> None:
        """Load the database from disk."""
        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return

        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.best_program_id = metadata.get("best_program_id")
            self.last_iteration = metadata.get("last_iteration", 0)

            logger.info(f"Loaded database metadata with last_iteration={self.last_iteration}")

        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path) as f:
                            program_data = json.load(f)

                        node = TreeNode.from_dict(program_data)
                        self.tree[node.id] = node
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")

        logger.info(f"Loaded tree search database with {len(self.tree)} nodes from {path}")

    def _update_best_program(self, program: TreeNode) -> None:
        if self.best_program_id is None:
            self.best_program_id = program.id
            logger.debug(f"Set initial best program to {program.id}")
            return

        if self.best_program_id not in self.tree:
            logger.warning(
                f"Best program {self.best_program_id} no longer exists, clearing reference"
            )
            self.best_program_id = program.id
            logger.info(f"Set new best program to {program.id}")
            return

        current_best = self.tree[self.best_program_id]

        if self._is_better(program, current_best):
            old_id = self.best_program_id
            self.best_program_id = program.id

            if "combined_score" in program.metrics and "combined_score" in current_best.metrics:
                old_score = current_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.info(
                    "New best %s replaces %s (score: %.4f -> %.4f, %+.4f)",
                    program.id,
                    old_id,
                    old_score,
                    new_score,
                    score_diff,
                )
            else:
                logger.info(f"New best program {program.id} replaces {old_id}")

    def _save_program(
        self,
        program: TreeNode,
        base_path: str | None = None,
        prompts: dict[str, dict[str, str]] | None = None,
    ) -> None:
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        program_dict = program.to_dict()
        if prompts:
            program_dict["prompts"] = prompts
        program_path = os.path.join(programs_dir, f"{program.id}.json")

        with open(program_path, "w") as f:
            json.dump(program_dict, f)

    def _is_better(self, program1: TreeNode, program2: TreeNode) -> bool:
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp

        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False

        fitness1 = get_fitness_score(program1.metrics)
        fitness2 = get_fitness_score(program2.metrics)

        return fitness1 > fitness2

    def log_prompt(
        self,
        program_id: str,
        template_key: str,
        prompt: dict[str, str],
        responses: list[str] | None = None,
    ) -> None:
        """Log a prompt and optional responses for a program."""
        if not self.config.log_prompts:
            return

        if responses is None:
            responses = []
        prompt["responses"] = responses

        if self.prompts_by_program is None:
            self.prompts_by_program = {}

        if program_id not in self.prompts_by_program:
            self.prompts_by_program[program_id] = {}
        self.prompts_by_program[program_id][template_key] = prompt
