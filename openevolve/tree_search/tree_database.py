"""
Program database for TreeSearch
"""

import base64
import json
import logging
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields

# FileLock removed - no longer needed with threaded parallel processing
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from openevolve.config import DatabaseConfig
from openevolve.utils.code_utils import calculate_edit_distance
from openevolve.utils.metrics_utils import safe_numeric_average, get_fitness_score
from openevolve.process_parallel import SerializableResult

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
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
    metrics: Dict[str, float] = field(default_factory=dict)    

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Prompts
    prompts: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Log if we're filtering out any fields
        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading Program: {filtered_out}")

        return cls(**filtered_data)


class TreeProgramDatabase:
    """
    Database for storing and sampling programs during evolution

    The database implements tree search.
    It is built off of openevolve.database.ProgramDatabase.
    """
    def __init__(self, config: DatabaseConfig):
        self.config = config

        # In-memory program storage
        self.tree: Dict[str, TreeNode] = {}
        
        # Track the absolute best program separately
        self.best_program_id: Optional[str] = None

        # Track the last iteration number (for resuming)
        self.last_iteration: int = 0

        # Load database from disk if path is provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

        # Prompt log
        self.prompts_by_program: Dict[str, Dict[str, Dict[str, str]]] = None

        # Set random seed for reproducible sampling if specified
        if config.random_seed is not None:
            import random

            random.seed(config.random_seed)
            logger.debug(f"Database: Set random seed to {config.random_seed}")

        logger.info(f"Initialized tree search database with {len(self.tree)} nodes")
    
    def add(
        self, node: TreeNode, iteration: int = None, root: bool = False,
    ) -> str:
        """
        Add a node to the tree

        Args:
            node: TreeNode to add
            iteration: Current iteration (defaults to last_iteration)

        Returns:
            Tree Node ID
        """
        # Store the program
        # If iteration is provided, update the program's iteration_found
        if iteration is not None:
            node.iteration_found = iteration
            # Update last_iteration if needed
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

        # Update the absolute best program tracking
        self._update_best_program(node)

        # Save to disk if configured
        if self.config.db_path:
            self._save_program(node)

        logger.debug(f"Added node {node.id} to tree")
        return node.id

    def ucb_expand(self, num_inspirations: Optional[int] = None) -> Tuple[TreeNode, List[TreeNode]]:
        """
        Deterministically choose a parent program using UCB to expand, and return top inspirations.

        - RankScore_A(i) uses ascending-order ranks (lowest score = 1, highest = N).
        It is normalized to [0, 1] via (rank-1)/(N-1) (and 1 if N == 1).
        - Flat prior P_A(i) = 1 / |A|.
        - PUCT(i) = RankScore_A(i) + c * P_A(i) * sqrt(total_visits) / (1 + visits_i)
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
            # Single candidate: RankScore = 1
            rank_scores = {sorted_ids_asc[0]: 1.0}

        # Flat prior
        prior_P = 1.0 / N

        # PUCT score
        ucb_scores = {}
        explore_const = getattr(self.config, "puct_exploration_constant", 1.0)
        for n in nodes:
            v = max(0, getattr(n, "visits", 0))
            ucb = rank_scores[n.id] + explore_const * prior_P * (total_visits ** 0.5) / (1.0 + v)
            ucb_scores[n.id] = ucb

        # Select parent by max PUCT
        best_id = max(ucb_scores, key=ucb_scores.get)
        parent = self.tree[best_id]

        # Inspirations: top-K by task score (descending), excluding the selected parent
        # TODO - consider other strategies (e.g., diversity-based)
        k = num_inspirations if num_inspirations is not None else min(5, max(0, N - 1))
        inspirations = [
            n for n in sorted(nodes, key=lambda n: (scores[n.id], n.id), reverse=True)
            if n.id != best_id
        ][:k]

        return parent, inspirations

    def get(self, node_id: str) -> Optional[TreeNode]:
        return self.tree.get(node_id)

    def get_best_program(self, metric: Optional[str] = None) -> Optional[TreeNode]:
        """
        Get the best program based on a metric

        Args:
            metric: Metric to use for ranking (uses combined_score or average if None)

        Returns:
            Best program or None if database is empty
        """
        if not self.tree:
            return None

        # If no specific metric and we have a tracked best program, return it
        if metric is None and self.best_program_id:
            if self.best_program_id in self.tree:
                logger.debug(f"Using tracked best program: {self.best_program_id}")
                return self.tree[self.best_program_id]
            else:
                logger.warning(
                    f"Tracked best program {self.best_program_id} no longer exists, will recalculate"
                )
                self.best_program_id = None

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in self.tree.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by metric '{metric}': {sorted_programs[0].id}")
        else:
            # Sort by fitness (excluding feature dimensions)
            sorted_programs = sorted(
                self.tree.values(),
                key=lambda p: get_fitness_score(p.metrics, self.config.feature_dimensions),
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by fitness score: {sorted_programs[0].id}")

        # Update the best program tracking if we found a better program
        if sorted_programs and (
            self.best_program_id is None or sorted_programs[0].id != self.best_program_id
        ):
            old_id = self.best_program_id
            self.best_program_id = sorted_programs[0].id
            logger.info(f"Updated best program tracking from {old_id} to {self.best_program_id}")

            # Also log the scores to help understand the update
            if (
                old_id
                and old_id in self.tree
                and "combined_score" in self.tree[old_id].metrics
                and "combined_score" in self.tree[self.best_program_id].metrics
            ):
                old_score = self.tree[old_id].metrics["combined_score"]
                new_score = self.tree[self.best_program_id].metrics["combined_score"]
                logger.info(
                    f"Score change: {old_score:.4f} → {new_score:.4f} ({new_score-old_score:+.4f})"
                )

        return sorted_programs[0] if sorted_programs else None


    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save the database to disk

        Args:
            path: Path to save to (uses config.db_path if None)
            iteration: Current iteration number
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for node in self.tree.values():
            prompts = None
            if (
                self.config.log_prompts
                and self.prompts_by_program
                and node.id in self.prompts_by_program
            ):
                prompts = self.prompts_by_program[node.id]
            self._save_program(node, save_path, prompts=prompts)

        # Save metadata
        metadata = {
            "best_program_id": self.best_program_id,
            "last_iteration": iteration or self.last_iteration,
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saved tree search database with {len(self.tree)} nodes to {save_path}")

    def load(self, path: str) -> None:
        """
        Load the database from disk

        Args:
            path: Path to load from
        """
        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return

        # Load metadata first
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.best_program_id = metadata.get("best_program_id")
            self.last_iteration = metadata.get("last_iteration", 0)

            logger.info(f"Loaded database metadata with last_iteration={self.last_iteration}")

        # Load programs
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)

                        node = TreeNode.from_dict(program_data)
                        self.tree[node.id] = node
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")

        logger.info(f"Loaded tree search database with {len(self.tree)} nodes from {path}")

    def _update_best_program(self, program: TreeNode) -> None:
        """
        Update the absolute best program tracking

        Args:
            program: Program to consider as the new best
        """
        # If we don't have a best program yet, this becomes the best
        if self.best_program_id is None:
            self.best_program_id = program.id
            logger.debug(f"Set initial best program to {program.id}")
            return

        # Compare with current best program (if it still exists)
        if self.best_program_id not in self.tree:
            logger.warning(
                f"Best program {self.best_program_id} no longer exists, clearing reference"
            )
            self.best_program_id = program.id
            logger.info(f"Set new best program to {program.id}")
            return

        current_best = self.tree[self.best_program_id]

        # Update if the new program is better
        if self._is_better(program, current_best):
            old_id = self.best_program_id
            self.best_program_id = program.id

            # Log the change
            if "combined_score" in program.metrics and "combined_score" in current_best.metrics:
                old_score = current_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.info(
                    f"New best program {program.id} replaces {old_id} (combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.info(f"New best program {program.id} replaces {old_id}")

    def _save_program(
        self,
        program: TreeNode,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """
        Save a program to disk

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
            prompts: Optional prompts to save with the program, in the format {template_key: { 'system': str, 'user': str }}
        """
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        # Create programs directory if it doesn't exist
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        # Save program
        program_dict = program.to_dict()
        if prompts:
            program_dict["prompts"] = prompts
        program_path = os.path.join(programs_dir, f"{program.id}.json")

        with open(program_path, "w") as f:
            json.dump(program_dict, f)

    def _is_better(self, program1: TreeNode, program2: TreeNode) -> bool:
        """
        Determine if program1 has better FITNESS than program2

        Args:
            program1: First program
            program2: Second program

        Returns:
            True if program1 is better than program2
        """
        # If no metrics, use newest
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp

        # If only one has metrics, it's better
        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False

        # Compare fitness (excluding feature dimensions)
        fitness1 = get_fitness_score(program1.metrics)
        fitness2 = get_fitness_score(program2.metrics)

        return fitness1 > fitness2

    def log_prompt(
        self,
        program_id: str,
        template_key: str,
        prompt: Dict[str, str],
        responses: Optional[List[str]] = None,
    ) -> None:
        """
        Log a prompt for a program.
        Only logs if self.config.log_prompts is True.

        Args:
        program_id: ID of the program to log the prompt for
        template_key: Key for the prompt template
        prompt: Prompts in the format {template_key: { 'system': str, 'user': str }}.
        responses: Optional list of responses to the prompt, if available.
        """

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
