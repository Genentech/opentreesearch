"""Command-line interface for Open Tree Search."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from opentreesearch.tree_controller import TreeSearch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Open Tree Search - LLM-guided code evolution")

    parser.add_argument("initial_program", help="Path to the initial program file")
    parser.add_argument(
        "evaluation_file", help="Path to the evaluation file containing an 'evaluate' function"
    )
    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)
    parser.add_argument("--output", "-o", help="Output directory for results", default=None)
    parser.add_argument(
        "--iterations", "-i", help="Maximum number of iterations", type=int, default=None
    )
    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )
    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from",
        default=None,
    )
    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)
    parser.add_argument("--primary-model", help="Primary LLM model name", default=None)
    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    return parser.parse_args()


async def main_async() -> int:
    """Run tree search evolution from CLI arguments."""
    args = parse_args()

    if not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found")
        return 1

    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1

    config = None
    if args.api_base or args.primary_model or args.secondary_model:
        from openevolve.config import load_config as oe_load_config

        config = oe_load_config(args.config)

        if args.api_base:
            config.llm.api_base = args.api_base
            print(f"Using API base: {config.llm.api_base}")

        if args.primary_model:
            config.llm.primary_model = args.primary_model
            print(f"Using primary model: {config.llm.primary_model}")

        if args.secondary_model:
            config.llm.secondary_model = args.secondary_model
            print(f"Using secondary model: {config.llm.secondary_model}")

        if args.primary_model or args.secondary_model:
            config.llm.rebuild_models()
            print("Applied CLI model overrides - active models:")
            for i, model in enumerate(config.llm.models):
                print(f"  Model {i + 1}: {model.name} (weight: {model.weight})")

    try:
        treesearch = TreeSearch(
            initial_program_path=args.initial_program,
            evaluation_file=args.evaluation_file,
            config=config,
            config_path=args.config if config is None else None,
            output_dir=args.output,
        )

        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint directory '{args.checkpoint}' not found")
                return 1
            print(f"Loading checkpoint from {args.checkpoint}")
            treesearch.database.load(args.checkpoint)
            print(
                f"Checkpoint loaded successfully (iteration {treesearch.database.last_iteration})"
            )

        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        best_program = await treesearch.run(
            iterations=args.iterations,
            target_score=args.target_score,
            checkpoint_path=args.checkpoint,
        )

        checkpoints = sorted(
            (
                p
                for p in Path(treesearch.output_dir, "checkpoints").glob("checkpoint_*")
                if p.is_dir()
            ),
            key=lambda p: int(p.name.rsplit("_", 1)[-1]),
        )
        latest_checkpoint = checkpoints[-1] if checkpoints else None

        print("\nEvolution complete!")
        print("Best program metrics:")
        for name, value in best_program.metrics.items():
            if isinstance(value, int | float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """Entry point for the treesearch-run command."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
