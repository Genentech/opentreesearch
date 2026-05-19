r"""Generate LLM summaries for programs in an OpenTreeSearch checkpoint.

For each program, calls the same LLM endpoint that tree search uses and asks
for two concise Markdown summaries: what the program does, and how it differs
from its parent. Results are written to ``<checkpoint>/summaries/<id>.json``.

Usage:
    python3 tools/generate_program_summaries.py <checkpoint_dir> \\
        --config examples/function_minimization/config.yaml
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from openevolve.llm.ensemble import LLMEnsemble
from tqdm import tqdm

from opentreesearch.tree_controller import load_config as load_treesearch_config

SYSTEM_MESSAGE = (
    "You are a senior software engineer reviewing programs produced by an "
    "evolutionary code-search system. Read the program (and its parent, when "
    "given) and write concise, faithful Markdown summaries. Describe only what "
    "the code actually does. Wrap symbol names in backticks."
)

PROMPT_TEMPLATE = """\
# Task
Summarize this {language} program{parent_clause}.

Respond with exactly these two Markdown sections, in this order, and nothing
else. Each section MUST start with one concise summary sentence, then a blank
line, then 3 short bullet points.

## LLM summary of solution {label}
<one sentence describing what the program does>

- bullet
- bullet
- bullet

## Differences from the parent solution
{diff_rule}

# Current program
```{language}
{code}
```{parent_block}
"""

CHILD_DIFF_RULE = (
    "<one sentence summarizing the change versus the parent>\n\n"
    "- 3 bullets covering only meaningful differences (algorithmic, "
    "structural, hyper-parameter, or behavioural). Skip purely cosmetic "
    "changes such as whitespace or comment-only edits."
)
ROOT_DIFF_RULE = "This is the root program; there is no parent."

SECTION_RE = re.compile(r"^##\s+.*?$", re.MULTILINE)


def label_for(program: dict) -> str:
    """Return a short human-readable label like ``#4`` or the first 8 id chars."""
    try:
        return f"#{int(program['iteration_found'])}"
    except (KeyError, TypeError, ValueError):
        return (program.get("id") or "root")[:8]


def build_prompt(program: dict, parent: dict | None, language: str) -> str:
    """Build the user prompt asking the LLM to summarize ``program`` (vs ``parent`` if any)."""
    parent_block = ""
    if parent is not None:
        parent_block = f"\n\n# Parent program\n```{language}\n{parent.get('code') or ''}\n```"
    return PROMPT_TEMPLATE.format(
        language=language,
        label=label_for(program),
        code=program.get("code") or "",
        parent_clause=" and explain how it differs from its parent" if parent else "",
        diff_rule=CHILD_DIFF_RULE if parent else ROOT_DIFF_RULE,
        parent_block=parent_block,
    )


def split_sections(response: str) -> tuple[str, str]:
    """Return (summary_md, diff_md) by splitting on the two ``##`` headings."""
    matches = list(SECTION_RE.finditer(response or ""))
    if not matches:
        return (response or "").strip(), ""
    ends = [m.end() for m in matches]
    starts = [m.start() for m in matches[1:]] + [len(response)]
    bodies = [response[s:e].strip() for s, e in zip(ends, starts)]
    return bodies[0], (bodies[1] if len(bodies) > 1 else "")


def load_programs(checkpoint: Path) -> dict[str, dict]:
    """Load all program JSON records from ``<checkpoint>/programs/`` keyed by id."""
    programs_dir = checkpoint / "programs"
    if not programs_dir.is_dir():
        raise SystemExit(f"Missing programs directory: {programs_dir}")
    programs = {}
    for path in sorted(programs_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("id"):
            programs[record["id"]] = record
    return programs


def build_ensemble(args: argparse.Namespace) -> tuple[LLMEnsemble, str]:
    """Build the LLM ensemble from CLI args, returning ``(ensemble, language)``."""
    config, _ = load_treesearch_config(str(args.config) if args.config else None)
    if args.api_base:
        config.llm.api_base = args.api_base
    if args.primary_model:
        config.llm.primary_model = args.primary_model
    if args.secondary_model:
        config.llm.secondary_model = args.secondary_model
    if args.primary_model or args.secondary_model:
        config.llm.rebuild_models()
    language = getattr(config, "language", None) or "python"
    return LLMEnsemble(config.llm.models), language


async def summarize_one(
    ensemble: LLMEnsemble,
    program: dict,
    parent: dict | None,
    language: str,
) -> dict:
    """Call the LLM once and return the summary payload to write to disk."""
    started = time.time()
    response = await ensemble.generate_with_context(
        system_message=SYSTEM_MESSAGE,
        messages=[{"role": "user", "content": build_prompt(program, parent, language)}],
    )
    summary_md, diff_md = split_sections(response or "")
    return {
        "program_id": program.get("id"),
        "parent_id": program.get("parent_id"),
        "iteration_found": program.get("iteration_found"),
        "label": label_for(program),
        "summary_markdown": summary_md,
        "diff_markdown": diff_md,
        "generated_at": time.time(),
        "elapsed_seconds": round(time.time() - started, 3),
    }


async def run(args: argparse.Namespace) -> int:
    """Summarize every program in the checkpoint and write results to ``summaries/``."""
    ensemble, language = build_ensemble(args)
    checkpoint = args.checkpoint.resolve()
    programs = load_programs(checkpoint)
    summaries_dir = checkpoint / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else None
    candidates = [r for pid, r in programs.items() if only is None or pid in only]
    targets = [
        r for r in candidates if args.force or not (summaries_dir / f"{r['id']}.json").exists()
    ]

    print(
        f"Checkpoint: {checkpoint}\n"
        f"Programs: {len(programs)} | to summarize: {len(targets)} | "
        f"already summarized: {len(candidates) - len(targets)}",
        flush=True,
    )
    if not targets:
        return 0

    print(
        f"Models: {[m.name for m in ensemble.models_cfg]} concurrency={args.concurrency}",
        flush=True,
    )

    sem = asyncio.Semaphore(max(1, args.concurrency))
    failures: list[tuple[str, str]] = []

    async def summarize(record: dict) -> tuple[str, str | None]:
        async with sem:
            pid = record["id"]
            try:
                payload = await summarize_one(
                    ensemble,
                    record,
                    programs.get(record.get("parent_id")),
                    language,
                )
                (summaries_dir / f"{pid}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return pid, None
            except Exception as exc:  # noqa: BLE001 - report and continue
                return pid, str(exc)

    started = time.time()
    tasks = [asyncio.create_task(summarize(r)) for r in targets]
    with tqdm(
        total=len(tasks),
        desc="Summarizing",
        unit="prog",
        mininterval=0,
        miniters=1,
        smoothing=0.3,
        dynamic_ncols=True,
    ) as pbar:
        for coro in asyncio.as_completed(tasks):
            pid, err = await coro
            if err:
                failures.append((pid, err))
            pbar.update(1)

    print(
        f"Wrote {len(targets) - len(failures)}/{len(targets)} summaries to "
        f"{summaries_dir} in {time.time() - started:.1f}s",
        flush=True,
    )
    for pid, err in failures:
        print(f"  failure: {pid}: {err}", flush=True)
    return 1 if failures else 0


def main() -> int:
    """Parse CLI arguments and run the summarizer."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", "-c", type=Path, default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--primary-model", default=None)
    parser.add_argument("--secondary-model", default=None)
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Number of concurrent LLM calls (default 8)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-summarize programs that already have a summary file",
    )
    parser.add_argument("--only", nargs="+", default=None, help="Only summarize these program ids")
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        raise SystemExit(f"Not a directory: {args.checkpoint}")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
