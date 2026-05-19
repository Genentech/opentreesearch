"""Generate a self-contained HTML tree-search viewer for an OpenTreeSearch checkpoint.

Usage:
    python3 tools/generate_tree_search_view.py <checkpoint_dir>

Example:
    python3 tools/generate_tree_search_view.py \
        output/circle_packing/phase_1_kimi/checkpoints/checkpoint_80

By default this writes:
    <checkpoint_dir>/tree_search_view.html

To choose a different output path:
    python3 tools/generate_tree_search_view.py <checkpoint_dir> --out path/to/view.html

If ``<checkpoint_dir>/summaries/`` exists (produced by
``tools/generate_program_summaries.py``), those LLM summaries are merged into
the viewer automatically. If not, the viewer simply omits the summary panel.

Open the generated HTML file directly in a browser. It embeds the checkpoint
data, so it does not need a local web server.
"""

import argparse
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("tree_search_view_template.html")


def compact_program(record: dict, summary: dict | None) -> dict:
    """Return the subset of program fields needed by the HTML viewer."""
    metrics = record.get("metrics") or {}
    metadata = record.get("metadata") or {}
    compact = {
        "id": record.get("id"),
        "parent_id": record.get("parent_id"),
        "children_ids": record.get("children_ids") or [],
        "depth": record.get("depth"),
        "iteration_found": record.get("iteration_found"),
        "visits": record.get("visits"),
        "language": record.get("language"),
        "timestamp": record.get("timestamp"),
        "metrics": metrics,
        "combined_score": metrics.get("combined_score"),
        "changes": metadata.get("changes"),
        "parent_metrics": metadata.get("parent_metrics") or {},
        "code": record.get("code") or "",
    }
    if summary:
        compact["llm_summary"] = {
            "label": summary.get("label"),
            "summary_markdown": summary.get("summary_markdown") or "",
            "diff_markdown": summary.get("diff_markdown") or "",
            "generated_at": summary.get("generated_at"),
        }
    return compact


def load_summaries(checkpoint: Path) -> dict[str, dict]:
    """Load <checkpoint>/summaries/*.json keyed by program_id (empty if absent)."""
    summaries_dir = checkpoint / "summaries"
    if not summaries_dir.is_dir():
        return {}
    summaries = {}
    for path in summaries_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries[payload.get("program_id") or path.stem] = payload
    return summaries


def load_json(path: Path) -> dict:
    """Read and parse a JSON file, returning ``{}`` if it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_checkpoint(checkpoint: Path) -> dict:
    """Load checkpoint metadata, programs, and (optionally) LLM summaries."""
    programs_dir = checkpoint / "programs"
    if not programs_dir.is_dir():
        raise SystemExit(f"Missing programs directory: {programs_dir}")

    summaries = load_summaries(checkpoint)
    programs = []
    for path in sorted(programs_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        programs.append(compact_program(record, summaries.get(record.get("id"))))
    return {
        "checkpoint": str(checkpoint),
        "program_count": len(programs),
        "summary_count": sum(1 for p in programs if p.get("llm_summary")),
        "best_program": load_json(checkpoint / "best_program_info.json"),
        "metadata": load_json(checkpoint / "metadata.json"),
        "programs": programs,
    }


def render_html(data: dict) -> str:
    """Render the HTML template with checkpoint data embedded as JSON."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__CHECKPOINT_DATA_JSON__", data_json)


def main() -> None:
    """Parse CLI arguments and write the generated HTML viewer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.out or checkpoint / "tree_search_view.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(load_checkpoint(checkpoint)), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
