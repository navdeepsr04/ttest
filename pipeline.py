# pipeline.py
"""
Main pipeline — runs all stages end to end.

Usage:
    # full run
    python pipeline.py --input data/input/tsne_srs.docx

    # run only up to requirements.json (then YOU review it)
    python pipeline.py --input data/input/tsne_srs.docx --stage extract

    # run only test generation (after reviewing requirements_clean.json)
    python pipeline.py --stage generate

    # resume extraction if it crashed midway
    python pipeline.py --input data/input/tsne_srs.docx --resume

    # validate existing requirements.json without re-extracting
    python pipeline.py --stage validate
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── import all modules ────────────────────────────────────────────────────────
from doc_reader      import extract_blocks,    print_summary  as reader_summary,  save_blocks
from chunker         import group_into_chunks, print_summary  as chunker_summary, save_chunks
from extractor       import extract_from_chunk, deduplicate,  renumber
from validator       import validate_all,       print_validation_summary
from test_generator  import run_generation


# ── checkpoint helpers ────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path("data/checkpoint.json")

def load_checkpoint() -> dict:
    """Load checkpoint if it exists."""
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {"completed_chunks": [], "partial_requirements": []}


def save_checkpoint(completed_chunks: list[int], partial_requirements: list[dict]):
    """Save progress after each chunk so we can resume on crash."""
    CHECKPOINT_PATH.write_text(
        json.dumps({
            "completed_chunks":      completed_chunks,
            "partial_requirements":  partial_requirements,
            "saved_at":              time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def clear_checkpoint():
    """Delete checkpoint after a successful full run."""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


# ── review gate ───────────────────────────────────────────────────────────────

def review_gate(requirements: list[dict]) -> bool:
    """
    Pause and ask the user to review requirements.json
    before test generation begins.

    Returns True if user wants to continue, False to abort.
    """
    needs_review = [r for r in requirements if r.get("_needs_review")]
    clean        = [r for r in requirements if not r.get("_needs_review")]

    print(f"\n{'='*55}")
    print(f"  ── HUMAN REVIEW GATE ──")
    print(f"{'='*55}")
    print(f"  Clean requirements  : {len(clean)}")
    print(f"  Flagged for review  : {len(needs_review)}")
    print(f"\n  requirements.json has been saved.")
    print(f"  requirements_clean.json contains only clean ones.")
    print()

    if needs_review:
        print(f"  {len(needs_review)} requirement(s) need your attention:")
        for r in needs_review:
            print(f"\n    [{r.get('id','?')}] {r.get('title','')}")
            for w in r.get("_warnings", [])[:3]:
                print(f"           ⚠  {w}")
        print()
        print(f"  Open data/requirements.json and fix the flagged ones.")
        print(f"  Or open data/requirements_clean.json to only use clean ones.")
        print()

    print(f"  Test generation will use: data/requirements_clean.json")
    print(f"  ({len(clean)} requirements → estimated {len(clean) * 4} test functions)")
    print()

    answer = input("  Continue to test generation? [y/n]: ").strip().lower()
    return answer == "y"


# ── stage functions ───────────────────────────────────────────────────────────

def stage_read(input_path: str) -> list[dict]:
    """Stage 1 — read docx, produce blocks."""
    print(f"\n{'─'*55}")
    print(f"  STAGE 1 — Document reader")
    print(f"{'─'*55}")

    if not Path(input_path).exists():
        print(f"  File not found: {input_path}")
        sys.exit(1)

    blocks = extract_blocks(input_path)
    reader_summary(blocks)
    save_blocks(blocks)
    return blocks


def stage_chunk(blocks: list[dict]) -> list[dict]:
    """Stage 2 — group blocks into chunks."""
    print(f"\n{'─'*55}")
    print(f"  STAGE 2 — Chunker")
    print(f"{'─'*55}")

    chunks = group_into_chunks(blocks)
    chunker_summary(chunks)
    save_chunks(chunks)
    return chunks


def stage_extract(chunks: list[dict], resume: bool = False) -> list[dict]:
    """
    Stage 3 — LLM extraction with checkpoint support.
    If resume=True, skip chunks already completed.
    """
    print(f"\n{'─'*55}")
    print(f"  STAGE 3 — LLM Extraction")
    print(f"{'─'*55}\n")

    # load checkpoint if resuming
    checkpoint         = load_checkpoint() if resume else {"completed_chunks": [], "partial_requirements": []}
    completed_indices  = set(checkpoint["completed_chunks"])
    all_requirements   = checkpoint["partial_requirements"]

    if resume and completed_indices:
        print(f"  Resuming from checkpoint — {len(completed_indices)} chunks already done\n")

    total = len(chunks)

    for i, chunk in enumerate(chunks):
        # skip already completed chunks
        if i in completed_indices:
            label = chunk["subsection"][:45]
            print(f"  [{i+1:02d}/{total}] {label:<45} ... skipped (checkpoint)")
            continue

        reqs = extract_from_chunk(chunk, i, total)
        all_requirements.extend(reqs)

        # save checkpoint after every chunk
        completed_indices.add(i)
        save_checkpoint(list(completed_indices), all_requirements)

    print(f"\n  Raw total: {len(all_requirements)} requirements extracted")

    # dedup + renumber
    all_requirements = deduplicate(all_requirements)
    all_requirements = renumber(all_requirements)

    # clear checkpoint after successful full extraction
    clear_checkpoint()
    print(f"  Checkpoint cleared\n")

    return all_requirements


def stage_validate(requirements: list[dict]) -> tuple[list[dict], dict]:
    """Stage 4 — validate all requirements."""
    print(f"\n{'─'*55}")
    print(f"  STAGE 4 — Validation")
    print(f"{'─'*55}")

    cleaned, stats = validate_all(requirements)
    print_validation_summary(cleaned, stats)

    # save both files
    Path("data/requirements.json").write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    clean_only = [r for r in cleaned if not r.get("_needs_review")]
    Path("data/requirements_clean.json").write_text(
        json.dumps(clean_only, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\n  Saved → data/requirements.json       ({len(cleaned)})")
    print(f"  Saved → data/requirements_clean.json ({len(clean_only)})")

    return cleaned, stats


def stage_generate():
    """Stage 5 — generate test files from clean requirements."""
    print(f"\n{'─'*55}")
    print(f"  STAGE 5 — Test Generation")
    print(f"{'─'*55}")
    run_generation()


# ── print final banner ────────────────────────────────────────────────────────

def print_banner(start_time: float, report_path: str = "data/generation_report.json"):
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    report = {}
    if Path(report_path).exists():
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    print(f"\n{'='*55}")
    print(f"  Pipeline complete  ({minutes}m {seconds}s)")
    print(f"{'='*55}")
    if report:
        print(f"  Generated     : {report.get('generated', 0)}")
        print(f"  Review needed : {report.get('review_needed', 0)}")
        print(f"  Failed        : {report.get('failed', 0)}")
    print(f"\n  Next steps:")
    print(f"    1. Review failed/     → fix manually")
    print(f"    2. Review review_needed/ → strengthen assertions")
    print(f"    3. Update conftest.py → correct app import path")
    print(f"    4. Run: pytest generated_tests/ -v")
    print(f"{'='*55}\n")


# ── argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-powered test case generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --input data/input/tsne_srs.docx
  python pipeline.py --input data/input/tsne_srs.docx --stage extract
  python pipeline.py --stage generate
  python pipeline.py --stage validate
  python pipeline.py --input data/input/tsne_srs.docx --resume
  python pipeline.py --input data/input/tsne_srs.docx --no-gate
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to input .docx file"
    )
    parser.add_argument(
        "--stage",
        choices=["extract", "validate", "generate", "all"],
        default="all",
        help=(
            "extract  → run up to requirements.json then stop\n"
            "validate → validate existing requirements.json\n"
            "generate → run test generation only\n"
            "all      → full pipeline (default)"
        )
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume extraction from checkpoint if previous run crashed"
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Skip the human review gate (useful for automation)"
    )

    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    start_time = time.time()

    print(f"\n{'='*55}")
    print(f"  AI Test Generation Pipeline")
    print(f"  stage={args.stage}  resume={args.resume}")
    print(f"{'='*55}")

    # ── stage: generate only ───────────────────────────────────────────────
    if args.stage == "generate":
        if not Path("data/requirements_clean.json").exists():
            print("  data/requirements_clean.json not found.")
            print("  Run with --stage extract first.")
            sys.exit(1)
        stage_generate()
        print_banner(start_time)
        return

    # ── stage: validate only ───────────────────────────────────────────────
    if args.stage == "validate":
        if not Path("data/requirements.json").exists():
            print("  data/requirements.json not found.")
            print("  Run with --stage extract first.")
            sys.exit(1)
        reqs = json.loads(
            Path("data/requirements.json").read_text(encoding="utf-8")
        )
        stage_validate(reqs)
        return

    # ── all other stages need --input ──────────────────────────────────────
    if not args.input:
        print("  --input is required for stages: extract, all")
        print("  Example: python pipeline.py --input data/input/tsne_srs.docx")
        sys.exit(1)

    # ── stage 1: read ──────────────────────────────────────────────────────
    blocks = stage_read(args.input)

    # ── stage 2: chunk ────────────────────────────────────────────────────
    chunks = stage_chunk(blocks)

    # ── stage 3: extract ──────────────────────────────────────────────────
    requirements = stage_extract(chunks, resume=args.resume)

    # ── stage 4: validate ─────────────────────────────────────────────────
    requirements, stats = stage_validate(requirements)

    # stop here if --stage extract
    if args.stage == "extract":
        print(f"\n  Stopped after extraction (--stage extract).")
        print(f"  Review data/requirements.json then run:")
        print(f"    python pipeline.py --stage generate\n")
        return

    # ── review gate ────────────────────────────────────────────────────────
    if not args.no_gate:
        proceed = review_gate(requirements)
        if not proceed:
            print(f"\n  Aborted at review gate.")
            print(f"  Fix requirements.json then run:")
            print(f"    python pipeline.py --stage generate\n")
            return

    # ── stage 5: generate ─────────────────────────────────────────────────
    stage_generate()
    print_banner(start_time)


if __name__ == "__main__":
    main()