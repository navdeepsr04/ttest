# extractor.py
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from preprocess.prompts import SYSTEM_PROMPT, build_user_message
from preprocess.validator import validate_all, print_validation_summary

load_dotenv()
client = OpenAI()


# ── helpers ───────────────────────────────────────────────────────────────────

def clean_json(raw: str) -> str:
    """Strip accidental markdown fences GPT sometimes adds."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",       "", raw)
    return raw.strip()


def token_count(text: str) -> int:
    """Rough token estimate — 1 token ≈ 4 characters."""
    return len(text) // 4


# ── single chunk → requirements ───────────────────────────────────────────────

def extract_from_chunk(chunk: dict, index: int, total: int) -> list[dict]:
    """Send one chunk to GPT-4o, return list of requirement dicts."""

    label = chunk["subsection"][:45]
    print(f"  [{index+1:02d}/{total}] {label:<45}", end=" ... ", flush=True)

    # skip chunks that are clearly not requirements
    skip_keywords = ["reference", "acknowledgement", "bibliography",
                     "table of content", "appendix"]
    if any(kw in chunk["subsection"].lower() for kw in skip_keywords):
        print("skipped (non-requirement section)")
        return []

    user_msg = build_user_message(chunk)

    # warn if chunk is very large
    tokens = token_count(SYSTEM_PROMPT + user_msg)
    if tokens > 6000:
        print(f"\n  ⚠ chunk is large (~{tokens} tokens) — consider splitting")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,                          # deterministic output
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ]
        )

        raw  = response.choices[0].message.content
        raw  = clean_json(raw)
        data = json.loads(raw)

        # GPT-4o with json_object sometimes wraps in {"requirements": [...]}
        # unwrap it if so
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                data = []

        if not isinstance(data, list):
            data = []

        # tag each requirement with its source section
        for req in data:
            req["_source"] = chunk["subsection"]

        print(f"found {len(data)}")
        return data

    except json.JSONDecodeError as e:
        print(f"JSON parse error ({e})")
        return []
    except Exception as e:
        print(f"API error ({e})")
        return []


# ── deduplication ─────────────────────────────────────────────────────────────

def deduplicate(requirements: list[dict]) -> list[dict]:
    """
    Ask GPT-4o to identify near-duplicate requirements
    that were extracted from overlapping sections.
    """
    if len(requirements) <= 3:
        return requirements

    print(f"\n  Deduplicating {len(requirements)} requirements ...",
          end=" ", flush=True)

    # send only title + description to keep the dedup prompt small
    slim = [
        {"id": r.get("id", f"REQ-{i}"),
         "title": r.get("title", ""),
         "description": r.get("description", "")}
        for i, r in enumerate(requirements)
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a QA analyst. Given a list of software requirements, "
                        "identify pairs that describe the exact same behaviour. "
                        "Return JSON: {\"duplicates\": [{\"keep\": \"id\", \"remove\": \"id\"}]} "
                        "Only flag clear duplicates. When in doubt keep both. "
                        "Return {\"duplicates\": []} if none found."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(slim, indent=2)
                }
            ]
        )

        result    = json.loads(response.choices[0].message.content)
        to_remove = {d["remove"] for d in result.get("duplicates", [])}
        deduped   = [r for r in requirements
                     if r.get("id", "") not in to_remove]

        removed = len(requirements) - len(deduped)
        print(f"removed {removed} → {len(deduped)} remaining")
        return deduped

    except Exception as e:
        print(f"dedup error ({e}) — keeping all")
        return requirements


# ── renumber IDs ──────────────────────────────────────────────────────────────

def renumber(requirements: list[dict]) -> list[dict]:
    prefix = {
        "functional":     "FR",
        "non_functional": "NFR",
        "ui":             "UI",
        "api":            "API",
        "validation":     "VAL",
    }
    counters = {k: 1 for k in prefix}

    for req in requirements:
        cat = req.get("category", "functional").lower()
        if cat not in prefix:
            cat = "functional"
            req["category"] = cat

        p         = prefix[cat]
        req["id"] = f"{p}-{counters[cat]:03d}"
        counters[cat] += 1

    return requirements


# ── main ──────────────────────────────────────────────────────────────────────

def run_extraction(
    chunks_path: str = "data/chunks.json",
    output_path: str = "data/requirements.json"
) -> list[dict]:

    if not Path(chunks_path).exists():
        print(f"chunks.json not found at {chunks_path}")
        print("Run chunker.py first.")
        raise SystemExit(1)

    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))

    print(f"\n{'='*55}")
    print(f"  LLM Extraction — {len(chunks)} chunks")
    print(f"  Model: gpt-4o  |  temperature: 0")
    print(f"{'='*55}\n")

    # ── step 1: extract from every chunk ──────────────────────────────────
    all_requirements = []
    for i, chunk in enumerate(chunks):
        reqs = extract_from_chunk(chunk, i, len(chunks))
        all_requirements.extend(reqs)

    print(f"\n  Raw total: {len(all_requirements)} requirements\n")

    # ── step 2: deduplicate ────────────────────────────────────────────────
    all_requirements = deduplicate(all_requirements)

    # ── step 3: renumber cleanly ───────────────────────────────────────────
    all_requirements = renumber(all_requirements)

    # added later
    print("\n  Validating requirements...")
    all_requirements, stats = validate_all(all_requirements)
    print_validation_summary(all_requirements, stats)

    # ── step 4: save ───────────────────────────────────────────────────────
    Path(output_path).write_text(
        json.dumps(all_requirements, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # save only clean ones separately for test generation
    clean_only = [r for r in all_requirements if not r.get("_needs_review")]
    Path("data/requirements_clean.json").write_text(
        json.dumps(clean_only, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\n  All  → data/requirements.json       ({len(all_requirements)})")
    print(f"  Clean → data/requirements_clean.json ({len(clean_only)})")

    # ── step 5: summary ────────────────────────────────────────────────────
    by_cat = {}
    for r in all_requirements:
        cat = r.get("category", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1

    print(f"\n{'='*55}")
    print(f"  Extraction complete")
    print(f"{'='*55}")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat:20s} : {count}")
    print(f"  {'TOTAL':20s} : {len(all_requirements)}")
    print(f"\n  Saved → {output_path}\n")

    return all_requirements


if __name__ == "__main__":
    run_extraction()