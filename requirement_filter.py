# requirement_filter.py
"""
Filters and compresses requirements_clean.json into
testable_requirements.json.

For each requirement, GPT-4o decides:
  KEEP   — real testable behaviour, compress to minimal format
  SKIP   — not testable in pytest (UI visual, doc, assumption etc.)

Output format per requirement (minimal JSON):
{
  "id":     "VAL-001",
  "what":   "perplexity must be < (group_size - 1)",
  "call":   "execute(node_setting)",
  "inputs": {"perplexity": 30, "group_size": 50},
  "expect": {"pass": "200 or no error", "fail": "400 perplexity too high"},
  "hints":  ["perplexity = group_size-1 → fail", "perplexity = group_size-2 → pass"]
}

If representable in one line (simple validation or output check):
{
  "id": "VAL-002",
  "what": "n_iter >= 250 else reject",
  "call": "execute(node_setting)",
  "inputs": {"n_iter": 100},
  "expect": {"fail": "400 n_iter below minimum"}
}
"""

import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# ── prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a senior QA engineer for a chip manufacturing analytics platform.
The backend is Python FastAPI. Nodes are Python classes with an execute() method.
Settings are Pydantic classes with validation rules.

You will receive a requirement object.

YOUR JOB:
  Decide if this requirement can produce a real pytest test.
  If yes — compress it to the minimal format needed to write the test.
  If no  — skip it.

KEEP if it describes ANY of:
  - input validation (wrong type, out of range, missing required field)
  - node execute() happy path (valid inputs → correct output)
  - node execute() error path (invalid inputs → specific error)
  - output schema (which fields exist, what type they are)
  - business rules that can be asserted in code

SKIP if it describes:
  - UI layout, colors, visual appearance
  - documentation or reference material
  - assumptions or design philosophy
  - performance benchmarks (unless a hard timeout is specified)
  - deployment or infrastructure concerns
  - anything that cannot be asserted with assert in pytest

OUTPUT FORMAT:
  If SKIP → return exactly: {"action": "skip", "reason": "one line why"}

  If KEEP → return minimal JSON with only these fields:
  {
    "action": "keep",
    "id":     "original ID",
    "what":   "one line: what behaviour is being tested",
    "call":   "which method/endpoint to call e.g. execute(node_setting) or POST /api/tsne/run",
    "inputs": {only the fields relevant to this test, with example values},
    "expect": {
      "pass": "what success looks like in one line, or null if error-only test",
      "fail": "what failure looks like in one line, or null if happy-path-only test"
    },
    "hints": ["boundary case 1", "boundary case 2"]
  }

RULES:
  - hints max 3 items
  - inputs only include fields that affect this specific test
  - what must be one line under 80 chars
  - no verbose descriptions, no explanations
  - if a requirement covers multiple distinct behaviours, split into multiple objects
    return a JSON array in that case
  - otherwise return a single JSON object
  - return ONLY JSON, no markdown, no explanation
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",       "", raw)
    return raw.strip()


def call_llm(req: dict) -> str | None:
    """Single LLM call with retry."""
    wait_times = [0, 5, 15]

    for attempt in range(3):
        if wait_times[attempt]:
            time.sleep(wait_times[attempt])
        try:
            response = client.chat.completions.create(
                model       = "gpt-4o",
                temperature = 0,
                response_format = {"type": "json_object"},
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(req, indent=2)},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"retry {attempt+1}: {e}")

    return None


def parse_response(raw: str) -> list[dict]:
    """
    Parse LLM response into a list of requirement dicts.
    Handles both single object and array responses.
    """
    try:
        data = json.loads(clean_json(raw))

        # unwrap if wrapped in a key
        if isinstance(data, dict):
            # check if it's a single requirement
            if "action" in data:
                return [data]
            # check if wrapped like {"requirements": [...]}
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]

        if isinstance(data, list):
            return data

    except json.JSONDecodeError:
        pass

    return []


# ── validate output ───────────────────────────────────────────────────────────

REQUIRED_KEEP_FIELDS = ["id", "what", "call", "inputs", "expect"]

def validate_keep(item: dict) -> tuple[bool, str]:
    """Check a keep item has all required fields."""
    for field in REQUIRED_KEEP_FIELDS:
        if field not in item:
            return False, f"missing field '{field}'"
    if len(item.get("what", "")) > 120:
        return False, "what field too long"
    return True, ""


# ── main filter ───────────────────────────────────────────────────────────────

def filter_requirements(
    input_path:  str = "data/requirements_clean.json",
    output_path: str = "data/testable_requirements.json",
) -> list[dict]:

    path = Path(input_path)
    if not path.exists():
        print(f"  {input_path} not found.")
        raise SystemExit(1)

    requirements = json.loads(path.read_text(encoding="utf-8"))

    print(f"\n{'='*55}")
    print(f"  Requirement Filter")
    print(f"  Input : {len(requirements)} requirements")
    print(f"  Model : gpt-4o  temperature=0")
    print(f"{'='*55}\n")

    kept    = []
    skipped = []
    errors  = []
    split_count = 0   # requirements that were split into multiple

    for i, req in enumerate(requirements):
        req_id = req.get("id", f"?-{i}")
        title  = req.get("title", "")[:45]

        print(f"  [{i+1:02d}/{len(requirements)}] {req_id:8} {title:<45}",
              end=" ... ", flush=True)

        raw = call_llm(req)

        if raw is None:
            print("ERROR (no response)")
            errors.append(req_id)
            continue

        items = parse_response(raw)

        if not items:
            print("ERROR (parse failed)")
            errors.append(req_id)
            continue

        # process each item (may be multiple if req was split)
        chunk_kept = 0
        chunk_skip = 0

        for item in items:
            action = item.get("action", "keep")

            if action == "skip":
                reason = item.get("reason", "")
                chunk_skip += 1
                skipped.append({
                    "id":     req_id,
                    "reason": reason,
                })

            else:
                # validate
                ok, err = validate_keep(item)
                if not ok:
                    print(f"\n    ⚠ validation failed for {req_id}: {err}")
                    errors.append(req_id)
                    continue

                # clean up — remove action field from output
                item.pop("action", None)

                # ensure hints is a list
                if "hints" not in item:
                    item["hints"] = []
                if isinstance(item["hints"], str):
                    item["hints"] = [item["hints"]]
                item["hints"] = item["hints"][:3]

                kept.append(item)
                chunk_kept += 1

        # print result for this requirement
        if len(items) > 1:
            split_count += 1
            print(f"split → {chunk_kept} kept, {chunk_skip} skipped")
        elif chunk_kept:
            print("kept")
        else:
            reason = skipped[-1]["reason"] if skipped else ""
            print(f"skipped  ({reason[:40]})")

    # ── save ──────────────────────────────────────────────────────────────
    Path(output_path).write_text(
        json.dumps(kept, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Filter complete")
    print(f"{'='*55}")
    print(f"  Input requirements  : {len(requirements)}")
    print(f"  Kept (testable)     : {len(kept)}")
    print(f"  Skipped             : {len(skipped)}")
    print(f"  Split into multiple : {split_count}")
    print(f"  Errors              : {len(errors)}")

    if skipped:
        print(f"\n  Skipped reasons:")
        for s in skipped[:10]:
            print(f"    [{s['id']:8}] {s['reason'][:60]}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped)-10} more")

    if errors:
        print(f"\n  Errors (re-run these manually):")
        for e in errors:
            print(f"    {e}")

    print(f"\n  Saved → {output_path}")
    print(f"  Size reduction: {len(requirements)} → {len(kept)} requirements\n")

    return kept


# ── run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kept = filter_requirements()

    # show sample output
    if kept:
        print(f"Sample filtered requirements:\n")
        for req in kept[:3]:
            print(json.dumps(req, indent=2))
            print()