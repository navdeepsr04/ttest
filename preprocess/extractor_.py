# extractor_v2.py
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

SKIP_SECTIONS = {
    "reference", "acknowledgement", "table of content",
    "traceability", "peer review", "srs review",
    "story details", "appendix", "bibliography",
}

SYSTEM_PROMPT = """
You are a QA engineer for a chip manufacturing analytics platform.
The platform has ML nodes (t-SNE, PCA, DBSCAN), data nodes, plot nodes.
Each node is a Python class with an execute(node_setting: dict) method.
Settings are Pydantic models — invalid inputs raise ValueError or ValidationError.

Read the given document section and extract test cases from it.

A test case covers ONE of:
  - Functional behaviour   : what execute() should do with valid input
  - Input validation       : what happens when a field is wrong/missing/out of range
  - Output check           : what columns/fields/types appear in the result
  - Boundary condition     : exact edge values that pass or fail
  - Constraint             : a rule that must hold (e.g. perplexity < group_size - 1)

Return a JSON array. Each element:
{
  "id":          "TC-?",
  "node":        "ClassName e.g. TSNENode, KlarfReaderNode, or null if not specified",
  "method":      "execute  (or specific method name if different)",
  "description": "one clear sentence: what is being tested and why",
  "input": {
    "field_name": value,
    "...only fields relevant to THIS test with realistic example values..."
  },
  "expected": {
    "outcome":  "pass | fail",
    "result":   "what the output contains on success, or null",
    "error":    "ValueError | ValidationError | specific exception, or null",
    "message":  "substring of error message to assert, or null"
  },
  "category": "functional | validation | output | boundary | constraint",
  "data_types": {
    "field_name": "int | float | str | bool | list | dict"
  }
}

IMPORTANT RULES:
  1. One test case per distinct behaviour — never combine two scenarios.
  2. For EVERY validation rule → create one fail test case.
  3. For EVERY validation rule → also create one pass test case (valid boundary).
  4. For output fields → create one test case checking field name and type.
  5. For parameters with default values → create one test case using defaults.
  6. data_types must list the type of every field in input.
  7. Keep input minimal — only fields that matter for this specific test.
  8. description must be specific, not vague. BAD: "test perplexity". 
     GOOD: "perplexity=49 with group_size=50 raises ValueError (boundary fail)".
  9. If section has nothing testable → return empty array [].
  10. Return ONLY the JSON array. No markdown fences. No explanation.
"""


def clean_raw(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def parse_array(raw: str) -> list[dict]:
    """Parse LLM string output into a list of dicts."""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        # sometimes wrapped: {"test_cases": [...]}
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
        return []
    except json.JSONDecodeError:
        return []


def extract_chunk(chunk: dict, index: int, total: int) -> list[dict]:
    label = chunk.get("subsection", "")[:45]

    # skip non-testable sections
    if any(kw in label.lower() for kw in SKIP_SECTIONS):
        print(f"  [{index:02d}/{total}] {label:<45} SKIP")
        return []

    user_message = (
        f"Document section: \"{chunk.get('subsection', '')}\"\n"
        f"Parent section  : \"{chunk.get('section', '')}\"\n\n"
        f"{chunk.get('text', '')}\n\n"
        f"Extract all test cases from this section."
    )

    for attempt in range(3):
        wait = [0, 8, 20][attempt]
        if wait:
            print(f"    waiting {wait}s before retry...", flush=True)
            time.sleep(wait)

        try:
            response = client.chat.completions.create(
                model    = "gpt-4o",
                temperature = 0,
                response_format = {"type": "json_object"},
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )

            raw    = response.choices[0].message.content
            raw    = clean_raw(raw)
            result = parse_array(raw)

            print(f"  [{index:02d}/{total}] {label:<45} {len(result)} test case(s)")
            return result

        except json.JSONDecodeError:
            print(f"  [{index:02d}/{total}] {label:<45} JSON error attempt {attempt+1}")
        except Exception as e:
            print(f"  [{index:02d}/{total}] {label:<45} error: {e} attempt {attempt+1}")

    print(f"  [{index:02d}/{total}] {label:<45} FAILED")
    return []


def deduplicate(test_cases: list[dict]) -> list[dict]:
    """Remove near-identical test cases by description."""
    seen  = set()
    clean = []
    for tc in test_cases:
        key = tc.get("description", "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            clean.append(tc)
    return clean


def renumber(test_cases: list[dict]) -> list[dict]:
    for i, tc in enumerate(test_cases, 1):
        tc["id"] = f"TC-{i:03d}"
    return test_cases


def run(
    chunks_path: str = "data/chunks.json",
    output_path: str = "data/test_cases.json",
) -> list[dict]:

    chunks_file = Path(chunks_path)
    if not chunks_file.exists():
        print(f"chunks.json not found at {chunks_path}")
        print("Run chunker.py first.")
        raise SystemExit(1)

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    total  = len(chunks)

    print(f"\n{'='*60}")
    print(f"  Extractor v2")
    print(f"  {total} chunks  |  model: gpt-4o  |  temp: 0")
    print(f"{'='*60}\n")

    all_cases = []
    for i, chunk in enumerate(chunks, 1):
        cases = extract_chunk(chunk, i, total)

        # tag source section on each case
        for c in cases:
            c["_section"] = chunk.get("subsection", "")

        all_cases.extend(cases)

    print(f"\n  Raw total     : {len(all_cases)}")
    all_cases = deduplicate(all_cases)
    print(f"  After dedup   : {len(all_cases)}")
    all_cases = renumber(all_cases)

    # save
    Path(output_path).write_text(
        json.dumps(all_cases, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # summary
    by_cat   = {}
    by_node  = {}
    by_outcome = {}

    for tc in all_cases:
        cat     = tc.get("category", "unknown")
        node    = tc.get("node") or "unknown"
        outcome = tc.get("expected", {}).get("outcome", "?")

        by_cat[cat]         = by_cat.get(cat, 0) + 1
        by_node[node]       = by_node.get(node, 0) + 1
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Complete — {len(all_cases)} test cases → {output_path}")
    print(f"{'='*60}")

    print(f"\n  By category:")
    for k, v in sorted(by_cat.items()):
        print(f"    {k:20}: {v}")

    print(f"\n  By node:")
    for k, v in sorted(by_node.items()):
        print(f"    {k:30}: {v}")

    print(f"\n  By outcome:")
    for k, v in sorted(by_outcome.items()):
        print(f"    {k:10}: {v}")

    print(f"\n  Sample (first 2):\n")
    for tc in all_cases[:2]:
        print(json.dumps(tc, indent=2))
        print()

    return all_cases


if __name__ == "__main__":
    run()