# extractor.py
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from prompts import SYSTEM_PROMPT, build_user_message

load_dotenv()
client = OpenAI()

SKIP_SECTIONS = {
    "reference", "acknowledgement", "bibliography",
    "table of content", "appendix", "traceability",
    "peer review", "srs review", "story details",
}


def clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def call_llm(chunk: dict) -> list[dict]:
    label = chunk["subsection"][:40]

    # skip non-requirement sections
    if any(kw in chunk["subsection"].lower() for kw in SKIP_SECTIONS):
        print(f"  {label:<42} skipped")
        return []

    for attempt in range(3):
        wait = [0, 5, 15][attempt]
        if wait:
            time.sleep(wait)
        try:
            resp = client.chat.completions.create(
                model           = "gpt-4o",
                temperature     = 0,
                response_format = {"type": "json_object"},
                messages        = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_message(chunk)},
                ],
            )
            raw  = clean_json(resp.choices[0].message.content)
            data = json.loads(raw)

            # unwrap if nested
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break
                else:
                    data = []

            result = data if isinstance(data, list) else []
            print(f"  {label:<42} {len(result)} test case(s)")
            return result

        except json.JSONDecodeError:
            print(f"  {label:<42} JSON error (attempt {attempt+1})")
        except Exception as e:
            print(f"  {label:<42} error: {e} (attempt {attempt+1})")

    print(f"  {label:<42} FAILED after 3 attempts")
    return []


def renumber(test_cases: list[dict]) -> list[dict]:
    for i, tc in enumerate(test_cases, 1):
        tc["id"] = f"TC-{i:03d}"
    return test_cases


def deduplicate(test_cases: list[dict]) -> list[dict]:
    """Remove exact duplicate 'what' descriptions."""
    seen  = set()
    clean = []
    for tc in test_cases:
        key = tc.get("what", "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            clean.append(tc)
    return clean


def run_extraction(
    chunks_path: str = "data/chunks.json",
    output_path: str = "data/test_cases.json",
) -> list[dict]:

    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))

    print(f"\n{'='*55}")
    print(f"  Extractor — {len(chunks)} chunks → test cases")
    print(f"{'='*55}\n")

    all_cases = []
    for chunk in chunks:
        cases = call_llm(chunk)
        # tag source section
        for c in cases:
            c["_section"] = chunk["subsection"]
        all_cases.extend(cases)

    print(f"\n  Raw: {len(all_cases)} test cases")

    all_cases = deduplicate(all_cases)
    print(f"  After dedup: {len(all_cases)}")

    all_cases = renumber(all_cases)

    Path(output_path).write_text(
        json.dumps(all_cases, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # summary
    by_type = {}
    for tc in all_cases:
        t = tc.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\n{'='*55}")
    print(f"  Extraction complete → {output_path}")
    print(f"{'='*55}")
    for t, count in sorted(by_type.items()):
        print(f"  {t:20}: {count}")
    print(f"  {'TOTAL':20}: {len(all_cases)}")

    return all_cases


if __name__ == "__main__":
    run_extraction()