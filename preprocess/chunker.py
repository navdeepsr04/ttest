# chunker.py
import json
from pathlib import Path


def group_into_chunks(blocks: list[dict]) -> list[dict]:
    """
    Group blocks by subsection heading.
    Each chunk = everything under one heading.

    Output per chunk:
    {
        "section":    "3  Document Understanding",
        "subsection": "3.1  Document knowledge",
        "text":       "...clean markdown-style text...",
        "has_table":  True/False,
        "word_count": 120
    }
    """
    grouped = {}
    order   = []

    for block in blocks:
        # use subsection as the grouping key
        # fall back to section if no subsection
        key = block.get("subsection") or block.get("section") or "General"

        if key not in grouped:
            grouped[key] = {
                "section":    block.get("section", "General"),
                "subsection": key,
                "blocks":     [],
            }
            order.append(key)

        grouped[key]["blocks"].append(block)

    chunks = []

    for key in order:
        group     = grouped[key]
        blist     = group["blocks"]
        has_table = any(b["type"] == "table" for b in blist)

        text_parts = []

        for b in blist:

            if b["type"] == "heading":
                # section title as a markdown heading
                text_parts.append(f"\n## {b['content']}\n")

            elif b["type"] == "label":
                # bold sub-labels like "Validation Rules:" or "Inputs:"
                text_parts.append(f"\n**{b['content']}**")

            elif b["type"] == "paragraph":
                text_parts.append(b["content"])

            elif b["type"] == "bullet":
                text_parts.append(f"  • {b['content']}")

            elif b["type"] == "table":
                rows = b["content"]
                if not rows:
                    continue

                # format as markdown table — LLMs read this well
                # first row = header
                header    = " | ".join(str(c) for c in rows[0])
                separator = " | ".join(["---"] * len(rows[0]))
                body_rows = "\n".join(
                    " | ".join(str(c) for c in row)
                    for row in rows[1:]
                )
                text_parts.append(f"\n{header}\n{separator}\n{body_rows}\n")

        # join everything into one clean string
        full_text = "\n".join(text_parts).strip()

        # skip empty or near-empty chunks (e.g. blank sections)
        if len(full_text) < 30:
            continue

        word_count = len(full_text.split())

        chunks.append({
            "section":    group["section"],
            "subsection": group["subsection"],
            "text":       full_text,
            "has_table":  has_table,
            "word_count": word_count,
        })

    return chunks


def print_summary(chunks: list[dict]):
    total_words = sum(c["word_count"] for c in chunks)

    print(f"\n{'='*55}")
    print(f"  Chunking summary")
    print(f"{'='*55}")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Total words  : {total_words:,}")
    print(f"  Avg per chunk: {total_words // max(len(chunks), 1)} words")
    print(f"{'='*55}\n")

    print(f"  {'#':<4} {'words':<7} {'tbl':<5} subsection")
    print(f"  {'-'*50}")

    for i, c in enumerate(chunks):
        table_flag = " ✓" if c["has_table"] else ""
        name = c["subsection"][:45]
        print(f"  {i+1:<4} {c['word_count']:<7} {table_flag:<5} {name}")

    print()


def save_chunks(chunks: list[dict], out_path: str = "data/chunks.json"):
    Path(out_path).write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    import sys

    # check blocks.json exists
    blocks_path = Path("data/blocks.json")
    if not blocks_path.exists():
        print("Run doc_reader.py first to generate data/blocks.json")
        sys.exit(1)

    blocks = json.loads(blocks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(blocks)} blocks from {blocks_path}")

    chunks = group_into_chunks(blocks)
    print_summary(chunks)
    save_chunks(chunks)