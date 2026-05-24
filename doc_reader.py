# doc_reader.py
import json
import re
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn


def is_heading(style: str, text: str) -> bool:
    """
    Detect headings by Word style name OR by numbered pattern.
    Handles:
      - Standard Word styles  : Heading1, Heading2, Heading 1 etc.
      - Numbered patterns     : "1 Introduction", "3.1 Data Section"
      - Title style           : Title, Subtitle
    """
    style_lower = style.lower()

    if "heading" in style_lower:
        return True

    if style_lower in ("title", "subtitle"):
        return True

    # numbered section: "1 Scope", "3.1 Model Configuration", "10. Reference"
    if re.match(r'^\d+[\.\d]*\.?\s+\w', text.strip()):
        return True

    return False


def extract_blocks(path: str) -> list[dict]:
    """
    Read a .docx file and return a list of structured blocks.

    Each block has:
        type       : heading | paragraph | bullet | label | table
        section    : top-level section name  e.g. "3  Functional Requirements"
        subsection : nearest heading         e.g. "3.1  Data Section"
        content    : str for text blocks, list[list[str]] for tables
    """
    doc = Document(path)

    blocks          = []
    current_section    = "General"
    current_subsection = "General"

    for element in doc.element.body:
        tag = element.tag.split('}')[-1]

        # ── Paragraph, Heading, Bullet ─────────────────────────────────────
        if tag == 'p':
            style_node = element.find(f'.//{qn("w:pStyle")}')
            style = (
                style_node.get(qn("w:val"), "")
                if style_node is not None
                else ""
            )

            # collect all text runs, preserving spaces
            text = "".join(
                node.text
                for node in element.findall(f'.//{qn("w:t")}')
                if node.text
            ).strip()

            if not text:
                continue

            # ── Heading ────────────────────────────────────────────────────
            if is_heading(style, text):
                # subsection = has a dot like "3.1" or "3.1.2"
                if re.match(r'^\d+\.\d+', text.strip()):
                    level              = 2
                    current_subsection = text.strip()
                else:
                    level              = 1
                    current_section    = text.strip()
                    current_subsection = text.strip()

                blocks.append({
                    "type":       "heading",
                    "section":    current_section,
                    "subsection": current_subsection,
                    "level":      level,
                    "content":    text,
                })

            # ── Bullet / List ──────────────────────────────────────────────
            elif any(kw in style.lower() for kw in ["list", "bullet", "number"]):
                blocks.append({
                    "type":       "bullet",
                    "section":    current_section,
                    "subsection": current_subsection,
                    "content":    text,
                })

            # ── Inline label e.g. "Inputs:", "Validation Rules:" ───────────
            elif text.endswith(":") and len(text) < 60:
                blocks.append({
                    "type":       "label",
                    "section":    current_section,
                    "subsection": current_subsection,
                    "content":    text,
                })

            # ── Normal paragraph ───────────────────────────────────────────
            else:
                blocks.append({
                    "type":       "paragraph",
                    "section":    current_section,
                    "subsection": current_subsection,
                    "content":    text,
                })

        # ── Table ──────────────────────────────────────────────────────────
        elif tag == 'tbl':
            rows = []
            for row in element.findall(f'.//{qn("w:tr")}'):
                cells = []
                for cell in row.findall(f'.//{qn("w:tc")}'):
                    cell_text = "".join(
                        n.text
                        for n in cell.findall(f'.//{qn("w:t")}')
                        if n.text
                    ).strip()
                    cells.append(cell_text)

                # skip fully empty rows
                if any(c.strip() for c in cells):
                    rows.append(cells)

            if rows:
                blocks.append({
                    "type":       "table",
                    "section":    current_section,
                    "subsection": current_subsection,
                    "content":    rows,
                })

    return blocks


def print_summary(blocks: list[dict]):
    counts   = {}
    sections = []

    for b in blocks:
        counts[b["type"]] = counts.get(b["type"], 0) + 1
        if b["type"] == "heading" and b["content"] not in sections:
            sections.append(b["content"])

    print(f"\n{'='*55}")
    print(f"  Document parsing summary")
    print(f"{'='*55}")
    for k, v in counts.items():
        print(f"  {k:12} : {v}")
    print(f"  {'total':12} : {sum(counts.values())}")
    print(f"{'='*55}")

    print(f"\n  Sections found ({len(sections)}):")
    for s in sections:
        indent = "      " if re.match(r'^\d+\.\d+', s) else "    "
        print(f"{indent}› {s}")

    print(f"\n  First 10 blocks:\n")
    for b in blocks[:10]:
        if b["type"] == "table":
            print(f"  [TABLE     ] subsection='{b['subsection']}'")
            for row in b["content"][:2]:
                print(f"               {row}")
        else:
            preview = (
                b["content"][:72] + "..."
                if len(b["content"]) > 72
                else b["content"]
            )
            print(f"  [{b['type'].upper():10}] sub='{b['subsection']}'")
            print(f"               {preview}")
        print()


def save_blocks(blocks: list[dict], out_path: str = "data/blocks.json"):
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(
        json.dumps(blocks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
    else:
        docs = list(Path("data").glob("*.docx"))
        if not docs:
            print("No .docx file found in data/  — drop your file there first.")
            sys.exit(1)
        doc_path = str(docs[0])
        print(f"Auto-detected: {doc_path}")

    print(f"Reading: {doc_path}\n")
    blocks = extract_blocks(doc_path)
    print_summary(blocks)
    save_blocks(blocks)