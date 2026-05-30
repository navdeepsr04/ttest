# preprocess/doc_pipeline.py
"""
End-to-end pipeline for one document.
Run this whenever a new doc is added or an existing one changes.

Usage:
  python -m preprocess.doc_pipeline --input data/input/tsne_srs.docx --doc-id tsne
  python -m preprocess.doc_pipeline --input data/input/klarf_srs.docx --doc-id klarf
  python -m preprocess.doc_pipeline --input data/input/pca_srs.docx   --doc-id pca --node PCANode
"""

import argparse
import json
import sys
from pathlib import Path

from preprocess.doc_reader  import extract_blocks
from preprocess.chunker     import group_into_chunks
from preprocess.extractor_v2 import run as run_extractor
from knowledge_base.tc_store import upsert_doc, stats


def run_pipeline(
    input_path: str,
    doc_id:     str,
    node:       str = None,
) -> list[dict]:

    print(f"\n{'='*55}")
    print(f"  Doc Pipeline")
    print(f"  input  : {input_path}")
    print(f"  doc_id : {doc_id}")
    print(f"  node   : {node or 'auto-detect'}")
    print(f"{'='*55}\n")

    # paths
    base          = Path("data") / doc_id
    base.mkdir(parents=True, exist_ok=True)
    blocks_path   = str(base / "blocks.json")
    chunks_path   = str(base / "chunks.json")
    tc_path       = str(base / "test_cases.json")

    # step 1 — read doc
    print("[1/4] Reading document...")
    blocks = extract_blocks(input_path)
    Path(blocks_path).write_text(
        json.dumps(blocks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  {len(blocks)} blocks extracted\n")

    # step 2 — chunk
    print("[2/4] Chunking...")
    chunks = group_into_chunks(blocks)
    Path(chunks_path).write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  {len(chunks)} chunks\n")

    # step 3 — extract test cases
    print("[3/4] Extracting test cases...")
    test_cases = run_extractor(
        chunks_path = chunks_path,
        output_path = tc_path,
    )
    print(f"  {len(test_cases)} test cases saved to {tc_path}\n")

    # step 4 — store in ChromaDB
    print("[4/4] Storing in ChromaDB...")
    upsert_doc(doc_id, test_cases, node=node)

    # final stats
    s = stats()
    print(f"\n{'='*55}")
    print(f"  Done — {doc_id} loaded")
    print(f"{'='*55}")
    print(f"  Total test cases in DB : {s['total_test_cases']}")
    print(f"  Total documents in DB  : {s['total_documents']}")
    print(f"\n  All documents:")
    for doc in s["documents"]:
        nodes_str = ", ".join(doc["nodes"])
        print(f"    {doc['doc_id']:20} {doc['count']:3} cases  nodes: {nodes_str}")

    return test_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="path to .docx file")
    parser.add_argument("--doc-id", required=True, help="unique id for this doc e.g. tsne, klarf")
    parser.add_argument("--node",   default=None,  help="node class name override e.g. TSNENode")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"File not found: {args.input}")
        sys.exit(1)

    run_pipeline(args.input, args.doc_id, args.node)