# knowledge_base/embedder.py
"""
Embeds requirement chunks into ChromaDB.
Run this once after requirements_clean.json is ready.
On new document version, run again to rebuild the store.
"""

import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ── embedding config ──────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, cheap + fast
BATCH_SIZE      = 20    # embed 20 chunks per API call
MAX_RETRIES     = 3


# ── text builder ──────────────────────────────────────────────────────────────

def requirement_to_text(req: dict) -> str:
    """
    Convert a requirement dict into a rich text string for embedding.

    We include title, description, inputs, error_cases and business_rules
    because all of these carry semantic meaning that helps matching.
    We do NOT include test_hints — those are for test generation, not search.
    """
    parts = []

    # core identity
    parts.append(f"ID: {req.get('id', '')}")
    parts.append(f"Category: {req.get('category', '')}")
    parts.append(f"Module: {req.get('module', '')}")
    parts.append(f"Title: {req.get('title', '')}")
    parts.append(f"Description: {req.get('description', '')}")

    # inputs — parameter names and types carry strong signal
    inputs = req.get("inputs", [])
    if inputs:
        input_strs = []
        for inp in inputs:
            s = f"{inp.get('name', '')}({inp.get('type', '')})"
            if inp.get("validation"):
                s += f" constraint: {inp['validation']}"
            if inp.get("default") is not None:
                s += f" default: {inp['default']}"
            input_strs.append(s)
        parts.append(f"Inputs: {', '.join(input_strs)}")

    # error cases — validation constraints are key for matching
    error_cases = req.get("error_cases", [])
    if error_cases:
        error_strs = [ec.get("trigger", "") for ec in error_cases]
        parts.append(f"Error conditions: {'; '.join(error_strs)}")

    # business rules
    rules = req.get("business_rules", [])
    if rules:
        parts.append(f"Business rules: {' | '.join(rules)}")

    return "\n".join(parts)


# ── embedding with retry ──────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using OpenAI.
    Returns list of embedding vectors.
    """
    wait_times = [0, 5, 15]

    for attempt in range(MAX_RETRIES):
        wait = wait_times[attempt]
        if wait > 0:
            print(f"    retrying in {wait}s...", end=" ", flush=True)
            time.sleep(wait)
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"embedding error: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
    return []


# ── main embed function ───────────────────────────────────────────────────────

def embed_requirements(
    requirements_path: str = "data/requirements_clean.json",
) -> tuple[list[str], list[list[float]], list[dict]]:
    """
    Load requirements, convert to text, embed all of them.

    Returns:
        ids         : list of requirement IDs  e.g. ["FR-001", "VAL-002"]
        embeddings  : list of embedding vectors
        metadatas   : list of metadata dicts for ChromaDB
    """
    path = Path(requirements_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{requirements_path} not found. "
            "Run extractor.py and validator.py first."
        )

    requirements = json.loads(path.read_text(encoding="utf-8"))
    print(f"  Loaded {len(requirements)} requirements from {requirements_path}")

    ids        = []
    texts      = []
    metadatas  = []

    for req in requirements:
        req_id = req.get("id", f"REQ-{len(ids)}")
        text   = requirement_to_text(req)

        ids.append(req_id)
        texts.append(text)
        metadatas.append({
            "id":          req_id,
            "category":    req.get("category", ""),
            "module":      req.get("module", ""),
            "title":       req.get("title", ""),
            "priority":    req.get("priority", ""),
            # store full requirement as JSON string for retrieval
            "full_json":   json.dumps(req),
        })

    # embed in batches
    print(f"  Embedding {len(texts)} requirements in batches of {BATCH_SIZE}...")
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch      = texts[i : i + BATCH_SIZE]
        batch_num  = (i // BATCH_SIZE) + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"    Batch {batch_num}/{total_batches} ({len(batch)} items)...",
              end=" ", flush=True)
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
        print("done")

    print(f"  Embedded {len(all_embeddings)} requirements total")
    return ids, all_embeddings, metadatas


# ── run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  Embedder — building knowledge base")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"{'='*55}\n")

    ids, embeddings, metadatas = embed_requirements()

    # quick sanity check
    print(f"\n  Sanity check:")
    print(f"    IDs count        : {len(ids)}")
    print(f"    Embeddings count : {len(embeddings)}")
    print(f"    Vector dimension : {len(embeddings[0]) if embeddings else 0}")
    print(f"    Metadatas count  : {len(metadatas)}")
    print(f"\n  Sample embedding for {ids[0]}:")
    print(f"    first 5 dims: {embeddings[0][:5]}")
    print(f"\n  Ready to store → run vector_store.py next")