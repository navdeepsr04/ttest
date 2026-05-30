# knowledge_base/tc_store.py
"""
Stores and retrieves test cases in ChromaDB.
Single collection "test_cases" — all docs, all nodes.

Each entry tagged with:
  doc_id  : source document identifier  e.g. "tsne_srs"
  node    : node class name             e.g. "TSNENode"
  tc_id   : original test case ID       e.g. "TC-001"

Operations:
  upsert_doc(doc_id, node, test_cases)  — add/update a doc's test cases
  search(node, query, top_k)            — find relevant test cases
  get_by_node(node)                     — get ALL test cases for a node
  delete_doc(doc_id)                    — remove a doc's test cases
  list_docs()                           — show all loaded documents
"""

import json
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from knowledge_base.embedder import embed_texts

load_dotenv()

CHROMA_PATH      = "data/chroma_db"
COLLECTION_NAME  = "test_cases"
MIN_SCORE        = 0.30
TOP_K            = 10


# ── client ────────────────────────────────────────────────────────────────────

def get_collection():
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path     = CHROMA_PATH,
        settings = Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},
    )


# ── build embeddable text from a test case ────────────────────────────────────

def tc_to_text(tc: dict) -> str:
    """
    Convert a test case to a rich text string for embedding.
    This is what gets semantically searched.
    """
    parts = [
        f"node: {tc.get('node', '')}",
        f"method: {tc.get('method', 'execute')}",
        f"description: {tc.get('description', '')}",
        f"category: {tc.get('category', '')}",
    ]

    inp = tc.get("input", {})
    if inp:
        parts.append(f"inputs: {' '.join(f'{k}={v}' for k, v in inp.items())}")

    exp = tc.get("expected", {})
    if exp.get("error"):
        parts.append(f"error: {exp['error']}")
    if exp.get("result"):
        parts.append(f"result: {exp['result']}")
    if exp.get("message"):
        parts.append(f"message: {exp['message']}")

    dtypes = tc.get("data_types", {})
    if dtypes:
        parts.append(f"types: {' '.join(f'{k}:{v}' for k, v in dtypes.items())}")

    return " | ".join(parts)


def make_chroma_id(doc_id: str, tc_id: str) -> str:
    """Unique ID for ChromaDB entry."""
    raw = f"{doc_id}::{tc_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── upsert ────────────────────────────────────────────────────────────────────

def upsert_doc(
    doc_id:     str,
    test_cases: list[dict],
    node:       str = None,   # override node tag if doc covers one node
) -> int:
    """
    Add or update all test cases from one document.
    If doc already exists → replaces its entries.

    Returns count of test cases stored.
    """
    if not test_cases:
        print(f"  No test cases to store for {doc_id}")
        return 0

    collection = get_collection()

    # delete existing entries for this doc first (clean update)
    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"  Removed {len(existing['ids'])} old entries for {doc_id}")

    # build texts for embedding
    texts     = [tc_to_text(tc) for tc in test_cases]
    ids       = [make_chroma_id(doc_id, tc.get("id", f"TC-{i}"))
                 for i, tc in enumerate(test_cases)]

    metadatas = []
    for tc in test_cases:
        node_tag = node or tc.get("node") or "unknown"
        metadatas.append({
            "doc_id":   doc_id,
            "node":     node_tag,
            "tc_id":    tc.get("id", ""),
            "category": tc.get("category", ""),
            "outcome":  tc.get("expected", {}).get("outcome", ""),
            "tc_json":  json.dumps(tc),
        })

    # embed in batches
    print(f"  Embedding {len(texts)} test cases...", end=" ", flush=True)
    embeddings = embed_texts(texts)
    print("done")

    # upsert in batches of 50
    batch = 50
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids        = ids[i:i+batch],
            embeddings = embeddings[i:i+batch],
            metadatas  = metadatas[i:i+batch],
            documents  = texts[i:i+batch],
        )

    count = len(test_cases)
    print(f"  Stored {count} test cases for doc='{doc_id}'")
    return count


# ── search ────────────────────────────────────────────────────────────────────

def search(
    query:     str,
    node:      str   = None,
    doc_id:    str   = None,
    top_k:     int   = TOP_K,
    min_score: float = MIN_SCORE,
) -> list[dict]:
    """
    Search for relevant test cases.

    Args:
        query     : text to search (function summary, method name, etc.)
        node      : filter by node class name
        doc_id    : filter by document
        top_k     : max results
        min_score : minimum cosine similarity

    Returns list of test case dicts with score attached.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # build where filter
    where = {}
    if node and doc_id:
        where = {"$and": [{"node": node}, {"doc_id": doc_id}]}
    elif node:
        where = {"node": node}
    elif doc_id:
        where = {"doc_id": doc_id}

    query_emb = embed_texts([query])
    if not query_emb:
        return []

    n = min(top_k, collection.count())
    results = collection.query(
        query_embeddings = query_emb,
        n_results        = n,
        where            = where if where else None,
        include          = ["metadatas", "distances"],
    )

    hits = []
    for chroma_id, distance, meta in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        score = round(1.0 - distance, 4)
        if score < min_score:
            continue
        try:
            tc = json.loads(meta["tc_json"])
            tc["_score"]  = score
            tc["_doc_id"] = meta["doc_id"]
            hits.append(tc)
        except Exception:
            pass

    hits.sort(key=lambda x: x["_score"], reverse=True)
    return hits


# ── get all by node ───────────────────────────────────────────────────────────

def get_by_node(node: str, doc_id: str = None) -> list[dict]:
    """
    Get ALL test cases for a node (no embedding search, direct filter).
    Use this when you want every test case for a node, not just top-k.
    """
    collection = get_collection()

    where = {"node": node}
    if doc_id:
        where = {"$and": [{"node": node}, {"doc_id": doc_id}]}

    results = collection.get(
        where   = where,
        include = ["metadatas"],
    )

    test_cases = []
    for meta in results["metadatas"]:
        try:
            tc = json.loads(meta["tc_json"])
            tc["_doc_id"] = meta["doc_id"]
            test_cases.append(tc)
        except Exception:
            pass

    return test_cases


# ── delete doc ────────────────────────────────────────────────────────────────

def delete_doc(doc_id: str) -> int:
    """Remove all test cases for a document."""
    collection = get_collection()
    existing   = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"  Deleted {len(existing['ids'])} entries for doc='{doc_id}'")
        return len(existing["ids"])
    print(f"  No entries found for doc='{doc_id}'")
    return 0


# ── list docs ─────────────────────────────────────────────────────────────────

def list_docs() -> list[dict]:
    """Show all documents loaded into the store."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    summary = {}
    for meta in results["metadatas"]:
        doc_id = meta.get("doc_id", "unknown")
        node   = meta.get("node",   "unknown")
        if doc_id not in summary:
            summary[doc_id] = {"doc_id": doc_id, "nodes": set(), "count": 0}
        summary[doc_id]["nodes"].add(node)
        summary[doc_id]["count"] += 1

    docs = []
    for v in summary.values():
        v["nodes"] = sorted(v["nodes"])
        docs.append(v)

    return sorted(docs, key=lambda x: x["doc_id"])


def stats() -> dict:
    collection = get_collection()
    docs       = list_docs()
    return {
        "total_test_cases": collection.count(),
        "total_documents":  len(docs),
        "documents":        docs,
    }