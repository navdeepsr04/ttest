# knowledge_base/vector_store.py
"""
ChromaDB interface for the requirements knowledge base.

All other modules use this — never ChromaDB directly.

Operations:
  build()   — embed all requirements and store them
  search()  — find top-k requirements matching a query text
  get()     — fetch a requirement by ID
  stats()   — collection info
  rebuild() — wipe and rebuild (use when document changes)
"""

import json
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from knowledge_base.embedder import embed_requirements, embed_texts

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

CHROMA_PATH       = "data/chroma_db"
COLLECTION_NAME   = "requirements"
TOP_K_DEFAULT     = 5      # how many requirements to return per search
MIN_SCORE         = 0.30   # minimum cosine similarity to be considered relevant
                           # 0.0 = no filter, 1.0 = exact match only
                           # 0.30 works well in practice


# ── client singleton ──────────────────────────────────────────────────────────

def get_client() -> chromadb.PersistentClient:
    """Get or create ChromaDB persistent client."""
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.PersistentClient):
    """Get or create the requirements collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )


# ── build ─────────────────────────────────────────────────────────────────────

def build(
    requirements_path: str = "data/requirements_clean.json",
    force_rebuild:     bool = False,
) -> int:
    """
    Embed all requirements and store them in ChromaDB.

    Args:
        requirements_path : path to requirements_clean.json
        force_rebuild     : if True, wipe existing collection first

    Returns:
        count of requirements stored
    """
    client     = get_client()
    collection = get_collection(client)

    # check if already built
    existing = collection.count()
    if existing > 0 and not force_rebuild:
        print(f"  Collection already has {existing} items.")
        print(f"  Use force_rebuild=True to rebuild from scratch.")
        return existing

    # wipe if rebuilding
    if force_rebuild and existing > 0:
        print(f"  Wiping existing collection ({existing} items)...")
        client.delete_collection(COLLECTION_NAME)
        collection = get_collection(client)

    # embed requirements
    print(f"  Building knowledge base from {requirements_path}...")
    ids, embeddings, metadatas = embed_requirements(requirements_path)

    if not ids:
        print("  No requirements to store.")
        return 0

    # store in ChromaDB
    # ChromaDB wants documents (text) alongside embeddings
    # we store the title+description as the document text
    documents = []
    for meta in metadatas:
        req = json.loads(meta["full_json"])
        doc = f"{req.get('title', '')}. {req.get('description', '')}"
        documents.append(doc)

    # print(f"  Storing {len(ids)} requirements in ChromaDB...")

    # # ChromaDB has a limit per upsert — batch it
    # batch_size = 50
    # for i in range(0, len(ids), batch_size):
    #     collection.upsert(
    #         ids         = ids[i : i + batch_size],
    #         embeddings  = embeddings[i : i + batch_size],
    #         metadatas   = metadatas[i : i + batch_size],
    #         documents   = documents[i : i + batch_size],
    #     )

    # count = collection.count()
    # print(f"  Stored {count} requirements in ChromaDB at {CHROMA_PATH}")
    # return count          # error 

    # deduplicate IDs before storing — safety net
    seen_ids = {}
    clean_ids, clean_embeddings, clean_metadatas, clean_documents = [], [], [], []

    for uid, emb, meta, doc in zip(ids, embeddings, metadatas, documents):
        if uid in seen_ids:
            # make unique by appending a counter
            counter = seen_ids[uid] + 1
            seen_ids[uid] = counter
            uid = f"{uid}_{counter}"
        else:
            seen_ids[uid] = 0

        clean_ids.append(uid)
        clean_embeddings.append(emb)
        clean_metadatas.append(meta)
        clean_documents.append(doc)

    if len(clean_ids) != len(ids):
        print(f"  ⚠ Deduplicated {len(ids) - len(clean_ids)} duplicate IDs")

    print(f"  Storing {len(clean_ids)} requirements in ChromaDB...")

    batch_size = 50
    for i in range(0, len(clean_ids), batch_size):
        collection.upsert(
            ids        = clean_ids[i : i + batch_size],
            embeddings = clean_embeddings[i : i + batch_size],
            metadatas  = clean_metadatas[i : i + batch_size],
            documents  = clean_documents[i : i + batch_size],
        )


# ── search ────────────────────────────────────────────────────────────────────

def search(
    query_text:   str,
    top_k:        int   = TOP_K_DEFAULT,
    min_score:    float = MIN_SCORE,
    category:     str   = None,   # filter by category if provided
) -> list[dict]:
    """
    Find the most relevant requirements for a given query text.

    Args:
        query_text : text to search with (function summary, code snippet, etc.)
        top_k      : max number of results to return
        min_score  : minimum similarity score (0-1)
        category   : optional filter e.g. "validation", "api", "functional"

    Returns:
        list of dicts, each containing:
          id          : requirement ID
          score       : cosine similarity (0-1, higher = more relevant)
          title       : requirement title
          requirement : full requirement dict
    """
    client     = get_client()
    collection = get_collection(client)

    if collection.count() == 0:
        raise RuntimeError(
            "Requirements collection is empty. Run vector_store.build() first."
        )

    # embed the query
    query_embeddings = embed_texts([query_text])
    if not query_embeddings:
        return []

    # build where filter if category specified
    where = {"category": category} if category else None

    # query ChromaDB
    results = collection.query(
        query_embeddings = query_embeddings,
        n_results        = min(top_k, collection.count()),
        where            = where,
        include          = ["metadatas", "distances", "documents"],
    )

    # ChromaDB returns distances (lower = more similar for cosine)
    # convert to similarity scores (higher = more similar)
    hits = []
    ids        = results["ids"][0]
    distances  = results["distances"][0]
    metadatas  = results["metadatas"][0]

    for req_id, distance, meta in zip(ids, distances, metadatas):
        # cosine distance → cosine similarity
        score = 1.0 - distance

        if score < min_score:
            continue

        # parse full requirement from stored JSON
        try:
            requirement = json.loads(meta["full_json"])
        except Exception:
            requirement = {}

        hits.append({
            "id":          req_id,
            "score":       round(score, 4),
            "title":       meta.get("title", ""),
            "category":    meta.get("category", ""),
            "module":      meta.get("module", ""),
            "requirement": requirement,
        })

    # sort by score descending
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits


# ── get by ID ─────────────────────────────────────────────────────────────────

def get_by_id(req_id: str) -> dict | None:
    """Fetch a single requirement by its ID."""
    client     = get_client()
    collection = get_collection(client)

    results = collection.get(
        ids     = [req_id],
        include = ["metadatas"],
    )

    if not results["ids"]:
        return None

    meta = results["metadatas"][0]
    try:
        return json.loads(meta["full_json"])
    except Exception:
        return None


# ── stats ─────────────────────────────────────────────────────────────────────

def stats() -> dict:
    """Return info about the current collection."""
    client     = get_client()
    collection = get_collection(client)
    count      = collection.count()

    result = {
        "collection":  COLLECTION_NAME,
        "count":       count,
        "chroma_path": CHROMA_PATH,
        "min_score":   MIN_SCORE,
        "top_k":       TOP_K_DEFAULT,
    }

    # sample a few IDs
    if count > 0:
        sample = collection.get(limit=5, include=[])
        result["sample_ids"] = sample["ids"]

    return result


# ── run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # parse simple CLI args
    rebuild = "--rebuild" in sys.argv
    test    = "--test"    in sys.argv

    print(f"\n{'='*55}")
    print(f"  Vector Store")
    print(f"{'='*55}\n")

    # ── build ──────────────────────────────────────────────────────────────
    print("[1] Building knowledge base...")
    count = build(force_rebuild=rebuild)

    # ── stats ──────────────────────────────────────────────────────────────
    print(f"\n[2] Collection stats:")
    s = stats()
    for k, v in s.items():
        print(f"    {k:15}: {v}")

    # ── test search ────────────────────────────────────────────────────────
    if test or True:   # always run a quick search test
        print(f"\n[3] Test searches:\n")

        test_queries = [
            # query from code side — simulates what ast_parser would produce
            "def run_tsne(perplexity, n_components, n_iter, features)",
            "KlarfReaderNode execute read klarf file input_file_path",
            "validation perplexity must be less than group size",
            "output tsne_x tsne_y embedding coordinates float columns",
        ]

        for query in test_queries:
            print(f"  Query: \"{query[:60]}\"")
            results = search(query, top_k=3)
            if results:
                for r in results:
                    print(f"    [{r['score']:.3f}] {r['id']:10} {r['title'][:50]}")
            else:
                print(f"    no results above min_score={MIN_SCORE}")
            print()