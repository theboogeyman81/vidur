from __future__ import annotations

import json
import os
import pathlib
import uuid

import voyageai
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

CHUNKS_PATH = pathlib.Path("rag/data/chunks.json")
COLLECTION = "ncert"
DENSE_DIM = 512
BATCH_SIZE = 50

_vc: voyageai.Client | None = None
_qc: QdrantClient | None = None
_sparse_model: SparseTextEmbedding | None = None


def _clients() -> tuple[voyageai.Client, QdrantClient, SparseTextEmbedding]:
    global _vc, _qc, _sparse_model
    if _vc is None:
        _vc = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        _qc = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _vc, _qc, _sparse_model  # type: ignore[return-value]


def _ensure_collection() -> None:
    _, qc, _ = _clients()
    existing = {c.name for c in qc.get_collections().collections}
    if COLLECTION not in existing:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        print(f"Created Qdrant collection '{COLLECTION}'")


def _embed_dense(texts: list[str]) -> list[list[float]]:
    vc, _, _ = _clients()
    result = vc.embed(texts, model="voyage-3-lite", input_type="document")
    return result.embeddings


def _embed_sparse(texts: list[str]) -> list[SparseVector]:
    _, _, sparse_model = _clients()
    embeddings = list(sparse_model.embed(texts))
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ]


def upsert_chunks(chunks: list[dict]) -> None:
    _, qc, _ = _clients()
    _ensure_collection()
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        dense_vecs = _embed_dense(texts)
        sparse_vecs = _embed_sparse(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_vecs[j], "sparse": sparse_vecs[j]},
                payload={"text": batch[j]["text"], "source": batch[j]["source"]},
            )
            for j in range(len(batch))
        ]
        qc.upsert(collection_name=COLLECTION, points=points)
        print(f"Upserted {min(i + BATCH_SIZE, total)}/{total} chunks")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run `uv run python -m rag.loader` first."
        )
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    upsert_chunks(chunks)
    print("Done.")
