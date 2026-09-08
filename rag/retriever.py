from __future__ import annotations

import os

import voyageai
from fastembed import SparseTextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

COLLECTION = "ncert"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

_vc: voyageai.Client | None = None
_qc: QdrantClient | None = None
_sparse_model: SparseTextEmbedding | None = None


def _clients() -> tuple[voyageai.Client, QdrantClient, SparseTextEmbedding]:
    global _vc, _qc, _sparse_model
    if _vc is None:
        _vc = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        _qc = QdrantClient(url=QDRANT_URL)
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _vc, _qc, _sparse_model  # type: ignore[return-value]


class Chunk(BaseModel):
    text: str
    source: str
    score: float


async def retrieve(query: str, top_k: int = 5) -> list[Chunk]:
    vc, qc, sparse_model = _clients()

    dense_vec: list[float] = vc.embed(
        [query], model="voyage-3-lite", input_type="query"
    ).embeddings[0]

    sparse_emb = next(sparse_model.embed([query]))
    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )

    results = qc.query_points(
        collection_name=COLLECTION,
        prefetch=[
            {"using": "dense", "query": dense_vec, "limit": top_k * 2},
            {"using": "sparse", "query": sparse_vec, "limit": top_k * 2},
        ],
        query={"fusion": "rrf"},
        limit=top_k,
    )

    return [
        Chunk(
            text=r.payload["text"],
            source=r.payload["source"],
            score=r.score,
        )
        for r in results.points
    ]
