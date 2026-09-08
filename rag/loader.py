from __future__ import annotations

import json
import pathlib

import tiktoken
from pypdf import PdfReader

DATA_DIR = pathlib.Path("rag/data/ncert")
CHUNKS_PATH = pathlib.Path("rag/data/chunks.json")
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50

_enc = tiktoken.get_encoding("cl100k_base")


def _pdf_to_text(path: pathlib.Path) -> str:
    reader = PdfReader(path)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def _chunk(text: str, source: str) -> list[dict]:
    tokens = _enc.encode(text)
    chunks: list[dict] = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk_text = _enc.decode(tokens[start:end])
        chunks.append({"text": chunk_text, "source": source, "tokens": end - start})
        if end == len(tokens):
            break
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks


def load_chunks() -> list[dict]:
    all_chunks: list[dict] = []
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}. Add NCERT chapter PDFs there.")
    for pdf in pdfs:
        text = _pdf_to_text(pdf)
        all_chunks.extend(_chunk(text, pdf.name))
    return all_chunks


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_count = len(list(DATA_DIR.glob("*.pdf")))
    print(f"Wrote {len(chunks)} chunks from {pdf_count} PDFs → {CHUNKS_PATH}")
