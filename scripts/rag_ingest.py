#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv


def _resolve_path(path: str, base_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _iter_docs(docs_dir: Path) -> List[Path]:
    allowed = {".md", ".txt"}
    files: List[Path] = []
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed:
            files.append(p)
    files.sort()
    return files


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _embed_texts(host: str, model: str, texts: List[str]) -> List[List[float]]:
    url = host.rstrip("/") + "/api/embed"
    out: List[List[float]] = []
    for t in texts:
        body = {"model": model, "input": t}
        r = requests.post(url, json=body, timeout=180)
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise RuntimeError(f"Unexpected embed response: {data}")
        emb = embeddings[0]
        if not isinstance(emb, list):
            raise RuntimeError(f"Unexpected embedding vector: {type(emb)}")
        out.append(emb)
    return out


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_json TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _build_rows(
    docs_dir: Path, chunk_size: int, overlap: int
) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, str]] = []

    files = _iter_docs(docs_dir)
    for file_path in files:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
        rel = file_path.relative_to(docs_dir).as_posix()
        for idx, chunk in enumerate(chunks):
            key = f"{rel}:{idx}:{hashlib.sha1(chunk.encode('utf-8')).hexdigest()[:12]}"
            ids.append(key)
            docs.append(chunk)
            metas.append({"source": rel, "chunk_index": str(idx)})
    return ids, docs, metas


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    p = argparse.ArgumentParser()
    p.add_argument("--docs-dir", default="data/docs", help="Folder with .md/.txt docs")
    p.add_argument("--db-path", default="data/vector_store.db", help="SQLite vector store file")
    p.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--embed-model", default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--chunk-overlap", type=int, default=120)
    p.add_argument("--reset", action="store_true", help="Delete existing stored chunks first")
    args = p.parse_args()

    docs_dir = _resolve_path(args.docs_dir, root)
    db_path = _resolve_path(args.db_path, root)

    if not docs_dir.exists():
        raise SystemExit(f"Docs dir not found: {docs_dir}")

    ids, docs, metas = _build_rows(
        docs_dir=docs_dir, chunk_size=args.chunk_size, overlap=args.chunk_overlap
    )
    if not docs:
        raise SystemExit(f"No chunks created from docs in: {docs_dir}")

    embeddings = _embed_texts(args.host, args.embed_model, docs)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_schema(conn)
        if args.reset:
            conn.execute("DELETE FROM chunks")
            conn.commit()

        rows = []
        for i in range(len(ids)):
            rows.append(
                (
                    ids[i],
                    metas[i]["source"],
                    int(metas[i]["chunk_index"]),
                    docs[i],
                    json.dumps(embeddings[i], ensure_ascii=False),
                )
            )
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (id, source, chunk_index, text, embedding_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    print(
        f"Indexed {len(docs)} chunks from {docs_dir} into SQLite vector store at {db_path}."
    )


if __name__ == "__main__":
    main()
