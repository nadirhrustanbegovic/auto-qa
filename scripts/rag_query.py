#!/usr/bin/env python3
import argparse
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


def _resolve_path(path: str, base_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _embed_text(host: str, model: str, text: str) -> List[float]:
    url = host.rstrip("/") + "/api/embed"
    body = {"model": model, "input": text}
    r = requests.post(url, json=body, timeout=180)
    r.raise_for_status()
    data = r.json()
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError(f"Unexpected embed response: {data}")
    emb = embeddings[0]
    if not isinstance(emb, list):
        raise RuntimeError(f"Unexpected embedding vector: {type(emb)}")
    return emb


def _chat(host: str, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    url = host.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    r = requests.post(url, json=body, timeout=300)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected chat response: {data}")
    return content.strip()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0:
        return -1.0
    return dot / denom


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    p = argparse.ArgumentParser()
    p.add_argument("question", help="User question")
    p.add_argument("--db-path", default="data/vector_store.db", help="SQLite vector store file")
    p.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--embed-model", default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    p.add_argument("--chat-model", default=os.getenv("OLLAMA_CHAT_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b")))
    p.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--output", default="", help="Optional JSON output path")
    args = p.parse_args()

    db_path = _resolve_path(args.db_path, root)
    if not db_path.exists():
        raise SystemExit(f"Vector store not found: {db_path}. Run rag_ingest.py first.")

    query_embedding = _embed_text(args.host, args.embed_model, args.question)
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT id, source, chunk_index, text, embedding_json FROM chunks"
        ).fetchall()
    finally:
        conn.close()

    scored: List[Dict[str, Any]] = []
    for row in rows:
        _, source, chunk_index, text, emb_json = row
        emb = json.loads(emb_json)
        score = _cosine_similarity(query_embedding, emb)
        scored.append(
            {
                "source": source,
                "chunk_index": chunk_index,
                "text": text,
                "score": score,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: max(1, args.k)]

    contexts: List[Dict[str, Any]] = []
    for i, item in enumerate(top):
        contexts.append(
            {
                "rank": i + 1,
                "source": item["source"],
                "chunk_index": str(item["chunk_index"]),
                "score": item["score"],
                "text": item["text"],
            }
        )

    context_block = "\n\n".join(
        [
            f"[{c['rank']}] source={c['source']} chunk={c['chunk_index']}\n{c['text']}"
            for c in contexts
        ]
    )

    system_prompt = (
        "You are a retrieval-augmented assistant. "
        "Answer only using the provided context. "
        "If the context is insufficient, say you do not have enough information. "
        "Cite source IDs like [1], [2] in the answer."
    )
    user_prompt = f"Question:\n{args.question}\n\nContext:\n{context_block}"
    answer = _chat(
        host=args.host,
        model=args.chat_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    out = {
        "question": args.question,
        "answer": answer,
        "sources": [
            {
                "rank": c["rank"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "score": c["score"],
            }
            for c in contexts
        ],
    }

    if args.output:
        out_path = _resolve_path(args.output, root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
