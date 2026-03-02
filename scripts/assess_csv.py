#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

DECISION_OUTPUT_SPEC = """
Return only valid JSON with exactly this schema:
{
  "pass": 1 or 0,
  "justification": "short reason that includes a direct quote",
  "source": {
    "type": "rag" or "prompt",
    "quote": "direct quote used as evidence"
  }
}

Rules:
- pass=1 means pass, pass=0 means fail.
- justification must include at least one direct quote from either policy_context (RAG) or this prompt text.
- source.type must match where the quote came from.
- source.quote must be the exact quoted text used as evidence.
"""


def _set_csv_field_limit() -> None:
    # Some datasets store long JSON transcripts in a single CSV cell.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _get_row_value(row: Dict[str, Any], *keys: str, default: str = "") -> str:
    if not row:
        return default
    keymap = {str(k).strip().lower(): v for k, v in row.items()}
    for k in keys:
        v = keymap.get(str(k).strip().lower())
        if v is not None:
            return str(v)
    return default


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _parse_blocked_words(value: str) -> List[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    # JSON array support
    if s.startswith("[") and s.endswith("]"):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
    # comma-separated
    return [w.strip() for w in s.split(",") if w.strip()]


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _find_blocked_words(message: str, blocked_words: List[str]) -> List[str]:
    msg = _normalize_text_for_match(message)
    found: List[str] = []
    for w in blocked_words:
        ww = _normalize_text_for_match(w)
        if not ww:
            continue
        # try word-boundary for simple tokens; fallback to substring
        if re.fullmatch(r"[a-z0-9']+", ww):
            if re.search(rf"\b{re.escape(ww)}\b", msg):
                found.append(w)
        else:
            if ww in msg:
                found.append(w)
    # unique, stable order
    seen = set()
    out: List[str] = []
    for x in found:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _parse_conversation(raw: Any | None) -> Any:
    # conversation is expected to be JSON (object or array) but may be double-escaped
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    for _ in range(2):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # attempt un-escape if it's a quoted JSON string
            try:
                s = json.loads(f'"{s}"')
            except Exception:
                break
    return None


def _extract_last_agent_message(conv: Any) -> Optional[str]:
    """Supports common shapes:
    - list of {role/content}
    - dict with "messages": [...]
    Agent roles supported: assistant, agent
    """
    if conv is None:
        return None

    if isinstance(conv, dict) and "messages" in conv and isinstance(conv["messages"], list):
        conv = conv["messages"]

    if isinstance(conv, dict):
        role = str(conv.get("role") or conv.get("sender") or conv.get("author") or "").lower()
        content = conv.get("content") or conv.get("text") or conv.get("message") or conv.get("body")
        if role in ("assistant", "agent") and isinstance(content, str) and content.strip():
            return content.strip()

    if isinstance(conv, list):
        for item in reversed(conv):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or item.get("sender") or item.get("author") or "").lower()
            content = (
                item.get("content")
                or item.get("text")
                or item.get("message")
                or item.get("body")
                or item.get("message_text")
            )
            if role in ("assistant", "agent") and isinstance(content, str) and content.strip():
                return content.strip()
            typ = str(item.get("type") or item.get("message_type") or "").lower()
            if typ == "agent" and isinstance(content, str) and content.strip():
                return content.strip()

    return None


def _resolve_ollama_model(host: str, configured_model: str) -> str:
    requested = (configured_model or "").strip()
    if not requested:
        requested = "llama3.1:8b"
    try:
        resp = requests.get(host.rstrip("/") + "/api/tags", timeout=20)
        resp.raise_for_status()
        models = resp.json().get("models") or []
        names = [str(m.get("name")) for m in models if isinstance(m, dict) and m.get("name")]
        if not names:
            return requested
        if requested in names:
            return requested
        preferred = ["llama3.1:8b", "qwen3:8b", "llama3.2:3b"]
        for cand in preferred:
            if cand in names:
                print(f"[warn] OLLAMA_MODEL '{requested}' not found. Falling back to '{cand}'.")
                return cand
        print(f"[warn] OLLAMA_MODEL '{requested}' not found. Falling back to '{names[0]}'.")
        return names[0]
    except Exception:
        # Keep configured model if tags endpoint is unavailable.
        return requested


def _ollama_chat(
    host: str,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    force_json: bool = True,
) -> str:
    url = host.rstrip("/") + "/api/chat"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    body = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    if force_json:
        body["format"] = "json"
    r = requests.post(url, json=body, timeout=300)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected response shape: {data}")
    return content.strip()


def _safe_json_loads(s: str) -> Dict[str, Any]:
    """Best-effort parse of model output as JSON."""
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _normalize_decision_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {"pass": 0, "justification": "", "source": {"type": "", "quote": ""}}
    if not isinstance(obj, dict):
        return out

    # Primary desired fields.
    p = obj.get("pass")
    if isinstance(p, bool):
        out["pass"] = 1 if p else 0
    elif isinstance(p, (int, float)):
        out["pass"] = 1 if int(p) == 1 else 0
    elif isinstance(p, str):
        pv = p.strip().lower()
        if pv in ("1", "pass", "true"):
            out["pass"] = 1
        else:
            out["pass"] = 0

    just = obj.get("justification")
    if isinstance(just, str):
        out["justification"] = just.strip()

    src = obj.get("source")
    if isinstance(src, dict):
        st = src.get("type")
        sq = src.get("quote")
        if isinstance(st, str):
            out["source"]["type"] = st.strip().lower()
        if isinstance(sq, str):
            out["source"]["quote"] = sq.strip()

    # Backward-compatible mapping from old schema.
    if "overall" in obj and isinstance(obj.get("overall"), dict):
        ov = obj["overall"]
        label = str(ov.get("label", "")).strip().upper()
        out["pass"] = 1 if label == "PASS" else 0
        if not out["justification"]:
            out["justification"] = str(ov.get("reason", "")).strip()

    if not out["justification"]:
        out["justification"] = "No justification provided."

    if out["source"]["type"] not in ("rag", "prompt"):
        out["source"]["type"] = "prompt"
    if not out["source"]["quote"]:
        out["source"]["quote"] = ""
    return out


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


def _ollama_embed(host: str, model: str, text: str) -> List[float]:
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


def _retrieve_policy_context(
    db_path: str,
    host: str,
    embed_model: str,
    query: str,
    k: int,
) -> Dict[str, Any]:
    if not os.path.exists(db_path):
        raise RuntimeError(f"RAG DB not found: {db_path}")

    query_embedding = _ollama_embed(host, embed_model, query)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT source, chunk_index, text, embedding_json FROM chunks"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("RAG DB has no indexed chunks. Run scripts/rag_ingest.py first.")

    scored: List[Dict[str, Any]] = []
    for source, chunk_index, text, emb_json in rows:
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
    top = scored[: max(1, k)]

    contexts = []
    sources = []
    for i, item in enumerate(top):
        contexts.append(
            f"[{i+1}] source={item['source']} chunk={item['chunk_index']} score={item['score']:.4f}\n{item['text']}"
        )
        sources.append(
            {
                "rank": i + 1,
                "source": item["source"],
                "chunk_index": str(item["chunk_index"]),
                "score": item["score"],
            }
        )
    return {
        "policy_context": "\n\n".join(contexts),
        "policy_sources": sources,
    }


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(project_root, ".env"))
    _set_csv_field_limit()

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/input.csv", help="Path to input CSV")
    p.add_argument("--output", default="data/output.jsonl", help="Path to output .jsonl or .csv")
    p.add_argument("--prompt", default="prompts/sample_prompt_1.md", help="System prompt path")
    p.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    p.add_argument(
        "--embed-model",
        default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        help="Embedding model used for policy retrieval from local RAG DB.",
    )
    p.add_argument(
        "--policy-db",
        default=os.getenv("POLICY_DB_PATH", "data/vector_store.db"),
        help="Local SQLite vector DB path generated by scripts/rag_ingest.py.",
    )
    p.add_argument("--rag-k", type=int, default=3, help="Top-k policy chunks to attach per graded row.")
    p.add_argument(
        "--allow-missing-rag",
        action="store_true",
        help="Allow grading to continue when policy retrieval fails (default: fail row).",
    )
    p.add_argument(
        "--strict-decision-output",
        action="store_true",
        help="Require output to include pass/justification/source with a non-empty quote.",
    )
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.1")))
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "900")))
    p.add_argument(
        "--no-force-json",
        action="store_true",
        help="Disable Ollama JSON mode (enabled by default).",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Process a random sample of N rows from input CSV (0 = all rows)",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )
    args = p.parse_args()
    args.input = _resolve_path(args.input, project_root)
    args.output = _resolve_path(args.output, project_root)
    args.prompt = _resolve_path(args.prompt, project_root)
    args.policy_db = _resolve_path(args.policy_db, project_root)
    args.model = _resolve_ollama_model(args.host, args.model)

    system_prompt = _load_prompt(args.prompt)
    system_prompt = f"{system_prompt}\n\n{DECISION_OUTPUT_SPEC}"

    out_is_csv = args.output.lower().endswith(".csv")
    out_is_jsonl = args.output.lower().endswith(".jsonl")
    if not (out_is_csv or out_is_jsonl):
        raise SystemExit("Output must end with .jsonl or .csv")

    rows: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("Input CSV has no headers")
        for row in reader:
            rows.append(row)

    if args.sample_size < 0:
        raise SystemExit("--sample-size must be >= 0")
    if args.sample_size > 0 and len(rows) > args.sample_size:
        rng = random.Random(args.sample_seed)
        rows = rng.sample(rows, args.sample_size)

    results: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="grading"):
        scenario_id = _get_row_value(row, "scenario_id", "send_id", "id").strip() or None
        conv_raw = _get_row_value(row, "conversation", "conversation_json")
        preferred_tone = _get_row_value(row, "message_tone", "tone").strip()
        notes = _get_row_value(row, "notes", "company_notes").strip()
        persona = _get_row_value(row, "persona").strip()
        blocked_words = _parse_blocked_words(_get_row_value(row, "blocked_words", "blocklisted_words"))
        policy_query = _get_row_value(row, "policy_query", "policy", "policy_topic", "qa_policy").strip()

        conv = _parse_conversation(conv_raw)
        last_agent_message = _extract_last_agent_message(conv) or ""

        found_blocked = _find_blocked_words(last_agent_message, blocked_words)

        if not policy_query:
            policy_query = (
                f"preferred_tone: {preferred_tone}\n"
                f"notes: {notes}\n"
                f"persona: {persona}\n"
                f"last_agent_message: {last_agent_message}"
            )

        policy_context = ""
        policy_sources: List[Dict[str, Any]] = []
        rag_error = ""
        try:
            rag = _retrieve_policy_context(
                db_path=args.policy_db,
                host=args.host,
                embed_model=args.embed_model,
                query=policy_query,
                k=args.rag_k,
            )
            policy_context = rag["policy_context"]
            policy_sources = rag["policy_sources"]
        except Exception as e:
            rag_error = str(e)
            if not args.allow_missing_rag:
                results.append({
                    "scenario_id": scenario_id,
                    "preferred_message_tone": preferred_tone,
                    "blocked_words_found": found_blocked,
                    "last_agent_message": last_agent_message,
                    "policy_query": policy_query,
                    "policy_sources": [],
                    "error": f"RAG policy retrieval failed: {rag_error}",
                })
                continue

        user_payload = {
            "scenario_id": scenario_id,
            "preferred_message_tone": preferred_tone,
            "notes": notes,
            "persona": persona,
            "blocked_words_found": found_blocked,
            "last_agent_message": last_agent_message,
            "policy_query": policy_query,
            "policy_context": policy_context,
            "policy_sources": policy_sources,
        }

        try:
            llm_text = _ollama_chat(
                host=args.host,
                model=args.model,
                system_prompt=system_prompt,
                user_payload=user_payload,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                force_json=not args.no_force_json,
            )
            result_row = {
                "scenario_id": scenario_id,
                "preferred_message_tone": preferred_tone,
                "blocked_words_found": found_blocked,
                "last_agent_message": last_agent_message,
                "policy_query": policy_query,
                "policy_sources": policy_sources,
            }
            if rag_error:
                result_row["rag_warning"] = rag_error
            try:
                llm_json = _safe_json_loads(llm_text)
                decision = _normalize_decision_output(llm_json)
                if args.strict_decision_output and not decision["source"]["quote"]:
                    raise RuntimeError("Missing required source.quote in model output.")
                result_row["grades"] = llm_json
                result_row["decision"] = decision
            except Exception:
                # Keep raw model text so custom prompt outputs do not fail the run.
                result_row["grades_text"] = llm_text
            results.append(result_row)
        except Exception as e:
            results.append({
                "scenario_id": scenario_id,
                "preferred_message_tone": preferred_tone,
                "blocked_words_found": found_blocked,
                "last_agent_message": last_agent_message,
                "error": str(e),
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if out_is_jsonl:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return

    # Flatten for CSV
    fieldnames = [
        "scenario_id",
        "preferred_message_tone",
        "blocked_words_found",
        "policy_query",
        "policy_sources",
        "pass",
        "justification",
        "source_type",
        "source_quote",
        "error",
    ]

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            out: Dict[str, Any] = {k: "" for k in fieldnames}
            out["scenario_id"] = r.get("scenario_id")
            out["preferred_message_tone"] = r.get("preferred_message_tone", "")
            out["blocked_words_found"] = json.dumps(r.get("blocked_words_found", []), ensure_ascii=False)
            out["policy_query"] = r.get("policy_query", "")
            out["policy_sources"] = json.dumps(r.get("policy_sources", []), ensure_ascii=False)
            out["error"] = r.get("error", "")

            d = r.get("decision") or {}
            if isinstance(d, dict):
                out["pass"] = d.get("pass", "")
                out["justification"] = d.get("justification", "")
                src = d.get("source") or {}
                if isinstance(src, dict):
                    out["source_type"] = src.get("type", "")
                    out["source_quote"] = src.get("quote", "")

            w.writerow(out)


if __name__ == "__main__":
    main()
