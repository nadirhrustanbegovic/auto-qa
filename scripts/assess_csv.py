#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm


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


def _ollama_chat(host: str, model: str, system_prompt: str, user_payload: Dict[str, Any],
                temperature: float, max_tokens: int) -> str:
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


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(project_root, ".env"))
    _set_csv_field_limit()

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/input.csv", help="Path to input CSV")
    p.add_argument("--output", default="data/output.jsonl", help="Path to output .jsonl or .csv")
    p.add_argument("--prompt", default="prompts/grader_system_prompt.md", help="System prompt path")
    p.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.1")))
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "900")))
    p.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Process only the first N rows from input CSV (0 = all rows)",
    )
    args = p.parse_args()
    args.input = _resolve_path(args.input, project_root)
    args.output = _resolve_path(args.output, project_root)
    args.prompt = _resolve_path(args.prompt, project_root)
    args.model = _resolve_ollama_model(args.host, args.model)

    system_prompt = _load_prompt(args.prompt)

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
            if args.sample_size > 0 and len(rows) >= args.sample_size:
                break

    results: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="grading"):
        scenario_id = _get_row_value(row, "scenario_id", "send_id", "id").strip() or None
        conv_raw = _get_row_value(row, "conversation", "conversation_json")
        preferred_tone = _get_row_value(row, "message_tone", "tone").strip()
        notes = _get_row_value(row, "notes", "company_notes").strip()
        persona = _get_row_value(row, "persona").strip()
        blocked_words = _parse_blocked_words(_get_row_value(row, "blocked_words", "blocklisted_words"))

        conv = _parse_conversation(conv_raw)
        last_agent_message = _extract_last_agent_message(conv) or ""

        found_blocked = _find_blocked_words(last_agent_message, blocked_words)

        user_payload = {
            "scenario_id": scenario_id,
            "preferred_message_tone": preferred_tone,
            "notes": notes,
            "persona": persona,
            "blocked_words_found": found_blocked,
            "last_agent_message": last_agent_message,
        }

        try:
            llm_text = _ollama_chat(
                host=args.host,
                model=args.model,
                system_prompt=system_prompt,
                user_payload=user_payload,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            llm_json = _safe_json_loads(llm_text)
            results.append({
                "scenario_id": scenario_id,
                "preferred_message_tone": preferred_tone,
                "blocked_words_found": found_blocked,
                "last_agent_message": last_agent_message,
                "grades": llm_json,
            })
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
        "detected_tone",
        "tone_match_label",
        "tone_match_reason",
        "blocked_words_check_label",
        "blocked_words_check_reason",
        "clarity_grammar_label",
        "clarity_grammar_reason",
        "clarity_typos_label",
        "clarity_typos_reason",
        "clarity_repetition_label",
        "clarity_repetition_reason",
        "clarity_understandable_label",
        "clarity_understandable_reason",
        "tone_empathy_label",
        "tone_empathy_reason",
        "tone_personalize_label",
        "tone_personalize_reason",
        "tone_preferred_label",
        "tone_preferred_reason",
        "overall_label",
        "overall_reason",
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
            out["error"] = r.get("error", "")

            g = r.get("grades") or {}
            if isinstance(g, dict):
                out["detected_tone"] = g.get("detected_tone", "")
                tm = g.get("tone_match") or {}
                out["tone_match_label"] = tm.get("label", "")
                out["tone_match_reason"] = tm.get("reason", "")
                bw = g.get("blocked_words_check") or {}
                out["blocked_words_check_label"] = bw.get("label", "")
                out["blocked_words_check_reason"] = bw.get("reason", "")

                cl = g.get("clarity") or {}
                ge = cl.get("grammar_errors_and_meaning") or {}
                ty = cl.get("typos") or {}
                rp = cl.get("repetition") or {}
                un = cl.get("understandable") or {}
                out["clarity_grammar_label"] = ge.get("label", "")
                out["clarity_grammar_reason"] = ge.get("reason", "")
                out["clarity_typos_label"] = ty.get("label", "")
                out["clarity_typos_reason"] = ty.get("reason", "")
                out["clarity_repetition_label"] = rp.get("label", "")
                out["clarity_repetition_reason"] = rp.get("reason", "")
                out["clarity_understandable_label"] = un.get("label", "")
                out["clarity_understandable_reason"] = un.get("reason", "")

                tn = g.get("tone") or {}
                em = tn.get("empathy") or {}
                ps = tn.get("personalize") or {}
                pt = tn.get("preferred_tone_followed") or {}
                out["tone_empathy_label"] = em.get("label", "")
                out["tone_empathy_reason"] = em.get("reason", "")
                out["tone_personalize_label"] = ps.get("label", "")
                out["tone_personalize_reason"] = ps.get("reason", "")
                out["tone_preferred_label"] = pt.get("label", "")
                out["tone_preferred_reason"] = pt.get("reason", "")

                ov = g.get("overall") or {}
                out["overall_label"] = ov.get("label", "")
                out["overall_reason"] = ov.get("reason", "")

            w.writerow(out)


if __name__ == "__main__":
    main()
