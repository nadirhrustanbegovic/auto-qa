# QA LLM Auto-Grader (Local)

This repo reads a CSV of scenarios, extracts the last agent message from each conversation JSON, checks blocked words, and asks a local instruct model to grade **clarity** and **tone**.

## Model recommendation (local, instruction-tuned)
- **Llama 3.1 8B Instruct** (good instruction-following, strong general quality, runs locally on a single GPU; can run on CPU with lower throughput).

## Local runtime (Ollama)
1. Install Ollama
2. Pull a model, e.g.
   - `ollama pull llama3.1:8b-instruct`
3. Create a venv and install deps:
   - `python -m venv .venv`
   - mac/linux: `source .venv/bin/activate`
   - windows: `.venv\Scripts\activate`
   - `pip install -r requirements.txt`
4. Copy env file and edit as needed:
   - `cp .env.example .env`

## Run
- `python scripts/assess_csv.py --input data/input.csv --output data/output.jsonl`

Output is JSON Lines (one record per scenario). Use `--output data/output.csv` to write CSV instead.

## CSV expectations
Required columns:
- `scenario_id`
- `conversation` (a JSON object/array as a string)
- `message_tone` (preferred tone label/description)
- `blocked_words` (comma-separated list or JSON array; optional)

## Notes
- Blocked words detection is a literal match (case-insensitive) with word-boundary checks where possible.
- Tone comparison is done by the model: it infers tone of the last agent message and compares against `message_tone`.
