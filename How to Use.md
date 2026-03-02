# How to Use (Simple)

This project checks support replies and gives:
- `pass` as `1` or `0`
- a short `justification` with a direct quote
- quote `source` (`rag` or `prompt`)

## First-time setup

1. Install project packages:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2. Download local AI models (one-time):

```powershell
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

## Before grading

1. Put policy docs (`.txt` or `.md`) in:

`data/docs/`

2. Build policy index:

```powershell
.\.venv\Scripts\python scripts/rag_ingest.py --reset
```

## Run grading

```powershell
.\.venv\Scripts\python scripts/assess_csv.py --input data/input.csv --output data/output.jsonl --prompt prompts/sample_prompt_1.md --strict-decision-output
```

## Day-to-day flow

1. Update policy files in `data/docs/`.
2. Run `rag_ingest.py --reset`.
3. Run `assess_csv.py`.
