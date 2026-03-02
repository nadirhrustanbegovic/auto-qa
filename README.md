# QA Auto-Grader (Local)

This project grades support-agent messages from CSV using:
- prompt rules
- policy retrieval (RAG) from local docs

Each result is a decision:
- `pass`: `1` or `0`
- `justification`: short reason with a direct quote
- `source`: `rag` or `prompt` + the quote used

## Quick Start

1. Install dependencies:
`.\.venv\Scripts\python -m pip install -r requirements.txt`

2. Pull Ollama models (once):
`ollama pull nomic-embed-text`
`ollama pull llama3.1:8b`

3. Build policy index from files in `data/docs/`:
`.\.venv\Scripts\python scripts/rag_ingest.py --reset`

4. Run grading:
`.\.venv\Scripts\python scripts/assess_csv.py --input data/input.csv --output data/output.jsonl --prompt prompts/sample_prompt_1.md --strict-decision-output`

## Main files

- Prompt examples:
  - `prompts/sample_prompt_1.md`
  - `prompts/sample_prompt_2.md`
- Grader: `scripts/assess_csv.py`
- Policy indexer: `scripts/rag_ingest.py`
- Optional RAG query tool: `scripts/rag_query.py`

For simple step-by-step instructions, see `How to Use.md`.
