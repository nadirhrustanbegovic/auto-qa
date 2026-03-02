Decide if the message passes QA using policy_context first, prompt wording second.

Return only:
{
  "pass": 1 or 0,
  "justification": "must include a direct quote",
  "source": {
    "type": "rag" or "prompt",
    "quote": "exact quoted evidence"
  }
}
