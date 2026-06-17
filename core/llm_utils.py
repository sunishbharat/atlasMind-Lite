"""
core/llm_utils.py

Shared utilities for LLM response post-processing.
Imported by all LLM clients (Groq, Claude, Bedrock, vLLM).
"""


def clean_llm_response(raw: str) -> str:
    """Strip markdown fences and any plain-text preamble/trailing text around the JSON object.

    LLMs sometimes prepend explanatory text before the JSON object without
    wrapping it in code fences:
        "Looking at the request: ...\n{"jql": "..."}"

    This function:
      1. Strips any markdown code fences (```json ... ```)
      2. Finds the first '{' (skipping any preamble)
      3. Tracks brace depth to return exactly the JSON object, excluding
         any trailing text the LLM may have appended

    The result is a clean JSON string starting with '{' and ending with '}'.
    """
    text = raw.strip()
    # Step 1: strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Step 2: find first '{' and track brace depth to find matching '}'
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]
