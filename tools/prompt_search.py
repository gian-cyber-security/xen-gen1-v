"""Prompt enrichment search for XEN image/video generation."""
from __future__ import annotations
from ddgs import DDGS

def enrich_prompt(prompt: str, max_results: int = 3) -> str:
    try:
        results = DDGS(timeout=8).text(
            prompt,
            region="us-en",
            safesearch="moderate",
            max_results=max_results,
        )
    except Exception:
        return prompt
    if not results:
        return prompt
    lines = [
        prompt,
        "",
        "Reference context from web search. Use only relevant visual facts; do not copy text into the generated media:",
    ]
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        if title or body:
            lines.append(f"- {title}: {body}")
    return "\n".join(lines)
