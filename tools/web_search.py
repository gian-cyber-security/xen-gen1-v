"""Optional web context for XEN multimodal generation.
Use it to enrich prompts with current public web information before passing the
final prompt to the image/video conditioner.
"""
from __future__ import annotations
from ddgs import DDGS

def should_search(prompt: str) -> bool:
    p=prompt.lower()
    return any(x in p for x in (
        "latest","current","today","right now","recent","news","this week",
        "this month","update","2026","price","weather","score","event"
    ))

def search_web(query: str, max_results: int=4) -> list[dict[str,str]]:
    try:
        rows=DDGS(timeout=8).text(query,region="us-en",safesearch="moderate",max_results=max_results)
        return [{"title":r.get("title",""),"url":r.get("href",""),"snippet":r.get("body","")} for r in rows]
    except Exception:
        return []

def enrich_prompt(prompt: str, max_results: int=4, max_chars: int=2500) -> str:
    if not should_search(prompt):
        return prompt
    results=search_web(prompt,max_results)
    if not results:
        return prompt
    parts=[prompt,"","CURRENT WEB CONTEXT:"]
    used=sum(map(len,parts))
    for i,r in enumerate(results,1):
        item=f"{i}. {r['title']}\n{r['snippet']}\nSource: {r['url']}\n"
        if used+len(item)>max_chars: break
        parts.append(item); used+=len(item)
    parts.append("Use the web context only to improve factual/current prompt details; do not invent unsupported facts.")
    return "\n".join(parts)
