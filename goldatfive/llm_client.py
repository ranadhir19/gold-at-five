"""Minimal chat clients for the LLM reranker, free tiers only.

Two providers, chosen by model id (``gemini-*`` → Google AI Studio,
anything else → Groq's OpenAI-compatible API):

- Groq: ``GROQ_API_KEY`` env var, else Muse's connector surrogate
  for ``custom.groq`` inside a Muse session.
- Gemini: ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` env var, else the
  surrogate for ``custom.google-ai-studio``. Generation runs on the
  Interactions API (the classic generateContent endpoint 404s for
  this account class). The Gemini free tier processes public material
  only — do not send private data through it.

No key is ever printed, logged, or written to disk by this module.
Returned ``model`` is always the SERVED model id from the response,
which can differ from the requested alias. Gemini completion tokens
include thought tokens: reasoning is a real cost and is counted.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
_CRED_BIN = "/opt/muse/skills/custom-connector-setup/bin"
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def _surrogate(request, provider: str, host: str) -> None:
    if _CRED_BIN not in sys.path:
        sys.path.insert(0, _CRED_BIN)
    from dynamic_credentials import add_surrogate_to_request
    add_surrogate_to_request(request, provider, allowed_hosts=[host])


def chat(model: str, prompt: str, max_tokens: int = 2500) -> dict:
    """One completion. Returns {text, model, prompt_tokens,
    completion_tokens}."""
    headers = {"Content-Type": "application/json", "User-Agent": _UA,
               "Accept": "application/json"}
    if model.startswith("gemini"):
        payload = {"model": "models/" + model, "input": prompt}
        request = urllib.request.Request(
            _GEMINI_URL, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")
        key = os.environ.get("GEMINI_API_KEY") \
            or os.environ.get("GOOGLE_API_KEY")
        if key:
            request.add_header("x-goog-api-key", key)
        else:
            _surrogate(request, "custom.google-ai-studio",
                       "generativelanguage.googleapis.com")
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = ""
        for step in result.get("steps") or []:
            if step.get("type") == "model_output":
                for part in step.get("content") or []:
                    if part.get("type") == "text":
                        text += part.get("text", "")
        usage = result.get("usage") or {}
        return {
            "text": text,
            "model": result.get("model"),
            "prompt_tokens": usage.get("total_input_tokens"),
            "completion_tokens":
                (usage.get("total_output_tokens") or 0)
                + (usage.get("total_thought_tokens") or 0),
        }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        _GROQ_URL, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST")
    key = os.environ.get("GROQ_API_KEY")
    if key:
        request.add_header("Authorization", "Bearer " + key)
    else:
        _surrogate(request, "custom.groq", "api.groq.com")
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = result.get("usage") or {}
    return {
        "text": message.get("content") or "",
        "model": result.get("model"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
