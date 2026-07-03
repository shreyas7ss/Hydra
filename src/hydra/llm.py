"""LLM abstraction + JSON parsing helpers.

The graph depends only on the `LLMClient` protocol, so the provider is a swap (plan
§8 #2). Three implementations ship:

* ``OpenAIClient`` — the directive's mandated GPT-4o baseline (lazy `openai` import).
* ``EchoLLM``      — a deterministic, offline demo LLM so the graph runs with no API key.
* ``ScriptedLLM``  — returns canned responses for fully deterministic unit tests.

Every node prompt embeds a ``TASK: <name>`` marker in its system message; the fakes
key off that marker so one client can serve every node in a single run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


# --------------------------------------------------------------------------- #
# Parsing helpers (LLMs wrap JSON in prose / code fences more often than not).
# --------------------------------------------------------------------------- #
def parse_json(text: str) -> Any | None:
    """Best-effort extraction of a JSON object/array from a model response."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def parse_str_list(text: str) -> list[str]:
    """Parse a JSON array of strings; fall back to non-empty lines."""
    data = parse_json(text)
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    return [line.strip("-* \t") for line in (text or "").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Real provider — OpenAI / GPT-4o baseline.
# --------------------------------------------------------------------------- #
class OpenAIClient:
    def __init__(self, model: str, api_key: str | None, temperature: float = 0.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "The 'openai' extra is not installed. Run "
                "`uv sync --extra openai`, or use the offline demo (`hydra --demo`)."
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Real provider — Google Gemini.
# --------------------------------------------------------------------------- #
class GeminiClient:
    def __init__(self, model: str, api_key: str | None, temperature: float = 0.0) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "The 'gemini' extra is not installed. Run `uv sync --extra gemini`, "
                "or use the offline demo (`hydra --demo`)."
            ) from exc
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, *, system: str, user: str) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
            ),
        )
        return resp.text or ""


# --------------------------------------------------------------------------- #
# Offline demo LLM — deterministic heuristics, no network.
# --------------------------------------------------------------------------- #
class EchoLLM:
    """A dependency-free stand-in so the whole graph runs without an API key.
    Not smart — just deterministic and good enough to demonstrate routing/flow."""

    def __init__(self, multi_query_count: int = 3) -> None:
        self.multi_query_count = multi_query_count

    def complete(self, *, system: str, user: str) -> str:
        q = user.strip()
        if "intent_classification" in system:
            lowered = q.lower()
            multi_signals = (
                " and ", " vs ", " versus ", "compare", "difference",
                "trend", "why", "how ", "impact", "relationship",
            )
            is_multi = any(s in lowered for s in multi_signals)
            looks_direct = (
                len(q.split()) <= 6
                or any(c.isdigit() for c in q)
                or '"' in q
                or "'" in q
            )
            intent = "direct" if (looks_direct and not is_multi) else "complex"
            confidence = 0.82 if intent == "direct" else 0.74
            return json.dumps(
                {"intent": intent, "confidence": confidence,
                 "reasoning": "offline heuristic demo classifier"}
            )
        if "multi_query" in system:
            variants = [q, f"What is {q}?", f"Explain in detail: {q}",
                        f"Key facts about {q}", f"{q} — overview"]
            return json.dumps(variants[: max(1, self.multi_query_count)])
        if "decompose" in system:
            parts = re.split(r"\band\b|\?|;", q, flags=re.IGNORECASE)
            subs = [p.strip(" .,") for p in parts if len(p.strip()) > 3]
            return json.dumps(subs or [q])
        if "hyde" in system:
            return f"A relevant document would state that, regarding '{q}', the answer is as follows: ..."
        if "retrieval_eval" in system:
            # Grade context relevance by content-word overlap (stopwords dropped).
            from hydra.retrieval.text import tokenize

            question_part = user.split("Context:", 1)[0]
            ctx_terms = set(tokenize(user.split("Context:", 1)[-1]))
            q_terms = set(tokenize(question_part))
            ratio = len(q_terms & ctx_terms) / len(q_terms) if q_terms else 0.0
            confidence = "high" if ratio >= 0.5 else "medium" if ratio >= 0.2 else "low"
            return json.dumps(
                {"confidence": confidence, "score": round(ratio, 3),
                 "reasoning": "offline heuristic retrieval grader"}
            )
        if "generate" in system:
            # Ground the answer in the supplied context (flattened top passage).
            ctx = user.split("Context:", 1)[-1]
            flat = " ".join(ctx.split())
            if not flat:
                return "I don't have enough retrieved context to answer that."
            return f"Based on the retrieved context: {flat[:260]}"
        if "reflect" in system:
            # Faithful if the answer's content words are supported by the context.
            from hydra.retrieval.text import tokenize

            answer_part = user.split("Answer:", 1)[-1].split("Context:", 1)[0]
            ctx_terms = set(tokenize(user.split("Context:", 1)[-1]))
            a_terms = set(tokenize(answer_part))
            faithful = bool(a_terms) and len(a_terms & ctx_terms) / len(a_terms) >= 0.5
            return json.dumps(
                {"faithful": faithful, "relevant": True,
                 "critique": "offline heuristic reflection"}
            )
        if "node_summary" in system:
            first = re.split(r"(?<=[.!?])\s", q.strip(), maxsplit=1)[0]
            return first[:160]
        if "segment" in system:
            # Split raw text into sections on blank lines; first line is the title.
            blocks = [b.strip() for b in re.split(r"\n\s*\n", q) if b.strip()]
            sections = []
            for b in blocks:
                lines = b.split("\n", 1)
                sections.append({"title": lines[0][:80], "content": lines[1] if len(lines) > 1 else ""})
            return json.dumps(sections or [{"title": "Document", "content": q}])
        if "tree_nav" in system:
            from hydra.retrieval.text import tokenize

            question_part = user.split("Children:", 1)[0]
            children_part = user.split("Children:", 1)[-1]
            q_terms = set(tokenize(question_part))
            best_idx, best_score = -1, 0
            for line in children_part.splitlines():
                m = re.match(r"\s*(\d+)\.", line)
                if not m:
                    continue
                score = len(q_terms & set(tokenize(line)))
                if score > best_score:
                    best_score, best_idx = score, int(m.group(1))
            return json.dumps({"choice": best_idx, "reason": "offline heuristic navigation"})
        return ""


# --------------------------------------------------------------------------- #
# Scripted LLM — exact canned responses for deterministic tests.
# --------------------------------------------------------------------------- #
class ScriptedLLM:
    """Returns ``responses[key]`` for the first ``key`` found in the system prompt
    (match on the node's ``TASK: <name>`` marker). Records calls for assertions."""

    def __init__(self, responses: dict[str, str] | None = None, default: str = "") -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        for key, value in self.responses.items():
            if key in system or key in user:
                return value
        return self.default


# --------------------------------------------------------------------------- #
# Factory.
# --------------------------------------------------------------------------- #
def build_llm(settings, *, demo: bool = False) -> LLMClient:
    if demo:
        return EchoLLM(multi_query_count=settings.multi_query_count)
    provider = settings.llm_provider.lower()
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in .env, or run the offline "
                "demo with `hydra --demo`."
            )
        return OpenAIClient(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
        )
    if provider == "gemini":
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. Set it in .env, or run "
                "the offline demo with `hydra --demo`."
            )
        return GeminiClient(
            model=settings.llm_model,
            api_key=settings.google_api_key,
            temperature=settings.temperature,
        )
    raise RuntimeError(f"Unknown LLM provider: {settings.llm_provider!r}")
