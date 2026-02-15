"""LLM service — build prompt, call OpenRouter (OpenAI-compatible), parse JSON."""

import asyncio
import json
import logging

import tiktoken
from openai import OpenAI

from src.backend.config import settings
from src.backend.services.retrieval_service import RetrievedChunk

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

# OpenRouter client (OpenAI-compatible)
_client = OpenAI(
    base_url=settings.OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = """You are an internal knowledge assistant.
Only answer using the provided context.
If the answer is not in the context, return:
{"answer": null, "reason": "insufficient_context"}
Always cite document IDs.
Return JSON only with this schema:
{"answer": "...", "citations": ["document_id"], "confidence": 0.0-1.0}"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    context_parts: list[str] = []
    total_tokens = 0

    for chunk in chunks:
        chunk_tokens = len(_enc.encode(chunk.chunk_text))
        if total_tokens + chunk_tokens > settings.MAX_CONTEXT_TOKENS:
            logger.info("Context truncated at %d tokens", total_tokens)
            break
        context_parts.append(
            f"[Document ID: {chunk.document_id}]\n{chunk.chunk_text}"
        )
        total_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts)


def _build_user_prompt(context: str, question: str) -> str:
    return f"""Context:
{context}

---

Question: {question}

Instructions:
- Answer ONLY using the context above.
- Cite the Document IDs used.
- If the context does not contain the answer, return: {{"answer": null, "reason": "insufficient_context"}}
- Return valid JSON only."""


async def generate_answer(
    chunks: list[RetrievedChunk],
    question: str,
) -> dict:
    """Build prompt from chunks, call Gemini, return parsed JSON answer.

    Uses asyncio.to_thread() to prevent blocking the event loop.
    """
    context = _build_context(chunks)
    user_prompt = _build_user_prompt(context, question)

    # Approximate token usage for logging
    prompt_tokens = len(_enc.encode(SYSTEM_PROMPT + user_prompt))
    logger.info("Approximate prompt tokens: %d", prompt_tokens)

    def _call_llm() -> str:
        # Gemma models don't support system instructions — prepend to user message
        combined_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        response = _client.chat.completions.create(
            model=settings.GENERATION_MODEL,
            messages=[
                {"role": "user", "content": combined_prompt},
            ],
            max_tokens=settings.MAX_OUTPUT_TOKENS,
            temperature=0.1,
        )
        return response.choices[0].message.content

    raw_text = await asyncio.to_thread(_call_llm)

    # Strip markdown code fences if present (e.g. ```json ... ```)
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Log approximate output tokens
    output_tokens = len(_enc.encode(raw_text))
    logger.info(
        "Approximate token usage — prompt: %d, output: %d, total: %d",
        prompt_tokens,
        output_tokens,
        prompt_tokens + output_tokens,
    )

    # Parse JSON response
    try:
        answer = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON response: %s", raw_text)
        answer = {
            "answer": raw_text,
            "citations": [],
            "confidence": 0.0,
        }

    # Normalise output keys
    return {
        "answer": answer.get("answer"),
        "citations": answer.get("citations", []),
        "confidence": answer.get("confidence", 0.0),
        "reason": answer.get("reason"),
    }
