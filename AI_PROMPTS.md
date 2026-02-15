# AI Prompts — Design & Rationale

This document details the exact prompts used in Aura's RAG pipeline, including design decisions, iteration history, and why human judgment remains essential.

---

## 1. System Prompt (Exact)

```
You are an internal knowledge assistant.
Only answer using the provided context.
If the answer is not in the context, return:
{"answer": null, "reason": "insufficient_context"}
Always cite document IDs.
Return JSON only with this schema:
{"answer": "...", "citations": ["document_id"], "confidence": 0.0-1.0}
```

### Why This Prompt?

- **Role anchoring** ("internal knowledge assistant") prevents the model from acting as a general chatbot.
- **Context-only constraint** eliminates hallucination from parametric knowledge.
- **Explicit refusal format** ensures the system fails gracefully when context is insufficient.
- **Citation requirement** provides traceability and auditability.
- **JSON-only** combined with `response_mime_type="application/json"` guarantees parseable output.

---

## 2. User Prompt Template (Exact)

```
Context:
{context}

---

Question: {question}

Instructions:
- Answer ONLY using the context above.
- Cite the Document IDs used.
- If the context does not contain the answer, return: {"answer": null, "reason": "insufficient_context"}
- Return valid JSON only.
```

### Variables

| Variable | Source |
|----------|--------|
| `{context}` | Top-5 retrieved chunks, each prefixed with `[Document ID: <uuid>]`, separated by `---`, truncated to 3,000 tokens max. |
| `{question}` | User's original question, verbatim. |

### Context Block Format

Each chunk is formatted as:

```
[Document ID: a1b2c3d4-...]
<chunk text here>

---

[Document ID: e5f6g7h8-...]
<chunk text here>
```

---

## 3. Output Schema

```json
{
  "answer": "string or null",
  "citations": ["document_id_1", "document_id_2"],
  "confidence": 0.85
}
```

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string \| null` | The generated answer, or null if context is insufficient. |
| `citations` | `string[]` | Document IDs used to form the answer. |
| `confidence` | `float` | Model's self-assessed confidence, 0.0–1.0. |
| `reason` | `string \| null` | Only present on refusal: `"insufficient_context"`. |

---

## 4. Example: Accepted Output

**Question:** "What is the refund policy for enterprise customers?"

**Context provided:** Contains a chunk from the billing policy document explaining refund procedures.

**LLM Response:**
```json
{
  "answer": "Enterprise customers are eligible for a full refund within 30 days of purchase. After 30 days, a prorated refund is available for the remaining contract period. Refund requests must be submitted through the account manager.",
  "citations": ["d4e5f6a7-1234-5678-9abc-def012345678"],
  "confidence": 0.91
}
```

**Why accepted:**
- Directly answers the question using context.
- Cites the specific document.
- Confidence is high and justified.
- Pure JSON, no extra text.

---

## 5. Example: Rejected Output

**Question:** "What is the weather today?"

**Context provided:** Contains only internal HR and engineering documents.

**LLM Response:**
```json
{
  "answer": null,
  "reason": "insufficient_context"
}
```

**Why rejected (correctly):**
- The question is unrelated to any provided context.
- The model correctly refuses rather than hallucinating.
- No citations because no documents were relevant.

---

## 6. Prompt Iteration History

### Iteration 1 — Minimal Prompt
```
Answer the question using the context.
```
**Problem:** Model would answer from its own knowledge when context was insufficient. No structured output.

### Iteration 2 — Added Constraints
```
Only answer using the provided context. Return JSON.
```
**Problem:** Model sometimes returned JSON wrapped in markdown code blocks. No citation requirement.

### Iteration 3 — Explicit Schema + Refusal
```
Return JSON only with this schema: {"answer": "...", "citations": [...], "confidence": ...}
If the answer is not in the context, return: {"answer": null, "reason": "insufficient_context"}
```
**Problem:** Model occasionally forgot to include citations or invented document IDs.

### Iteration 4 — Final (Current)
Added:
- `response_mime_type="application/json"` at the API level to force pure JSON.
- Explicit `[Document ID: ...]` prefixes in context so the model can cite real IDs.
- Redundant instruction in both system and user prompts for robustness.

**Result:** Consistent, parseable JSON with real document IDs and appropriate refusals.

---

## 7. Why Human Judgment Is Required

Despite rigorous prompt engineering, **human review is essential** for:

1. **Confidence calibration** — The model's self-reported confidence is not externally validated. A 0.9 confidence doesn't mean 90% accuracy. Human reviewers should periodically sample answers and assess actual quality.

2. **Edge cases in context** — When context is ambiguous or contradictory (e.g., two policy documents with different dates), the model may pick arbitrarily. A human can detect and flag outdated information.

3. **Sensitive domains** — HR policies, legal terms, and compliance documents require human sign-off. An AI answer about termination procedures should never be the sole source of truth.

4. **Prompt drift** — As the document corpus evolves, the prompt may need adjustment. What works for 10 documents may not work for 10,000. Regular human evaluation of answer quality is necessary.

5. **Refusal tuning** — The 0.25 distance threshold is a heuristic. Too strict = too many refusals (bad UX). Too loose = hallucinated answers. Only human evaluation of real queries can find the right balance.

6. **Citation accuracy** — The model may cite a document ID that appears in context but isn't actually the source of the answer. Human spot-checks catch this.

> **Bottom line:** AI generates, humans validate. This system is a tool to accelerate knowledge retrieval, not a replacement for human judgment.
