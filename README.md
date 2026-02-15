# Aura — Multi-Tenant RAG Knowledge Assistant

A Dockerized FastAPI backend implementing a deterministic multi-tenant RAG pipeline for internal knowledge retrieval. Employees ask natural language questions against tenant-scoped document stores and receive grounded, citation-backed answers.

Built for a 90-minute AI Engineer assessment (Option C — Internal Knowledge Assistant).

---

## Overview

Aura ingests plain-text documents per tenant, chunks and embeds them into PostgreSQL via pgvector, and answers questions through a retrieval-augmented generation pipeline. The system enforces strict tenant isolation at every layer — database queries, vector search, cache keys, and LLM prompts.

| Component | Technology |
|-----------|-----------|
| API server | FastAPI 0.111, Python 3.10, async |
| Database | PostgreSQL 16 + pgvector (cosine distance) |
| Cache | Redis 7 (alpine) |
| Embedding | Gemini `models/gemini-embedding-001` (768 dimensions) |
| LLM | OpenRouter `google/gemma-3-12b-it:free` via OpenAI SDK |
| Chunking | tiktoken `cl100k_base` (500–800 tokens, 100 overlap) |
| ORM | SQLAlchemy 2.0 (async, asyncpg) |
| Validation | Pydantic v2 + pydantic-settings |
| Containers | Docker Compose (3 services) |

---

## 1. Problem Framing

**Target user:** Employees of B2B SaaS companies who need instant, accurate answers from internal knowledge bases — onboarding guides, policy documents, technical runbooks.

**Why keyword search fails:**

- Keyword search cannot resolve semantic intent. A query like "What's our refund policy?" will not match a document titled "Customer Satisfaction Procedures."
- Employees waste time scanning dozens of documents for a single answer.
- Each tenant's knowledge base is private and must be strictly isolated.

**Why RAG:**

Retrieval-Augmented Generation grounds LLM responses in actual tenant documents, eliminating hallucination while providing natural language answers with document-level citations. The LLM never invents information — it either answers from retrieved context or explicitly refuses.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Client (curl / Postman)                 │
└────────────────────────────┬─────────────────────────────────┘
                             │ HTTP (port 8000)
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (:8000)                      │
│                                                              │
│  POST /tenants     — register a new tenant                   │
│  POST /documents   — ingest, chunk, embed, store             │
│  POST /ask         — full RAG pipeline                       │
│  GET  /health      — liveness check (DB + Redis)             │
│                                                              │
│  ┌────────────────┐ ┌───────────────┐ ┌────────────────────┐ │
│  │ Document       │ │ Embedding     │ │ LLM Service        │ │
│  │ Service        │ │ Service       │ │ (OpenRouter)       │ │
│  │ (tiktoken      │ │ (Gemini API)  │ │ gemma-3-12b-it     │ │
│  │  chunking)     │ │ 768-dim vecs  │ │ max 500 tokens out │ │
│  └───────┬────────┘ └───────┬───────┘ └──────────┬─────────┘ │
│          │                  │                    │           │
│  ┌───────┴──────────────────┴────────────────────┴─────────┐ │
│  │              Retrieval Service                           │ │
│  │    pgvector cosine distance (<=>), top-5, tenant filter  │ │
│  └─────────────────────────┬───────────────────────────────┘ │
│                            │                                 │
│  ┌─────────────────────────┴───────────────────────────────┐ │
│  │              Cache Service (Redis, TTL 900s)            │ │
│  │         key = {tenant_id}:{sha256(question)}            │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────┬──────────────────────────────────┬─────────────┘
              │                                  │
              ▼                                  ▼
┌──────────────────────┐          ┌──────────────────────────┐
│   PostgreSQL 16      │          │       Redis 7-alpine     │
│   + pgvector         │          │                          │
│                      │          │   Tenant-scoped keys     │
│  tenants             │          │   TTL = 900 seconds      │
│  documents           │          └──────────────────────────┘
│  document_chunks     │
│  ai_requests         │
│  ai_responses        │
└──────────────────────┘
```

**Layers:**

| Layer | Responsibility |
|-------|---------------|
| API | Request validation (Pydantic), routing, error handling, HTTP status codes |
| Retrieval | Tenant-filtered pgvector cosine distance search, top-5 selection, threshold gating |
| LLM | Prompt construction, context truncation (3000 tokens), OpenRouter API call, JSON parsing |
| Cache | Redis lookup/store with tenant-scoped keys, 900-second TTL |
| Persistence | SQLAlchemy async ORM, 5 tables, pgvector Vector(768) column, JSONB audit logs |

### RAG Pipeline (POST /ask)

```
① Validate tenant ──→ not found? → HTTP 404
② Check Redis cache ─→ hit? → return cached answer (cached: true)
③ Embed question ────→ Gemini API → 768-dim vector
④ Retrieve chunks ───→ pgvector top-5, WHERE tenant_id = :tid
⑤ Threshold gate ────→ no chunk with distance < 0.35? → refuse (skip LLM)
⑥ Generate answer ───→ OpenRouter LLM → JSON response
⑦ Log to database ───→ ai_requests + ai_responses
⑧ Cache in Redis ────→ TTL 900s
⑨ Return response ───→ answer, citations, confidence, cached: false
```

---

## 3. Data Model

Five tables, created automatically on startup via `Base.metadata.create_all()`. No migration tool.

```
┌──────────────┐       ┌──────────────────┐       ┌────────────────────┐
│   tenants    │──1:N──│   documents      │──1:N──│  document_chunks   │
│              │       │                  │       │                    │
│  id (UUID PK)│       │  id (UUID PK)    │       │  id (UUID PK)      │
│  name        │       │  tenant_id (FK)  │       │  document_id (FK)  │
│  created_at  │       │  title           │       │  tenant_id (idx)   │
└──────────────┘       │  content         │       │  chunk_text        │
                       │  created_at      │       │  embedding Vec(768)│
                       └──────────────────┘       │  created_at        │
                                                  └────────────────────┘

┌──────────────────┐       ┌──────────────────┐
│   ai_requests    │──1:1──│  ai_responses    │
│                  │       │                  │
│  id (UUID PK)    │       │  id (UUID PK)    │
│  tenant_id (idx) │       │  request_id (FK) │
│  question        │       │  answer_json     │
│  created_at      │       │  created_at      │
└──────────────────┘       └──────────────────┘
```

| Table | Purpose |
|-------|---------|
| `tenants` | Registry of B2B tenants. UUID primary key. |
| `documents` | Raw documents per tenant. `tenant_id` FK → `tenants.id`, indexed. |
| `document_chunks` | Chunked text + embedding. `document_id` FK → `documents.id`. `tenant_id` indexed for filtered vector search. `embedding` is pgvector `Vector(768)`. |
| `ai_requests` | Audit log of every question asked. `tenant_id` indexed. |
| `ai_responses` | Audit log of every LLM response. `answer_json` stored as PostgreSQL `JSONB`. FK → `ai_requests.id`. |

**Key constraints:**

- All primary keys are UUID v4, generated server-side.
- `documents.tenant_id` has a foreign key to `tenants.id`.
- `document_chunks.document_id` has a foreign key to `documents.id`.
- `ai_responses.request_id` has a foreign key to `ai_requests.id`.
- `tenant_id` is indexed on `documents`, `document_chunks`, and `ai_requests`.
- LLM responses are stored as `JSONB` in `ai_responses.answer_json` for flexible querying.

---

## 4. Prompt Design

### System Prompt

```
You are an internal knowledge assistant.
Only answer using the provided context.
If the answer is not in the context, return:
{"answer": null, "reason": "insufficient_context"}
Always cite document IDs.
Return JSON only with this schema:
{"answer": "...", "citations": ["document_id"], "confidence": 0.0-1.0}
```

The system prompt enforces three invariants:

1. **Context-only answering** — the model must not use its parametric knowledge.
2. **Mandatory citations** — every answer must reference the document IDs it used.
3. **Structured output** — JSON only, no prose.

### User Prompt Template

```
Context:
[Document ID: <uuid>]
<chunk text>

---

[Document ID: <uuid>]
<chunk text>

---

Question: <user question>

Instructions:
- Answer ONLY using the context above.
- Cite the Document IDs used.
- If the context does not contain the answer, return: {"answer": null, "reason": "insufficient_context"}
- Return valid JSON only.
```

Since Gemma models do not support a separate system instruction parameter, the system prompt is prepended to the user message as a single combined prompt.

### Output Schema

```json
{
  "answer": "string or null",
  "citations": ["document_id_1", "document_id_2"],
  "confidence": 0.85,
  "reason": "null or insufficient_context"
}
```

The API response wraps this with an additional `cached` field:

```json
{
  "answer": "...",
  "citations": ["..."],
  "confidence": 0.85,
  "reason": null,
  "cached": false
}
```

### Hallucination Prevention

| Layer | Mechanism |
|-------|-----------|
| Retrieval | Distance threshold (0.35) gates LLM invocation — if no chunk is semantically close enough, the LLM is never called |
| Prompt | System prompt explicitly instructs the model to refuse when context is insufficient |
| Prompt | Only tenant-owned chunks are injected into context — no external knowledge leakage |
| Output | JSON parsing with fallback — malformed responses are caught and returned with `confidence: 0.0` |
| Temperature | Set to 0.1 for near-deterministic output |

---

## 5. RAG Design

### Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Tokenizer | tiktoken `cl100k_base` | Accurate subword token counting |
| Max chunk size | 800 tokens | Balances context richness with retrieval precision |
| Min chunk size | 500 tokens | Prevents fragments too small to carry meaning |
| Overlap | 100 tokens | Preserves context continuity at chunk boundaries |
| Trailing fragments | Merged into previous chunk | Fragments < 250 tokens (half of min) are appended to the last chunk |

**Algorithm:**

1. Encode the full document into tokens using tiktoken.
2. Slide a window of 800 tokens, stepping forward by 700 tokens (800 − 100 overlap).
3. Decode each window back to text.
4. If the final fragment is shorter than 250 tokens, merge it into the previous chunk.

### Embedding

| Parameter | Value |
|-----------|-------|
| Model | `models/gemini-embedding-001` |
| Dimensions | 768 (set via `output_dimensionality`) |
| Task type (documents) | `RETRIEVAL_DOCUMENT` |
| Task type (queries) | `RETRIEVAL_QUERY` |
| Async wrapping | `asyncio.to_thread()` (Gemini SDK is synchronous) |

### Retrieval

```sql
SELECT id, document_id, chunk_text,
       embedding <=> :query_vec AS distance
FROM document_chunks
WHERE tenant_id = :tenant_id
ORDER BY embedding <=> :query_vec
LIMIT 5
```

- `<=>` is pgvector's **cosine distance** operator. Range: `[0, 2]`. `0` = identical vectors.
- Results are always filtered by `tenant_id` — no cross-tenant data leakage.
- Top-5 candidates are retrieved, then filtered by the distance threshold.

### Distance Threshold

- **Threshold:** `distance < 0.35`
- Chunks with `distance >= 0.35` are discarded.
- If **zero chunks** survive the threshold, the system returns a refusal response and the LLM is **never called**.
- This prevents hallucination and saves API cost on irrelevant queries.

---

## 6. Cost Control (Section C)

| Control | Value | Effect |
|---------|-------|--------|
| `max_output_tokens` | 500 | Caps LLM output per call |
| Context truncation | 3,000 tokens | Limits prompt size sent to LLM |
| Redis cache | TTL 900s (15 min) | Identical questions (per tenant) skip embedding + LLM entirely |
| Distance gate | threshold 0.35 | If no relevant chunks exist, LLM is skipped — zero generation cost |
| Token logging | Every call | Approximate prompt and output tokens logged for cost monitoring |

### When the LLM is NOT called

| Scenario | Behavior |
|----------|----------|
| Question was asked before (same tenant, same text) | Cache hit → return immediately |
| No chunk passes distance threshold | Refusal response → LLM skipped |
| Tenant does not exist | HTTP 404 → pipeline aborted before embedding |

---

## 7. Tenant Isolation (Section D)

| Layer | Mechanism |
|-------|-----------|
| **Database queries** | Every query includes `WHERE tenant_id = :tenant_id` |
| **Vector search** | pgvector similarity query is filtered by `tenant_id` — a tenant's question will never match another tenant's documents |
| **Cache keys** | Key format: `{tenant_id}:{sha256(question)}` — inherently scoped, no cross-tenant collision |
| **LLM prompt** | Only chunks belonging to the validated tenant are injected into context |
| **Tenant validation** | If `tenant_id` does not exist in the `tenants` table, the request is rejected with HTTP 404 before any processing |

---

## 8. Assumptions & Trade-offs

### Assumptions

1. Documents are plain text or markdown. No PDF/DOCX parsing.
2. A single Gemini API key is shared across all tenants. Rate limits apply globally.
3. The system runs behind an API gateway or reverse proxy that handles authentication.
4. Document sizes are reasonable (< 100K tokens per document).
5. PostgreSQL handles the vector search load at current scale (< 1M chunks per tenant).

### Trade-offs

| Decision | Benefit | Risk |
|----------|---------|------|
| No Alembic migrations | Simpler setup, faster iteration | Schema changes require manual intervention or data reset |
| `metadata.create_all()` on startup | Zero-config database initialization | Not safe for production schema evolution |
| No authentication | Reduces scope for 90-minute assessment | API is open — must be placed behind a gateway |
| Synchronous Gemini SDK + `asyncio.to_thread()` | Non-blocking without native async client | Thread pool overhead |
| Single API key for all tenants | Simple configuration | No per-tenant rate limiting or billing |
| Free-tier LLM (OpenRouter) | Zero cost | Rate limits, potential downtime, no SLA |
| No background workers | Simpler architecture | Large documents block the request until all chunks are embedded |
| tiktoken dependency | Accurate token counting | Adds ~5MB to Docker image |

---

## 9. Future Improvements

**2-week roadmap:**

- JWT authentication and per-tenant API key management.
- Alembic migrations for safe schema evolution.
- Background document processing with a task queue (Celery or arq).
- Per-tenant rate limiting at the application layer.
- IVFFlat or HNSW index on pgvector for sub-linear search at scale.
- Admin dashboard — document listing, usage statistics, cost tracking.
- Configurable chunking strategies per tenant.
- Fallback LLM provider if OpenRouter is unavailable.

**Not yet justified:**

- Fine-tuned embeddings or custom models.
- Multi-step agent workflows.
- Real-time document sync from external sources.
- Multi-language support.

---

## 10. Runbook

Step-by-step guide to start the system and verify every feature.

### Prerequisites

| Requirement | Check |
|-------------|-------|
| Docker Desktop | `docker --version` |
| Docker Compose | `docker compose version` |
| Gemini API key | [aistudio.google.com](https://aistudio.google.com) |
| OpenRouter API key | [openrouter.ai/keys](https://openrouter.ai/keys) |

### 1. Start the System

```bash
# Clone and enter the project
git clone <repo-url> && cd aura

# Create environment file from template
cp .env.example .env

# Edit .env and set both API keys:
#   GEMINI_API_KEY=<your-gemini-key>
#   OPENROUTER_API_KEY=<your-openrouter-key>

# Build and start all services
cd src/infra
docker compose up --build
```

Three containers will start:

| Container | Image | Port |
|-----------|-------|------|
| `infra-backend-1` | python:3.10-slim (custom) | 8000 |
| `infra-postgres-1` | pgvector/pgvector:pg16 | 5432 |
| `infra-redis-1` | redis:7-alpine | 6379 |

Wait for the log line: `Database initialised successfully.`

### 2. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "database": "ok",
  "redis": "ok"
}
```

### 3. Create a Tenant

```bash
curl -X POST http://localhost:8000/tenants \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp"}'
```

Expected response (201 Created):

```json
{
  "id": "a1b2c3d4-0000-0000-0000-000000000000",
  "name": "Acme Corp",
  "created_at": "2025-02-15T10:00:00.000000"
}
```

Save the `id` value — it is used as `tenant_id` in all subsequent requests.

### 4. Upload a Document

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<TENANT_ID>",
    "title": "Employee Onboarding Guide",
    "content": "Welcome to Acme Corp. Your first week involves orientation, setting up your workstation, and meeting your team lead. All new employees must complete the security training module within 5 business days. Contact HR at hr@acme.com for badge access. IT will provide your laptop on day 1."
  }'
```

Expected response (201 Created):

```json
{
  "id": "d1e2f3a4-0000-0000-0000-000000000000",
  "tenant_id": "<TENANT_ID>",
  "title": "Employee Onboarding Guide",
  "chunks_created": 1,
  "created_at": "2025-02-15T10:01:00.000000"
}
```

**What happens internally:**

1. Document record is created in `documents`.
2. Content is split into chunks (500–800 tokens, 100-token overlap).
3. Each chunk is embedded via Gemini API (768-dim vector).
4. Chunks + embeddings are stored in `document_chunks`.

### 5. Ask a Question (Normal RAG Flow)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<TENANT_ID>",
    "question": "What do I need to do in my first week?"
  }'
```

Expected response (200 OK):

```json
{
  "answer": "In your first week, you need to complete orientation, set up your workstation, and meet your team lead. You must also complete the security training module within 5 business days.",
  "citations": ["d1e2f3a4-0000-0000-0000-000000000000"],
  "confidence": 0.95,
  "reason": null,
  "cached": false
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string \| null` | LLM-generated answer, or `null` if context is insufficient |
| `citations` | `string[]` | Document IDs used to generate the answer |
| `confidence` | `float` | Model's self-assessed confidence (0.0–1.0) |
| `reason` | `string \| null` | Refusal reason, e.g. `"insufficient_context"` |
| `cached` | `bool` | `true` if the response came from Redis cache |

### 6. Cache Hit Demonstration

Repeat the exact same question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<TENANT_ID>",
    "question": "What do I need to do in my first week?"
  }'
```

Expected response:

```json
{
  "answer": "In your first week, you need to complete orientation...",
  "citations": ["d1e2f3a4-0000-0000-0000-000000000000"],
  "confidence": 0.95,
  "reason": null,
  "cached": true
}
```

Note `"cached": true`. The system did not call the embedding API or the LLM. The response was served directly from Redis using the cache key `{tenant_id}:{sha256(question)}` with a TTL of 900 seconds (15 minutes).

### 7. Refusal Example (Insufficient Context)

Ask a question unrelated to any uploaded document:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<TENANT_ID>",
    "question": "What is the current stock price of Tesla?"
  }'
```

Expected response:

```json
{
  "answer": null,
  "citations": [],
  "confidence": 0.0,
  "reason": "insufficient_context",
  "cached": false
}
```

**What happened:** The question was embedded and compared against all chunks belonging to this tenant. No chunk had a cosine distance below 0.35, so the distance threshold gate rejected the query. The LLM was **never called**. This saves cost and prevents hallucination.

### 8. Tenant Isolation Demonstration

Create a second tenant:

```bash
curl -X POST http://localhost:8000/tenants \
  -H "Content-Type: application/json" \
  -d '{"name": "Beta Inc"}'
```

Ask the same question using the new tenant's ID:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<BETA_TENANT_ID>",
    "question": "What do I need to do in my first week?"
  }'
```

Expected response:

```json
{
  "answer": null,
  "citations": [],
  "confidence": 0.0,
  "reason": "insufficient_context",
  "cached": false
}
```

Beta Inc has no documents. The vector search query includes `WHERE tenant_id = :tenant_id`, so it returns zero results despite Acme Corp having a relevant document. Tenant isolation is enforced at the database level.

### Stopping the System

```bash
cd src/infra

# Stop containers (preserve data)
docker compose down

# Stop containers and delete all data
docker compose down -v
```
