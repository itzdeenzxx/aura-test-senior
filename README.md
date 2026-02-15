# Aura — Multi-Tenant RAG Knowledge Assistant

ระบบ Backend สำหรับ Internal Knowledge Assistant แบบ Multi-Tenant  
ใช้เทคนิค RAG (Retrieval-Augmented Generation) ให้พนักงานถามคำถามจากเอกสารภายในองค์กรได้เป็นภาษาธรรมชาติ  
ระบบจะตอบจากเอกสารจริงเท่านั้น พร้อมอ้างอิง Document ID — ไม่มีการ hallucinate

**Tech Stack:**
| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python 3.10, async) |
| Database | PostgreSQL 16 + pgvector |
| Cache | Redis 7 |
| Embedding | Google Gemini (`gemini-embedding-001`, 768 dims) |
| LLM | OpenRouter (`google/gemma-3-12b-it:free`) via OpenAI SDK |
| Containerization | Docker & Docker Compose |

---

## สารบัญ

1. [วิธีรันระบบ (Quick Start)](#1-วิธีรันระบบ-quick-start)
2. [โครงสร้างโปรเจกต์](#2-โครงสร้างโปรเจกต์)
3. [System Architecture](#3-system-architecture)
4. [API Endpoints](#4-api-endpoints)
5. [ทดสอบระบบ (Testing Guide)](#5-ทดสอบระบบ-testing-guide)
6. [Data Model](#6-data-model)
7. [RAG Design (Section B)](#7-rag-design-section-b)
8. [Cost Control Strategy (Section C)](#8-cost-control-strategy-section-c)
9. [Tenant Isolation Strategy (Section D)](#9-tenant-isolation-strategy-section-d)
10. [Problem Framing (Section A)](#10-problem-framing-section-a)
11. [Trade-offs & Assumptions](#11-trade-offs--assumptions)
12. [Future Improvements (Section E)](#12-future-improvements-section-e)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. วิธีรันระบบ (Quick Start)

### สิ่งที่ต้องมี

| ต้องมี | วิธีตรวจสอบ |
|--------|-------------|
| Docker Desktop | `docker --version` |
| Docker Compose | `docker compose version` |
| Google Gemini API Key | ขอได้ที่ [aistudio.google.com](https://aistudio.google.com) |
| OpenRouter API Key | ขอได้ที่ [openrouter.ai/keys](https://openrouter.ai/keys) |

### ขั้นตอนการรัน

```bash
# 1. Clone โปรเจกต์
git clone <repo-url>
cd aura

# 2. สร้างไฟล์ .env จาก template
cp .env.example .env

# 3. แก้ไข .env — ใส่ API keys ทั้ง 2 ตัว
#    GEMINI_API_KEY=your_gemini_key_here
#    OPENROUTER_API_KEY=your_openrouter_key_here

# 4. สั่ง Docker build + start ทุก container
cd src/infra
docker compose up --build
```

> ⏱ การ build ครั้งแรกใช้เวลาประมาณ 2-3 นาที (ดาวน์โหลด Python packages)

### ตรวจสอบว่าระบบพร้อมใช้งาน

เมื่อ build เสร็จ จะเห็น log ว่า `Database initialised successfully` ให้ทดสอบด้วย:

```bash
curl http://localhost:8000/health
```

ถ้าได้ผลลัพธ์นี้ แปลว่าระบบพร้อมทำงาน:

```json
{"status": "healthy", "database": "ok", "redis": "ok"}
```

### Docker Containers ที่รันอยู่

| Container | Image | Port | หน้าที่ |
|-----------|-------|------|---------|
| `infra-backend-1` | python:3.10-slim (custom build) | 8000 | FastAPI application server |
| `infra-postgres-1` | pgvector/pgvector:pg16 | 5432 | ฐานข้อมูล + vector search |
| `infra-redis-1` | redis:7-alpine | 6379 | Cache คำตอบที่เคยถามแล้ว |

### การหยุดระบบ

```bash
cd src/infra

# หยุดทุก container (เก็บข้อมูลไว้)
docker compose down

# หยุดทุก container + ลบข้อมูลทั้งหมด (reset)
docker compose down -v
```

---

## 2. โครงสร้างโปรเจกต์

```
aura/
├── .env                    ← API keys (ไม่ commit ขึ้น git)
├── .env.example            ← Template สำหรับ .env
├── .gitignore
├── Dockerfile              ← Build image สำหรับ backend
├── README.md               ← ไฟล์นี้
├── AI_PROMPTS.md           ← บันทึก Prompt iteration history
├── requirements.txt        ← Python dependencies (11 packages)
└── src/
    ├── __init__.py
    ├── backend/
    │   ├── __init__.py
    │   ├── config.py           ← Configuration (env vars + RAG constants)
    │   ├── database.py         ← SQLAlchemy async engine + session
    │   ├── dependencies.py     ← FastAPI deps (DB, Redis, tenant validation)
    │   ├── main.py             ← FastAPI app + 4 endpoints + lifecycle
    │   ├── models.py           ← ORM models (5 tables)
    │   ├── schemas.py          ← Pydantic request/response schemas
    │   └── services/
    │       ├── __init__.py
    │       ├── cache_service.py      ← Redis caching (tenant-scoped)
    │       ├── document_service.py   ← Ingest → chunk → embed → store
    │       ├── embedding_service.py  ← Gemini embedding API
    │       ├── llm_service.py        ← OpenRouter LLM generation
    │       └── retrieval_service.py  ← pgvector cosine distance search
    └── infra/
        └── docker-compose.yml  ← Docker Compose (3 services)
```

**ไฟล์สำคัญและหน้าที่:**

| ไฟล์ | หน้าที่ |
|------|---------|
| `config.py` | โหลด environment variables ผ่าน pydantic-settings, fail-fast ถ้าไม่มี API key |
| `main.py` | FastAPI app, 4 endpoints, startup สร้างตาราง, shutdown ปิด connection |
| `models.py` | SQLAlchemy ORM — 5 ตาราง (tenants, documents, document_chunks, ai_requests, ai_responses) |
| `schemas.py` | Pydantic schemas สำหรับ validate request/response ทุก endpoint |
| `dependencies.py` | FastAPI Depends — inject DB session, Redis client, tenant validation |
| `embedding_service.py` | เรียก Gemini API สร้าง embedding vector (768 dims) ผ่าน `asyncio.to_thread()` |
| `llm_service.py` | สร้างคำตอบจาก LLM ผ่าน OpenRouter, parse JSON, ตัด context ไม่เกิน 3,000 tokens |
| `retrieval_service.py` | ค้นหา chunks ที่เกี่ยวข้องด้วย pgvector cosine distance, filter by tenant |
| `document_service.py` | pipeline: รับเอกสาร → ตัดเป็น chunks → สร้าง embeddings → เก็บลง DB |
| `cache_service.py` | Redis cache ด้วย key format `{tenant_id}:{sha256(question)}`, TTL 15 นาที |

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Client (curl / Postman)                 │
└────────────────────────────┬─────────────────────────────────┘
                             │ HTTP (port 8000)
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (:8000)                      │
│                                                              │
│  POST /tenants     — สร้าง tenant ใหม่                        │
│  POST /documents   — อัปโหลดเอกสาร (chunk + embed + store)    │
│  POST /ask         — ถามคำถาม (RAG pipeline)                  │
│  GET  /health      — ตรวจสอบสถานะระบบ                         │
│                                                              │
│  ┌────────────────┐ ┌───────────────┐ ┌────────────────────┐ │
│  │ Document       │ │ Embedding     │ │ LLM Service        │ │
│  │ Service        │ │ Service       │ │ (OpenRouter)       │ │
│  │ (chunk text)   │ │ (Gemini API)  │ │ gemma-3-12b-it     │ │
│  └───────┬────────┘ └───────┬───────┘ └──────────┬─────────┘ │
│          │                  │                    │           │
│  ┌───────┴──────────────────┴────────────────────┴─────────┐ │
│  │              Retrieval Service                           │ │
│  │     pgvector cosine distance + tenant filter             │ │
│  └─────────────────────────┬───────────────────────────────┘ │
│                            │                                 │
│  ┌─────────────────────────┴───────────────────────────────┐ │
│  │              Cache Service (Redis, TTL 15 min)          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────┬──────────────────────────────────┬─────────────┘
              │                                  │
              ▼                                  ▼
┌──────────────────────┐          ┌──────────────────────────┐
│   PostgreSQL 16      │          │       Redis 7            │
│   + pgvector         │          │   (cache, TTL=900s)      │
│                      │          │                          │
│  tenants             │          │  key format:             │
│  documents           │          │  {tenant_id}:{sha256(q)} │
│  document_chunks     │          └──────────────────────────┘
│  ai_requests         │
│  ai_responses        │
└──────────────────────┘
```

### RAG Pipeline Flow (POST /ask)

เมื่อมีคำถามเข้ามา ระบบทำงาน 9 ขั้นตอนตามลำดับ:

```
  ① Validate Tenant ──→ tenant ไม่มี? → 404
  ② Check Cache ──────→ เคยถามแล้ว? → return cached answer
  ③ Embed Question ───→ แปลงคำถามเป็น vector (Gemini API)
  ④ Retrieve Chunks ──→ ค้นหา top-5 chunks จาก pgvector (cosine distance)
  ⑤ Threshold Gate ───→ ไม่มี chunk ที่ distance < 0.35? → refuse (ไม่เรียก LLM)
  ⑥ Generate Answer ──→ ส่ง context + question ไป OpenRouter LLM
  ⑦ Log to DB ────────→ บันทึก question + answer ลง ai_requests / ai_responses
  ⑧ Cache Answer ─────→ เก็บคำตอบใน Redis (TTL 15 นาที)
  ⑨ Return JSON ──────→ ส่ง answer + citations + confidence กลับ client
```

---

## 4. API Endpoints

### `GET /health` — ตรวจสอบสถานะระบบ

ตรวจสอบว่า Database และ Redis เชื่อมต่อได้

```bash
curl http://localhost:8000/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "database": "ok",
  "redis": "ok"
}
```

---

### `POST /tenants` — สร้าง Tenant ใหม่

ทุกเอกสารและคำถามจะผูกกับ tenant — ข้อมูลแต่ละ tenant แยกจากกันโดยสมบูรณ์

**Request:**
```bash
curl -X POST http://localhost:8000/tenants \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp"}'
```

**Response (201 Created):**
```json
{
  "id": "643d7e56-2ca5-4781-91c2-0f0a67b657d6",
  "name": "Acme Corp",
  "created_at": "2025-02-15T10:00:00"
}
```

> 💡 เก็บค่า `id` ไว้ — ต้องใช้ทุกครั้งตอนอัปโหลดเอกสารและถามคำถาม

---

### `POST /documents` — อัปโหลดเอกสาร

ส่งเอกสาร (plain text / markdown) เข้ามา ระบบจะ:
1. ตัดเป็น chunks (500-800 tokens, overlap 100 tokens)
2. สร้าง embedding vector ให้แต่ละ chunk (Gemini API)
3. เก็บ chunks + vectors ลง PostgreSQL

**Request:**
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "643d7e56-2ca5-4781-91c2-0f0a67b657d6",
    "title": "Employee Onboarding Guide",
    "content": "# Onboarding\n\nWelcome to Acme Corp. Your first week involves orientation, setting up your workstation, and meeting your team lead. All new employees must complete the security training module within 5 business days. Contact HR at hr@acme.com for badge access."
  }'
```

**Response (201 Created):**
```json
{
  "id": "b8e3f1a2-...",
  "tenant_id": "643d7e56-...",
  "title": "Employee Onboarding Guide",
  "chunks_created": 1,
  "created_at": "2025-02-15T10:01:00"
}
```

> 📌 `chunks_created` บอกว่าเอกสารถูกตัดเป็นกี่ chunks — เอกสารสั้นจะได้ 1 chunk, เอกสารยาวจะได้หลาย chunks

---

### `POST /ask` — ถามคำถาม (RAG Pipeline)

ส่งคำถามเป็นภาษาธรรมชาติ ระบบจะค้นหาเอกสารที่เกี่ยวข้องแล้วสร้างคำตอบ

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "643d7e56-2ca5-4781-91c2-0f0a67b657d6",
    "question": "What do I need to do in my first week?"
  }'
```

**Response — มีคำตอบ (200 OK):**
```json
{
  "answer": "In your first week, you need to complete orientation, set up your workstation, and meet your team lead. You must also complete the security training module within 5 business days.",
  "citations": ["b8e3f1a2-..."],
  "confidence": 0.85,
  "reason": null,
  "cached": false
}
```

**Response — ไม่มีเอกสารที่เกี่ยวข้อง (200 OK):**
```json
{
  "answer": null,
  "citations": [],
  "confidence": 0.0,
  "reason": "insufficient_context",
  "cached": false
}
```

**Response — ถามซ้ำ (cache hit):**
```json
{
  "answer": "...",
  "citations": ["b8e3f1a2-..."],
  "confidence": 0.85,
  "reason": null,
  "cached": true
}
```

**Response fields อธิบาย:**

| Field | Type | คำอธิบาย |
|-------|------|----------|
| `answer` | `string \| null` | คำตอบจาก LLM, หรือ `null` ถ้าไม่มี context เพียงพอ |
| `citations` | `string[]` | รายการ document_id ที่ใช้ตอบ |
| `confidence` | `float` | ระดับความมั่นใจ 0.0 - 1.0 |
| `reason` | `string \| null` | สาเหตุที่ตอบไม่ได้ เช่น `"insufficient_context"` |
| `cached` | `bool` | `true` = คำตอบมาจาก cache, `false` = สร้างใหม่ |

---

## 5. ทดสอบระบบ (Testing Guide)

หลังจากรันระบบแล้ว ให้ทดสอบ end-to-end ตาม flow ด้านล่าง:

### ขั้นตอนที่ 1: ตรวจสอบ Health

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

✅ ต้องได้ `"status": "healthy"`

### ขั้นตอนที่ 2: สร้าง Tenant

```bash
TENANT=$(curl -s -X POST http://localhost:8000/tenants \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp"}')
echo "$TENANT" | python3 -m json.tool

# เก็บ tenant_id ไว้ใช้ต่อ
TENANT_ID=$(echo "$TENANT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Tenant ID: $TENANT_ID"
```

### ขั้นตอนที่ 3: อัปโหลดเอกสาร

```bash
curl -s -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"$TENANT_ID\",
    \"title\": \"Employee Onboarding Guide\",
    \"content\": \"Welcome to Acme Corp. Your first week involves orientation, setting up your workstation, and meeting your team lead. All new employees must complete the security training module within 5 business days. Contact HR at hr@acme.com for badge access. IT will provide your laptop on day 1. You will be assigned a buddy who will help you navigate the first month.\"
  }" | python3 -m json.tool
```

✅ ต้องได้ `"chunks_created": 1` (หรือมากกว่าถ้าเอกสารยาว)

### ขั้นตอนที่ 4: ถามคำถามที่มีคำตอบในเอกสาร

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"$TENANT_ID\",
    \"question\": \"What do I need to do in my first week?\"
  }" | python3 -m json.tool
```

✅ ต้องได้ `"answer"` ที่พูดถึง orientation, workstation, team lead  
✅ `"citations"` ต้องมี document_id อยู่  
✅ `"cached": false` (ถามครั้งแรก)

### ขั้นตอนที่ 5: ถามซ้ำ — ทดสอบ Cache

```bash
# ถามคำถามเดิมอีกครั้ง
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"$TENANT_ID\",
    \"question\": \"What do I need to do in my first week?\"
  }" | python3 -m json.tool
```

✅ `"cached": true` — คำตอบมาจาก Redis ไม่ได้เรียก LLM ใหม่

### ขั้นตอนที่ 6: ถามคำถามที่ไม่เกี่ยวข้อง — ทดสอบ Refusal

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"$TENANT_ID\",
    \"question\": \"What is the current stock price of Tesla?\"
  }" | python3 -m json.tool
```

✅ `"answer": null`  
✅ `"reason": "insufficient_context"` — ระบบปฏิเสธเพราะไม่มีเอกสารที่เกี่ยวข้อง  
✅ LLM ไม่ถูกเรียก (ประหยัด cost)

### ขั้นตอนที่ 7: ทดสอบ Tenant Isolation

```bash
# สร้าง tenant ใหม่
TENANT2=$(curl -s -X POST http://localhost:8000/tenants \
  -H "Content-Type: application/json" \
  -d '{"name": "Beta Inc"}')
TENANT2_ID=$(echo "$TENANT2" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# ใช้ tenant ใหม่ ถามคำถามเกี่ยวกับเอกสารของ Acme Corp
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{
    \"tenant_id\": \"$TENANT2_ID\",
    \"question\": \"What do I need to do in my first week?\"
  }" | python3 -m json.tool
```

✅ `"answer": null`, `"reason": "insufficient_context"`  
✅ Beta Inc ไม่สามารถเข้าถึงเอกสารของ Acme Corp ได้ — tenant isolation ทำงานถูกต้อง

---

## 6. Data Model

ระบบมี 5 ตาราง สร้างอัตโนมัติตอน startup ด้วย `Base.metadata.create_all()` (ไม่ใช้ Alembic):

```
┌──────────────┐       ┌──────────────────┐       ┌────────────────────┐
│   tenants    │──1:N──│   documents      │──1:N──│  document_chunks   │
│              │       │                  │       │                    │
│  id (UUID)   │       │  id (UUID)       │       │  id (UUID)         │
│  name        │       │  tenant_id (FK)  │       │  document_id (FK)  │
│  created_at  │       │  title           │       │  tenant_id (idx)   │
└──────────────┘       │  content         │       │  chunk_text        │
                       │  created_at      │       │  embedding Vec(768)│
                       └──────────────────┘       │  created_at        │
                                                  └────────────────────┘

┌──────────────────┐       ┌──────────────────┐
│   ai_requests    │──1:1──│  ai_responses    │
│                  │       │                  │
│  id (UUID)       │       │  id (UUID)       │
│  tenant_id (idx) │       │  request_id (FK) │
│  question        │       │  answer_json     │
│  created_at      │       │  created_at      │
└──────────────────┘       └──────────────────┘
```

| ตาราง | หน้าที่ |
|-------|---------|
| `tenants` | ทะเบียน tenant (องค์กร). UUID primary key. |
| `documents` | เอกสารดิบ ผูกกับ tenant ผ่าน FK → tenants. |
| `document_chunks` | เอกสารที่ตัดเป็น chunks + embedding vector(768). FK → documents. มี index บน tenant_id. |
| `ai_requests` | Audit log ของทุกคำถามที่ถูกถาม ผูกกับ tenant. |
| `ai_responses` | Audit log ของทุกคำตอบ (JSONB) ผูกกับ request. |

**Key points:**
- ทุกตารางที่มีข้อมูล tenant มี `tenant_id` column พร้อม index
- `document_chunks.embedding` เป็น `pgvector Vector(768)` สำหรับ Gemini `gemini-embedding-001`
- ไม่ใช้ migration — สร้างตารางด้วย `metadata.create_all()` ตอน startup
- `ai_responses.answer_json` เป็น JSONB — เก็บ answer, citations, confidence ทั้งหมด

---

## 7. RAG Design (Section B)

### 7.1 Chunking Strategy

| Parameter | ค่า | เหตุผล |
|-----------|-----|--------|
| Tokenizer | tiktoken `cl100k_base` | นับ token ได้แม่นยำ ใช้ encoding เดียวกับ GPT models |
| Chunk size | 500-800 tokens | สมดุลระหว่าง granularity กับ context — ไม่สั้นเกินไป (context น้อย) ไม่ยาวเกินไป (relevance ลดลง) |
| Overlap | 100 tokens | ป้องกันการตัดข้อมูลสำคัญที่อยู่ตรงขอบ chunk |
| Trailing fragment | ต่อท้าย chunk สุดท้าย | ถ้า fragment สั้นกว่า 250 tokens (ครึ่งของ min) จะไม่สร้าง chunk ใหม่ แต่ต่อเข้า chunk ก่อนหน้า |

**อัลกอริทึม:**
1. Tokenize เอกสารทั้งหมดด้วย tiktoken
2. เลื่อน window ขนาด 800 tokens ไปทีละ 700 tokens (800 - 100 overlap)
3. Decode แต่ละ window กลับเป็น text
4. Fragment สุดท้ายที่สั้นกว่า 250 tokens จะถูกต่อเข้า chunk ก่อนหน้า

### 7.2 Embedding

| Parameter | ค่า |
|-----------|-----|
| Model | `models/gemini-embedding-001` |
| Dimensions | 768 (กำหนดผ่าน `output_dimensionality`) |
| Task type (documents) | `RETRIEVAL_DOCUMENT` |
| Task type (questions) | `RETRIEVAL_QUERY` |
| Async wrapping | `asyncio.to_thread()` (Gemini SDK เป็น sync) |

### 7.3 Retrieval — Tenant-Filtered Vector Search

```sql
SELECT id, document_id, chunk_text,
       embedding <=> :query_vec AS distance
FROM document_chunks
WHERE tenant_id = :tenant_id
ORDER BY embedding <=> :query_vec
LIMIT 5
```

- `<=>` คือ cosine distance operator ของ pgvector (range [0, 2]; 0 = เหมือนกันที่สุด)
- Filter `WHERE tenant_id = :tenant_id` **เสมอ** — ไม่มี cross-tenant leakage
- ดึง top-5 chunks ที่ใกล้คำถามที่สุด

### 7.4 Similarity Threshold Logic

- **Threshold:** `distance < 0.35` (cosine distance, เทียบเท่า similarity > 0.65)
- ถ้า **ไม่มี chunk ใด** ผ่าน threshold → ระบบ return refusal ทันที:
  ```json
  {"answer": null, "reason": "insufficient_context", "confidence": 0.0}
  ```
- **LLM จะไม่ถูกเรียก** ถ้าไม่มี chunks ที่เกี่ยวข้อง — ประหยัด cost และป้องกัน hallucination

### 7.5 LLM Generation

| Parameter | ค่า |
|-----------|-----|
| Provider | OpenRouter (OpenAI-compatible API) |
| Model | `google/gemma-3-12b-it:free` |
| Max output tokens | 500 |
| Temperature | 0.1 (ให้ตอบ deterministic) |
| Output format | JSON only |
| Context budget | ≤ 3,000 tokens (ตัดถ้าเกิน) |

**Prompt structure:**
- Gemma ไม่รองรับ system instruction → merge system prompt เข้าไปใน user message
- System prompt สั่งให้ตอบจาก context ที่ให้เท่านั้น ถ้าไม่มีให้ return `insufficient_context`
- ถ้า LLM ตอบมาเป็น markdown code fence (`` ```json...``` ``) ระบบจะ strip ออกก่อน parse

---

## 8. Cost Control Strategy (Section C)

| Control | Implementation | ผลลัพธ์ |
|---------|----------------|---------|
| **Distance gate** | ถ้าไม่มี chunk ที่ distance < 0.35 → ไม่เรียก LLM | ไม่เสีย cost เมื่อไม่มีข้อมูลที่เกี่ยวข้อง |
| **Redis cache** | คำถามเดิม (per tenant) cache 15 นาที | ไม่เรียก embedding + LLM ซ้ำ |
| **max_output_tokens** | จำกัด 500 tokens ต่อ generation call | ป้องกันคำตอบยาวเกินจำเป็น |
| **Context truncation** | ตัด context ไม่เกิน 3,000 tokens ก่อนส่ง LLM | ลดจำนวน prompt tokens |
| **Token logging** | Log prompt + output tokens ทุก LLM call | ติดตาม cost ได้ |
| **JSON-only output** | บังคับ LLM ตอบเป็น JSON เท่านั้น | ป้องกัน verbose prose |

### เมื่อไหร่ที่ AI ไม่ควรถูกเรียกใช้

| สถานการณ์ | การจัดการ |
|-----------|----------|
| คำถามเดิมที่เคยถามแล้ว | → Cache hit, return ทันที |
| ไม่มี document chunks ที่เกี่ยวข้อง | → Refuse, ไม่เรียก LLM |
| คำถามที่ตอบได้ด้วย keyword search / FAQ | → ควรมี FAQ endpoint แยก (future) |
| การตัดสินใจที่ต้องใช้วิจารณญาณมนุษย์ (HR, legal) | → ระบุไว้ในนโยบายการใช้งาน |

---

## 9. Tenant Isolation Strategy (Section D)

### 9.1 Database-Level Isolation
- ทุก query มี `WHERE tenant_id = :tenant_id`
- `tenant_id` มี index บนทุกตารางที่มีข้อมูล tenant
- ไม่มี shared query ที่อาจ leak ข้อมูลข้าม tenant

### 9.2 Vector Search Isolation
- pgvector similarity query มี `WHERE tenant_id = :tenant_id` เสมอ
- คำถามของ tenant A จะ **ไม่มีทาง** match กับเอกสารของ tenant B

### 9.3 Cache Scoping
- Redis key format: `{tenant_id}:{sha256(question)}`
- Cache lookup เป็น tenant-scoped โดย design
- ไม่มี key collision ข้าม tenant

### 9.4 Prompt-Level Protection
- เฉพาะ chunks ของ tenant ที่ verified เท่านั้นที่จะถูก inject เข้า prompt
- System prompt สั่งให้ model ตอบ **เฉพาะจาก context ที่ให้**
- Fail-fast: ถ้า tenant ไม่มีอยู่ → HTTP 404 ทันที ก่อนทำอะไรทั้งสิ้น

---

## 10. Problem Framing (Section A)

### ผู้ใช้คือใคร?
พนักงานของบริษัท B2B SaaS ที่ต้องการคำตอบที่รวดเร็วและแม่นยำจาก knowledge base ภายใน — เช่น onboarding guides, policy documents, technical runbooks

### ทำไม Keyword Search ถึงไม่เพียงพอ?
- Keyword search ตอบ semantic intent ไม่ได้ — เช่น ถาม "นโยบายคืนเงินคืออะไร?" จะไม่ match กับเอกสารชื่อ "Customer Satisfaction Procedures"
- พนักงานเสียเวลาค้นหาเอกสารจำนวนมาก
- Knowledge base ของแต่ละ tenant ต้องแยกจากกันอย่างเคร่งครัด

### ทำไมต้อง RAG?
RAG (Retrieval-Augmented Generation) ให้ LLM ตอบจากเอกสารจริงของ tenant โดยเฉพาะ:
- **Retrieval** — ค้นหาเอกสารที่เกี่ยวข้องด้วย semantic search (vector similarity)
- **Augmented** — นำเอกสารที่ค้นเจอมาเป็น context ให้ LLM
- **Generation** — LLM สร้างคำตอบจาก context เท่านั้น ไม่ hallucinate

---

## 11. Trade-offs & Assumptions

### Trade-offs

| การตัดสินใจ | ข้อดี | ข้อเสีย |
|-------------|-------|---------|
| ไม่ใช้ Alembic migrations | Setup ง่าย, build เร็ว | เปลี่ยน schema ต้องทำเอง |
| Gemini SDK (sync) + `asyncio.to_thread()` | ไม่ block event loop | เพิ่ม thread overhead |
| API key เดียวสำหรับทุก tenant | Config ง่าย | ไม่มี per-tenant rate limiting |
| `metadata.create_all()` ตอน startup | Dev cycle เร็ว | ไม่เหมาะกับ production schema evolution |
| tiktoken dependency | นับ token แม่นยำ | เพิ่มขนาด Docker image ~5MB |
| ไม่มี authentication | ง่ายต่อการทดสอบ | ต้องวาง API gateway ข้างหน้าใน production |
| LLM ผ่าน OpenRouter (free tier) | ฟรี, ไม่ต้องจ่าย | อาจมี rate limit, latency สูงกว่า |

### Assumptions

1. เอกสารเป็น plain text หรือ markdown (ไม่มี PDF/DOCX parsing)
2. Gemini API key เดียวใช้ร่วมกันทุก tenant (rate limit เป็น global)
3. Embedding dimension คือ 768 (Gemini `gemini-embedding-001`)
4. ระบบรันหลัง API gateway / reverse proxy ที่จัดการ auth
5. เอกสารมีขนาดสมเหตุสมผล (< 100K tokens ต่อเอกสาร)
6. PostgreSQL รองรับ vector search load ที่ scale ปัจจุบัน (< 1M chunks ต่อ tenant)

---

## 12. Future Improvements (Section E)

### สิ่งที่จะทำใน 2 สัปดาห์ถัดไป
- Authentication & API key management ต่อ tenant
- Alembic migrations สำหรับ schema evolution ที่ปลอดภัย
- Background document processing (async chunking สำหรับเอกสารขนาดใหญ่)
- Rate limiting ต่อ tenant
- Admin dashboard (รายการเอกสาร, สถิติการใช้งาน)
- IVFFlat / HNSW index บน pgvector สำหรับ performance ที่ scale

### สิ่งที่ยังไม่ควรสร้าง
- Fine-tuned embeddings หรือ custom models
- Complex agent workflows หรือ multi-step reasoning
- Real-time document sync จาก external sources
- Multi-language support

### ความเสี่ยงที่สำคัญ
1. **API rate limits / downtime** — ไม่มี fallback model สำหรับทั้ง Gemini (embedding) และ OpenRouter (LLM)
2. **pgvector performance at scale** — ยังเป็น linear scan ไม่มี IVFFlat/HNSW index
3. **ไม่มี auth** — ใครก็ตามที่รู้ tenant UUID สามารถเข้าถึงได้
4. **Single-region deployment** — ไม่มี HA หรือ disaster recovery

---

## 13. Troubleshooting

### ปัญหาที่พบบ่อย

| ปัญหา | สาเหตุ | วิธีแก้ |
|-------|--------|---------|
| Health check ได้ `"database": "error"` | PostgreSQL ยังไม่พร้อม | รอ 10 วินาทีแล้วลองใหม่ หรือ `docker compose logs postgres` |
| Health check ได้ `"redis": "error"` | Redis ยังไม่พร้อม | `docker compose logs redis` ดู error |
| Error 500 ตอนอัปโหลดเอกสาร | GEMINI_API_KEY ไม่ถูกต้อง | ตรวจสอบ `.env` ว่า key ถูกต้อง |
| Error 429 ตอนอัปโหลดเอกสาร | Gemini API rate limit | รอ 30-60 วินาทีแล้วลองใหม่ |
| Error 429 ตอนถามคำถาม | OpenRouter rate limit | รอสักครู่แล้วลองใหม่ (free tier มี limit) |
| Error 404 ตอนถามคำถาม | tenant_id ไม่ถูกต้อง | ตรวจสอบว่าใช้ tenant_id ที่ได้จาก POST /tenants |
| ได้ `insufficient_context` ทั้งที่มีเอกสาร | เอกสารไม่ตรงกับคำถาม หรือ distance > 0.35 | ลองถามคำถามที่ตรงกับเนื้อหาเอกสารมากขึ้น |
| Cache ไม่ update หลังแก้โค้ด | Redis ยังเก็บคำตอบเก่า | `docker exec infra-redis-1 redis-cli FLUSHALL` |
| Backend ไม่ start | Port 8000 ถูกใช้อยู่ | `lsof -i :8000` หา process ที่ใช้อยู่แล้ว kill |

### ดู Logs

```bash
# ดู logs ทุก container
cd src/infra
docker compose logs -f

# ดู logs เฉพาะ backend
docker compose logs -f backend

# ดู logs เฉพาะ database
docker compose logs -f postgres
```

### Reset ข้อมูลทั้งหมด

```bash
cd src/infra
docker compose down -v
docker compose up --build
```

### ตรวจสอบ Containers

```bash
cd src/infra
docker compose ps
```

ต้องเห็น 3 containers ที่ status เป็น `Up (healthy)`:
```
NAME               STATUS
infra-backend-1    Up (healthy)
infra-postgres-1   Up (healthy)
infra-redis-1      Up (healthy)
```
