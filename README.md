# RAG Persona Chatbot

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://rag-persona-chatbot-kastack.streamlit.app)
[![Video Demo](https://img.shields.io/badge/Video%20Demo-Loom-625DF5?logo=loom&logoColor=white)](https://www.loom.com/share/e965953ba23b4d31a50c0a4310e24198)

> 🌐 **Hosted app:** [https://rag-persona-chatbot-kastack.streamlit.app](https://rag-persona-chatbot-kastack.streamlit.app)  
> 🎬 **Video demo:** [https://www.loom.com/share/e965953ba23b4d31a50c0a4310e24198](https://www.loom.com/share/e965953ba23b4d31a50c0a4310e24198)

> **ML Engineer Take-Home Assignment — KaStack Labs**  
> A two-stage pipeline: an offline indexer (`build_index.py`) and a Streamlit chatbot UI (`app.py`).

---

## Architecture Overview

```
conversations.csv
       │
       ▼
 build_index.py  ──► chroma_db/  (topic summaries + chunk summaries + raw messages)
                 ──► personas/persona.json
                              │
                              ▼
                          app.py  (Streamlit UI — reads index, serves chatbot)
```

**The two components are strictly separated**: `build_index.py` is a one-time offline job; `app.py` is a read-only runtime that never touches the CSV.

---

## 1. Topic Change Detection (Cosine Similarity)

### How it works

All messages are embedded with **`all-MiniLM-L6-v2`** (sentence-transformers), producing a 384-dimensional dense vector per message.

The conversation is divided into **non-overlapping adjacent windows** of `N = 12` consecutive messages. For each window `i`, a **mean embedding** is computed:

```
win_emb[i] = mean(embed(msg[i·N]), embed(msg[i·N+1]), ..., embed(msg[i·N+11]))
```

Consecutive window embeddings are compared using **cosine similarity**:

```
similarity(win[i], win[i+1]) = (win[i] · win[i+1]) / (‖win[i]‖ · ‖win[i+1]‖)
```

A topic boundary is fired when **both** conditions are met:
1. `similarity < 0.62` (genuine semantic shift detected)
2. At least **30 messages** have accumulated since the last boundary (prevents micro-fragments)

### Why non-overlapping windows?

Step-1 sliding windows share `N−1` messages with their neighbour, so their mean embeddings are nearly identical regardless of topic — similarity is always > 0.9 and no threshold can discriminate. Non-overlapping windows share **zero** messages, making the similarity a true measure of topical continuity.

### Why threshold 0.62?

With 12-message windows:
- Same-topic adjacent windows cluster at similarity **0.65–0.85**
- A genuine topic shift (e.g. cooking → job change) drops to **0.2–0.45**
- The threshold of **0.62** sits cleanly in the gap

### What happens next

Each detected segment is passed to **local Ollama (`llama3.1:8b`)** to generate a 2-3 sentence summary. The summary, along with the mean embedding of the segment, is upserted into the **`topic_summaries`** ChromaDB collection:

```json
{"type": "topic_summary", "start_id": 42, "end_id": 87}
```

---

## 2. Three-Stage Retrieval

At query time, `app.py` performs **three independent semantic queries** against ChromaDB using the embedded user question:

### Stage 1 — Macro Context: Topic Summaries (`top-3`)

The `topic_summaries` collection is queried for the **3 most semantically relevant topic summaries**. These are high-level, LLM-generated abstractions of conversation segments.

**Purpose**: Give the LLM broad thematic context — *what topics were discussed that relate to this question?*

### Stage 2 — Chronological Context: Chunk Summaries (`top-2`)

The `chunk_summaries` collection is queried for the **2 most relevant 100-message rolling summaries**. These provide a chronological narrative of what was happening around the relevant time period.

**Purpose**: Bridge the gap between macro themes and individual messages — *what was the conversation about in that window of time?*

### Stage 3 — Micro Context: Raw Messages (`top-5`)

The `raw_messages` collection is queried for the **5 most semantically similar individual messages**. These are verbatim lines from the conversation.

**Purpose**: Ground the LLM's answer in specific, concrete evidence — *what was actually said?*

### Fusion

All three result sets are concatenated into a single context block sent to the LLM as part of its system prompt:

```
### 📌 MACRO CONTEXT (Topic Summaries)
<summary_1>  <summary_2>  <summary_3>

### 📅 CHRONOLOGICAL CONTEXT (Rolling Summaries)
<chunk_summary_1>  <chunk_summary_2>

### 🔍 MICRO CONTEXT (Exact Quotes)
<raw_msg_1> ... <raw_msg_5>
```

This three-layer structure ensures the LLM has both high-level thematic awareness and verbatim evidence before answering.

---

## 3. Map-Reduce Persona Consolidation

Extracting a persona from 191,000+ messages in a single LLM call is infeasible (context limits, quality). A **map-reduce** strategy is used, independently for **both User 1 and User 2**.

### Map Phase

The full message list is divided into **chunks of 200 messages**. For each chunk and each user, an Ollama call is made in **JSON mode** (`format="json"`). The required schema is injected into the system prompt:

```json
{
  "user_id": "User 1",
  "habits": ["..."],
  "communication_style": {"tone": "...", "emoji_usage": "..."},
  "personality_traits": ["..."],
  "personal_facts": {
    "facts_with_history": [
      {"fact": "occupation", "history": [{"value": "student"}], "latest": "student"}
    ]
  }
}
```

This produces a **list of partial personas** — each capturing what was observable in that 200-message window.

### Reduce Phase

All partial personas for a given user are passed to a **single final Ollama call** with instructions to:
- **Deduplicate** habits and personality traits.
- **Merge fact histories** — if `occupation` was `"student"` early and `"software engineer"` later, both are preserved in `history[]` with the latest value surfaced in `"latest"`.
- **Synthesize** a single communication style from all observations.

The result is written to **`personas/persona.json`** as a list of two consolidated persona objects (one per user).

---

## 4. Setup & Run Instructions

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.com) installed and running locally (for `build_index.py`)
- A free [Groq API key](https://console.groq.com) (for `app.py` chatbot)

### Step 1 — Clone and configure

```bash
# Copy and fill in your Groq API key (used by app.py only)
cp .env.example .env
# Edit .env:  GROQ_API_KEY=your_key_here
```

### Step 2 — Pull the Ollama model

```bash
# Make sure Ollama is running, then pull the model:
ollama pull llama3.1:8b
```

### Step 3 — Install dependencies

```bash
uv sync
```

Or with plain pip:

```bash
pip install -r requirements.txt
```

### Step 4 — Place the dataset

Make sure `conversations.csv` is in the project root:

```
rag-persona-chatbot/
├── conversations.csv   ← here
├── build_index.py
└── app.py
```

### Step 5 — Build the index (run ONCE)

```bash
uv run python build_index.py
```

This will:
1. Parse the CSV and embed all messages using `all-MiniLM-L6-v2` (local, CPU).
2. Detect topic boundaries and summarize each segment via local Ollama.
3. Create 100-message rolling summaries via local Ollama.
4. Store all data in `./chroma_db/`.
5. Run map-reduce persona extraction → `./personas/persona.json`.

> ℹ️ **No rate limits**: All ingestion LLM calls use local Ollama — they run as fast as your hardware allows with no API quotas.

### Step 6 — Launch the chatbot

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| LLM — ingestion | `ollama` (`llama3.1:8b`, local, no rate limits) |
| LLM — chatbot UI | `groq` SDK (`llama-3.1-8b-instant` via Groq API) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local) |
| Vector DB | `chromadb` (PersistentClient) |
| UI | `streamlit` |
| Package Manager | `uv` |

---

## Project Structure

```
rag-persona-chatbot/
├── conversations.csv          # Input dataset (not committed to repo)
├── build_index.py             # Offline ingestion pipeline
├── app.py                     # Streamlit chatbot UI
├── pyproject.toml             # uv project manifest
├── requirements.txt           # pip-compatible deps
├── .env.example               # API key template (copy to .env)
├── chroma_db/                 # Generated: ChromaDB persistent storage
└── personas/
    └── persona.json           # Generated: extracted user personas
```
