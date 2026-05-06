"""
build_index.py — Offline ingestion script (run ONCE before launching app.py).

Steps executed:
  1. Parse conversations.csv into a flat chronological message list.
  2. Embed all messages with sentence-transformers (all-MiniLM-L6-v2).
  3A. Topic checkpoints  — sliding-window cosine similarity → llama-3.1-8b summary → ChromaDB.
  3B. Chunk checkpoints  — every-100-message rolling summaries → ChromaDB.
  3C. Raw messages       — store raw texts in ChromaDB for retrieval.
  4.  Persona extraction — map-reduce with llama-3.1-8b JSON-mode → persona.json.

Usage:
    uv run python build_index.py
"""

import os
import re
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import ollama

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
load_dotenv()
# No cloud client needed — LLM calls go to local Ollama during ingestion.
# app.py uses Groq for the live chatbot.

CSV_PATH = "conversations.csv"
CHROMA_PATH = "./chroma_db"
PERSONAS_DIR = "./personas"
PERSONAS_PATH = f"{PERSONAS_DIR}/persona.json"

WINDOW_SIZE = 12             # messages per adjacent window — larger = more stable mean embedding
SIM_THRESHOLD = 0.62         # cosine similarity below this → topic change
MIN_SEGMENT_LEN = 30         # never create a segment shorter than this many messages
CHUNK_SIZE = 100             # messages per rolling summary
PERSONA_CHUNK = 200          # messages per persona map-phase chunk
OLLAMA_MODEL = "llama3.1:8b" # local Ollama model (pull with: ollama pull llama3.1:8b)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# JSON schema injected into Groq system prompt for persona extraction
# ──────────────────────────────────────────────────────────────
PERSONA_SCHEMA_PROMPT = """
You must respond with a single JSON object that strictly follows this schema:
{
  "user_id": "<string: exactly as given>",
  "habits": ["<string>", ...],
  "communication_style": {
    "tone": "<string>",
    "emoji_usage": "<string>"
  },
  "personality_traits": ["<string>", ...],
  "personal_facts": {
    "facts_with_history": [
      {
        "fact": "<string: e.g. occupation>",
        "history": [{"value": "<string>"}, ...],
        "latest": "<string: most recent value>"
      }
    ]
  }
}
Do not include any text outside the JSON object.
"""


# ──────────────────────────────────────────────────────────────
# CSV Parsing
# ──────────────────────────────────────────────────────────────
_MSG_RE = re.compile(r"^(User \d+):\s*(.*)")


def parse_conversation(raw: str) -> list[dict]:
    """Split one CSV row (a day's conversation) into message dicts."""
    msgs = []
    for line in raw.split("\n"):
        m = _MSG_RE.match(line.strip())
        if m:
            msgs.append({"sender": m.group(1), "text": m.group(2).strip()})
    return msgs


def load_all_messages(csv_path: str) -> list[dict]:
    """Return a flat chronological list of all messages across every CSV row."""
    log.info("Loading CSV → %s", csv_path)
    df = pd.read_csv(csv_path, header=None, on_bad_lines="skip")
    all_msgs: list[dict] = []
    msg_id = 0
    for _, row in df.iterrows():
        for m in parse_conversation(str(row[0])):
            m["message_id"] = msg_id
            all_msgs.append(m)
            msg_id += 1
    log.info("Parsed %d total messages.", len(all_msgs))
    return all_msgs


# ──────────────────────────────────────────────────────────────
# Ollama LLM helpers  (local inference — no rate limits, no API key)
# ──────────────────────────────────────────────────────────────
def _llm_call(prompt: str) -> str:
    """Plain text generation via local Ollama. No sleep needed — no rate limits."""
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"]
    except Exception as exc:
        log.error("Ollama call failed: %s", exc)
        raise


def _llm_json_call(system_prompt: str, user_prompt: str) -> dict:
    """
    JSON-mode generation via local Ollama.
    format='json' instructs the model to emit only a valid JSON object.
    The schema is still injected into the system prompt so the model
    knows exactly which fields to populate.
    """
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            format="json",
        )
        return json.loads(resp["message"]["content"])
    except Exception as exc:
        log.error("Ollama JSON call failed: %s", exc)
        raise


# ──────────────────────────────────────────────────────────────
# ChromaDB setup
# ──────────────────────────────────────────────────────────────
def setup_chroma(path: str):
    client = chromadb.PersistentClient(path=path)
    return (
        client,
        client.get_or_create_collection("topic_summaries"),
        client.get_or_create_collection("chunk_summaries"),
        client.get_or_create_collection("raw_messages"),
    )


# ──────────────────────────────────────────────────────────────
# Part A — Topic Checkpoints
# ──────────────────────────────────────────────────────────────
def build_topic_checkpoints(
    messages: list[dict],
    embedder: SentenceTransformer,
    topic_col,
) -> np.ndarray:
    """
    1. Embed all messages.
    2. Build NON-OVERLAPPING adjacent windows of WINDOW_SIZE messages.
       Each window shares ZERO messages with its neighbours, so cosine
       similarity between consecutive windows is a genuine measure of
       topical continuity — not positional overlap.
       (Step-1 sliding windows share N-1 messages → similarity is always
       0.9+ even across topic changes, making detection impossible.)
    3. When similarity(win[i-1], win[i]) < SIM_THRESHOLD AND the new
       segment would be >= MIN_SEGMENT_LEN messages → topic boundary.
    4. Summarize each segment with Groq Llama3; upsert into ChromaDB.
    Returns the full embeddings array (reused by later steps).
    """
    log.info("Embedding all messages …")
    texts = [m["text"] for m in messages]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    # Non-overlapping adjacent windows (step = WINDOW_SIZE, not 1).
    # win_starts[i] is the message index where window i begins.
    win_starts = list(range(0, len(messages), WINDOW_SIZE))
    win_embs = [
        np.mean(embeddings[s : min(s + WINDOW_SIZE, len(messages))], axis=0)
        for s in win_starts
    ]

    # Compare adjacent window pairs; emit a boundary only when:
    #   (a) similarity drops below SIM_THRESHOLD, AND
    #   (b) the segment since the last boundary is >= MIN_SEGMENT_LEN.
    # Guard (b) prevents micro-fragments when similarity briefly dips.
    boundaries = [0]
    for i in range(1, len(win_embs)):
        sim = cosine_similarity([win_embs[i - 1]], [win_embs[i]])[0][0]
        segment_so_far = win_starts[i] - boundaries[-1]
        if sim < SIM_THRESHOLD and segment_so_far >= MIN_SEGMENT_LEN:
            boundaries.append(win_starts[i])  # actual message index, not window index
    boundaries.append(len(messages))
    log.info("Detected %d topic segments.", len(boundaries) - 1)

    existing = set(topic_col.get()["ids"])
    for idx in range(len(boundaries) - 1):
        start, end = boundaries[idx], boundaries[idx + 1]
        seg_id = f"topic_{start}_{end}"
        if seg_id in existing:
            continue

        seg_text = "\n".join(
            f"{m['sender']}: {m['text']}" for m in messages[start:end]
        )
        prompt = (
            f"Summarize this conversation segment in 2-3 sentences, "
            f"capturing the main topic:\n\n{seg_text[:3000]}"
        )
        log.info("  Topic segment %d/%d (msgs %d–%d) …", idx + 1, len(boundaries) - 1, start, end)
        summary = _llm_call(prompt)

        seg_emb = np.mean(embeddings[start:end], axis=0).tolist()
        topic_col.upsert(
            ids=[seg_id],
            documents=[summary],
            embeddings=[seg_emb],
            metadatas=[{"type": "topic_summary", "start_id": start, "end_id": end}],
        )

    log.info("Topic checkpoints done.")
    return embeddings


# ──────────────────────────────────────────────────────────────
# Part B — 100-Message Chunk Checkpoints
# ──────────────────────────────────────────────────────────────
def build_chunk_checkpoints(
    messages: list[dict],
    embeddings: np.ndarray,
    chunk_col,
) -> None:
    log.info("Building 100-message chunk checkpoints …")
    existing = set(chunk_col.get()["ids"])

    for start in range(0, len(messages), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(messages))
        chunk_id = f"chunk_{start}_{end}"
        if chunk_id in existing:
            continue

        chunk_text = "\n".join(
            f"{m['sender']}: {m['text']}" for m in messages[start:end]
        )
        prompt = (
            "Write a 3-4 sentence rolling summary of this 100-message block, "
            f"covering key topics and context:\n\n{chunk_text[:4000]}"
        )
        log.info("  Chunk %d (msgs %d–%d) …", start // CHUNK_SIZE + 1, start, end)
        summary = _llm_call(prompt)

        chunk_emb = np.mean(embeddings[start:end], axis=0).tolist()
        chunk_col.upsert(
            ids=[chunk_id],
            documents=[summary],
            embeddings=[chunk_emb],
            metadatas=[{"type": "chunk_summary", "start_id": start, "end_id": end}],
        )

    log.info("Chunk checkpoints done.")


# ──────────────────────────────────────────────────────────────
# Part C — Raw Messages
# ──────────────────────────────────────────────────────────────
def store_raw_messages(
    messages: list[dict],
    embeddings: np.ndarray,
    raw_col,
) -> None:
    log.info("Storing raw messages in ChromaDB …")
    existing = set(raw_col.get()["ids"])
    BATCH = 500

    for b_start in range(0, len(messages), BATCH):
        batch = messages[b_start : b_start + BATCH]
        batch_embs = embeddings[b_start : b_start + len(batch)]

        new_ids, new_docs, new_embs, new_meta = [], [], [], []
        for msg, emb in zip(batch, batch_embs):
            mid = f"msg_{msg['message_id']}"
            if mid not in existing:
                new_ids.append(mid)
                new_docs.append(msg["text"])
                new_embs.append(emb.tolist())
                new_meta.append({"sender": msg["sender"], "message_id": msg["message_id"]})

        if new_ids:
            raw_col.upsert(ids=new_ids, documents=new_docs, embeddings=new_embs, metadatas=new_meta)
            log.info("  Stored batch starting at %d (%d msgs).", b_start, len(new_ids))

    log.info("Raw messages stored.")


# ──────────────────────────────────────────────────────────────
# Part D — Persona Extraction (Map-Reduce)
# ──────────────────────────────────────────────────────────────
def _extract_partial_persona(user_id: str, messages: list[dict]) -> dict | None:
    """MAP phase: extract partial persona for user_id from a message chunk."""
    user_msgs = [m for m in messages if m["sender"] == user_id]
    if not user_msgs:
        return None

    conv_ctx = "\n".join(f"{m['sender']}: {m['text']}" for m in messages)[:3000]
    user_lines = "\n".join(f"- {m['text']}" for m in user_msgs)[:2000]

    system_prompt = PERSONA_SCHEMA_PROMPT
    user_prompt = f"""Analyze the conversation and extract a persona for {user_id} ONLY.

CONVERSATION CONTEXT:
{conv_ctx}

{user_id}'s MESSAGES:
{user_lines}

For personal_facts, track evolving facts using history lists.
The user_id field must be exactly "{user_id}"."""

    return _llm_json_call(system_prompt, user_prompt)


def _consolidate_persona(user_id: str, partials: list[dict]) -> dict:
    """REDUCE phase: merge all partial personas into one final persona."""
    system_prompt = PERSONA_SCHEMA_PROMPT
    user_prompt = f"""Consolidate these partial personas for {user_id} into ONE final persona.

PARTIAL PERSONAS:
{json.dumps(partials, indent=2)[:6000]}

Rules:
- Deduplicate habits and personality_traits.
- Merge personal_facts histories; set "latest" to the most recent value.
- Synthesize a single communication_style.
- user_id must be exactly "{user_id}"."""

    return _llm_json_call(system_prompt, user_prompt)


def build_personas(messages: list[dict]) -> None:
    log.info("Starting persona extraction (map-reduce) …")
    Path(PERSONAS_DIR).mkdir(exist_ok=True)

    final_personas = []
    for user_id in ["User 1", "User 2"]:
        log.info("  Processing %s …", user_id)
        partials = []

        # MAP
        for i, start in enumerate(range(0, len(messages), PERSONA_CHUNK)):
            chunk = messages[start : start + PERSONA_CHUNK]
            log.info("    Map chunk %d for %s …", i + 1, user_id)
            partial = _extract_partial_persona(user_id, chunk)
            if partial:
                partials.append(partial)

        # REDUCE
        log.info("  Reducing %d partials for %s …", len(partials), user_id)
        final = partials[0] if len(partials) == 1 else _consolidate_persona(user_id, partials)
        final_personas.append(final)

    with open(PERSONAS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_personas, f, indent=2, ensure_ascii=False)
    log.info("Personas saved → %s", PERSONAS_PATH)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=== build_index.py starting ===")

    messages = load_all_messages(CSV_PATH)

    log.info("Loading sentence-transformers model …")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    _, topic_col, chunk_col, raw_col = setup_chroma(CHROMA_PATH)

    # Build all index components
    embeddings = build_topic_checkpoints(messages, embedder, topic_col)
    build_chunk_checkpoints(messages, embeddings, chunk_col)
    store_raw_messages(messages, embeddings, raw_col)
    build_personas(messages)

    log.info("=== Done! ChromaDB → %s | Personas → %s ===", CHROMA_PATH, PERSONAS_PATH)


if __name__ == "__main__":
    main()
