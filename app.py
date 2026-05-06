"""
app.py — Streamlit chatbot UI.

Run AFTER build_index.py has finished.
Usage:
    uv run streamlit run app.py
"""

import os
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# ──────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────
load_dotenv()
_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

CHROMA_PATH = "./chroma_db"
PERSONAS_PATH = "./personas/persona.json"
TOP_K_TOPICS = 3
TOP_K_CHUNKS = 2
TOP_K_RAW = 5
CONVERSATION_MEMORY = 5  # number of recent messages passed to Groq for follow-up context
GROQ_MODEL = "llama-3.1-8b-instant"

st.set_page_config(
    page_title="RAG Persona Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #e2e8f0; }

    /* Card container */
    .persona-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
    }
    .persona-card h3 { margin-top: 0; color: #a78bfa; }

    /* Trait badges */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78em;
        font-weight: 500;
        margin: 3px;
    }

    /* Section label */
    .section-label {
        font-size: 0.72em;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #818cf8;
        margin-bottom: 6px;
    }

    /* Chat bubbles */
    .stChatMessage { border-radius: 12px !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        background: rgba(255,255,255,0.05);
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
    }

    /* Input */
    .stChatInputContainer { border-radius: 12px !important; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Cached resource loaders
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ChromaDB & embedder …")
def load_chroma_and_embedder():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    topic_col  = client.get_collection("topic_summaries")
    chunk_col  = client.get_collection("chunk_summaries")   # chronological rolling summaries
    raw_col    = client.get_collection("raw_messages")
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    return topic_col, chunk_col, raw_col, embedder


@st.cache_data(show_spinner=False)
def load_personas() -> list[dict]:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# RAG retrieval — three-layer memory
# ──────────────────────────────────────────────────────────────
def retrieve_context(query: str, topic_col, chunk_col, raw_col, embedder) -> str:
    """
    Three-layer retrieval:
      Macro   — topic_summaries : high-level thematic context
      Chrono  — chunk_summaries : chronological rolling summaries
      Micro   — raw_messages    : exact verbatim quotes
    """
    q_emb = embedder.encode([query])[0].tolist()

    topic_hits = topic_col.query(query_embeddings=[q_emb], n_results=TOP_K_TOPICS)
    chunk_hits = chunk_col.query(query_embeddings=[q_emb], n_results=TOP_K_CHUNKS)
    raw_hits   = raw_col.query(query_embeddings=[q_emb],   n_results=TOP_K_RAW)

    ctx  = "### 📌 MACRO CONTEXT (Topic Summaries)\n"
    ctx += "\n\n".join(topic_hits["documents"][0])
    ctx += "\n\n### 📅 CHRONOLOGICAL CONTEXT (Rolling Summaries)\n"
    ctx += "\n\n".join(chunk_hits["documents"][0])
    ctx += "\n\n### 🔍 MICRO CONTEXT (Exact Quotes)\n"
    ctx += "\n\n".join(raw_hits["documents"][0])
    return ctx


# ──────────────────────────────────────────────────────────────
# System prompt builder  (AI Analyst — 3rd-person, never roleplay)
# ──────────────────────────────────────────────────────────────
def build_system_prompt(context: str, personas: list[dict]) -> str:
    """
    Constructs the system message that positions the bot as a neutral
    AI Analyst. Rules injected explicitly:
      • Always refer to users in the 3rd person.
      • Never roleplay as or speak on behalf of a user.
      • Distinguish past history from current status when facts evolved.
    """
    persona_block = ""
    for p in personas:
        uid    = p.get("user_id", "Unknown")
        traits = ", ".join(p.get("personality_traits", [])) or "N/A"
        habits = ", ".join(p.get("habits", []))             or "N/A"
        cs     = p.get("communication_style", {})

        # Render fact history so the LLM sees evolution
        facts_block = ""
        for item in p.get("personal_facts", {}).get("facts_with_history", []):
            hist_vals = " → ".join(
                h.get("value", str(h)) if isinstance(h, dict) else str(h)
                for h in item.get("history", [])
            )
            facts_block += (
                f"    - {item.get('fact', '?')}: "
                f"history [{hist_vals}], latest = {item.get('latest', '?')}\n"
            )

        persona_block += (
            f"\n[{uid}]\n"
            f"  Personality  : {traits}\n"
            f"  Habits       : {habits}\n"
            f"  Tone         : {cs.get('tone', 'N/A')}\n"
            f"  Emoji usage  : {cs.get('emoji_usage', 'N/A')}\n"
            f"  Personal facts (with evolution history):\n{facts_block}"
        )

    return f"""You are a neutral, third-person AI Analyst. \
Your role is to answer questions ABOUT User 1 and User 2 based on their private conversation data.

CORE RULES — follow these strictly:
1. NEVER roleplay as User 1 or User 2. Never say "I am User 1" or speak in their voice.
2. Always refer to them in the third person: "User 1 is ...", "User 2 mentioned ...", "They tend to ...".
3. TEMPORAL AWARENESS: When a personal fact has evolved over time, explicitly distinguish past \
from present. For example: "User 1 used to be a student, but is now a software engineer."
4. Ground every claim in the retrieved context below. Do not invent information.
5. If the answer cannot be found in the provided context, say so honestly.

## EXTRACTED PERSONA PROFILES
{persona_block}

## RETRIEVED CONVERSATION CONTEXT
{context}
"""


# ──────────────────────────────────────────────────────────────
# Persona card renderer
# ──────────────────────────────────────────────────────────────
def render_persona_card(persona: dict) -> None:
    uid = persona.get("user_id", "Unknown")
    traits = persona.get("personality_traits", [])
    habits = persona.get("habits", [])
    cs = persona.get("communication_style", {})
    facts = persona.get("personal_facts", {}).get("facts_with_history", [])

    badges_html = "".join(f'<span class="badge">{t}</span>' for t in traits)

    st.markdown(
        f"""
        <div class="persona-card">
            <h3>👤 {uid}</h3>
            <div class="section-label">Personality Traits</div>
            <div style="margin-bottom:16px">{badges_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">💬 Communication Style</div>', unsafe_allow_html=True)
        st.markdown(f"- **Tone:** {cs.get('tone', 'N/A')}")
        st.markdown(f"- **Emoji Usage:** {cs.get('emoji_usage', 'N/A')}")

        st.markdown("")
        st.markdown('<div class="section-label">🏃 Habits</div>', unsafe_allow_html=True)
        for h in habits:
            st.markdown(f"- {h}")

    with col2:
        if facts:
            st.markdown('<div class="section-label">📋 Personal Facts</div>', unsafe_allow_html=True)
            for item in facts:
                fact_label = item.get("fact", "").replace("_", " ").title()
                latest = item.get("latest", "N/A")
                with st.expander(f"**{fact_label}** — latest: *{latest}*"):
                    for i, h in enumerate(item.get("history", []), 1):
                        val = h.get("value", str(h)) if isinstance(h, dict) else str(h)
                        st.markdown(f"{i}. {val}")

    st.divider()


# ──────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────
def main() -> None:
    # Header
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <h1 style="font-size:2.4rem; font-weight:700; background:linear-gradient(135deg,#818cf8,#a78bfa);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
                🤖 RAG Persona Chatbot
            </h1>
            <p style="color:#94a3b8; margin-top:6px;">
                Powered by Groq Llama 3 8B · ChromaDB · Sentence Transformers
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Guard: ensure index is built
    if not Path(CHROMA_PATH).exists() or not Path(PERSONAS_PATH).exists():
        st.error(
            "⚠️ Index not found. Please run `build_index.py` first:\n\n"
            "```bash\nuv run python build_index.py\n```"
        )
        st.stop()

    try:
        topic_col, chunk_col, raw_col, embedder = load_chroma_and_embedder()
        personas = load_personas()
    except Exception as exc:
        st.error(f"❌ Failed to load resources: {exc}")
        st.stop()

    tab_chat, tab_persona = st.tabs(["💬  Chat", "🪪  Persona Cards"])

    # ── Tab 1: Chat ─────────────────────────────────────────────
    with tab_chat:
        st.markdown(
            "<p style='color:#94a3b8'>Ask anything about the conversations between User 1 and User 2.</p>",
            unsafe_allow_html=True,
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the conversation …"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving context …"):
                    context = retrieve_context(
                        prompt, topic_col, chunk_col, raw_col, embedder
                    )
                    system_prompt = build_system_prompt(context, personas)

                # Build message list:
                #   [system] + last CONVERSATION_MEMORY turns + current user query
                history_window = st.session_state.messages[-(CONVERSATION_MEMORY * 2):]
                api_messages = (
                    [{"role": "system", "content": system_prompt}]
                    + [{"role": m["role"], "content": m["content"]} for m in history_window]
                )

                # Streaming response with typing effect
                placeholder = st.empty()
                full_response = ""
                rate_limited = False
                try:
                    stream = _groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=api_messages,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        full_response += delta
                        placeholder.markdown(full_response + "▮")  # blinking cursor effect
                    placeholder.markdown(full_response)  # final render without cursor
                except Exception as exc:
                    err = str(exc)
                    if "429" in err or "rate" in err.lower() or "limit" in err.lower():
                        rate_limited = True
                    else:
                        raise

                if rate_limited:
                    st.warning(
                        "⏳ Rate limit reached (HTTP 429). "
                        "Groq allows 30 RPM on the free tier — please wait a moment and try again."
                    )
                elif full_response:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )

        # Clear chat button
        if st.session_state.get("messages"):
            if st.button("🗑️ Clear chat", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()

    # ── Tab 2: Persona Cards ─────────────────────────────────────
    with tab_persona:
        st.markdown(
            "<p style='color:#94a3b8'>Extracted profiles for each user based on conversation analysis.</p>",
            unsafe_allow_html=True,
        )
        if not personas:
            st.info("No persona data found. Run `build_index.py` to generate personas.")
        else:
            for persona in personas:
                render_persona_card(persona)


if __name__ == "__main__":
    main()
