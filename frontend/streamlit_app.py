import json
import uuid
import requests
import streamlit as st

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")

BACKEND_URL = "http://localhost:8000"


# ── State bootstrap ───────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_load" not in st.session_state:
    st.session_state.pending_load = None  # thread_id to load, set by sidebar buttons


# ── Handle pending session load (must run before any rendering) ───────────────
if st.session_state.pending_load:
    tid = st.session_state.pending_load
    st.session_state.pending_load = None
    try:
        resp = requests.get(f"{BACKEND_URL}/sessions/{tid}/messages", timeout=10)
        st.session_state.messages = resp.json().get("messages", []) if resp.ok else []
    except Exception:
        st.session_state.messages = []
    st.session_state.thread_id = tid


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💬 Conversations")

    if st.button("＋ New Conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Fetch unique sessions from backend
    try:
        resp = requests.get(f"{BACKEND_URL}/sessions", timeout=5)
        sessions = resp.json().get("sessions", []) if resp.ok else []
    except Exception:
        sessions = []

    if sessions:
        st.caption("Previous conversations")
        for s in sessions:
            is_active = s["thread_id"] == st.session_state.thread_id
            col1, col2 = st.columns([5, 1])
            with col1:
                label = ("▶ " if is_active else "") + s["title"][:38]
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"load_{s['thread_id']}",
                             use_container_width=True, type=btn_type):
                    if not is_active:
                        # Set pending load — actual fetch happens at top of next rerun
                        st.session_state.pending_load = s["thread_id"]
                        st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{s['thread_id']}", help="Delete session"):
                    try:
                        requests.delete(f"{BACKEND_URL}/sessions/{s['thread_id']}", timeout=5)
                    except Exception:
                        pass
                    if st.session_state.thread_id == s["thread_id"]:
                        st.session_state.thread_id = str(uuid.uuid4())
                        st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("No past conversations yet.")

    st.divider()

    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT or MD files", type=["pdf", "txt", "md"],
        accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploaded_files and st.button("Ingest", use_container_width=True):
        with st.spinner("Processing..."):
            files_payload = [("files", (f.name, f.getvalue())) for f in uploaded_files]
            r = requests.post(f"{BACKEND_URL}/upload", files=files_payload)
            if r.ok:
                st.success(f"Done: {r.json()['details']}")
            else:
                st.error("Ingestion failed.")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("NaviGraph")
st.caption(f"Session: `{st.session_state.thread_id}`")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} used)"):
                for src in msg["sources"]:
                    st.caption(f"**{src['type']}**: {src['content'][:200]}...")

# ── Chat input ────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        answer_box = st.empty()

        full_answer = ""
        sources = []
        thinking_steps = []

        def render_thinking(steps, done=False):
            if not steps:
                return
            label = "🧠 Agent steps" if done else "🤔 Agent thinking..."
            with thinking_placeholder.expander(label, expanded=not done):
                for step in steps:
                    st.markdown(f"- {step}")

        try:
            with requests.post(
                f"{BACKEND_URL}/chat/stream",
                json={"query": query, "thread_id": st.session_state.thread_id},
                stream=True, timeout=120
            ) as resp:
                resp.raise_for_status()
                event_type = None
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        event_type = None
                        continue
                    if raw_line.startswith("event:"):
                        event_type = raw_line[len("event:"):].strip()
                    elif raw_line.startswith("data:"):
                        payload = json.loads(raw_line[len("data:"):].strip())
                        if event_type == "thinking":
                            thinking_steps.append(payload["text"])
                            render_thinking(thinking_steps, done=False)
                        elif event_type == "token":
                            full_answer += payload["text"]
                            answer_box.markdown(full_answer + "▌")
                        elif event_type == "sources":
                            sources = payload
        except Exception as e:
            st.error(f"Stream error: {e}")
            full_answer = full_answer or "Something went wrong."

        answer_box.markdown(full_answer)
        render_thinking(thinking_steps, done=True)  # collapse + relabel

        if sources:
            with st.expander(f"Sources ({len(sources)} used)"):
                for src in sources:
                    st.caption(f"**{src['type']}**: {src['content'][:200]}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources
    })
