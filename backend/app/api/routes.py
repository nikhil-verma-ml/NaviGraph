# api/routes.py
import os
import shutil
import json
from fastapi import UploadFile, File, APIRouter
from fastapi.responses import StreamingResponse
from retrievers.ingest import ingest_files
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
from graph.builder import get_graph
from api.schemas import ChatRequest, ChatResponse, Source, SessionListResponse, SessionInfo
from memory.session_store import get_session_store
from config import DATA_DIR


router = APIRouter()
UPLOAD_DIR = str(DATA_DIR / "raw")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    graph = get_graph()
    session_store = get_session_store()
    session_store.create_or_update_session(
        thread_id=request.thread_id,
        title=request.query[:50]
    )
    config = {
        "configurable": {"thread_id": request.thread_id},
        "metadata": {"session_id": request.thread_id}
    }
    result = graph.invoke(
        {"messages": [HumanMessage(content=request.query)]},
        config=config
    )
    final_message = result["messages"][-1]
    sources = [
        Source(type=msg.name, content=str(msg.content)[:300])
        for msg in result["messages"]
        if msg.__class__.__name__ == "ToolMessage"
    ]
    return ChatResponse(answer=_extract_text(final_message.content), sources=sources)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) and part.get("type") == "text"
            else part.get("text", "") if isinstance(part, dict) and "text" in part
            else ""
            for part in content
        ).strip()
    return str(content)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    graph = get_graph()
    session_store = get_session_store()
    session_store.create_or_update_session(
        thread_id=request.thread_id,
        title=request.query[:50]
    )
    config = {
        "configurable": {"thread_id": request.thread_id},
        "metadata": {"session_id": request.thread_id}
    }
    from nodes.agent import _get_llm_with_tools

    def event_stream():
        sources = []
        prev_last_used = "primary"
        tool_calls_made = []   # track which tools actually fired this turn

        for stream_mode, chunk in graph.stream(
            {"messages": [HumanMessage(content=request.query)]},
            config=config,
            stream_mode=["messages", "updates"]
        ):
            # ── updates mode: node-level events (what actually ran) ──────────
            if stream_mode == "updates":
                for node_name, node_output in chunk.items():
                    msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []

                    if node_name == "agent":
                        # Detect fallback
                        llm_wrapper = _get_llm_with_tools()
                        if hasattr(llm_wrapper, "last_used") and llm_wrapper.last_used != prev_last_used:
                            if llm_wrapper.last_used == "fallback":
                                yield f"event: thinking\ndata: {json.dumps({'text': '⚠️ Gemini quota exceeded — switched to Groq (llama-3.3-70b)'})}\n\n"
                            prev_last_used = llm_wrapper.last_used

                        # Check if agent decided to call tools or answer directly
                        for msg in msgs:
                            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                                for tc in msg.tool_calls:
                                    label = {
                                        "vector_search": "🔍 Decided to search knowledge base",
                                        "web_search": "🌐 Decided to search the web"
                                    }.get(tc["name"], f"⚙️ Calling {tc['name']}")
                                    yield f"event: thinking\ndata: {json.dumps({'text': label})}\n\n"
                                    tool_calls_made.append(tc["name"])
                            elif isinstance(msg, AIMessage) and _extract_text(msg.content) and not getattr(msg, "tool_calls", None):
                                # Agent is giving final answer — no tools needed or done looping
                                if not tool_calls_made:
                                    yield f"event: thinking\ndata: {json.dumps({'text': '💬 No retrieval needed — answering from knowledge'})}\n\n"
                                else:
                                    yield f"event: thinking\ndata: {json.dumps({'text': '✍️ Composing final answer from retrieved results'})}\n\n"

                    elif node_name == "tools":
                        for msg in msgs:
                            if msg.__class__.__name__ == "ToolMessage":
                                done_label = {
                                    "vector_search": "✅ Knowledge base search complete",
                                    "web_search": "✅ Web search complete"
                                }.get(msg.name, f"✅ {msg.name} complete")
                                yield f"event: thinking\ndata: {json.dumps({'text': done_label})}\n\n"
                                sources.append({"type": msg.name, "content": str(msg.content)[:300]})

            # ── messages mode: token-by-token streaming ──────────────────────
            elif stream_mode == "messages":
                msg_chunk, metadata = chunk
                if isinstance(msg_chunk, AIMessageChunk):
                    text = _extract_text(msg_chunk.content)
                    if text:
                        yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"
                elif isinstance(msg_chunk, AIMessage) and not isinstance(msg_chunk, AIMessageChunk):
                    text = _extract_text(msg_chunk.content)
                    if text:
                        yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"

        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions():
    session_store = get_session_store()
    sessions = session_store.list_sessions()
    return SessionListResponse(sessions=[SessionInfo(**s) for s in sessions])


@router.get("/sessions/{thread_id}/messages")
def get_session_messages(thread_id: str):
    """Returns filtered conversation history (Human + AI only) for a thread."""
    from langchain_core.messages import HumanMessage, AIMessage

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    if not state or not state.values:
        return {"messages": []}

    messages = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": _extract_text(msg.content)})
        elif isinstance(msg, AIMessage) and msg.content:
            text = _extract_text(msg.content)
            if text:
                messages.append({"role": "assistant", "content": text})

    return {"messages": messages}


@router.delete("/sessions/{thread_id}")
def delete_session(thread_id: str):
    get_session_store().delete_session(thread_id)
    return {"status": "deleted"}


@router.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(file_path)
    result = ingest_files(saved_paths)
    return {"message": "Documents ingested successfully", "details": result}