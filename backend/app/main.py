# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import config
from api.routes import router


app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    """Pre-warm all singletons so the first request isn't slow."""
    from graph.builder import get_graph
    from nodes.agent import _get_llm_with_tools
    from retrievers.vector_retriever import get_vector_retriever
    from memory.session_store import get_session_store
    get_graph()
    _get_llm_with_tools()
    get_vector_retriever()
    get_session_store()


@app.get("/")
def health_check():
    return {"status": "ok"}