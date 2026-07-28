# api/schemas.py

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default-thread"


class Source(BaseModel):
    type: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []


class SessionInfo(BaseModel):
    thread_id: str
    created_at: str
    last_active_at: str
    title: str


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]