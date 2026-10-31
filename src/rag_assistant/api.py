"""HTTP API.

Identity: behind the API gateway the user's groups come from a validated token
(e.g. Microsoft Entra ID group claims) and are forwarded as ``X-User-Groups``.
The service must therefore only be reachable through that gateway.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

from . import __version__
from .assistant import Assistant


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    allow_actions: bool = Field(False, description="user confirmed side-effecting tools")


@lru_cache(maxsize=1)
def get_assistant() -> Assistant:
    return Assistant.from_settings()


app = FastAPI(title="Enterprise RAG Assistant", version=__version__)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    a = get_assistant()
    return {
        "status": "ready",
        "chunks": len(a.chunks),
        "retriever": a.retriever.name,
        "llm": type(a.llm).__name__,
    }


@app.post("/ask")
def ask(req: AskRequest, x_user_groups: str = Header("all")) -> dict:
    groups = {g.strip() for g in x_user_groups.split(",") if g.strip()} | {"all"}
    return get_assistant().ask(req.question, groups, req.allow_actions).to_dict()
