# handler.py
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from rag_core import answer_query, make_citation

app = FastAPI(title="Charity FAQ RAG API")


# ==============================
# REQUEST / RESPONSE MODELS
# ==============================
class HistoryItem(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    history: Optional[List[HistoryItem]] = None


class Passage(BaseModel):
    faq_id: str
    lang: str
    question: str
    answer: str
    tags: List[str]
    citation: str


class QueryResponse(BaseModel):
    answer: str
    passages: List[Passage]


# ==============================
# API ENDPOINT
# ==============================
@app.post("/query", response_model=QueryResponse)
async def query_faq(req: QueryRequest):
    try:
        # Convert history to plain dicts for rag_core
        hist = [h.dict() for h in req.history] if req.history else None
        top_k = req.top_k if req.top_k is not None else 5

        answer, passages_raw = answer_query(req.question, top_k=top_k, history=hist)

        # Format passages for response
        passages = [
            Passage(
                faq_id=d.get("faq_id", "?"),
                lang=d.get("lang", "en"),
                question=d.get("question", ""),
                answer=d.get("answer", ""),
                tags=d.get("tags") or [],
                citation=make_citation(d),
            )
            for d in passages_raw
        ]

        return QueryResponse(answer=answer, passages=passages)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "answer": "", "passages": []},
        )


# ==============================
# HEALTHCHECK ENDPOINT
# ==============================
@app.get("/health")
async def healthcheck():
    return {"status": "ok"}
