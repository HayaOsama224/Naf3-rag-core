# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from rag_core import (
    answer_query,
    TOP_K,
    detect_lang,
    INSUFFICIENT_EN,
    INSUFFICIENT_AR,
)

description = """
Naf3 Charity FAQ RAG Microservice

This service answers Naf3 donor/recipient FAQ questions in Arabic and English
using a Retrieval-Augmented Generation pipeline:

- FAISS vector index over charity FAQ entries
- Multilingual E5 embeddings
- Qwen2.5-3B-Instruct (GGUF, llama.cpp) for generation
"""

app = FastAPI(
    title="Naf3 Charity FAQ RAG",
    description=description,
    version="1.0.0",
)
class HistoryTurn(BaseModel):
    question: str
    answer: str

class AskRequest(BaseModel):
    """
    Request body for asking a FAQ question.
    """
    question: str
    top_k: Optional[int] = None
    # Optional short chat history (used only for resolving references)
    # A list of {question, answer} pairs.
    history: Optional[List["HistoryTurn"]] = None



class Passage(BaseModel):
    """
    A retrieved FAQ entry.
    """
    id: Optional[str] = None
    faq_id: Optional[str] = None
    lang: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    tags: Optional[List[str]] = None
    source_file: Optional[str] = None


class AskResponse(BaseModel):
    """
    Response body for a FAQ question.
    """
    answer: str
    insufficient: bool
    lang: str
    passages: List[Passage]


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a Naf3 FAQ question (AR/EN)",
    tags=["faq"],
)
def ask(req: AskRequest):
    """
    Ask a charity FAQ question in Arabic or English and get a short answer
    grounded in the FAQ JSON.

    If the information is not present in the FAQ corpus, the service responds
    with a fixed insufficient-context message and sets `insufficient = true`.
    """
    k = req.top_k or TOP_K
    history = None
    if req.history:
        history = []
        for t in req.history:
            if hasattr(t, "model_dump"):
                history.append(t.model_dump())
            else:
                history.append(t.dict())

    answer, docs = answer_query(req.question, k, history=history)

    lang = detect_lang(req.question or "")
    insufficient_flag = answer.strip() in {INSUFFICIENT_EN, INSUFFICIENT_AR}

    return AskResponse(
        answer=answer,
        insufficient=insufficient_flag,
        lang=lang,
        passages=docs
    )

