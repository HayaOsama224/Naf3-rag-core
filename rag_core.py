# rag_core.py
import os
import json
import glob
import pickle
import re
from typing import List, Dict, Any

import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ===============================
# CONFIG
# ===============================
DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_PATH = os.getenv("INDEX_PATH", "./artifacts/faq.index")
DOC_STORE_PATH = os.getenv("DOC_STORE_PATH", "./artifacts/faq_docs.pkl")
ARTIFACT_DIR = os.path.dirname(INDEX_PATH) or "."
os.makedirs(ARTIFACT_DIR, exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

GGUF_REPO_ID = os.getenv("GGUF_REPO_ID", "Qwen/Qwen2.5-3B-Instruct-GGUF")
GGUF_FILENAME = os.getenv("GGUF_FILENAME", "qwen2.5-3b-instruct-q4_k_m.gguf")

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "4000"))

N_CTX = int(os.getenv("N_CTX", "4096"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))

INSUFFICIENT_EN = "Insufficient FAQ context"
INSUFFICIENT_AR = "لا توجد إجابة في الأسئلة الشائعة الحالية"

# ===============================
# HELPERS
# ===============================
AR_REGEX = re.compile(r"[\u0600-\u06FF]")


def detect_lang(text: str) -> str:
    """Detect AR / EN with a simple Arabic-character shortcut."""
    if AR_REGEX.search(text or ""):
        return "ar"
    try:
        return "ar" if detect(text or "") == "ar" else "en"
    except Exception:
        return "en"


def normalize_q(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def make_citation(d: Dict[str, Any]) -> str:
    """
    Build a tiny citation string from FAQ id + first tag.
    """
    fid = d.get("id", "?")
    tags = d.get("tags") or []
    tag = tags[0] if tags else ""
    return f"FAQ {fid}" + (f" — {tag}" if tag else "")


def truncate_ctx(s: str, limit: int = MAX_CTX_CHARS) -> str:
    return s if len(s) <= limit else s[:limit] + "\n[...]"


# ===============================
# DATA LOADING & INDEXING
# ===============================

def load_faq_jsons(folder: str) -> List[Dict[str, Any]]:
    """
    Load all *.json files under DATA_DIR, expecting the structure:

    {
      "meta": {...},
      "faqs": [
        {
          "id": "...",
          "question_ar": "...",
          "answer_ar": "...",
          "question_en": "...",
          "answer_en": "...",
          "tags": [...]
        },
        ...
      ]
    }
    """
    docs: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not files:
        print(f"[load_faq_jsons] No JSON files found in {folder}")
        return docs

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            faqs = data.get("faqs", [])
            for faq in faqs:
                faq_id = faq.get("id") or os.path.basename(fp)
                q_ar = normalize_q(faq.get("question_ar", ""))
                a_ar = normalize_q(faq.get("answer_ar", ""))
                q_en = normalize_q(faq.get("question_en", ""))
                a_en = normalize_q(faq.get("answer_en", ""))
                tags = faq.get("tags") or []

                if q_ar or a_ar:
                    docs.append(
                        {
                            "id": f"{faq_id}::ar",
                            "faq_id": faq_id,
                            "lang": "ar",
                            "question": q_ar,
                            "answer": a_ar,
                            "tags": tags,
                            "source_file": fp,
                        }
                    )

                if q_en or a_en:
                    docs.append(
                        {
                            "id": f"{faq_id}::en",
                            "faq_id": faq_id,
                            "lang": "en",
                            "question": q_en,
                            "answer": a_en,
                            "tags": tags,
                            "source_file": fp,
                        }
                    )

        except Exception as e:
            print(f"[load_faq_jsons] Error reading {fp}: {e}")

    print(f"Loaded {len(docs)} FAQ QA entries from {len(files)} file(s).")
    return docs


def passages_text(d: Dict[str, Any]) -> str:
    """Build the text we actually embed."""
    q = d.get("question") or ""
    a = d.get("answer") or ""
    tags = d.get("tags") or []
    faq_id = d.get("faq_id", "?")
    base = f"Q: {q}\nA: {a}\nFAQ ID: {faq_id}\nTags: {', '.join(tags)}"
    # e5-style prefix
    return "passage: " + base


def build_index(docs: List[Dict[str, Any]],
                embedder: SentenceTransformer,
                index_path: str,
                doc_store_path: str):
    if not docs:
        raise ValueError("No documents found to index.")

    texts = [passages_text(d) for d in docs]
    emb = embedder.encode(
        texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64
    )
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension())
    index.add(emb)
    faiss.write_index(index, index_path)

    with open(doc_store_path, "wb") as f:
        pickle.dump(docs, f)

    print(f"[build_index] Index built with {len(docs)} vectors.")


def load_index():
    """
    Load (or build) FAISS index and document store.
    """
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOC_STORE_PATH)):
        if not os.path.isdir(DATA_DIR):
            raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
        docs = load_faq_jsons(DATA_DIR)
        if not docs:
            raise FileNotFoundError(
                f"No FAQ JSON files found in {DATA_DIR}. "
                f"Please add your charity_faq_eg_ar_en.json there."
            )

        print("[load_index] Building index from FAQ JSON...")
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        build_index(docs, embedder, INDEX_PATH, DOC_STORE_PATH)

    index = faiss.read_index(INDEX_PATH)
    with open(DOC_STORE_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs


# Global initialization (for Spaces / long-lived server)
try:
    INDEX, DOCS = load_index()
except Exception as e:
    print("[init] Failed to load/build FAISS index:", e)
    INDEX, DOCS = None, []

try:
    EMBEDDER = SentenceTransformer(EMBED_MODEL, device="cpu")
except Exception as e:
    print("[init] Failed to load embedder:", e)
    EMBEDDER = None


# ===============================
# LLM (llama.cpp CPU) setup
# ===============================

def get_llm() -> Llama:
    local_path = hf_hub_download(
        repo_id=GGUF_REPO_ID,
        filename=GGUF_FILENAME,
        local_dir="./models",
        local_dir_use_symlinks=False,
    )
    return Llama(
        model_path=local_path,
        n_threads=max(2, os.cpu_count() or 2),
        n_ctx=N_CTX,
        chat_format="qwen",
        verbose=False,
    )


try:
    LLM = get_llm()
except Exception as e:
    print("[init] Failed to init LLM:", e)
    LLM = None


# ===============================
# RETRIEVAL + GENERATION
# ===============================

def retrieve(query_text: str, top_k: int = TOP_K, lang_hint: str | None = None):
    if EMBEDDER is None or INDEX is None:
        return []

    q_emb = EMBEDDER.encode(
        ["query: " + (query_text or "")],
        convert_to_numpy=True,
    )
    faiss.normalize_L2(q_emb)
    D, I = INDEX.search(q_emb, top_k * 2)  # pull extra, filter by language

    lang = lang_hint or detect_lang(query_text or "")

    same_lang, others = [], []
    for i in I[0]:
        if i < 0 or i >= len(DOCS):
            continue
        d = DOCS[i]
        (same_lang if d.get("lang") == lang else others).append(d)

    out = same_lang[:top_k]
    if len(out) < top_k:
        out.extend(others[: top_k - len(out)])
    return out[:top_k]


def build_messages(user_q: str, passages: List[Dict[str, Any]]):
    lang = detect_lang(user_q or "")

    sys_en = (
        "You are Naf3 Charity FAQ Assistant. Answer ONLY using the provided FAQ context. "
        "If the requested information is NOT present verbatim in the context, "
        f"reply EXACTLY: \"{INSUFFICIENT_EN}\". "
        "Add short FAQ citations like (FAQ 012). Answer in the user's language."
    )
    sys_ar = (
        "أنت مساعد الأسئلة الشائعة لمنصة نفع الخيرية. أجب فقط من السياق المقدم. "
        f"إذا لم تظهر المعلومة المطلوبة نصًا داخل السياق فأجِب نصًا: \"{INSUFFICIENT_AR}\". "
        "أضف إشارة موجزة لرقم السؤال مثل (FAQ 012). أجب بلغة المستخدم."
    )

    sys = sys_ar if lang == "ar" else sys_en

    seen = set()
    blocks: List[str] = []
    for d in passages:
        key = (d.get("lang"), d.get("question"), d.get("answer"), d.get("faq_id"))
        if key in seen:
            continue
        seen.add(key)

        cite = make_citation(d)
        q = d.get("question") or ""
        a = d.get("answer") or ""
        if d.get("lang") == "ar":
            blocks.append(f"س: {q}\nج: {a}\nالمصدر: {cite}")
        else:
            blocks.append(f"Q: {q}\nA: {a}\nSource: {cite}")

    ctx = truncate_ctx("\n\n---\n\n".join(blocks))

    if lang == "ar":
        user = (
            f"أجب في جملة أو جملتين فقط بالاعتماد على السياق التالي. "
            f"إن لم يكن الجواب موجودًا في السياق فأجِب نصًا: \"{INSUFFICIENT_AR}\".\n\n"
            f"السؤال: {user_q}\n\nالسياق:\n{ctx}"
        )
    else:
        user = (
            f"Answer in 1–2 sentences using ONLY the FAQ context below. "
            f"If the answer isn’t in the context, reply EXACTLY: \"{INSUFFICIENT_EN}\".\n\n"
            f"Question: {user_q}\n\nContext:\n{ctx}"
        )

    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def llm_generate(messages, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if LLM is None:
        return INSUFFICIENT_EN
    out = LLM.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=max_new_tokens,
        repeat_penalty=1.15,
        stop=None,
    )
    try:
        return out["choices"][0]["message"]["content"].strip()
    except Exception:
        return INSUFFICIENT_EN


def answer_query(user_q: str, top_k: int = TOP_K):
    """
    Main entrypoint: given a user question, returns (answer, passages).
    """
    if INDEX is None or EMBEDDER is None or LLM is None:
        lang = detect_lang(user_q or "")
        msg = INSUFFICIENT_AR if lang == "ar" else INSUFFICIENT_EN
        return msg, []

    lang = detect_lang(user_q or "")
    passages = retrieve(user_q, top_k=top_k, lang_hint=lang)

    if not passages:
        msg = INSUFFICIENT_AR if lang == "ar" else INSUFFICIENT_EN
        return msg, []

    msgs = build_messages(user_q, passages)
    resp = llm_generate(msgs)
    return resp, passages
