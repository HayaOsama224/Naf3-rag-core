# rag_core.py
import os
import json
import glob
import pickle
import re
from typing import List, Dict, Any, Optional, Tuple

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

# Important: for FAQs, 0.30–0.40 usually works well
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

INSUFFICIENT_EN = "Insufficient FAQ context"
INSUFFICIENT_AR = "لا توجد إجابة في الأسئلة الشائعة الحالية"

# ===============================
# HELPERS
# ===============================
AR_REGEX = re.compile(r"[\u0600-\u06FF]")
AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")


def detect_lang(text: str) -> str:
    """Detect AR / EN with Arabic-char shortcut first."""
    if AR_REGEX.search(text or ""):
        return "ar"
    try:
        return "ar" if detect(text or "") == "ar" else "en"
    except Exception:
        return "en"


def normalize_q(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_arabic(text: str) -> str:
    text = (text or "").strip()
    text = AR_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    # remove punctuation/symbols but keep Arabic letters/numbers/spaces
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_en(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_retrieval(text: str) -> str:
    base = normalize_q(text)
    lang = detect_lang(base)
    return normalize_arabic(base) if lang == "ar" else normalize_en(base)


def make_citation(d: Dict[str, Any]) -> str:
    fid = d.get("faq_id") or d.get("id", "?")
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
    Load all *.json files under DATA_DIR, expecting structure:
    {
      "meta": {...},
      "faqs": [
        { "id": "...", "question_ar": "...", "answer_ar": "...",
          "question_en": "...", "answer_en": "...", "tags": [...] }
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

                q_ar_norm = normalize_arabic(q_ar) if q_ar else ""
                q_en_norm = normalize_en(q_en) if q_en else ""

                if q_ar or a_ar:
                    docs.append({
                        "id": f"{faq_id}::ar",
                        "faq_id": faq_id,
                        "lang": "ar",
                        "question": q_ar,
                        "question_norm": q_ar_norm,
                        "answer": a_ar,
                        "tags": tags,
                        "source_file": fp,
                    })

                if q_en or a_en:
                    docs.append({
                        "id": f"{faq_id}::en",
                        "faq_id": faq_id,
                        "lang": "en",
                        "question": q_en,
                        "question_norm": q_en_norm,
                        "answer": a_en,
                        "tags": tags,
                        "source_file": fp,
                    })

        except Exception as e:
            print(f"[load_faq_jsons] Error reading {fp}: {e}")

    print(f"Loaded {len(docs)} FAQ QA entries from {len(files)} file(s).")
    return docs


def passages_text(d: Dict[str, Any]) -> str:
    """
    Build the text we embed.
    Use normalized question to stabilize Arabic retrieval.
    """
    q = d.get("question_norm") or d.get("question") or ""
    a = d.get("answer") or ""
    tags = d.get("tags") or []
    faq_id = d.get("faq_id", "?")

    base = f"Q: {q}\nA: {a}\nFAQ ID: {faq_id}\nTags: {', '.join(tags)}"
    return "passage: " + base  # e5 passage prefix


def build_index(docs: List[Dict[str, Any]],
                embedder: SentenceTransformer,
                index_path: str,
                doc_store_path: str) -> None:
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


def load_index() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
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
                f"Please add your JSON FAQ file(s) there."
            )

        print("[load_index] Building index from FAQ JSON...")
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        build_index(docs, embedder, INDEX_PATH, DOC_STORE_PATH)

    index = faiss.read_index(INDEX_PATH)
    with open(DOC_STORE_PATH, "rb") as f:
        docs = pickle.load(f)

    return index, docs


# ===============================
# GLOBAL INIT
# ===============================
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
def get_llm() -> Optional[Llama]:
    if Llama is None:
        return None

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
def retrieve(query_text: str, top_k: int = TOP_K, lang_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    if EMBEDDER is None or INDEX is None:
        return []

    # normalize query for stable retrieval
    query_norm = normalize_for_retrieval(query_text or "")

    q_emb = EMBEDDER.encode(
        ["query: " + query_norm],
        convert_to_numpy=True,
    )
    faiss.normalize_L2(q_emb)

    D, I = INDEX.search(q_emb, top_k * 2)  # pull extra, filter by language
    lang = lang_hint or detect_lang(query_text or "")

    same_lang, others = [], []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(DOCS):
            continue
        if float(score) < SIMILARITY_THRESHOLD:
            continue
        d = DOCS[idx]
        (same_lang if d.get("lang") == lang else others).append(d)

    out = same_lang[:top_k]
    if len(out) < top_k:
        out.extend(others[: max(0, top_k - len(out))])
    return out[:top_k]


def build_messages(user_q: str,
                   passages: List[Dict[str, Any]],
                   history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    lang = detect_lang(user_q or "")

    sys_en = (
        "You are Naf3 Charity FAQ Assistant. Answer ONLY using the provided FAQ context. "
        "You may use the chat history ONLY to understand what the user refers to, "
        "but do NOT treat history as a source of facts. "
        "If the requested information is NOT present verbatim in the context, "
        f"reply EXACTLY: \"{INSUFFICIENT_EN}\". "
        "Add short FAQ citations like (FAQ 012). Answer in the user's language."
    )
    sys_ar = (
        "أنت مساعد الأسئلة الشائعة لمنصة نفع الخيرية. أجب فقط من السياق المقدم. "
        "يمكنك استخدام سجل المحادثة فقط لفهم مرجع السؤال (مثل الضمائر)، لكن لا تعتبره مصدرًا للمعلومة. "
        f"إذا لم تظهر المعلومة المطلوبة نصًا داخل السياق فأجِب نصًا: \"{INSUFFICIENT_AR}\". "
        "أضف إشارة موجزة لرقم السؤال مثل (FAQ 012). أجب بلغة المستخدم."
    )

    sys = sys_ar if lang == "ar" else sys_en

    seen = set()
    blocks = []
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

    history_block = ""
    if history:
        turns = []
        for t in history[-6:]:
            qh = (t.get("question") or "").strip()
            ah = (t.get("answer") or "").strip()
            if not qh or not ah:
                continue
            if lang == "ar":
                turns.append(f"س: {qh}\nج: {ah}")
            else:
                turns.append(f"Q: {qh}\nA: {ah}")
        if turns:
            if lang == "ar":
                history_block = "\n\nسجل المحادثة (للمرجع فقط):\n" + "\n\n".join(turns)
            else:
                history_block = "\n\nChat history (for reference only):\n" + "\n\n".join(turns)

    if lang == "ar":
        user = (
            f"أجب في جملة أو جملتين فقط بالاعتماد على السياق التالي. "
            f"إن لم يكن الجواب موجودًا في السياق فأجِب نصًا: \"{INSUFFICIENT_AR}\".\n\n"
            f"السؤال: {user_q}{history_block}\n\nالسياق:\n{ctx}"
        )
    else:
        user = (
            f"Answer in 1–2 sentences using ONLY the FAQ context below. "
            f"If the answer isn’t in the context, reply EXACTLY: \"{INSUFFICIENT_EN}\".\n\n"
            f"Question: {user_q}{history_block}\n\nContext:\n{ctx}"
        )

    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def llm_generate(messages: List[Dict[str, str]], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if LLM is None:
        # return insufficient in correct language based on system prompt
        sys = (messages[0].get("content") or "")
        return INSUFFICIENT_AR if "أنت" in sys else INSUFFICIENT_EN

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
        sys = (messages[0].get("content") or "")
        return INSUFFICIENT_AR if "أنت" in sys else INSUFFICIENT_EN


def answer_query(user_q: str, top_k: int = TOP_K, history: Optional[List[Dict[str, Any]]] = None):
    """
    Main entrypoint: given a user question, returns (answer, passages).
    """
    if INDEX is None or EMBEDDER is None:
        lang = detect_lang(user_q or "")
        msg = INSUFFICIENT_AR if lang == "ar" else INSUFFICIENT_EN
        return msg, []

    lang = detect_lang(user_q or "")
    uq_norm = normalize_for_retrieval(user_q or "")

    # 1) Exact match fallback:
    # If question exists in JSON (after normalization), return it immediately (no insufficient).
    for d in DOCS:
        if d.get("lang") != lang:
            continue
        if (d.get("question_norm") or "") == uq_norm and (d.get("answer") or "").strip():
            ans = d["answer"].strip() + f" ({make_citation(d)})"
            return ans, [d]

    # 2) Normal retrieval
    passages = retrieve(user_q, top_k=top_k, lang_hint=lang)
    if not passages:
        msg = INSUFFICIENT_AR if lang == "ar" else INSUFFICIENT_EN
        return msg, []

    # 3) LLM answer grounded in passages
    msgs = build_messages(user_q, passages, history=history)
    resp = llm_generate(msgs)
    return resp, passages
