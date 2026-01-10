# handler.py
import runpod
from rag_core import answer_query, TOP_K, detect_lang, INSUFFICIENT_EN, INSUFFICIENT_AR

def handler(event):
    """
    RunPod Serverless handler.
    Expected input:
      {
        "input": {
          "question": "...",
          "top_k": 5,          # optional
          "history": [         # optional
            {"question": "...", "answer": "..."}
          ]
        }
      }
    """
    inp = (event or {}).get("input") or {}
    question = (inp.get("question") or "").strip()
    if not question:
        return {"error": "Missing 'question' in input"}

    k = int(inp.get("top_k") or TOP_K)
    history = inp.get("history")

    answer, docs = answer_query(question, k, history=history)

    lang = detect_lang(question)
    insufficient = answer.strip() in {INSUFFICIENT_EN, INSUFFICIENT_AR}

    return {
        "answer": answer,
        "insufficient": insufficient,
        "lang": lang,
        "passages": docs,
    }

runpod.serverless.start({"handler": handler})
