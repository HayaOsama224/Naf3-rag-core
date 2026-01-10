# handler.py
import traceback
import threading
import runpod

import rag_core

# Protect llama-cpp calls (safe default for serverless concurrency)
_INFER_LOCK = threading.Lock()

# Optional: validate on cold start that core loaded
_COLD_START_STATUS = {
    "index_loaded": rag_core.INDEX is not None,
    "embedder_loaded": rag_core.EMBEDDER is not None,
    "llm_loaded": rag_core.LLM is not None,
}

def _extract_payload(event: dict) -> dict:
    # RunPod usually sends {"input": {...}}
    if isinstance(event, dict) and isinstance(event.get("input"), dict):
        return event["input"]
    return event if isinstance(event, dict) else {}

def handler(event):
    try:
        payload = _extract_payload(event)

        question = payload.get("question", "") or ""
        paragraph_history = payload.get("paragraph_history", "") or ""
        top_k = int(payload.get("top_k", getattr(rag_core, "TOP_K", 5)))

        # Optional: allow caller to ask for health
        if payload.get("health") is True:
            return {"health": "ok", **_COLD_START_STATUS}

        # Lock around model inference for safety
        with _INFER_LOCK:
            result = rag_core.answer_with_json_io(
                question=question,
                paragraph_history=paragraph_history,
                top_k=top_k,
            )

        # Return everything
        return result

    except Exception as e:
        return {
            "error": str(e),
            "details": traceback.format_exc(),
            "cold_start_status": _COLD_START_STATUS,
        }

runpod.serverless.start({"handler": handler})
