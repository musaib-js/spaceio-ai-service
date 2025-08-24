from typing import List, Dict, Any, Literal
from settings import Settings
from fastapi import HTTPException
import google.generativeai as genai
from main import redis

settings = Settings()

async def generate_llm_answer(
    contexts: List[str], history: List[Dict[str, str]], message: str
):
    if not settings.GEMINI_API_KEY:
        raise HTTPException(500, detail="GEMINI_API_KEY not configured")

    model = genai.GenerativeModel(settings.GEMINI_CHAT_MODEL)

    system_prompt = (
        "You are a helpful assistant for a student or a company. Answer strictly using the provided context. "
        "If the answer cannot be derived from context, say you don't have enough information. "
        "Answer should be in great detail. "
        "You also have access to the history. Sometimes, the question might be a follow-up to the previous messages, "
        "where the context might be irrelevant or not present. Handle such cases effectively. "
        "The answer should be in proper markdown format with correct formatting."
        "Ensure proper line breaks and spacing."
    )

    ctx_block = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])

    chat_history = []
    for turn in reversed(history):  # reverse to chronological
        role = turn["role"]
        content = turn["content"]
        chat_history.append({"role": role, "parts": [content]})
        
    messages = chat_history + [
        {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{ctx_block}\n\nUser: {message}"}]}
    ]

    # Stream response from Gemini
    def stream_chunks():
        response_stream = model.generate_content(messages, stream=True)
        for chunk in response_stream:
            if not chunk.candidates:
                continue
            parts = chunk.candidates[0].content.parts
            if parts and parts[0].text:
                yield parts[0].text

    # Wrap generator into async
    for chunk in stream_chunks():
        yield chunk


async def cache_push_message(
    space_id: str, user_id: str, role: Literal["user", "assistant"], content: str
):
    key = f"chat:{space_id}:{user_id}"
    await redis.lpush(key, {"role": role, "content": content})
    await redis.ltrim(key, 0, 4)  # keep last 5


async def cache_get_recent(space_id: str, user_id: str) -> List[Dict[str, str]]:
    key = f"chat:{space_id}:{user_id}"
    raw = await redis.lrange(key, 0, 4)
    # Redis JSON not guaranteed; we stored Python dict; use repr/ast? Safer: store as JSON
    # Backward-compat: if bytes, try eval fallback
    out = []
    import json, ast

    for item in raw:
        try:
            if isinstance(item, (bytes, bytearray)):
                item = item.decode()
            obj = json.loads(item)
        except Exception:
            try:
                obj = ast.literal_eval(item)
            except Exception:
                obj = {"role": "user", "content": str(item)}
        out.append(obj)
    return out