from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import uuid
import json
from main import mdb, redis
from auth.helpers import get_current_user, now_utc, require_space_role
from activity.helpers import update_space_last_activity, _log_activity
from chat.helpers import generate_llm_answer, cache_get_recent
from data.vectorisation import query_pinecone
from settings import Settings
from typing import Literal

settings = Settings()

from chat.types import ChatRequest

chat_router = APIRouter()

@chat_router.post("/{space_id}")
async def chat(
    space_id: str,
    background_tasks: BackgroundTasks,
    body: ChatRequest,
    user=Depends(get_current_user),
):
    space = mdb.spaces.find_one({"_id": space_id})
    if not space:
        raise HTTPException(404, detail="Space not found")
    require_space_role("member", user, space)
    update_space_last_activity(space_id)

    # Store user message
    msg_id = str(uuid.uuid4())
    message_doc = {
        "_id": msg_id,
        "space_id": space_id,
        "user_id": user["_id"],
        "user_name": user["name"],
        "role": "user",
        "content": body.message,
        "created_at": now_utc(),
    }
    mdb.messages.insert_one(message_doc)

    background_tasks.add_task(
        _log_activity,
        space["_id"],
        user,
        "message",
        f"{user['name']} sent a message in space - {space['name']}"
    )

    await redis.lpush(
        f"chat:{space_id}:{user['_id']}",
        json.dumps({"role": "user", "content": body.message}),
    )
    await redis.ltrim(f"chat:{space_id}:{user['_id']}", 0, 4)

    # Retrieve last 5
    history = await cache_get_recent(space_id, user["_id"])

    # Retrieve contexts
    matches = await query_pinecone(space_id, body.message, settings.TOP_K)
    contexts = [m["text"] for m in matches]
    
    async def event_stream():
        collected = []
        async for chunk in generate_llm_answer(contexts, history, body.message):
            if chunk:
                collected.append(chunk)
                yield chunk
        # Save the full assistant message after stream ends
        full_answer = "".join(collected)
        as_id = str(uuid.uuid4())
        as_doc = {
            "_id": as_id,
            "space_id": space_id,
            "user_id": user["_id"],
            "user_name": "Spaceio AI",
            "role": "assistant",
            "content": full_answer,
            "created_at": now_utc(),
        }
        mdb.messages.insert_one(as_doc)
        await redis.lpush(
            f"chat:{space_id}:{user['_id']}",
            json.dumps({"role": "assistant", "content": full_answer}),
        )
        await redis.ltrim(f"chat:{space_id}:{user['_id']}", 0, 4)

    return StreamingResponse(event_stream(), media_type="text/plain")

@chat_router.get("/{space_id}/conversations")
async def list_conversations(
    space_id: str,
    scope: Literal["mine", "all"] = "mine",
    user=Depends(get_current_user),
):
    space = mdb.spaces.find_one({"_id": space_id})
    if not space:
        raise HTTPException(404, detail="Space not found")
    if scope == "all":
        require_space_role("owner", user, space)
        cursor = mdb.messages.find({"space_id": space_id}).sort("created_at", -1)
    else:
        require_space_role("member", user, space)
        cursor = mdb.messages.find({"space_id": space_id, "user_id": user["_id"]}).sort(
            "created_at", -1
        )
    items = list(cursor)
    if scope != "all":
        for d in items:
            d.pop("_id", None)
    return {"messages": items}
