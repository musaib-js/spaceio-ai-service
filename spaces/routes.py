from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks, APIRouter
from fastapi.responses import StreamingResponse 
from auth.helpers import get_current_user, now_utc, require_space_role
from spaces.types import SpaceCreate, InviteRequest, ChatRequest
import uuid
from main import mdb, pc_index, redis, settings
from typing import List, Literal
from data.extractor import extract_text
import os
from data.vectorisation import upsert_chunks_to_pinecone, query_pinecone
from datetime import timedelta
from activity.helpers import _log_activity, update_space_last_activity
from email_invite import send_invite_email
from chat.helpers import generate_llm_answer, cache_get_recent


spaces_router = APIRouter()

@spaces_router.post("/")
async def create_space(space: SpaceCreate, user=Depends(get_current_user)):
    doc = {
        "_id": str(uuid.uuid4()),
        "name": space.name,
        "description": space.description,
        "owner_id": user["_id"],
        "members": [],
        "created_at": now_utc(),
    }
    mdb.spaces.insert_one(doc)
    return {"space_id": doc["_id"]}


@spaces_router.get("/")
async def get_spaces(user=Depends(get_current_user)):
    spaces = list(
        mdb.spaces.find({"$or": [{"owner_id": user["_id"]}, {"members": user["_id"]}]})
    )
    for space in spaces:
        space["messages"] = mdb.messages.count_documents({"space_id": space["_id"]})
        space["documents"] = mdb.documents.count_documents({"space_id": space["_id"]})
    return {"spaces": spaces}


@spaces_router.get("/{space_id}")
async def get_space(space_id: str, user=Depends(get_current_user)):
    space_details = mdb.spaces.find_one({"_id": space_id})
    return {"space": space_details}


@spaces_router.get("/{space_id}/documents")
async def list_documents(space_id: str, user=Depends(get_current_user)):
    documents = list(mdb.documents.find({"space_id": space_id}))
    return {"documents": documents}


@spaces_router.post("/{space_id}/documents")
async def upload_documents(
    space_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user=Depends(get_current_user),
):
    space = mdb.spaces.find_one({"_id": space_id})
    if not space:
        raise HTTPException(404, detail="Space not found")
    require_space_role("member", user, space)

    total_chunks = 0
    for f in files:

        doc_id = str(uuid.uuid4())

        res = mdb.documents.insert_one(
            {
                "_id": doc_id,
                "file_name": f.filename,
                "file_size": f.size,
                "file_type": f.content_type,
                "uploaded_at": now_utc(),
                "space_id": space_id,
                "_uploaded_by": user["_id"],
                "status": "processing",
            }
        )

        raw = await f.read()

        # Save uploaded file temporarily
        tmp_path = f"/tmp/{doc_id}_{f.filename}"
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(raw)

        print(f"Extracting text from {tmp_path}")
        try:
            chunks = extract_text(
                tmp_path, settings.DOC_CHUNK_SIZE, settings.DOC_CHUNK_OVERLAP
            )
        finally:
            os.remove(tmp_path)

        if not pc_index:
            raise HTTPException(500, detail="Pinecone not configured")
        await upsert_chunks_to_pinecone(space_id, doc_id, chunks)
        total_chunks += len(chunks)

        mdb.documents.update_one(
            {"_id": res.inserted_id},
            {"$set": {"status": "uploaded", "size": len(raw), "chunks": len(chunks)}},
        )

        background_tasks.add_task(
            _log_activity, space_id, user, "document", f"{user['name']} uploaded a new document - {f.filename}"
        )

    return {
        "_id": doc_id,
        "file_name": f.filename,
        "file_size": f.size,
        "file_type": f.content_type,
        "uploaded_at": now_utc(),
        "space_id": space_id,
        "_uploaded_by": user["_id"],
        "status": "uploaded",
        "chunks": len(chunks),
    }


@spaces_router.post("/{space_id}/invite")
async def invite_members(
    space_id: str,
    body: InviteRequest,
    bg: BackgroundTasks,
    user=Depends(get_current_user),
):
    space = mdb.spaces.find_one({"_id": space_id})
    if not space:
        raise HTTPException(404, detail="Space not found")
    require_space_role("owner", user, space)

    tokens = []
    for email in body.emails:
        token = str(uuid.uuid4())
        invite = {
            "_id": token,
            "token": token,
            "space_id": space_id,
            "email": str(email),
            "created_at": now_utc(),
            "expires_at": now_utc() + timedelta(days=7),
            "used": False,
        }
        mdb.invites.insert_one(invite)
        tokens.append({"email": email, "token": token})
        bg.add_task(send_invite_email, email, token)
        bg.add_task(
            _log_activity, space["_id"], user, "invite", f"{user['name']} invited {email} to the space {space_id}"
        )
    return {"invites": tokens}


@spaces_router.get("/join")
async def join_space(token: str, background_tasks: BackgroundTasks, user=Depends(get_current_user)):
    print(f"User {user['_id']} joining space with token {token}")
    inv = mdb.invites.find_one(
        {"token": token, "used": False, "expires_at": {"$gt": now_utc()}}
    )
    if not inv:
        raise HTTPException(400, detail="Invalid or expired invite")
    space = mdb.spaces.find_one({"_id": inv["space_id"]})
    if not space:
        raise HTTPException(404, detail="Space not found")
    if user["_id"] == space["owner_id"] or user["_id"] in space.get("members", []):
        pass
    else:
        mdb.spaces.update_one(
            {"_id": space["_id"]}, {"$addToSet": {"members": user["_id"]}}
        )
        
        background_tasks.add_task(
            _log_activity, space["_id"], user, "join", f"{user['name']} joined the space"
        )

    mdb.invites.update_one(
        {"_id": inv["_id"]},
        {"$set": {"used": True, "used_by": user["_id"], "used_at": now_utc()}},
    )
    return {"joined": inv["space_id"]}


@spaces_router.post("/{space_id}/chat")
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

    import json
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


@spaces_router.get("/{space_id}/conversations")
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
