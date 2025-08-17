import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Literal, Dict, Any
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    BackgroundTasks,
)
from fastapi import status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr

# MongoDB Async (PyMongo Async API)
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Redis asyncio client
from redis.asyncio import Redis

# Pinecone SDK v7 (API 2025-04)
from pinecone import Pinecone, ServerlessSpec

# Google Gemini SDK
import google.generativeai as genai

# JWT
import jwt
from passlib.context import CryptContext

from contextlib import asynccontextmanager

from dotenv import load_dotenv

from extractor import extract_text
from jose import JWTError
from email_invite import send_invite_email
from fastapi.responses import StreamingResponse

load_dotenv()  # Load environment variables from .env file
# ----------------------------
# Config & Globals
# ----------------------------


class Settings(BaseModel):
    MONGO_URI: str = Field(
        default_factory=lambda: os.getenv("MONGO_URI", "mongodb://localhost:27017")
    )
    MONGO_DB: str = os.getenv("MONGO_DB", "spaces_rag")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    PINECONE_API_KEY: str = Field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", "")
    )
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "spaces-rag")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_INDEX_HOST: str = os.getenv("PINECONE_HOST", "index.pinecone.io")

    GEMINI_API_KEY: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
    GEMINI_EMBED_MODEL: str = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

    JWT_SECRET: str = Field(
        default_factory=lambda: os.getenv("JWT_SECRET", "dev-secret-change-me")
    )
    JWT_ALG: str = os.getenv("JWT_ALG", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    DOC_CHUNK_SIZE: int = int(os.getenv("DOC_CHUNK_SIZE", "1200"))
    DOC_CHUNK_OVERLAP: int = int(os.getenv("DOC_CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.getenv("TOP_K", "6"))


settings = Settings()

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Clients (initialized in startup)
mongo_client: None
mdb = None  # db handle
redis: Optional[Redis] = None
pc: Optional[Pinecone] = None
pc_index = None

# Gemini
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)


# ----------------------------
# Lifespan Startup/Shutdown
# ----------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mongo_client, mdb, redis, pc, pc_index
    # Mongo Async
    mongo_client = MongoClient(settings.MONGO_URI)
    mdb = mongo_client[settings.MONGO_DB]
    # Indexes
    mdb["spaces"].create_index("owner_id")
    mdb["spaces"].create_index("members")
    mdb["messages"].create_index([("space_id", 1), ("user_id", 1), ("created_at", -1)])
    mdb["invites"].create_index("space_id")
    mdb["invites"].create_index("token", unique=True)
    mdb["invites"].create_index("expires_at", expireAfterSeconds=0)

    # Redis
    redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)

    # Pinecone
    if not settings.PINECONE_API_KEY:
        print("[WARN] PINECONE_API_KEY missing – vector ops will fail.")
    else:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        pc_index = Pinecone.Index(
            self=pc,
            name=settings.PINECONE_INDEX,
            host=settings.PINECONE_INDEX_HOST,
        )

    # Gemini check
    if not settings.GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY missing – LLM calls will fail.")

    yield

    # Shutdown
    mongo_client.close()
    if redis:
        await redis.close()


# FastAPI app
app = FastAPI(title="Spaces RAG Backend", version="0.1.0", lifespan=lifespan)

# ----------------------------
# Utility Helpers
# ----------------------------


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    return pwd_context.verify(pw, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = now_utc() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALG)


def create_refresh_token(data: dict):
    expire = now_utc() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"sub": data["sub"], "exp": expire}
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALG)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = mdb.users.find_one({"_id": user_id})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")


def require_space_role(
    role: Literal["owner", "member", "any"], user: dict, space: dict
):
    if role == "any":
        return
    if role == "owner" and space["owner_id"] != user["_id"]:
        raise HTTPException(403, detail="Owner access required")
    if role == "member":
        # either owner or in members list
        if space["owner_id"] == user["_id"]:
            return
        if user["_id"] not in space.get("members", []):
            raise HTTPException(403, detail="Member access required")


def update_space_last_activity(space_id: str):
    update_data = {"last_activity": datetime.now(timezone.utc)}
    mdb.spaces.update_one({"_id": space_id}, {"$set": update_data})


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def get_pinecone_index():
    global pc_index
    if pc_index is None:
        pc_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        pc_index = pc_client.Index(host=settings.PINECONE_INDEX_HOST)
    return pc_index


async def pinecone_upsert(namespace: str, records: List[Dict[str, Any]]):
    index = get_pinecone_index()
    index.upsert_records(namespace, records)


async def pinecone_query(namespace: str, query_text: str, top_k: int = 5):
    index = get_pinecone_index()
    results = index.search(
        namespace=namespace,
        query={"inputs": {"text": query_text}},
        top_k=top_k,
        fields=["chunk_text", "category"],
    )
    return results


async def upsert_chunks_to_pinecone(space_id: str, doc_id: str, chunks: List[str]):
    index = get_pinecone_index()
    records = []
    for i, ch in enumerate(chunks):
        records.append(
            {
                "_id": f"{doc_id}:{i}",
                "text": ch,
                "space_id": space_id,
                "doc_id": doc_id,
                "chunk_no": i,
            }
        )
    index.upsert_records(space_id, records)


async def query_pinecone(space_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    index = get_pinecone_index()
    print("space id", space_id, query, top_k)
    res = index.search(
        namespace=space_id,
        query={"inputs": {"text": query}, "top_k": top_k},
        fields=["text", "doc_id", "chunk_no"],
    )
    res = res.get("result").get("hits", [])
    matches = []
    print("matches", res)
    for m in res:
        matches.append(
            {
                "id": m["_id"],
                "score": m["_score"],
                "text": m.get("fields", {}).get("text", ""),
                "doc_id": m.get("fields", {}).get("doc_id", ""),
            }
        )
    return matches


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


async def ensure_index():
    # Create serverless index if it doesn't exist
    indexes = await pc.list_indexes()
    if settings.PINECONE_INDEX not in [ix.name for ix in indexes]:
        await pc.create_index(
            name=settings.PINECONE_INDEX,
            dimension=3072,  # Gemini text-embedding-004 dim
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION
            ),
        )
    return pc.Index(settings.PINECONE_INDEX)


# ----------------------------
# Pydantic Schemas
# ----------------------------


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: str
    user_id: str = Field(default=None, description="User ID of the authenticated user")
    user_email: EmailStr = Field(
        default=None, description="Email of the authenticated user"
    )
    user_name: str = Field(default=None, description="Name of the authenticated user")


class SpaceCreate(BaseModel):
    name: str
    description: Optional[str] = None


class InviteRequest(BaseModel):
    emails: List[EmailStr]


class ChatRequest(BaseModel):
    message: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Activity Logger
# ----------------------------


def _log_activity(space_id: str, actor: dict, type_: str, description: str):
    mdb.activities.insert_one(
        {
            "space_id": space_id,
            "actor": {
                "id": actor["_id"],
                "name": actor.get("name"),
                "team": actor.get("team"),
            },
            "type": type_,
            "description": description,
            "created_at": now_utc(),
        }
    )


# ----------------------------
# Auth Endpoints
# ----------------------------


@app.post("/auth/register", response_model=Token)
async def register(body: UserCreate):
    existing = mdb.users.find_one({"email": body.email})
    if existing:
        raise HTTPException(400, detail="Email already registered")
    user = {
        "_id": str(uuid.uuid4()),
        "email": body.email,
        "name": body.name or body.email.split("@")[0],
        "password": hash_password(body.password),
        "created_at": now_utc(),
    }
    mdb.users.insert_one(user)
    token = create_access_token({"sub": user["_id"], "email": user["email"]})
    refresh_token = create_refresh_token({"sub": user["_id"], "email": user["email"]})
    return Token(
        access_token=token,
        refresh_token=refresh_token,
        user_id=user["_id"],
        user_email=user["email"],
        user_name=user["name"],
    )


@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = mdb.users.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(401, detail="Invalid credentials")
    token = create_access_token({"sub": user["_id"], "email": user["email"]})
    refresh_token = create_refresh_token({"sub": user["_id"], "email": user["email"]})
    return Token(
        access_token=token,
        refresh_token=refresh_token,
        user_id=user["_id"],
        user_email=user["email"],
        user_name=user["name"],
    )


@app.post("/refresh", response_model=Token)
async def refresh_tokens(body: TokenRefreshRequest):
    try:
        payload = jwt.decode(
            body.refresh_token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

        # issue new tokens
        access_token = create_access_token({"sub": user_id})
        refresh_token = create_refresh_token({"sub": user_id})
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )


# ----------------------------
# Space Management
# ----------------------------


@app.post("/spaces")
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


@app.get("/spaces")
async def get_spaces(user=Depends(get_current_user)):
    spaces = list(
        mdb.spaces.find({"$or": [{"owner_id": user["_id"]}, {"members": user["_id"]}]})
    )
    for space in spaces:
        space["messages"] = mdb.messages.count_documents({"space_id": space["_id"]})
        space["documents"] = mdb.documents.count_documents({"space_id": space["_id"]})
    return {"spaces": spaces}


@app.get("/spaces/{space_id}")
async def get_space(space_id: str, user=Depends(get_current_user)):
    space_details = mdb.spaces.find_one({"_id": space_id})
    return {"space": space_details}


@app.get("/spaces/{space_id}/documents")
async def list_documents(space_id: str, user=Depends(get_current_user)):
    documents = list(mdb.documents.find({"space_id": space_id}))
    return {"documents": documents}


@app.post("/spaces/{space_id}/documents")
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


@app.post("/spaces/{space_id}/invite")
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


@app.get("/join/space")
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


# ----------------------------
# Chat (RAG)
# ----------------------------


@app.post("/spaces/{space_id}/chat")
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

# ----------------------------
# Conversation listing
# ----------------------------


@app.get("/spaces/{space_id}/conversations")
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


# ------------------------
# Activity 
# ------------------------
@app.get("/activities")
async def list_activities(
    space_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    query = {}

    if space_id:
        # Specific space feed
        space = mdb.spaces.find_one({"_id": space_id})
        if not space:
            raise HTTPException(404, detail="Space not found")

        # Check membership or ownership
        if user["_id"] != space["owner_id"] and user["_id"] not in space.get("members", []):
            raise HTTPException(403, detail="Not authorized to view activities")

        query["space_id"] = space_id
    else:
        # Dashboard feed (all spaces where user is owner or member)
        user_spaces = mdb.spaces.find({
            "$or": [
                {"owner_id": user["_id"]},
                {"members": user["_id"]}
            ]
        }, {"_id": 1})

        allowed_space_ids = [s["_id"] for s in user_spaces]
        query["space_id"] = {"$in": allowed_space_ids}

    cursor = (
        mdb.activities.find(query)
        .sort("created_at", -1)
        .skip(0)
        .limit(10)
    )

    activities = []
    for act in cursor:
        activities.append({
            "id": str(act["_id"]),
            "space_id": act["space_id"],
            "actor": act["actor"],
            "type": act["type"],
            "description": act["description"],
            "created_at": act["created_at"],
        })

    return {"activities": activities}


# ----------------------------
# Health
# ----------------------------


@app.get("/health")
async def health():
    return {
        "ok": True,
        "mongo": True if mongo_client else False,
        "redis": True if redis else False,
        "pinecone": True if pc else False,
        "time": now_utc().isoformat(),
    }
