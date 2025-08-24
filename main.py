from typing import Optional
from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from redis.asyncio import Redis
from pinecone import Pinecone
import google.generativeai as genai
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from settings import Settings

load_dotenv()


# Clients (initialized in startup)
mongo_client: None
mdb = None  # db handle
redis: Optional[Redis] = None
pc: Optional[Pinecone] = None
pc_index = None
settings = Settings()

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


app = FastAPI(title="Spaces RAG Backend", version="0.1.0", lifespan=lifespan)

# ----------------------------
# Routers
# ----------------------------
from auth.routes import auth_router
from activity.routes import activity_router
from chat.routes import chat_router
from spaces.routes import spaces_router

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(spaces_router, prefix="/spaces", tags=["spaces"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(activity_router, prefix="/activity", tags=["activity"])


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Health
# ----------------------------


@app.get("/health")
async def health():
    from auth.helpers import now_utc
    return {
        "ok": True,
        "mongo": True if mongo_client else False,
        "redis": True if redis else False,
        "pinecone": True if pc else False,
        "time": now_utc().isoformat(),
    }
