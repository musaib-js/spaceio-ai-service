from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import jwt
from jose import JWTError
from auth.data_types import UserCreate, Token, TokenRefreshRequest
import uuid
from main import mdb
import settings
from auth.helpers import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    now_utc
)


auth_router = APIRouter()
settings = settings.Settings()

# ----------------------------
# Auth Endpoints
# ----------------------------


@auth_router.post("/register", response_model=Token)
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
        "verified": False
    }
    mdb.users.insert_one(user)
    
    return {
        "message": "You've successfully joined the waiting list. Stay tuned in your inbox for updates!",
        "status": "Success"
    }
    # After full launch
    token = create_access_token({"sub": user["_id"], "email": user["email"]})
    refresh_token = create_refresh_token({"sub": user["_id"], "email": user["email"]})
    return Token(
        access_token=token,
        refresh_token=refresh_token,
        user_id=user["_id"],
        user_email=user["email"],
        user_name=user["name"],
    )


@auth_router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = mdb.users.find_one({"email": form_data.username})
    if not user["verified"]:
        raise HTTPException(403, detail="You're still there in the waiting list. Stay tuned for updates!")
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


@auth_router.post("/refresh", response_model=Token)
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