from pydantic import BaseModel, EmailStr, Field
from typing import Optional

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
    
class TokenRefreshRequest(BaseModel):
    refresh_token: str