from pydantic import BaseModel, EmailStr
from typing import Optional, List

class SpaceCreate(BaseModel):
    name: str
    description: Optional[str] = None


class InviteRequest(BaseModel):
    emails: List[EmailStr]


class ChatRequest(BaseModel):
    message: str
