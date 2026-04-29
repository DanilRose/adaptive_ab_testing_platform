# backend/auth/models.py

from enum import Enum
from pydantic import BaseModel
from typing import Optional


class UserRole(str, Enum):
    user = "user"


class User(BaseModel):
    id: int
    username: str
    role: UserRole
    full_name: str
    job_title: Optional[str] = None
    permissions: list[str] = []
    email: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    full_name: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
