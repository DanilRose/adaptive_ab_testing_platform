# backend/auth/service.py

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.microservices.auth_core.models import User, UserInDB, TokenData, UserRole
from backend.microservices.database.models import UserORM
from backend.microservices.database.session import SessionLocal

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "adaptive-ab-testing-secret-key-2024-very-secure")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def _to_user_in_db(user: UserORM) -> UserInDB:
    return UserInDB(
        id=user.id,
        username=user.username,
        role=UserRole(user.role),
        full_name=user.full_name,
        hashed_password=user.hashed_password,
    )


def get_user(username: str, db: Optional[Session] = None) -> Optional[UserInDB]:
    if db is not None:
        user = db.query(UserORM).filter(UserORM.username == username).first()
        return _to_user_in_db(user) if user else None

    with SessionLocal() as local_db:
        user = local_db.query(UserORM).filter(UserORM.username == username).first()
        return _to_user_in_db(user) if user else None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Не удалось проверить учётные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception

    user = await run_in_threadpool(get_user, token_data.username)
    if user is None:
        raise credentials_exception

    return User(
        id=user.id,
        username=user.username,
        role=user.role,
        full_name=user.full_name,
    )


def require_role(*roles: str):

    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Недостаточно прав. Требуется одна из ролей: {', '.join(roles)}",
            )
        return current_user

    return role_checker
