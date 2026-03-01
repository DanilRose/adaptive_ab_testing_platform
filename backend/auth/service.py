# backend/auth/service.py

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.auth.models import User, UserInDB, TokenData, UserRole

# Конфигурация
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "adaptive-ab-testing-secret-key-2024-very-secure")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

# Контекст для хэширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 схема
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_password_hash(password: str) -> str:
    """Хэширование пароля через bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверка пароля через bcrypt"""
    return pwd_context.verify(plain_password, hashed_password)


# Хранилище пользователей в памяти (без БД)
_users_db: dict[str, UserInDB] = {
    "developer": UserInDB(
        id=1,
        username="developer",
        role=UserRole.developer,
        full_name="Разработчик",
        hashed_password=get_password_hash("dev123"),
    ),
    "analyst": UserInDB(
        id=2,
        username="analyst",
        role=UserRole.analyst,
        full_name="Аналитик",
        hashed_password=get_password_hash("analyst123"),
    ),
    "manager": UserInDB(
        id=3,
        username="manager",
        role=UserRole.manager,
        full_name="Проект-менеджер",
        hashed_password=get_password_hash("manager123"),
    ),
}


def get_user(username: str) -> Optional[UserInDB]:
    """Получить пользователя из словаря"""
    return _users_db.get(username)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Проверить логин и пароль пользователя"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Создать JWT токен"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Dependency для FastAPI — декодирует JWT и возвращает пользователя"""
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

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception

    return User(
        id=user.id,
        username=user.username,
        role=user.role,
        full_name=user.full_name,
    )


def require_role(*roles: str):
    """Фабрика dependency — проверяет, что у пользователя нужная роль"""

    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Недостаточно прав. Требуется одна из ролей: {', '.join(roles)}",
            )
        return current_user

    return role_checker
