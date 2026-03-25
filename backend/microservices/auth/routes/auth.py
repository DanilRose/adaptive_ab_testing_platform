# backend/api/routes/auth.py

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordRequestForm

from backend.auth.models import Token, User
from backend.auth.service import (
    ACCESS_TOKEN_EXPIRE_HOURS,
    authenticate_user,
    create_access_token,
    get_current_user,
)

router = APIRouter(tags=["auth"])


@router.post("/login", response_model=Token, summary="Авторизация пользователя")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Принимает username и password, возвращает JWT токен"""
    user = await run_in_threadpool(authenticate_user, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires,
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user.role,
        full_name=user.full_name,
    )


@router.get("/me", response_model=User, summary="Получить текущего пользователя")
async def get_me(current_user: User = Depends(get_current_user)):
    """Возвращает информацию о текущем авторизованном пользователе"""
    return current_user


@router.post("/logout", summary="Выход из системы")
async def logout():
    """Выход из системы (токен инвалидируется на стороне клиента)"""
    return {"message": "Logged out successfully"}
