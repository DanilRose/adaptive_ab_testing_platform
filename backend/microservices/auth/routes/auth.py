# backend/api/routes/auth.py

from datetime import timedelta

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, constr
from sqlalchemy.orm import Session

from backend.microservices.auth_core.models import Token, User, UserRole
from backend.microservices.auth_core.service import (
    ACCESS_TOKEN_EXPIRE_HOURS,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_role,
)
from backend.microservices.database import crud
from backend.microservices.auth_core.service import get_password_hash
from backend.microservices.database.session import get_db

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


class AdminUserRoleUpdateRequest(BaseModel):
    role: UserRole = Field(..., description="Legacy: базовая роль пользователя")


class AdminUserPermissionsUpdateRequest(BaseModel):
    permissions: list[str] = Field(default_factory=list, description="Набор granular-ролей (permissions)")


class ProfileUpdateRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=128)
    email: str | None = Field(None, max_length=255)
    phone: str | None = Field(None, max_length=32)
    avatar_url: str | None = Field(None, max_length=2048)


class AdminCreateUserRequest(BaseModel):
    username: constr(strip_whitespace=True, min_length=3, max_length=64)
    password: constr(min_length=6, max_length=128)
    full_name: constr(strip_whitespace=True, min_length=2, max_length=128)
    role: str = Field(default="user", max_length=32)
    job_title: str | None = Field(default=None, max_length=64)
    permissions: list[str] = Field(default_factory=list)
    email: str | None = Field(default=None, max_length=255)
    phone: str | None = Field(default=None, max_length=32)
    avatar_url: str | None = Field(default=None, max_length=2048)


@router.put("/me/profile", response_model=User, summary="Обновить профиль текущего пользователя")
async def update_my_profile(
    payload: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    updated = crud.update_user_profile(
        db,
        user_id=current_user.id,
        full_name=payload.full_name.strip(),
        email=(payload.email.strip() if payload.email else None),
        phone=(payload.phone.strip() if payload.phone else None),
        avatar_url=(payload.avatar_url.strip() if payload.avatar_url else None),
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return User(
        id=updated.id,
        username=updated.username,
        role=UserRole(updated.role),
        full_name=updated.full_name,
        job_title=updated.job_title,
        permissions=updated.permissions_json or [],
        email=updated.email,
        phone=updated.phone,
        avatar_url=updated.avatar_url,
    )


@router.post("/me/avatar", response_model=User, summary="Загрузить фото профиля (BLOB)")
async def upload_my_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    content_type = (file.content_type or "").lower()
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Допустимы только JPEG/PNG/WEBP")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Пустой файл")
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Максимальный размер 5MB")

    updated = crud.update_user_avatar_blob(
        db,
        user_id=current_user.id,
        avatar_blob=content,
        mime_type=content_type,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return User(
        id=updated.id,
        username=updated.username,
        role=UserRole(updated.role),
        full_name=updated.full_name,
        job_title=updated.job_title,
        permissions=updated.permissions_json or [],
        email=updated.email,
        phone=updated.phone,
        avatar_url=updated.avatar_url,
    )


@router.get("/me/avatar", summary="Получить фото профиля (BLOB)")
async def get_my_avatar(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = crud.get_user_by_username(db, current_user.username)
    if user is None or user.avatar_blob is None:
        raise HTTPException(status_code=404, detail="Фото профиля не найдено")

    return Response(content=user.avatar_blob, media_type=user.avatar_mime_type or "application/octet-stream")


@router.post("/logout", summary="Выход из системы")
async def logout():
    """Выход из системы (токен инвалидируется на стороне клиента)"""
    return {"message": "Logged out successfully"}


@router.get("/admin/users", summary="Список пользователей (только developer)")
async def admin_list_users(
    current_user: User = Depends(require_role("developer")),
    db: Session = Depends(get_db),
):
    users = crud.list_users(db, limit=1000)
    return {
        "items": [
            {
                "id": user.id,
                "username": user.username,
                "role": user.role,
                "full_name": user.full_name,
                "job_title": user.job_title,
                "permissions": user.permissions_json or [],
                "email": user.email,
                "phone": user.phone,
                "avatar_url": user.avatar_url,
                "has_avatar": bool(user.avatar_blob) or bool(user.avatar_url),
            }
            for user in users
        ],
        "count": len(users),
    }


@router.put("/admin/users/{user_id}/role", summary="Обновить роль пользователя (только developer)")
async def admin_update_user_role(
    user_id: int,
    payload: AdminUserRoleUpdateRequest,
    current_user: User = Depends(require_role("developer")),
    db: Session = Depends(get_db),
):
    updated_user = crud.update_user_role(db, user_id=user_id, role=payload.role.value)
    if updated_user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {
        "id": updated_user.id,
        "username": updated_user.username,
        "role": updated_user.role,
        "full_name": updated_user.full_name,
        "job_title": updated_user.job_title,
        "permissions": updated_user.permissions_json or [],
        "email": updated_user.email,
        "phone": updated_user.phone,
        "avatar_url": updated_user.avatar_url,
        "message": "Роль пользователя обновлена",
    }


@router.put("/admin/users/{user_id}/permissions", summary="Обновить permissions пользователя (только developer)")
async def admin_update_user_permissions(
    user_id: int,
    payload: AdminUserPermissionsUpdateRequest,
    current_user: User = Depends(require_role("developer")),
    db: Session = Depends(get_db),
):
    updated_user = crud.update_user_permissions(db, user_id=user_id, permissions=payload.permissions)
    if updated_user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {
        "id": updated_user.id,
        "username": updated_user.username,
        "role": updated_user.role,
        "full_name": updated_user.full_name,
        "job_title": updated_user.job_title,
        "permissions": updated_user.permissions_json or [],
        "email": updated_user.email,
        "phone": updated_user.phone,
        "avatar_url": updated_user.avatar_url,
        "message": "Права пользователя обновлены",
    }


@router.get("/admin/users/{user_id}/avatar", summary="Получить фото пользователя (только developer)")
async def admin_get_user_avatar(
    user_id: int,
    current_user: User = Depends(require_role("developer")),
    db: Session = Depends(get_db),
):
    from backend.microservices.database.models import UserORM
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    if user.avatar_blob is not None:
        return Response(content=user.avatar_blob, media_type=user.avatar_mime_type or "application/octet-stream")

    if user.avatar_url:
        raise HTTPException(status_code=409, detail="У пользователя фото задано как URL")

    raise HTTPException(status_code=404, detail="Фото профиля не найдено")


@router.post("/admin/users", summary="Создать пользователя (только developer)")
async def admin_create_user(
    payload: AdminCreateUserRequest,
    current_user: User = Depends(require_role("developer")),
    db: Session = Depends(get_db),
):
    existing = crud.get_user_by_username(db, payload.username)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Пользователь с таким username уже существует")

    hashed_password = get_password_hash(payload.password)

    try:
        created_user = crud.create_user(
            db,
            username=payload.username,
            hashed_password=hashed_password,
            full_name=payload.full_name,
            role=(payload.role or "user").strip() or "user",
            job_title=(payload.job_title.strip() if payload.job_title else None),
            permissions=payload.permissions,
            email=(str(payload.email).strip() if payload.email else None),
            phone=(payload.phone.strip() if payload.phone else None),
            avatar_url=(payload.avatar_url.strip() if payload.avatar_url else None),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось создать пользователя: {exc}")

    return {
        "id": created_user.id,
        "username": created_user.username,
        "role": created_user.role,
        "full_name": created_user.full_name,
        "job_title": created_user.job_title,
        "permissions": created_user.permissions_json or [],
        "email": created_user.email,
        "phone": created_user.phone,
        "avatar_url": created_user.avatar_url,
        "message": "Пользователь создан",
    }
