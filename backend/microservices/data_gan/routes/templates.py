"""
backend/api/routes/templates.py
API для управления шаблонами (GAN конфиги, синтетические данные, A/B тесты).
"""
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.microservices.auth_core.models import User
from backend.microservices.auth_core.service import require_role
from backend.microservices.database import crud
from backend.microservices.database.session import get_db

router = APIRouter(prefix="/api/v1/templates", tags=["Шаблоны"])


# ============================================================================
# Pydantic модели
# ============================================================================

class TemplateCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Название шаблона")
    description: Optional[str] = Field(None, description="Описание шаблона")
    template_type: str = Field(
        ...,
        description="Тип шаблона: gan_config | synthetic_data | ab_test",
        pattern="^(gan_config|synthetic_data|ab_test)$",
    )
    config_json: dict[str, Any] = Field(..., description="Конфигурация шаблона (JSON)")
    tags: Optional[List[str]] = Field(None, description="Теги для поиска и фильтрации")


class TemplateUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    config_json: Optional[dict[str, Any]] = None
    tags: Optional[List[str]] = None


class TemplateResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    template_type: str
    config_json: dict[str, Any]
    tags: Optional[List[str]]
    created_by: Optional[str]
    created_at: str
    updated_at: str


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/", summary="Список шаблонов")
async def list_templates(
    template_type: Optional[str] = Query(None, description="Фильтр по типу: gan_config | synthetic_data | ab_test"),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(require_role("developer", "analyst", "manager")),
    db: Session = Depends(get_db),
):
    """Получить список всех шаблонов (с возможностью фильтрации по типу)."""
    templates = crud.list_templates(db, template_type=template_type, limit=limit)
    return {
        "items": [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "template_type": t.template_type,
                "config_json": t.config_json,
                "tags": t.tags or [],
                "created_by": t.created_by,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            }
            for t in templates
        ],
        "count": len(templates),
    }


@router.post("/", summary="Создать шаблон")
async def create_template(
    body: TemplateCreateRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """Создать новый шаблон."""
    template = crud.create_template(
        db,
        name=body.name,
        description=body.description,
        template_type=body.template_type,
        config_json=body.config_json,
        tags=body.tags,
        created_by=current_user.username,
    )
    return {
        "id": template.id,
        "name": template.name,
        "template_type": template.template_type,
        "message": "Шаблон успешно создан",
    }


@router.get("/{template_id}", summary="Получить шаблон по ID")
async def get_template(
    template_id: int,
    current_user: User = Depends(require_role("developer", "analyst", "manager")),
    db: Session = Depends(get_db),
):
    """Получить детали конкретного шаблона."""
    template = crud.get_template_by_id(db, template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Шаблон не найден")
    return {
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "template_type": template.template_type,
        "config_json": template.config_json,
        "tags": template.tags or [],
        "created_by": template.created_by,
        "created_at": template.created_at.isoformat() if template.created_at else None,
        "updated_at": template.updated_at.isoformat() if template.updated_at else None,
    }


@router.put("/{template_id}", summary="Обновить шаблон")
async def update_template(
    template_id: int,
    body: TemplateUpdateRequest,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """Обновить существующий шаблон."""
    template = crud.update_template(
        db,
        template_id,
        name=body.name,
        description=body.description,
        config_json=body.config_json,
        tags=body.tags,
    )
    if not template:
        raise HTTPException(status_code=404, detail="Шаблон не найден")
    return {
        "id": template.id,
        "name": template.name,
        "template_type": template.template_type,
        "message": "Шаблон успешно обновлён",
    }


@router.delete("/{template_id}", summary="Удалить шаблон")
async def delete_template(
    template_id: int,
    current_user: User = Depends(require_role("developer", "analyst")),
    db: Session = Depends(get_db),
):
    """Удалить шаблон по ID."""
    success = crud.delete_template_by_id(db, template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Шаблон не найден")
    return {"status": "deleted", "id": template_id}


@router.post("/seed-defaults", summary="Создать стандартные шаблоны")
async def seed_default_templates(
    current_user: User = Depends(require_role("developer")),
    db: Session = Depends(get_db),
):
    """Создать стандартные шаблоны (только если база пустая)."""
    count = crud.seed_default_templates(db)
    if count == 0:
        return {"message": "Стандартные шаблоны уже существуют", "created": 0}
    return {"message": f"Создано {count} стандартных шаблонов", "created": count}
