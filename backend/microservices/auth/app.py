from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.microservices.auth.routes.auth import router as auth_router
from backend.microservices.common import ServiceHealthResponse, get_service_settings

settings = get_service_settings(default_name="auth-service", default_port=8001)

app = FastAPI(
    title="Adaptive A/B Testing Auth Service",
    description="Auth микросервис",
    version=settings.service_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])


@app.get("/health", response_model=ServiceHealthResponse)
async def health_check() -> ServiceHealthResponse:
    return ServiceHealthResponse.build(settings.service_name, settings.service_version)
