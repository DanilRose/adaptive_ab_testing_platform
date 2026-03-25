from __future__ import annotations

from datetime import datetime, timezone
import math

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.microservices.auth.routes.auth import router as auth_router
from backend.microservices.ab_testing.routes.tests import router as tests_router
from backend.microservices.data_gan.routes.data import router as data_router
from backend.microservices.ab_testing.routes.results import router as results_router
from backend.microservices.data_gan.routes.templates import router as templates_router
from backend.microservices.common import ServiceHealthResponse, get_service_settings

settings = get_service_settings(default_name="gateway-service", default_port=8000)


def sanitize_data(data):
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_data(item) for item in data]
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data


class SanitizedJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(sanitize_data(content))


app = FastAPI(
    title="Adaptive A/B Testing Platform Gateway",
    description="API gateway для микросервисов Adaptive A/B Testing",
    version=settings.service_version,
    default_response_class=SanitizedJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Сохраняем публичные контракты для frontend без изменений
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(tests_router)
app.include_router(data_router)
app.include_router(results_router)
app.include_router(templates_router)


@app.get("/")
async def root():
    return {
        "message": "Adaptive A/B Testing Platform Gateway",
        "version": settings.service_version,
        "status": "running",
        "endpoints": {
            "auth": "/api/v1/auth",
            "tests": "/api/v1/tests",
            "data": "/api/v1/data",
            "results": "/api/v1/results",
            "templates": "/api/v1/templates",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=ServiceHealthResponse)
async def health_check() -> ServiceHealthResponse:
    return ServiceHealthResponse.build(settings.service_name, settings.service_version)


@app.get("/api/v1/status")
async def api_status():
    return {
        "api_version": settings.service_version,
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
