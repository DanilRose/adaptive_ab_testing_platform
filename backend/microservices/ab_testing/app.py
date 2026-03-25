from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.microservices.ab_testing.routes.tests import router as tests_router
from backend.microservices.ab_testing.routes.results import router as results_router
from backend.microservices.common import ServiceHealthResponse, get_service_settings

logger = logging.getLogger(__name__)
settings = get_service_settings(default_name="ab-testing-service", default_port=8002)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting AB Testing service")
    yield
    logger.info("🛑 Stopping AB Testing service")


app = FastAPI(
    title="Adaptive A/B Testing Service",
    description="AB testing and analytics микросервис",
    version=settings.service_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tests_router)
app.include_router(results_router)


@app.get("/health", response_model=ServiceHealthResponse)
async def health_check() -> ServiceHealthResponse:
    return ServiceHealthResponse.build(settings.service_name, settings.service_version)
