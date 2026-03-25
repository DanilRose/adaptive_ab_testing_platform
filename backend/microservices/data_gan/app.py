from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.microservices.data_gan.routes.data import router as data_router
from backend.microservices.data_gan.routes.templates import router as templates_router
from backend.microservices.common import ServiceHealthResponse, get_service_settings

logger = logging.getLogger(__name__)
settings = get_service_settings(default_name="data-gan-service", default_port=8003)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Data GAN service")
    yield
    logger.info("🛑 Stopping Data GAN service")


app = FastAPI(
    title="Adaptive A/B Testing Data GAN Service",
    description="Data generation and templates микросервис",
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

app.include_router(data_router)
app.include_router(templates_router)


@app.get("/health", response_model=ServiceHealthResponse)
async def health_check() -> ServiceHealthResponse:
    return ServiceHealthResponse.build(settings.service_name, settings.service_version)
