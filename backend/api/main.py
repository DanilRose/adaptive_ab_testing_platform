# backend/api/main.py

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes.tests import router as tests_router
from backend.api.routes.data import router as data_router
from backend.api.routes.results import router as results_router

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Adaptive A/B Testing Platform API")
    yield
    # Shutdown
    logger.info("🛑 Shutting down Adaptive A/B Testing Platform API")

app = FastAPI(
    title="Adaptive A/B Testing Platform",
    description="Профессиональная платформа для адаптивного A/B тестирования с использованием ML",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://frontend:80").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(tests_router)
app.include_router(data_router)
app.include_router(results_router)

@app.get("/")
async def root():
    return {
        "message": "Adaptive A/B Testing Platform API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "tests": "/api/v1/tests",
            "data": "/api/v1/data", 
            "results": "/api/v1/results",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "ok",
            "ab_testing": "ok",
            "data_generation": "ok"
        }
    }

@app.get("/api/v1/status")
async def api_status():
    return {
        "api_version": "1.0.0",
        "status": "operational",
        "active_tests": 0,
        "total_requests": 0,
        "uptime": "0 days 0 hours 0 minutes"
    }

# Глобальный обработчик ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": "mock-request-id"  # В реальном приложении генерировать ID
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )