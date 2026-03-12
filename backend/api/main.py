# backend/api/main.py

import os
import math
import json
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
from backend.api.routes.auth import router as auth_router
from backend.database.init_db import bootstrap_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_data(data):
    """Recursively sanitize data to replace NaN/Infinity with None"""
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data


class SanitizedJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        sanitized = sanitize_data(content)
        return super().render(sanitized)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(" Starting Adaptive A/B Testing Platform API")
    try:
        bootstrap_database()
        logger.info("✅ Database initialized")
        
        # Исправление бага 1: Сброс "висячих" статусов симуляции при перезапуске
        try:
            from backend.database.session import SessionLocal
            from backend.database import crud
            
            with SessionLocal() as db:
                # Находим все тесты с simulation_status='running' и сбрасываем их
                from backend.database.models import ABTestORM
                running_simulations = db.query(ABTestORM).filter(
                    ABTestORM.simulation_status == 'running'
                ).all()
                
                for test in running_simulations:
                    test.simulation_status = None
                    logger.warning(f"⚠️ Reset stuck simulation status for test {test.test_id}")
                
                if running_simulations:
                    db.commit()
                    logger.info(f"✅ Reset {len(running_simulations)} stuck simulation(s)")
        except Exception as e:
            logger.error(f"⚠️ Failed to reset stuck simulations: {e}")
            
    except Exception as db_exc:
        logger.error(f"❌ Database bootstrap failed: {db_exc}", exc_info=True)
    yield
    logger.info("Shutting down Adaptive A/B Testing Platform API")

app = FastAPI(
    title="Adaptive A/B Testing Platform",
    description="Профессиональная платформа для адаптивного A/B тестирования с использованием ML",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=SanitizedJSONResponse
)


cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://frontend:80").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
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

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": "mock-request-id"  
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