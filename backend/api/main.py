# backend/api/main.py

from __future__ import annotations

import uvicorn

from backend.microservices.gateway.app import app
from backend.database.init_db import bootstrap_database


@app.on_event("startup")
async def _legacy_bootstrap_startup() -> None:
    """
    Legacy bootstrap hook kept for backward compatibility.
    Реальная API-точка теперь находится в gateway микросервисе.
    """
    bootstrap_database()


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
