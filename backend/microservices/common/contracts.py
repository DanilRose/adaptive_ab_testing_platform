from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel


class ServiceInfo(BaseModel):
    name: str
    version: str


class ServiceHealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    service: ServiceInfo

    @staticmethod
    def build(name: str, version: str) -> "ServiceHealthResponse":
        return ServiceHealthResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            service=ServiceInfo(name=name, version=version),
        )
