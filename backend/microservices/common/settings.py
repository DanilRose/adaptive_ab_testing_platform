from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceSettings:
    service_name: str
    service_version: str
    service_port: int
    cors_origins: list[str]



def get_service_settings(default_name: str, default_port: int) -> ServiceSettings:
    raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://frontend:80")
    origins = [item.strip() for item in raw_origins.split(",") if item.strip()]

    return ServiceSettings(
        service_name=os.getenv("SERVICE_NAME", default_name),
        service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
        service_port=int(os.getenv("SERVICE_PORT", str(default_port))),
        cors_origins=origins,
    )
