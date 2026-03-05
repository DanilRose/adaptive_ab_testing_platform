# backend/database/config.py

from __future__ import annotations

import os


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ab_testing")
DB_USER = os.getenv("DB_USER", "ab_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ab_password")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)
