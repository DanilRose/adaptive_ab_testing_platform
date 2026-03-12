# backend/database/models.py

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from backend.database.session import Base


class UserORM(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    tests: Mapped[list[ABTestORM]] = relationship("ABTestORM", back_populates="created_by_user")


class ABTestORM(Base):
    __tablename__ = "ab_tests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    test_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    test_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False, default="prepared")  # prepared, active, paused, completed, archived
    simulation_status: Mapped[str | None] = mapped_column(String(32), nullable=True)  # null, running, stopped
    total_users: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_percentage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    archive_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ОБЯЗАТЕЛЬНАЯ ПРИВЯЗКА К СИНТЕТИЧЕСКОМУ ДАТАСЕТУ
    dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("generated_data.id", ondelete="RESTRICT"),
        nullable=True,  # Временно nullable для миграции
        index=True
    )

    variants: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    primary_metric: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_type: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence_level: Mapped[float] = mapped_column(Float, nullable=False, default=0.95)
    power: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    min_effect_size: Mapped[float] = mapped_column(Float, nullable=False, default=0.1)

    # НОВЫЕ ПОЛЯ ДЛЯ GOOGLE-STANDARD TESTING
    # Временные параметры (реальное время vs симуляция)
    real_world_duration_days: Mapped[int] = mapped_column(Integer, nullable=False, default=14)
    simulation_duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    
    # MDE и статистическая мощность
    mde_percent: Mapped[float] = mapped_column(Float, nullable=False, default=10.0)
    actual_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Sequential Testing параметры
    max_sequential_looks: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    current_sequential_look: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stopped_early: Mapped[bool] = mapped_column(Integer, default=0, nullable=False)  # SQLite: bool as int
    early_stop_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Sample Ratio Mismatch (SRM) проверка
    srm_check_passed: Mapped[bool | None] = mapped_column(Integer, nullable=True)  # SQLite: bool as int
    srm_p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Traffic split тип (fixed, adaptive)
    traffic_split_type: Mapped[str] = mapped_column(String(32), nullable=False, default="fixed")
    traffic_split_seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)

    created_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    created_by_user: Mapped[UserORM | None] = relationship("UserORM", back_populates="tests")
    
    # Связь с датасетом (НОВОЕ)
    dataset: Mapped[GeneratedDataORM | None] = relationship(
        "GeneratedDataORM",
        foreign_keys=[dataset_id],
        back_populates="ab_tests_using_dataset"
    )
    
    # Старая связь для обратной совместимости
    generated_data: Mapped[list[GeneratedDataORM]] = relationship(
        "GeneratedDataORM",
        back_populates="ab_test",
        foreign_keys="GeneratedDataORM.ab_test_id",
        cascade="all, delete-orphan",
    )
    checkpoints: Mapped[list[CheckpointORM]] = relationship(
        "CheckpointORM",
        back_populates="ab_test",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list[TestSessionORM]] = relationship(
        "TestSessionORM",
        backref="ab_test_ref",
        foreign_keys="TestSessionORM.test_id",
        primaryjoin="ABTestORM.test_id == TestSessionORM.test_id",
        lazy="dynamic"
    )

    __table_args__ = ()


class GeneratedDataORM(Base):
    __tablename__ = "generated_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ab_test_id: Mapped[int | None] = mapped_column(ForeignKey("ab_tests.id", ondelete="SET NULL"), nullable=True, index=True)

    data_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)  # real|synthetic
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    schema_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    preview_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    ab_test: Mapped[ABTestORM | None] = relationship(
        "ABTestORM",
        back_populates="generated_data",
        foreign_keys=[ab_test_id]
    )

    # Новая связь: какие тесты используют этот датасет
    ab_tests_using_dataset: Mapped[list[ABTestORM]] = relationship(
        "ABTestORM",
        back_populates="dataset",
        foreign_keys="ABTestORM.dataset_id"
    )

    __table_args__ = ()


class TestSessionORM(Base):
    __tablename__ = "test_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    test_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    variant: Mapped[str] = mapped_column(String(32), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True, default={})

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = ()


class ABTestTimeSeriesORM(Base):
    __tablename__ = "ab_test_time_series"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    test_id: Mapped[str] = mapped_column(String(64), ForeignKey("ab_tests.test_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Срез данных (например, по количеству пользователей)
    users_processed: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Вариант теста (A, B, C...)
    variant: Mapped[str] = mapped_column(String(32), nullable=False)
    
    # Метрики на момент среза
    cumulative_metric: Mapped[float] = mapped_column(Float, nullable=False)  # Накопленная метрика
    mean_metric: Mapped[float] = mapped_column(Float, nullable=False)  # Среднее значение
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)  # Размер выборки
    
    # Статистика
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)  # p-value на момент среза
    confidence_interval_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_interval_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = ()


class CheckpointORM(Base):
    __tablename__ = "checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ab_test_id: Mapped[int | None] = mapped_column(ForeignKey("ab_tests.id", ondelete="SET NULL"), nullable=True, index=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    ab_test: Mapped[ABTestORM | None] = relationship("ABTestORM", back_populates="checkpoints")

    __table_args__ = (
        UniqueConstraint("file_path", name="uq_checkpoints_file_path"),
    )
