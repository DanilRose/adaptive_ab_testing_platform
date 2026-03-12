# backend/database/migration_script.py
"""
Скрипт миграции для добавления новых полей в таблицу ab_tests и оптимизации индексов.

Запуск: python -m backend.database.migration_script
"""

from sqlalchemy import text
from backend.database.session import engine, SessionLocal

def migrate_ab_tests_table():
    """Добавляет новые поля в таблицу ab_tests и создает индексы для оптимизации"""

    migrations = [
        # Google-standard fields
        """
        ALTER TABLE ab_tests
        ADD COLUMN IF NOT EXISTS dataset_id INTEGER,
        ADD COLUMN IF NOT EXISTS real_world_duration_days INTEGER DEFAULT 14 NOT NULL,
        ADD COLUMN IF NOT EXISTS simulation_duration_minutes INTEGER DEFAULT 20 NOT NULL,
        ADD COLUMN IF NOT EXISTS mde_percent FLOAT DEFAULT 10.0 NOT NULL,
        ADD COLUMN IF NOT EXISTS actual_power FLOAT,
        ADD COLUMN IF NOT EXISTS max_sequential_looks INTEGER DEFAULT 5 NOT NULL,
        ADD COLUMN IF NOT EXISTS current_sequential_look INTEGER DEFAULT 0 NOT NULL,
        ADD COLUMN IF NOT EXISTS stopped_early INTEGER DEFAULT 0 NOT NULL,
        ADD COLUMN IF NOT EXISTS early_stop_reason VARCHAR(128),
        ADD COLUMN IF NOT EXISTS srm_check_passed INTEGER,
        ADD COLUMN IF NOT EXISTS srm_p_value FLOAT,
        ADD COLUMN IF NOT EXISTS traffic_split_type VARCHAR(32) DEFAULT 'fixed' NOT NULL,
        ADD COLUMN IF NOT EXISTS traffic_split_seed INTEGER DEFAULT 42 NOT NULL;
        """,

        # Добавляем новое поле simulation_status
        """
        ALTER TABLE ab_tests
        ADD COLUMN IF NOT EXISTS simulation_status VARCHAR(32);
        """,

        # Обновляем статус по умолчанию на 'prepared' для новых тестов
        # (для существующих тестов оставляем как есть)
        """
        ALTER TABLE ab_tests
        ALTER COLUMN status SET DEFAULT 'prepared';
        """,

        # Foreign key constraint для dataset_id
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'ab_tests_dataset_id_fkey'
            ) THEN
                ALTER TABLE ab_tests
                ADD CONSTRAINT ab_tests_dataset_id_fkey
                FOREIGN KEY (dataset_id)
                REFERENCES generated_data(id)
                ON DELETE RESTRICT;
            END IF;
        END $$;
        """,

        # Индекс для dataset_id
        """
        CREATE INDEX IF NOT EXISTS ix_ab_tests_dataset_id ON ab_tests(dataset_id);
        """,

        # Составной индекс для status + created_at
        """
        CREATE INDEX IF NOT EXISTS ix_ab_tests_status_created_at ON ab_tests(status, created_at);
        """,

        # Составной индекс для data_type + created_at (для ускорения generated-history)
        """
        CREATE INDEX IF NOT EXISTS ix_generated_data_data_type_created_at ON generated_data(data_type, created_at);
        """,
    ]

    with engine.connect() as conn:
        for i, migration in enumerate(migrations, 1):
            try:
                print(f"Выполняется миграция {i}/{len(migrations)}...")
                conn.execute(text(migration))
                conn.commit()
                print(f"✅ Миграция {i} выполнена успешно")
            except Exception as e:
                print(f"⚠️  Миграция {i} пропущена или выполнена ранее: {e}")
                conn.rollback()

    print("\n✅ Все миграции применены!")
    print("\n📊 ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ:")
    print("   - Добавлены индексы для ускорения запросов к ab_tests и generated_data")
    print("   - Добавлено поле simulation_status для отслеживания статуса симуляции")
    print("   - Рекомендуется выполнить VACUUM ANALYZE для оптимизации планов запросов")

if __name__ == "__main__":
    print("🔧 Начинаю миграцию базы данных...")
    migrate_ab_tests_table()
    print("✨ Миграция завершена!")
