# backend/database/migration_script.py

from sqlalchemy import text
from backend.microservices.database.session import engine, SessionLocal

def migrate_ab_tests_table():

    migrations = [
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
        ADD COLUMN IF NOT EXISTS traffic_split_seed INTEGER DEFAULT 42 NOT NULL,
        ADD COLUMN IF NOT EXISTS extra_config JSONB,
        ADD COLUMN IF NOT EXISTS analysis_mode VARCHAR(32) DEFAULT 'fixed_experiment' NOT NULL,
        ADD COLUMN IF NOT EXISTS guardrails_config JSONB,
        ADD COLUMN IF NOT EXISTS guardrails_status JSONB,
        ADD COLUMN IF NOT EXISTS analysis_validity VARCHAR(32) DEFAULT 'valid_for_inference' NOT NULL;
        """,

        """
        ALTER TABLE ab_tests
        ADD COLUMN IF NOT EXISTS simulation_status VARCHAR(32);
        """,

        """
        ALTER TABLE ab_tests
        ALTER COLUMN status SET DEFAULT 'prepared';
        """,

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

        """
        CREATE INDEX IF NOT EXISTS ix_ab_tests_dataset_id ON ab_tests(dataset_id);
        """,

        """
        CREATE INDEX IF NOT EXISTS ix_ab_tests_status_created_at ON ab_tests(status, created_at);
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_generated_data_data_type_created_at ON generated_data(data_type, created_at);
        """,
        """
        CREATE TABLE IF NOT EXISTS assignment_audit (
            id SERIAL PRIMARY KEY,
            test_id VARCHAR(64) NOT NULL REFERENCES ab_tests(test_id) ON DELETE CASCADE,
            session_id VARCHAR(64) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            variant VARCHAR(32) NOT NULL,
            splitter_type VARCHAR(32) NOT NULL,
            analysis_mode VARCHAR(32) NOT NULL,
            traffic_split_type VARCHAR(32) NOT NULL,
            hash_bucket INTEGER,
            hash_space_size INTEGER,
            seed INTEGER,
            assignment_metadata JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_assignment_audit_test_id_created_at ON assignment_audit(test_id, created_at);
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_assignment_audit_session_id ON assignment_audit(session_id);
        """,
        """
        CREATE TABLE IF NOT EXISTS user_assignments (
            id SERIAL PRIMARY KEY,
            test_id VARCHAR(64) NOT NULL REFERENCES ab_tests(test_id) ON DELETE CASCADE,
            user_id VARCHAR(64) NOT NULL,
            variant VARCHAR(32) NOT NULL,
            splitter_type VARCHAR(32) NOT NULL,
            hash_bucket INTEGER,
            hash_space_size INTEGER,
            seed INTEGER,
            assignment_metadata JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_user_assignments_test_user UNIQUE (test_id, user_id)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_user_assignments_test_id ON user_assignments(test_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_user_assignments_user_id ON user_assignments(user_id);
        """,
        """
        CREATE TABLE IF NOT EXISTS metric_events (
            id SERIAL PRIMARY KEY,
            event_id VARCHAR(128) NOT NULL UNIQUE,
            session_id VARCHAR(64) NOT NULL,
            test_id VARCHAR(64) NOT NULL REFERENCES ab_tests(test_id) ON DELETE CASCADE,
            metric_name VARCHAR(128) NOT NULL,
            value FLOAT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_metric_events_session_id ON metric_events(session_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_metric_events_test_id ON metric_events(test_id);
        """,
    ]

    with engine.connect() as conn:
        for i, migration in enumerate(migrations, 1):
            try:
                print(f"Выполняется миграция {i}/{len(migrations)}...")
                conn.execute(text(migration))
                conn.commit()
                print(f"Миграция {i} выполнена успешно")
            except Exception as e:
                print(f"Миграция {i} пропущена или выполнена ранее: {e}")
                conn.rollback()


if __name__ == "__main__":
    print(" Начинаю миграцию базы данных")
    migrate_ab_tests_table()
    print(" Миграция завершена!")
