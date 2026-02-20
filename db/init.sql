-- SpendLens Database Schema

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert default user for single-user MVP
INSERT INTO users (id, username) VALUES (1, 'default') ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id),
    date TIMESTAMPTZ NOT NULL,
    description TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    merchant VARCHAR(255) NOT NULL,
    category VARCHAR(50),
    confidence DOUBLE PRECISION,
    anomaly_score DOUBLE PRECISION,
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_transactions_is_anomaly ON transactions(is_anomaly);

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    accuracy DOUBLE PRECISION,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry(model_name, is_active);

CREATE TABLE IF NOT EXISTS drift_log (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    psi_score DOUBLE PRECISION NOT NULL,
    is_drifted BOOLEAN NOT NULL DEFAULT FALSE,
    details JSONB,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_log_checked_at ON drift_log(checked_at);
