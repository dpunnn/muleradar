-- MuleRadar Database Schema

CREATE TABLE IF NOT EXISTS accounts (
    account_id      VARCHAR(50) PRIMARY KEY,
    institution_id  VARCHAR(20) NOT NULL,          -- BANK_A, BANK_B
    account_type    VARCHAR(20) DEFAULT 'personal', -- personal, corporate, merchant
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS devices (
    device_id   VARCHAR(50) PRIMARY KEY,
    device_type VARCHAR(20) DEFAULT 'mobile'
);

CREATE TABLE IF NOT EXISTS account_devices (
    account_id  VARCHAR(50) REFERENCES accounts(account_id),
    device_id   VARCHAR(50) REFERENCES devices(device_id),
    PRIMARY KEY (account_id, device_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    tx_id           VARCHAR(100) PRIMARY KEY,
    from_account    VARCHAR(50) REFERENCES accounts(account_id),
    to_account      VARCHAR(50) REFERENCES accounts(account_id),
    amount          NUMERIC(18, 2) NOT NULL,
    currency        VARCHAR(30) DEFAULT 'IDR',
    channel         VARCHAR(20),                    -- mobile, atm, internet, teller, qris
    payment_format  VARCHAR(30),
    tx_timestamp    TIMESTAMP NOT NULL,
    device_id       VARCHAR(50),
    institution_id  VARCHAR(20),
    is_laundering   SMALLINT DEFAULT 0,             -- 0=licit, 1=illicit (label SynthAML)
    typology        VARCHAR(50)                     -- judol, scam, dormant, dll
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        VARCHAR(50) PRIMARY KEY,         -- ALT-xxxxxxxxxxxx dari consumer
    account_id      VARCHAR(50),
    tx_id           VARCHAR(100),
    cluster_id      VARCHAR(100),
    typology        VARCHAR(100),
    risk_score      NUMERIC(5, 4),
    rule_triggered  TEXT,
    severity        VARCHAR(10) DEFAULT 'MEDIUM',    -- HIGH, MEDIUM, LOW
    node_count      INTEGER,
    status          VARCHAR(20) DEFAULT 'NEW',       -- NEW, IN_REVIEW, CONFIRM, FP, CLOSED
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cases (
    case_id         SERIAL PRIMARY KEY,
    alert_id        VARCHAR(50) REFERENCES alerts(alert_id),
    assigned_to     VARCHAR(100),
    status          VARCHAR(20) DEFAULT 'NEW',
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- Indexes untuk query graph dan alert
CREATE INDEX IF NOT EXISTS idx_tx_from    ON transactions(from_account);
CREATE INDEX IF NOT EXISTS idx_tx_to      ON transactions(to_account);
CREATE INDEX IF NOT EXISTS idx_tx_time    ON transactions(tx_timestamp);
CREATE INDEX IF NOT EXISTS idx_tx_label   ON transactions(is_laundering);
CREATE INDEX IF NOT EXISTS idx_tx_typology ON transactions(typology);
CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status);
