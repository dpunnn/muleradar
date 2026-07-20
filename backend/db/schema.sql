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
    detection_layer VARCHAR(20),                     -- AML_CORE/TYPOLOGY_ID/STATISTICAL/GRAPH_MOTIF/ML_ENSEMBLE (fix 6.7, 20-Jul)
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

-- Timeline/Audit Trail Case Detail (15-Jul) — event asli (bukan fabrikasi):
-- diisi tiap kali case dibuat/assign/status berubah/note disimpan.
CREATE TABLE IF NOT EXISTS audit_log (
    log_id          SERIAL PRIMARY KEY,
    case_id         INTEGER REFERENCES cases(case_id),
    event_type      VARCHAR(30) NOT NULL,   -- ALERT_CREATED, CASE_CREATED, ASSIGNED, STATUS_CHANGED, NOTE_ADDED
    actor           VARCHAR(100) DEFAULT 'System',
    detail          TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_log(case_id, created_at);

-- Indexes untuk query graph dan alert
--
-- CATATAN INSIDEN OOM (4-Jul): retrain XGBoost crash di batch 138/235
-- (~489 menit jalan) dengan "out of memory for query result" pada query
-- window-function _SQL_TS_FILTERED (detection/features.py) — ROW_NUMBER()
-- OVER (PARTITION BY from_account/to_account ORDER BY tx_timestamp DESC).
-- Index dasar (idx_tx_from, idx_tx_to, dst di bawah) SUDAH ADA sebelumnya —
-- itu BUKAN akar masalahnya (diagnosis awal sempat keliru krn salah pakai
-- `\di transactions*`, yang cuma cocok index BERNAMA "transactions*", bukan
-- index MILIK tabel transactions — verifikasi index yang benar pakai
-- `SELECT indexname FROM pg_indexes WHERE tablename='transactions'`).
-- Akar masalah SEBENARNYA: index tunggal (from_account) tidak menyimpan
-- urutan tx_timestamp, jadi window function tetap harus SORT manual di
-- memori per-partition — kalau satu batch kebetulan berisi akun "hub"
-- (rekening kolektor bertransaksi sangat banyak — pola mule network yang
-- justru kita cari), partition-nya besar dan sort itu bisa OOM.
-- FIX: index COMPOSITE (from_account/to_account, tx_timestamp DESC) di
-- bawah — leading-column composite tetap melayani query equality/ANY()
-- SEKALIGUS memberi urutan yang window function butuhkan LANGSUNG dari
-- index scan, tanpa sort terpisah di memori sama sekali.
CREATE INDEX IF NOT EXISTS idx_tx_from_ts  ON transactions(from_account, tx_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tx_to_ts    ON transactions(to_account, tx_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tx_time     ON transactions(tx_timestamp);
CREATE INDEX IF NOT EXISTS idx_tx_label    ON transactions(is_laundering);
CREATE INDEX IF NOT EXISTS idx_tx_typology ON transactions(typology);
CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status);

-- ============================================================
-- PHASE 4.5 — OSINT INTELLIGENCE MODULE
-- Proaktif hunting rekening judol dari situs publik (Kominfo blocklist)
-- lalu seed ke Neo4j sebagai starting node untuk graph tracing.
-- ============================================================

-- Antrean URL yang menunggu di-crawl (priority queue, bukan cron).
-- Diisi harian oleh kominfo_sync.py, dikonsumsi 24/7 oleh crawler workers.
CREATE TABLE IF NOT EXISTS osint_queue (
    url         TEXT PRIMARY KEY,
    priority    INTEGER DEFAULT 100,             -- kecil = didahulukan (newest blocked first)
    status      VARCHAR(12) DEFAULT 'PENDING',   -- PENDING, IN_PROGRESS, DONE, FAILED, SKIP
    attempts    SMALLINT DEFAULT 0,              -- jumlah percobaan crawl (untuk retry cap)
    queued_at   TIMESTAMP DEFAULT NOW(),
    crawled_at  TIMESTAMP
);

-- Metadata per situs yang sudah di-crawl (audit + statistik coverage).
CREATE TABLE IF NOT EXISTS osint_sites (
    url             TEXT PRIMARY KEY,
    status          VARCHAR(10),                 -- DONE, FAILED, SKIP
    http_status     INTEGER,                     -- 200, 404, dll (NULL kalau timeout/SSL)
    rekening_count  INTEGER DEFAULT 0,           -- jumlah rekening ditemukan di situs ini
    screenshot_path TEXT,
    error_type      VARCHAR(20),                 -- PROXY_ERROR/TIMEOUT/SITE_ERROR/BOT_BLOCKED/NULL
    crawled_at      TIMESTAMP DEFAULT NOW()
);

-- Rekening bank/e-wallet yang diekstrak dari situs judol.
-- shared_count = di berapa situs berbeda rekening ini muncul (indikator bandar).
CREATE TABLE IF NOT EXISTS osint_accounts (
    rekening        VARCHAR(50) PRIMARY KEY,     -- nomor rekening/HP ternormalisasi (digit saja)
    bank            VARCHAR(30),                 -- BCA, BNI, MANDIRI, BRI, BSI, GOPAY, OVO, DANA
    account_type    VARCHAR(10) DEFAULT 'bank',  -- bank, ewallet
    sumber_url      TEXT[],                      -- daftar URL tempat rekening ditemukan
    shared_count    INTEGER DEFAULT 1,           -- = cardinality(sumber_url)
    screenshot_path TEXT,                        -- bukti pertama kali ditemukan
    confidence      NUMERIC(4, 3) DEFAULT 1.0,   -- 1.0 regex bersih, <1 hasil OCR/deobfuscation
    seeded_to_graph BOOLEAN DEFAULT FALSE,       -- sudah dibuat node :OsintAccount di Neo4j?
    first_seen      TIMESTAMP DEFAULT NOW(),
    last_seen       TIMESTAMP DEFAULT NOW()
);

-- Jaringan bandar: cluster rekening yang berbagi situs yang sama.
CREATE TABLE IF NOT EXISTS osint_networks (
    network_id      SERIAL PRIMARY KEY,
    rekening_list   TEXT[],                      -- rekening anggota jaringan
    site_list       TEXT[],                      -- situs yang menghubungkan mereka
    risk_level      VARCHAR(10),                 -- HIGH (3+ situs), MED (2), LOW (1)
    detected_at     TIMESTAMP DEFAULT NOW()
);

-- Indexes OSINT
CREATE INDEX IF NOT EXISTS idx_osint_queue_status ON osint_queue(status, priority);
CREATE INDEX IF NOT EXISTS idx_osint_acc_seeded   ON osint_accounts(seeded_to_graph);
CREATE INDEX IF NOT EXISTS idx_osint_acc_shared   ON osint_accounts(shared_count);
CREATE INDEX IF NOT EXISTS idx_osint_acc_bank     ON osint_accounts(bank);
CREATE INDEX IF NOT EXISTS idx_osint_net_risk     ON osint_networks(risk_level);
CREATE INDEX IF NOT EXISTS idx_osint_acc_last_seen ON osint_accounts(last_seen);
-- Diagnosa ops: "berapa banyak PROXY_ERROR minggu ini?" → tunnel putus,
-- bukan situsnya yang hilang. Lihat crawler.py _classify_error (4.5.10).
CREATE INDEX IF NOT EXISTS idx_osint_sites_error   ON osint_sites(error_type) WHERE error_type IS NOT NULL;

-- ============================================================
-- PHASE 4.5.9 — OSINT TERPUSAT vs DETECTION ON-PREMISE
-- Dua tabel ini TIDAK hidup di database yang sama secara produksi:
--   osint_api_keys       → database TERPUSAT (infra MuleRadar)
--   osint_watchlist_sync → database LOKAL tiap bank (on-premise)
-- Didefinisikan di file schema.sql yang sama untuk kemudahan deployment
-- single-tenant (satu DB untuk semuanya) — lihat PIPELINE.txt 4.5.9.
-- ============================================================

-- TERPUSAT: otentikasi bank klien untuk endpoint GET /osint/watchlist.
-- Hanya hash yang disimpan — plaintext key ditampilkan sekali saat
-- diterbitkan (osint/api_keys.py::issue_key), sama seperti pola personal
-- access token GitHub/Stripe.
CREATE TABLE IF NOT EXISTS osint_api_keys (
    bank_id       VARCHAR(50) NOT NULL,
    key_hash      VARCHAR(64) PRIMARY KEY,        -- SHA-256 hex digest
    is_active     BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMP DEFAULT NOW(),
    last_used_at  TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_osint_key_bank ON osint_api_keys(bank_id);

-- ON-PREMISE: cursor sinkronisasi watchlist_consumer.py di sisi bank —
-- menyimpan last_seen terbaru yang sudah diproses supaya polling
-- berikutnya hanya minta data BARU (incremental, bukan re-fetch semua).
CREATE TABLE IF NOT EXISTS osint_watchlist_sync (
    consumer_id     VARCHAR(50) PRIMARY KEY DEFAULT 'default',
    last_synced_at  TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT NOW()
);
