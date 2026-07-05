"""SQLite schema for the experiment registry.

Single-file database (sessions/registry.db).

Tables:
  architectures  — model architecture definitions (fingerprint = primary key)
  runs           — individual training runs (one per unique arch+training combo)
  experiments    — sweep/experiment metadata
  experiment_runs — M:N link: which runs belong to which experiment

Design principles:
  - config_json columns store full asdict() blobs → new fields don't need migrations
  - view run_summary exposes frequently-queried fields via json_extract
  - ddl_version enables future schema evolution
"""

SCHEMA_VERSION = 1

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS architectures (
    fingerprint  TEXT PRIMARY KEY,
    n_params     INTEGER NOT NULL,
    seq_len      INTEGER NOT NULL,
    shape        TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    architecture_fp  TEXT NOT NULL REFERENCES architectures(fingerprint),
    training_hash    TEXT NOT NULL,
    model_name       TEXT NOT NULL,

    final_train_loss REAL,
    final_val_loss   REAL,
    total_samples    INTEGER,
    duration_seconds REAL,
    status           TEXT NOT NULL DEFAULT 'pending',

    arch_config_json  TEXT NOT NULL,
    train_config_json TEXT NOT NULL,

    started_at   TEXT,
    finished_at  TEXT,
    error_message TEXT,

    UNIQUE(architecture_fp, training_hash)
);

CREATE TABLE IF NOT EXISTS experiments (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    strategy          TEXT NOT NULL,
    vary_param        TEXT NOT NULL,
    vary_values_json  TEXT NOT NULL,
    config_json       TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending',
    created_at        TEXT NOT NULL,
    finished_at       TEXT
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    experiment_id  TEXT NOT NULL REFERENCES experiments(id),
    run_id         TEXT NOT NULL REFERENCES runs(id),
    vary_value     TEXT NOT NULL,
    PRIMARY KEY (experiment_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_arch_fp      ON runs(architecture_fp);
CREATE INDEX IF NOT EXISTS idx_runs_status       ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_train_loss   ON runs(final_train_loss);
CREATE INDEX IF NOT EXISTS idx_exp_runs_exp      ON experiment_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_exp_runs_run      ON experiment_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_experiments_name   ON experiments(name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);

-- Convenience view: expose common fields from JSON blobs
CREATE VIEW IF NOT EXISTS run_summary AS
SELECT
    r.id, r.architecture_fp, r.training_hash, r.model_name,
    r.final_train_loss, r.final_val_loss, r.total_samples,
    r.duration_seconds, r.status,
    a.shape, a.seq_len, a.n_params,
    json_extract(r.arch_config_json, '$.activation')   AS activation,
    json_extract(r.arch_config_json, '$.normalization') AS normalization,
    json_extract(r.arch_config_json, '$.dropout')       AS dropout,
    json_extract(r.train_config_json, '$.scheduler')    AS scheduler,
    json_extract(r.train_config_json, '$.lr')           AS lr,
    json_extract(r.train_config_json, '$.optimizer')    AS optimizer,
    json_extract(r.train_config_json, '$.batch_size')   AS batch_size,
    json_extract(r.train_config_json, '$.noise_prob')   AS noise_prob,
    json_extract(r.train_config_json, '$.target_samples') AS target_samples
FROM runs r
JOIN architectures a ON a.fingerprint = r.architecture_fp;

-- View: experiment results with run details
CREATE VIEW IF NOT EXISTS experiment_results AS
SELECT
    e.name        AS experiment,
    e.strategy,
    e.vary_param,
    er.vary_value,
    r.id          AS run_id,
    r.model_name,
    r.final_train_loss,
    r.final_val_loss,
    r.total_samples,
    r.duration_seconds,
    r.status       AS run_status,
    e.status       AS experiment_status
FROM experiments e
JOIN experiment_runs er ON er.experiment_id = e.id
JOIN runs r ON r.id = er.run_id;
"""
