from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def _db_path() -> Path:
    env_val = os.environ.get("BRAIN_DB_PATH")
    if env_val:
        return Path(env_val)
    return Path(__file__).resolve().parent.parent / "app_data.db"


def get_connection() -> sqlite3.Connection:
    con = sqlite3.connect(_db_path(), check_same_thread=False)
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    return con


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name  TEXT    NOT NULL,
    user_role  TEXT    NOT NULL,
    user_email TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS mri_scans (
    scan_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(user_id),
    scan_filename    TEXT    NOT NULL,
    scan_sequence    TEXT    NOT NULL DEFAULT 'unknown',
    scan_upload_date TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name         TEXT    NOT NULL,
    model_backbone     TEXT    NOT NULL,
    model_iou_score    REAL    NOT NULL,
    model_trained_date TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS segmentation_results (
    result_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id           INTEGER NOT NULL REFERENCES mri_scans(scan_id),
    model_id          INTEGER NOT NULL REFERENCES model_versions(model_id),
    tumor_pixel_count INTEGER NOT NULL,
    tumor_area_cm2    REAL    NOT NULL,
    mask_opacity_alpha REAL   NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    report_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id                  INTEGER NOT NULL REFERENCES segmentation_results(result_id),
    report_filename            TEXT    NOT NULL,
    report_tumor_area_cm2      REAL    NOT NULL,
    report_pixel_count         INTEGER NOT NULL,
    report_diagnostic_impression TEXT  NOT NULL,
    report_generated_date      TEXT    NOT NULL
);
"""


def init_db() -> None:
    con = get_connection()
    try:
        con.executescript(_SCHEMA)
        con.commit()
    finally:
        con.close()
