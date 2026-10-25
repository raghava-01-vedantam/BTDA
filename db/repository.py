from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from db.database import get_connection


def get_or_create_user(name: str, role: str) -> int:
    con = get_connection()
    try:
        row = con.execute(
            "SELECT user_id FROM users WHERE user_name = ? AND user_role = ?",
            (name, role),
        ).fetchone()
        if row:
            return int(row["user_id"])
        cur = con.execute(
            "INSERT INTO users (user_name, user_role) VALUES (?, ?)",
            (name, role),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def list_users() -> list[dict]:
    con = get_connection()
    try:
        rows = con.execute("SELECT * FROM users ORDER BY user_id").fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def insert_scan(user_id: int, filename: str, sequence: str = "unknown") -> int:
    con = get_connection()
    try:
        cur = con.execute(
            "INSERT INTO mri_scans (user_id, scan_filename, scan_sequence, scan_upload_date)"
            " VALUES (?, ?, ?, ?)",
            (user_id, filename, sequence, datetime.now().isoformat()),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def get_or_create_model_version(
    name: str,
    backbone: str,
    iou_score: float,
    trained_date: str,
) -> int:
    con = get_connection()
    try:
        row = con.execute(
            "SELECT model_id FROM model_versions WHERE model_name = ?",
            (name,),
        ).fetchone()
        if row:
            return int(row["model_id"])
        cur = con.execute(
            "INSERT INTO model_versions"
            " (model_name, model_backbone, model_iou_score, model_trained_date)"
            " VALUES (?, ?, ?, ?)",
            (name, backbone, iou_score, trained_date),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def list_model_versions() -> list[dict]:
    con = get_connection()
    try:
        rows = con.execute("SELECT * FROM model_versions ORDER BY model_id DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def insert_segmentation_result(
    scan_id: int,
    model_id: int,
    pixel_count: int,
    area_cm2: float,
    opacity: float,
) -> int:
    con = get_connection()
    try:
        cur = con.execute(
            "INSERT INTO segmentation_results"
            " (scan_id, model_id, tumor_pixel_count, tumor_area_cm2, mask_opacity_alpha)"
            " VALUES (?, ?, ?, ?, ?)",
            (scan_id, model_id, pixel_count, area_cm2, opacity),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def insert_report(
    result_id: int,
    filename: str,
    area_cm2: float,
    pixel_count: int,
    impression: str,
) -> int:
    con = get_connection()
    try:
        cur = con.execute(
            "INSERT INTO reports"
            " (result_id, report_filename, report_tumor_area_cm2,"
            "  report_pixel_count, report_diagnostic_impression, report_generated_date)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                result_id,
                filename,
                area_cm2,
                pixel_count,
                impression,
                datetime.now().isoformat(),
            ),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def has_report_for_result(result_id: int) -> bool:
    con = get_connection()
    try:
        row = con.execute(
            "SELECT 1 FROM reports WHERE result_id = ? LIMIT 1",
            (result_id,),
        ).fetchone()
        return row is not None
    finally:
        con.close()


def list_history(user_id: Optional[int] = None) -> list[dict]:
    con = get_connection()
    try:
        where = "WHERE u.user_id = ?" if user_id else ""
        params = (user_id,) if user_id else ()
        rows = con.execute(
            f"""
            SELECT
                sr.result_id,
                ms.scan_upload_date  AS date,
                u.user_name          AS user,
                ms.scan_filename     AS filename,
                sr.tumor_area_cm2    AS area_cm2,
                sr.tumor_pixel_count AS pixels,
                CASE WHEN r.report_id IS NOT NULL THEN 'Yes' ELSE 'No' END AS report
            FROM segmentation_results sr
            JOIN mri_scans ms ON ms.scan_id = sr.scan_id
            JOIN users u       ON u.user_id  = ms.user_id
            LEFT JOIN reports r ON r.result_id = sr.result_id
            {where}
            ORDER BY ms.scan_upload_date DESC
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def get_eval_metrics() -> Optional[dict]:
    import json
    from pathlib import Path

    promoted_path = (
        Path(__file__).resolve().parent.parent
        / "model" / "checkpoints" / "promoted.json"
    )
    if not promoted_path.exists():
        return None
    try:
        with open(promoted_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
