import json
import sqlite3
from pathlib import Path


def init_history_db(db_path):
    if isinstance(db_path, str):
        db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                audio_name TEXT,
                video_name TEXT,
                original_av_prediction TEXT,
                new_lora_prediction TEXT,
                fused_prediction TEXT,
                selected_prediction TEXT,
                english_translation TEXT,
                processing_time REAL,
                metrics TEXT,
                performance TEXT,
                selected_model TEXT,
                device TEXT,
                audio_duration REAL,
                extracted_frames INTEGER
            )
            """
        )


def load_history(db_path):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT run_id, timestamp, audio_name, video_name, original_av_prediction, new_lora_prediction, fused_prediction, selected_prediction, english_translation, processing_time, metrics, performance, selected_model, device, audio_duration, extracted_frames FROM runs ORDER BY timestamp DESC"
        ).fetchall()
    items = []
    for row in rows:
        items.append({
            "run_id": row[0],
            "timestamp": row[1],
            "audio_name": row[2],
            "video_name": row[3],
            "original_av_prediction": row[4],
            "new_lora_prediction": row[5],
            "fused_prediction": row[6],
            "selected_prediction": row[7],
            "english_translation": row[8],
            "processing_time": row[9],
            "metrics": json.loads(row[10] or "{}"),
            "performance": json.loads(row[11] or "{}"),
            "selected_model": row[12],
            "device": row[13],
            "audio_duration": row[14],
            "extracted_frames": row[15],
        })
    return items


def save_history_record(db_path, record):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, timestamp, audio_name, video_name, original_av_prediction,
                new_lora_prediction, fused_prediction, selected_prediction,
                english_translation, processing_time, metrics, performance,
                selected_model, device, audio_duration, extracted_frames
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("run_id"),
                record.get("timestamp"),
                record.get("audio_name"),
                record.get("video_name"),
                record.get("original_av_prediction"),
                record.get("new_lora_prediction"),
                record.get("fused_prediction"),
                record.get("selected_prediction"),
                record.get("english_translation"),
                float(record.get("processing_time") or 0.0),
                json.dumps(record.get("metrics") or {}),
                json.dumps(record.get("performance") or {}),
                record.get("selected_model") or "Whisper AV Studio",
                record.get("device") or "cpu",
                float(record.get("audio_duration") or 0.0),
                int(record.get("extracted_frames") or 0),
            ),
        )


def delete_history_record(db_path, run_id):
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


def clear_history(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM runs")
