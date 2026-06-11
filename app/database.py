"""
database.py
本地 SQLite 資料庫，負責儲存 qa_feedback 紀錄。
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "qa_feedback.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """建立 qa_feedback 資料表（若不存在）"""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS qa_feedback (
                id                TEXT PRIMARY KEY,
                session_id        TEXT NOT NULL,
                user_id           TEXT,           -- ✨ 新增，NULL = 尚未登入
                user_question     TEXT NOT NULL,
                llm_answer        TEXT NOT NULL,
                retrieved_sources TEXT NOT NULL,   -- JSON 字串
                status            TEXT NOT NULL,   -- 'refused' | 'negative'
                timestamp         TEXT NOT NULL
            )
        """)
        conn.commit()


def insert_feedback(
    session_id: str,
    user_question: str,
    llm_answer: str,
    retrieved_sources: list,
    status: str,           # 'refused' | 'negative'
) -> str:
    """寫入一筆 qa_feedback，回傳新紀錄的 id"""
    record_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO qa_feedback
                (id, session_id, user_question, llm_answer, retrieved_sources, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                session_id,
                user_question,
                llm_answer,
                json.dumps(retrieved_sources, ensure_ascii=False),
                status,
                timestamp,
            ),
        )
        conn.commit()

    return record_id
