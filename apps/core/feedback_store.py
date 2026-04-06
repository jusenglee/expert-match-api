"""
사용자의 추천 결과에 대한 피드백(채택/탈락 전문가)을 저장하는 모듈입니다.
SQLite를 사용하여 로컬 환경에서 가볍게 피드백 데이터를 관리합니다.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FeedbackStore:
    """추천 시스템의 피드백 데이터를 데이터베이스에 기록하는 클래스입니다."""
    db_path: Path
    table_name: str

    def initialize(self) -> None:
        """피드백 저장용 데이터베이스 파일과 테이블을 초기화합니다."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, -- 고유 식별자
                    created_at TEXT NOT NULL,             -- 피드백 일시
                    query TEXT NOT NULL,                  -- 사용자 질의어
                    selected_expert_ids TEXT NOT NULL,    -- 채택된 전문가 ID 목록 (JSON)
                    rejected_expert_ids TEXT NOT NULL,    -- 탈락된 전문가 ID 목록 (JSON)
                    notes TEXT,                           -- 추가 의견/메모
                    metadata TEXT NOT NULL                -- 기타 분석 메타데이터 (JSON)
                )
                """
            )
            conn.commit()

    def save_feedback(
        self,
        *,
        query: str,
        selected_expert_ids: list[str],
        rejected_expert_ids: list[str],
        notes: str | None,
        metadata: dict[str, Any],
    ) -> int:
        """
        사용자가 제출한 피드백을 데이터베이스에 저장합니다.
        ID 목록과 메타데이터는 JSON 형식으로 직렬화되어 저장됩니다.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO {self.table_name} (
                    created_at,
                    query,
                    selected_expert_ids,
                    rejected_expert_ids,
                    notes,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    query,
                    json.dumps(selected_expert_ids, ensure_ascii=False),
                    json.dumps(rejected_expert_ids, ensure_ascii=False),
                    notes,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)  # 저장된 레코드의 ID 반환

