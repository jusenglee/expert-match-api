from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FeedbackStore:
    db_path: Path
    table_name: str

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    query TEXT NOT NULL,
                    selected_expert_ids TEXT NOT NULL,
                    rejected_expert_ids TEXT NOT NULL,
                    notes TEXT,
                    metadata TEXT NOT NULL
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
            return int(cursor.lastrowid)

