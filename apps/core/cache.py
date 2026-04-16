from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Type, TypeVar, Generic

from pydantic import BaseModel

from apps.domain.models import PlannerOutput

logger = logging.getLogger(__name__)

T = TypeVar("T")

class BaseCache(Generic[T]):
    def __init__(self, cache_dir: Path, ttl_seconds: int = 86400) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _generate_key(self, *args: Any) -> str:
        payload = "|".join(str(a) for a in args)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"


class PlanCache(BaseCache[PlannerOutput]):
    """L1: Canonical Plan Cache"""
    def get(self, query: str, filters: dict[str, Any], version: str) -> PlannerOutput | None:
        key = self._generate_key(query.lower().strip(), json.dumps(filters, sort_keys=True), version)
        path = self._get_path(key)
        if not path.exists(): return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return PlannerOutput.model_validate(json.load(f))
        except Exception: return None

    def set(self, query: str, filters: dict[str, Any], version: str, output: PlannerOutput) -> None:
        key = self._generate_key(query.lower().strip(), json.dumps(filters, sort_keys=True), version)
        try:
            with open(self._get_path(key), "w", encoding="utf-8") as f:
                json.dump(output.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        except Exception as exc: logger.warning("L1 Cache save failed: %s", exc)


class BranchCompileCache:
    """L2: Branch Compile Cache"""
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, plan_json: str, lexicon_version: str) -> dict[str, Any] | None:
        key = hashlib.sha256(f"{plan_json}|{lexicon_version}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if not path.exists(): return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception: return None

    def set(self, plan_json: str, lexicon_version: str, compiled_queries: dict[str, Any]) -> None:
        key = hashlib.sha256(f"{plan_json}|{lexicon_version}".encode()).hexdigest()
        try:
            with open(self.cache_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(compiled_queries, f, ensure_ascii=False)
        except Exception: pass


class RetrievalResultCache:
    """L3: Retrieval Result Cache"""
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, compiled_queries_json: str, filter_json: str, snapshot_id: str) -> list[dict[str, Any]] | None:
        key = hashlib.sha256(f"{compiled_queries_json}|{filter_json}|{snapshot_id}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if not path.exists(): return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception: return None

    def set(self, compiled_queries_json: str, filter_json: str, snapshot_id: str, results: list[dict[str, Any]]) -> None:
        key = hashlib.sha256(f"{compiled_queries_json}|{filter_json}|{snapshot_id}".encode()).hexdigest()
        try:
            with open(self.cache_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
        except Exception: pass
