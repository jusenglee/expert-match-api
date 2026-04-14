"""
범용 유틸리티 함수 모듈입니다.

judge.py / service.py 양쪽에 중복 존재하던 _merge_unique_strings 를
단일 권위 있는 구현으로 통합합니다.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def merge_unique_strings(*groups: list[str]) -> list[str]:
    """여러 문자열 리스트를 순서를 유지하며 중복 없이 병합합니다.

    각 그룹을 순서대로 순회하며, 공백 제거 후 중복되지 않은 항목만 결과에 추가합니다.
    빈 문자열(공백 포함)은 제외됩니다.

    Args:
        *groups: 병합할 문자열 리스트들 (가변 인수).

    Returns:
        중복이 제거된 문자열 리스트.

    Examples:
        >>> merge_unique_strings(["a", "b"], ["b", "c"])
        ['a', 'b', 'c']
        >>> merge_unique_strings(["  a  ", "b"], ["a"])
        ['a', 'b']
    """
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged


def build_deterministic_seed(*parts: Any) -> int:
    """Build a stable 32-bit seed from structured inputs."""
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
