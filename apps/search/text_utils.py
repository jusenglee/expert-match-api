from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import UTC, date, datetime

TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


def normalize_org_name(value: str | None) -> str | None:
    if not value:
        return None
    collapsed = re.sub(r"[^0-9A-Za-z가-힣]", "", value).upper()
    return collapsed or None


def tokenize_korean_text(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def stable_unit_vector(text: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{text}::{counter}".encode("utf-8")).digest()
        for index in range(0, len(digest), 4):
            chunk = digest[index : index + 4]
            if len(chunk) < 4:
                continue
            raw = int.from_bytes(chunk, byteorder="big", signed=False)
            values.append((raw / 2**32) - 0.5)
            if len(values) == dimension:
                break
        counter += 1
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def as_rfc3339(value: str | int | None) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        origin = date(1899, 12, 30)
        return datetime.combine(origin.fromordinal(origin.toordinal() + value), datetime.min.time(), UTC).date().isoformat()
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return f"{text}-01"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if text.isdigit():
        serial = int(text)
        origin = date(1899, 12, 30)
        return (origin.fromordinal(origin.toordinal() + serial)).isoformat()
    return text


def sparse_term_counts(text: str) -> Counter[str]:
    return Counter(tokenize_korean_text(text))

