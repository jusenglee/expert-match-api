"""flat chunk payload의 doc_type / family / chunk_id 단일 출처.

실제 적재 데이터 기준(v2.1). 검색·evidence·집계가 doc_type/family/codec을 여기 한 곳에서만
참조하도록 한다(다른 곳에서 손수 재정의 금지).

실제 계약(메모리 flat-payload-contract / DATA_MODEL.md):
- doc_type 5종: paper / patent / project / assessor_activity / specialty
- chunk_id codec: <doc_type>_<숫자doc_id>_c<NNN>  (예: paper_100000045256_c000)
  · doc_id = <doc_type>_<숫자doc_id>            (예: paper_100000045256)
  · chunk_index 3자리 zero-pad
- family: achievement={paper,patent,project} / assessment={assessor_activity} /
          expertise={specialty} / identity=합성(전용 doc_type 없음, root 필드로 구성)

근거: apps/docs/architecture/DATA_MODEL.md, ADR/0002(chunk-level point model), ADR/0004(evidence=chunk_id)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DocType",
    "DOC_TYPES",
    "Family",
    "DOC_TYPE_TO_FAMILY",
    "FAMILY_EVIDENCE_CAP",
    "CHUNK_INDEX_PAD",
    "ChunkIdParts",
    "build_doc_id",
    "build_chunk_id",
    "parse_chunk_id",
    "doc_type_of_chunk_id",
]


class DocType(StrEnum):
    """실제 적재 데이터의 doc_type 5종 단일 출처."""

    PAPER = "paper"
    PATENT = "patent"
    PROJECT = "project"
    ASSESSOR_ACTIVITY = "assessor_activity"
    SPECIALTY = "specialty"


#: doc_type 문자열 튜플(순서 고정). `set(DOC_TYPES)`로 커버리지 검증에 사용.
DOC_TYPES: tuple[str, ...] = tuple(dt.value for dt in DocType)


class Family(StrEnum):
    """4 family 단일 출처.

    identity는 전용 doc_type이 없다(연구자 신원/누적 실적은 모든 chunk의 flat root 필드에
    비정규화 반복 저장됨). 합성 profile evidence가 identity family를 차지한다.
    """

    IDENTITY = "identity"
    ACHIEVEMENT = "achievement"
    ASSESSMENT = "assessment"
    EXPERTISE = "expertise"


#: doc_type → family. 5 doc_type 전부를 덮는다(identity는 doc_type 없음).
DOC_TYPE_TO_FAMILY: dict[str, str] = {
    DocType.PAPER: Family.ACHIEVEMENT,
    DocType.PATENT: Family.ACHIEVEMENT,
    DocType.PROJECT: Family.ACHIEVEMENT,
    DocType.ASSESSOR_ACTIVITY: Family.ASSESSMENT,
    DocType.SPECIALTY: Family.EXPERTISE,
}

#: family별 evidence top-N cap 기본값(grounding 선별 한정 — 후보 순위 영향 0).
#: identity는 합성 profile evidence 1건.
FAMILY_EVIDENCE_CAP: dict[str, int] = {
    Family.ACHIEVEMENT: 10,
    Family.ASSESSMENT: 6,
    Family.EXPERTISE: 6,
    Family.IDENTITY: 1,
}


# ---------------------------------------------------------------------------
# 로드 시 invariant 검증 (단일 출처 무결성 보장)
# ---------------------------------------------------------------------------
assert set(DOC_TYPE_TO_FAMILY) == set(DOC_TYPES), "DOC_TYPE_TO_FAMILY는 5 doc_type을 빠짐없이/중복없이 덮어야 한다"
assert set(DOC_TYPE_TO_FAMILY.values()) <= set(Family), "family 값은 4종 enum 중 하나여야 한다"
assert set(FAMILY_EVIDENCE_CAP) == set(Family), "FAMILY_EVIDENCE_CAP는 4 family 전부를 덮어야 한다"


# ---------------------------------------------------------------------------
# chunk_id 코덱
#   형식: <doc_type>_<doc_num>_c<NNN>    (예: paper_100000045256_c000)
#   doc_id: <doc_type>_<doc_num>          (예: paper_100000045256)
#   계약: ① 결정성(같은 chunk → 같은 id) ② 전역 유일성 ③ payload evidence id
#   참고: Point ID로 chunk_id를 쓰면 멱등 upsert가 쉬우나 런타임은 payload.chunk_id를 기준으로 한다.
# ---------------------------------------------------------------------------

#: chunk_index zero-pad 자릿수(샘플 c000과 일치).
CHUNK_INDEX_PAD = 3

#: prefix 토큰은 doc_type 그대로(소문자). doc_type에 `_`가 있어도(assessor_activity)
#: 알려진 doc_type 집합으로 앵커링해 모호성 없이 파싱한다.
#: doc_id 본문은 실데이터상 doc_type마다 다르다 — paper/patent/project는 숫자 실적 id
#: (예: paper_100000045256_c000), specialty/assessor_activity는 연구자ID 기반
#: (예: specialty_M1013800_c000). 따라서 본문은 숫자로 제한하지 않고 끝의 `_c<NNN>`로만 앵커링한다.
_DOC_TYPE_ALT = "|".join(re.escape(dt) for dt in DOC_TYPES)
_CHUNK_ID_RE = re.compile(
    rf"^(?P<doc_type>{_DOC_TYPE_ALT})_(?P<doc_num>.+)_c(?P<cidx>\d+)$"
)
_DOC_ID_RE = re.compile(rf"^(?P<doc_type>{_DOC_TYPE_ALT})_(?P<doc_num>.+)$")


@dataclass(frozen=True)
class ChunkIdParts:
    doc_type: str
    doc_num: str
    chunk_index: int

    @property
    def doc_id(self) -> str:
        return f"{self.doc_type}_{self.doc_num}"


def build_doc_id(doc_type: str, doc_num: int | str) -> str:
    """`<doc_type>_<doc_num>`. doc_num 본문은 숫자(실적) 또는 연구자ID형(specialty 등) 모두 허용."""
    if doc_type not in DOC_TYPE_TO_FAMILY:
        raise ValueError(f"unknown doc_type: {doc_type!r}")
    body = str(doc_num).strip()
    if not body:
        raise ValueError("doc id body must be non-empty")
    return f"{doc_type}_{body}"


def build_chunk_id(doc_type: str, doc_num: int | str, chunk_index: int) -> str:
    """결정적 chunk_id 생성. 동일 입력 → 동일 출력(순수 함수)."""
    return f"{build_doc_id(doc_type, doc_num)}_c{int(chunk_index):0{CHUNK_INDEX_PAD}d}"


def parse_chunk_id(chunk_id: str) -> ChunkIdParts:
    """chunk_id를 분해. 알 수 없는 doc_type/형식 위반은 ValueError."""
    match = _CHUNK_ID_RE.match(chunk_id or "")
    if match is None:
        raise ValueError(f"malformed chunk_id: {chunk_id!r}")
    return ChunkIdParts(
        doc_type=match["doc_type"],
        doc_num=match["doc_num"],
        chunk_index=int(match["cidx"]),
    )


def doc_type_of_chunk_id(chunk_id: str) -> str | None:
    """chunk_id(또는 doc_id)에서 doc_type만 빠르게 추출. 형식 위반이면 None."""
    text = chunk_id or ""
    match = _CHUNK_ID_RE.match(text) or _DOC_ID_RE.match(text)
    return match["doc_type"] if match else None
