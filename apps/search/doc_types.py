"""v2.0 chunk 모델의 doc_type / family / chunk_id 단일 출처 (WO-0 산출물).

이 모듈은 v1.x→v2.0 마이그레이션에서 WO-A(소스→chunk 변환)와 WO-C(런타임 파이프라인
chunk화 · evidence id 교체)가 **공통으로 의존하는 단일 계약점**이다. 여기 정의된 상수/유틸을
양쪽에서 import해 쓰며, doc_type/PREFIX/family 매핑을 다른 곳에서 손수 재정의하지 않는다.

본 모듈은 **순수 상수/유틸**이며 런타임 동작을 바꾸지 않는다(WO-0은 비파괴). 실제 적용
(named-vector 제거, RRF 집계 chunk화, family cap 적용, reasoner evidence id 교체 등)은 WO-C.

근거 문서:
- apps/docs/architecture/DATA_MODEL.md  §4(11 doc_type·4 family), §5(인덱스), §6(불변식)
- apps/docs/architecture/ADR/0002-chunk-level-point-model.md (Point ID==chunk_id, 멱등 적재)
- apps/docs/architecture/ADR/0004-chunk-id-evidence-contract.md (evidence 참조=chunk_id)
- apps/docs/전체 샘플 payload (도메인별).txt (실제 chunk_id 예: PUB_M1006328_0001_c0)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DocType",
    "DOC_TYPES",
    "Family",
    "DOC_TYPE_TO_PREFIX",
    "PREFIX_TO_DOC_TYPE",
    "DOC_TYPE_TO_FAMILY",
    "FAMILY_EVIDENCE_CAP",
    "ChunkIdParts",
    "build_doc_id",
    "build_chunk_id",
    "parse_chunk_id",
]


class DocType(StrEnum):
    """v2.0 11 doc_type 단일 출처 (DATA_MODEL §4)."""

    PUBLICATION = "publication"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    RESEARCH_PROJECT = "research_project"
    RESEARCHER_ASSESSOR = "researcher_assessor"
    EXPERT_ASSESSOR = "expert_assessor"
    RESEARCHER_TECH = "researcher_tech"
    EXPERT_TECH = "expert_tech"
    RESEARCHER_CORE = "researcher_core"
    RESEARCHER_MAJOR = "researcher_major"
    EXPERT_SPECIFIC = "expert_specific"
    PROFILE = "profile"


#: 11 doc_type 문자열 튜플(순서 고정). `set(DOC_TYPES)`로 커버리지 검증에 사용.
DOC_TYPES: tuple[str, ...] = tuple(dt.value for dt in DocType)


class Family(StrEnum):
    """4 family 단일 출처 (DATA_MODEL §4)."""

    IDENTITY = "identity"
    ACHIEVEMENT = "achievement"
    ASSESSMENT = "assessment"
    EXPERTISE = "expertise"


#: doc_type → doc_id/chunk_id PREFIX. 단일 출처(역매핑은 아래에서 파생).
#: 샘플 payload의 실제 chunk_id PREFIX와 1:1 일치.
DOC_TYPE_TO_PREFIX: dict[str, str] = {
    DocType.PUBLICATION: "PUB",
    DocType.INTELLECTUAL_PROPERTY: "IP",
    DocType.RESEARCH_PROJECT: "PJT",
    DocType.RESEARCHER_ASSESSOR: "RAS",
    DocType.EXPERT_ASSESSOR: "EAS",
    DocType.RESEARCHER_TECH: "RTC",
    DocType.EXPERT_TECH: "ETC",
    DocType.RESEARCHER_CORE: "RCO",
    DocType.RESEARCHER_MAJOR: "RMJ",
    DocType.EXPERT_SPECIFIC: "ESP",
    DocType.PROFILE: "PRF",
}

#: PREFIX → doc_type (역매핑, 단일 출처에서 파생 — 손수 작성 금지).
PREFIX_TO_DOC_TYPE: dict[str, str] = {prefix: dt for dt, prefix in DOC_TYPE_TO_PREFIX.items()}


#: doc_type → family (DATA_MODEL §4).
#: assessment(researcher_assessor/expert_assessor)는 v1.x 4브랜치에 없던 신규 1급 신호.
DOC_TYPE_TO_FAMILY: dict[str, str] = {
    DocType.PROFILE: Family.IDENTITY,
    DocType.PUBLICATION: Family.ACHIEVEMENT,
    DocType.INTELLECTUAL_PROPERTY: Family.ACHIEVEMENT,
    DocType.RESEARCH_PROJECT: Family.ACHIEVEMENT,
    DocType.RESEARCHER_ASSESSOR: Family.ASSESSMENT,
    DocType.EXPERT_ASSESSOR: Family.ASSESSMENT,
    DocType.RESEARCHER_TECH: Family.EXPERTISE,
    DocType.EXPERT_TECH: Family.EXPERTISE,
    DocType.RESEARCHER_CORE: Family.EXPERTISE,
    DocType.RESEARCHER_MAJOR: Family.EXPERTISE,
    DocType.EXPERT_SPECIFIC: Family.EXPERTISE,
}

#: family별 evidence top-N cap 기본값(값만 정의, 실제 cap 적용은 WO-C).
#: HARD 제약: evidence 리랭커는 grounding 선별만 — 후보 순위에 영향 0.
FAMILY_EVIDENCE_CAP: dict[str, int] = {
    Family.ACHIEVEMENT: 10,
    Family.ASSESSMENT: 6,
    Family.EXPERTISE: 6,
    Family.IDENTITY: 1,
}


# ---------------------------------------------------------------------------
# 로드 시 invariant 검증 (단일 출처 무결성 보장)
# ---------------------------------------------------------------------------
assert len(DOC_TYPE_TO_PREFIX) == len(DOC_TYPES), "DOC_TYPE_TO_PREFIX는 11 doc_type 전부를 덮어야 한다"
assert len(PREFIX_TO_DOC_TYPE) == len(DOC_TYPE_TO_PREFIX), "PREFIX 충돌(전역 유일성 위반)"
assert set(DOC_TYPE_TO_FAMILY) == set(DOC_TYPES), "DOC_TYPE_TO_FAMILY는 11 doc_type을 빠짐없이/중복없이 덮어야 한다"
assert set(DOC_TYPE_TO_FAMILY.values()) <= set(Family), "family 값은 4종 enum 중 하나여야 한다"
assert set(FAMILY_EVIDENCE_CAP) == set(Family), "FAMILY_EVIDENCE_CAP는 4 family 전부를 덮어야 한다"


# ---------------------------------------------------------------------------
# chunk_id 코덱
#   형식: {PREFIX}_{researcher_id}_{seq:04d}_c{chunk_index}
#   계약: ① 결정성(같은 chunk → 같은 id) ② 전역 유일성 ③ 멱등 upsert(Point ID==chunk_id)
# ---------------------------------------------------------------------------

# 끝에서부터 `_c{int}`(chunk_index)와 직전 `_{digits}`(seq)를 떼고, 첫 토큰을 PREFIX로,
# 가운데를 researcher_id로 해석한다(researcher_id에 `_`가 들어와도 깨지지 않음).
_CHUNK_ID_RE = re.compile(r"^(?P<prefix>[A-Z]+)_(?P<rid>.+)_(?P<seq>\d+)_c(?P<cidx>\d+)$")


@dataclass(frozen=True)
class ChunkIdParts:
    prefix: str
    doc_type: str
    researcher_id: str
    seq: str
    chunk_index: int


def _normalize_seq(seq: int | str) -> str:
    """seq를 4자리 zero-pad 문자열로 정규화(샘플 `0001`과 일치)."""
    try:
        return f"{int(seq):04d}"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seq must be a non-negative integer: {seq!r}") from exc


def build_doc_id(doc_type: str, researcher_id: str, seq: int | str) -> str:
    """`{PREFIX}_{researcher_id}_{seq:04d}` (chunk 접미사 없음). 샘플 doc_id와 일치."""
    prefix = DOC_TYPE_TO_PREFIX.get(doc_type)
    if prefix is None:
        raise ValueError(f"unknown doc_type: {doc_type!r}")
    if not researcher_id:
        raise ValueError("researcher_id must be non-empty")
    return f"{prefix}_{researcher_id}_{_normalize_seq(seq)}"


def build_chunk_id(
    doc_type: str, researcher_id: str, seq: int | str, chunk_index: int
) -> str:
    """결정적 chunk_id 생성. 동일 입력 → 동일 출력(순수 함수)."""
    return f"{build_doc_id(doc_type, researcher_id, seq)}_c{int(chunk_index)}"


def parse_chunk_id(chunk_id: str) -> ChunkIdParts:
    """chunk_id를 분해. 알 수 없는 PREFIX/형식 위반은 ValueError."""
    match = _CHUNK_ID_RE.match(chunk_id or "")
    if match is None:
        raise ValueError(f"malformed chunk_id: {chunk_id!r}")
    prefix = match["prefix"]
    doc_type = PREFIX_TO_DOC_TYPE.get(prefix)
    if doc_type is None:
        raise ValueError(f"unknown chunk_id prefix: {prefix!r} (in {chunk_id!r})")
    return ChunkIdParts(
        prefix=prefix,
        doc_type=doc_type,
        researcher_id=match["rid"],
        seq=match["seq"],
        chunk_index=int(match["cidx"]),
    )
