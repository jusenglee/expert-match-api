from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from apps.domain.chunk_view import clean_value, normalize_doc_date

# 실제 적재 데이터의 doc_type 5종(메모리 flat-payload-contract). evidence/카드 표시용 라벨 포함.
DocTypeLiteral = Literal["paper", "patent", "project", "assessor_activity", "specialty"]

#: 연구자 공통 집계 count 필드(모든 chunk의 flat root에 비정규화 반복).
COUNT_FIELDS: tuple[str, ...] = (
    "publication_count",
    "scie_publication_count",
    "intellectual_property_count",
    "research_project_count",
    "researcher_assessor_activity_count",
)


def _is_blank_string(value: Any) -> bool:
    return isinstance(value, str) and not value.strip()


def _normalize_string_list(value: Any) -> Any:
    if value is None or _is_blank_string(value):
        return []
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            if item is None or _is_blank_string(item):
                continue
            normalized_value = item.strip() if isinstance(item, str) else str(item).strip()
            if normalized_value:
                normalized.append(normalized_value)
        return normalized
    return value


def _normalize_int(value: Any) -> Any:
    """카운트 정규화: None/공백/'NONE' → 0, 숫자문자열 → int."""
    if value is None or _is_blank_string(value):
        return 0
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            try:
                return int(float(stripped))
            except ValueError:
                return 0
    if isinstance(value, float):
        return int(value)
    return value


# ---------------------------------------------------------------------------
# flat chunk payload (실제 적재 데이터 = 1 chunk = 1 Point). Point ID == chunk_id.
# 연구자 공통 메타는 root에 비정규화, doc_type별 상세만 doc_attrs(passthrough).
# ---------------------------------------------------------------------------


class ChunkPayload(BaseModel):
    """Qdrant point 1개의 flat payload (메모리 flat-payload-contract).

    실데이터는 root에 연구자 공통 메타가 평탄화되어 있고, doc_type별 상세는 doc_attrs에 들어간다.
    알 수 없는 추가 키는 무시(extra=ignore)하여 스키마 진화에 견딘다.
    """

    model_config = ConfigDict(extra="ignore")

    # 1층 · 공통 식별
    researcher_id: str
    researcher_name: str = ""
    doc_type: str
    doc_id: str = ""
    chunk_id: str
    chunk_text: str = ""
    doc_date: str | None = None
    # 연구자 공통 메타(flat root, 비정규화)
    affiliated_organization: str | None = None
    highest_degree: str | None = None
    publication_count: int = 0
    scie_publication_count: int = 0
    intellectual_property_count: int = 0
    research_project_count: int = 0
    researcher_assessor_activity_count: int = 0
    # doc_type별 상세(passthrough — assessor_activity/specialty 키는 미상)
    doc_attrs: dict[str, Any] = Field(default_factory=dict)

    @field_validator(*COUNT_FIELDS, mode="before")
    @classmethod
    def _normalize_counts(cls, value: Any) -> Any:
        return _normalize_int(value)

    @field_validator("doc_date", "affiliated_organization", "highest_degree", mode="before")
    @classmethod
    def _normalize_optional_str(cls, value: Any) -> Any:
        return clean_value(value)

    def counts(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in COUNT_FIELDS}


class ChunkHit(BaseModel):
    """검색에서 회수된 chunk 1건(점수 + flat payload). rank=그룹 내 융합점수 순위(1-base)."""

    score: float = 0.0
    payload: ChunkPayload
    rank: int | None = None

    @property
    def chunk_id(self) -> str:
        return self.payload.chunk_id

    @property
    def doc_type(self) -> str:
        return self.payload.doc_type

    @property
    def researcher_id(self) -> str:
        return self.payload.researcher_id


class ResearcherCandidate(BaseModel):
    """검색 시점에 researcher_id로 집계된 연구자 후보(구 GroupedSearchHit/ExpertPayload 대체).

    정체성/누적 실적은 chunk root에서 가져오고(모든 chunk 동일), 매칭된 chunk를 doc_type별로 보유한다.
    """

    researcher_id: str
    researcher_name: str = ""
    affiliated_organization: str | None = None
    highest_degree: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    group_score: float = 0.0
    rank_score: float = 0.0
    chunks: list[ChunkHit] = Field(default_factory=list)

    @property
    def doc_types_present(self) -> list[str]:
        seen: list[str] = []
        for hit in self.chunks:
            if hit.doc_type not in seen:
                seen.append(hit.doc_type)
        return seen

    def chunks_of(self, doc_type: str) -> list[ChunkHit]:
        return [hit for hit in self.chunks if hit.doc_type == doc_type]


class ChunkEvidence(BaseModel):
    """표시/grounding용 evidence 단위(doc_type 무관 통일 모델). 참조 id == chunk_id."""

    chunk_id: str
    doc_type: str
    title: str | None = None
    date: str | None = None
    snippet: str = ""
    doc_attrs: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class PlannerOutput(BaseModel):
    intent_summary: str
    hard_filters: dict[str, Any] = Field(default_factory=dict)
    include_orgs: list[str] = Field(default_factory=list)
    exclude_orgs: list[str] = Field(default_factory=list)
    task_terms: list[str] = Field(default_factory=list)
    core_keywords: list[str] = Field(default_factory=list)
    retrieval_core: list[str] = Field(default_factory=list)
    role_terms: list[str] = Field(default_factory=list)
    action_terms: list[str] = Field(default_factory=list)
    bundle_ids: list[str] = Field(default_factory=list)
    intent_flags: dict[str, Any] = Field(default_factory=dict)
    semantic_query: str = ""
    top_k: int = 15

    @field_validator(
        "include_orgs",
        "exclude_orgs",
        "task_terms",
        "core_keywords",
        "retrieval_core",
        "role_terms",
        "action_terms",
        "bundle_ids",
        mode="before",
    )
    @classmethod
    def _normalize_string_lists(cls, value: Any) -> Any:
        return _normalize_string_list(value)


class EvidenceItem(BaseModel):
    """LLM 추천 결정에 포함되는 근거 1건. type=5 doc_type 또는 합성 'profile'."""

    type: Literal["paper", "patent", "project", "assessor_activity", "specialty", "profile"]
    title: str
    date: str | None = None
    detail: str | None = None
    snippet: str | None = None
    chunk_id: str | None = None


class CandidateCard(BaseModel):
    """후보 카드(cards→reasoner 인터페이스). evidence는 doc_type별 ChunkEvidence 묶음."""

    expert_id: str
    name: str
    organization: str | None = None
    degree: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    evidence_by_type: dict[str, list[ChunkEvidence]] = Field(default_factory=dict)
    matched_filter_summary: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    shortlist_score: float = 0.0
    rank_score: float = 0.0

    @property
    def doc_types_present(self) -> list[str]:
        return [dt for dt, items in self.evidence_by_type.items() if items]

    def evidence_of(self, doc_type: str) -> list[ChunkEvidence]:
        return self.evidence_by_type.get(doc_type, [])

    def all_evidence(self) -> list[ChunkEvidence]:
        items: list[ChunkEvidence] = []
        for bucket in self.evidence_by_type.values():
            items.extend(bucket)
        return items


class RecommendationDecision(BaseModel):
    rank: int
    expert_id: str
    name: str
    organization: str | None = None
    fit: Literal["높음", "중간", "보통"]
    recommendation_reason: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    rank_score: float = 0.0

    @computed_field(return_type=list[str])
    @property
    def reasons(self) -> list[str]:
        normalized_reason = " ".join(self.recommendation_reason.split())
        return [normalized_reason] if normalized_reason else []
