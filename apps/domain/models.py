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
# flat chunk payload (실제 적재 데이터 = 1 chunk = 1 Point).
# payload.chunk_id가 evidence authoritative id이며 Point ID와 다를 수 있다.
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
    # v2.1 관련도 검색: 이 chunk가 충족하는 concept id 목록 + 어느 검색 view에서 잡혔나.
    concepts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    # doc_attrs(제목/키워드)엔 있으나 본문 확정엔 없는 concept(표시 전용 — 점수/확정 무관, 진단용).
    display_only_concepts: list[str] = Field(default_factory=list)

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
    # v2.1 관련도 검색: concept 커버리지 + capped evidence score 분해.
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    coverage_type: str = ""  # "joint" | "separate" | "partial" | ""
    evidence_by_concept: dict[str, list[ChunkHit]] = Field(default_factory=dict)  # {"joint","<concept>","optional"}
    score_breakdown: dict[str, float] = Field(default_factory=dict)

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
    """표시/grounding용 evidence 단위(doc_type 무관 통일 모델). 참조 id == payload chunk_id."""

    chunk_id: str
    doc_type: str
    title: str | None = None
    date: str | None = None
    snippet: str = ""
    doc_attrs: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    evidence_kind: str = "matched"  # "matched"=질의 매칭 / "profile"=researcher hydration(참고 실적)


class ConceptSpec(BaseModel):
    """질의별 동적 Concept Evidence Plan 단위(planner 산출, registry 보강).

    하드코딩 사전 대신 planner(LLM)가 질의마다 concept를 만들고, app-layer는 이를 검증 규칙으로만 쓴다.
    - query_terms: 검색(recall)용 — concept별 SPLADE 검색문 생성.
    - evidence_terms: 태깅 '확정'(strong) — 이 term이 chunk에 직접 등장해야 concept 확정.
    - weak_terms: 약신호(확정 불가) — 단독으로는 concept 근거가 되지 못함(초기엔 보관만).
    source/confidence는 디버깅·추적용(planner/registry/fallback/query_exact 구분).
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    label: str = ""
    role: Literal["required", "optional"] = "required"
    query_terms: list[str] = Field(default_factory=list)
    evidence_terms: list[str] = Field(default_factory=list)
    weak_terms: list[str] = Field(default_factory=list)
    source: Literal["planner", "registry", "fallback", "query_exact"] = "planner"
    confidence: float = 1.0

    @field_validator("query_terms", "evidence_terms", "weak_terms", mode="before")
    @classmethod
    def _normalize_term_lists(cls, value: Any) -> Any:
        return _normalize_string_list(value)


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
    # v2.1 관련도 검색 view + concept (planner 동적 산출; 실패 시 raw_query/빈값 fallback).
    dense_query: str = ""
    sparse_raw: str = ""
    focused_sparse_query: str = ""
    concept_queries: dict[str, str] = Field(default_factory=dict)  # {concept_id: sparse query text}
    required_concepts: list[str] = Field(default_factory=list)
    optional_concepts: list[str] = Field(default_factory=list)
    planner_concept_terms: dict[str, list[str]] = Field(default_factory=dict)  # {concept_id: [alias...]} 동적 alias(구버전 compat)
    # ⭐ 동적 Concept Evidence Plan(권장). 채워지면 resolve_concept_plan이 이걸 최우선 사용한다.
    concept_specs: list[ConceptSpec] = Field(default_factory=list)

    @field_validator(
        "include_orgs",
        "exclude_orgs",
        "task_terms",
        "core_keywords",
        "retrieval_core",
        "role_terms",
        "action_terms",
        "bundle_ids",
        "required_concepts",
        "optional_concepts",
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
    evidence_kind: str = "matched"  # "matched"=질의 매칭 / "profile"=researcher hydration(참고 실적)


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
    raw_score: float = 0.0
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    coverage_type: str = ""
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    top_chunks: list[dict[str, Any]] = Field(default_factory=list)
    # researcher_id hydration으로 보강한 '참고 실적'(질의 매칭 아님 — 표시/맥락용, 점수/랭킹 무영향).
    profile_evidence: list[ChunkEvidence] = Field(default_factory=list)

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
    match_badges: list[str] = Field(default_factory=list)
    match_summary: str = ""
    match_details: dict[str, Any] = Field(default_factory=dict)
    score_explanation: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    profile_evidence: list[EvidenceItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    rank_score: float = 0.0

    @computed_field(return_type=list[str])
    @property
    def reasons(self) -> list[str]:
        normalized_reason = " ".join(self.recommendation_reason.split())
        return [normalized_reason] if normalized_reason else []
