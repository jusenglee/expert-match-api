"""최종 선별된 전문가 후보군을 LLM이 심사하여 '추천 사유(Reason)'를 생성하는 모듈.

flat chunk 모델(v2.1): 후보 카드는 doc_type별 ChunkEvidence를 갖고, evidence 참조 id는 chunk_id다.
LLM에는 선별 evidence(relevant_evidence)와 맥락 evidence(context_evidence)를 제공하고, 인용은
제공된 evidence_id(=chunk_id)만 사용하도록 강제한다.

HARD 제약: LLM은 후보 재정렬/탈락/새 ID 생성을 하지 않는다 — 사유 생성과 evidence 선택만.
"""
from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from apps.core.config import Settings
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text
from apps.core.llm_policies import build_consistency_invoke_kwargs
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.core.utils import build_deterministic_seed
from apps.domain.models import CandidateCard, PlannerOutput
from apps.recommendation.evidence_selector import RelevantEvidenceBundle
from apps.search.doc_types import DOC_TYPES

logger = logging.getLogger(__name__)

FIT_HIGH = "높음"
FIT_MEDIUM = "중간"
FIT_NORMAL = "보통"
FIT_VALUES = {FIT_HIGH, FIT_MEDIUM, FIT_NORMAL}

MAX_SELECTED_EVIDENCE_IDS = 4
# recommendation_reason 서버측 상한(시스템 프롬프트 지시와 동일). LLM이 초과해도 결정론적으로 컷한다.
REASON_MAX_CHARS = 320
REASON_TOOL_NAME = "submit_recommendation_batch"
REASON_GENERATION_MAX_TOKENS = 32,768
# evidence id == chunk_id 코덱: <doc_type>_<doc_id본문>_c<NNN>
# doc_id 본문은 숫자(paper_100000045256_c000) 또는 연구자ID형(specialty_M1013800_c000) 모두 가능 → 끝의 _c<NNN>로만 앵커링.
_DOC_TYPE_ALT = "|".join(re.escape(dt) for dt in DOC_TYPES)
VALID_EVIDENCE_ID_PATTERN = re.compile(rf"^(?:{_DOC_TYPE_ALT})_.+_c\d+$")
# recommendation_reason 본문에 누출된 chunk_id 토큰 제거용(비앵커). 괄호로 감싼 경우 괄호째,
# 그렇지 않으면 토큰만 제거(주변 공백은 이후 collapse가 정리 → 단어가 붙지 않도록).
_EVIDENCE_ID_CORE = rf"(?:{_DOC_TYPE_ALT})_[A-Za-z0-9]+_c\d+"
_INLINE_EVIDENCE_ID_PATTERN = re.compile(
    rf"[\(\[（]\s*{_EVIDENCE_ID_CORE}\s*[\)\]）]|{_EVIDENCE_ID_CORE}"
)

PRIMARY_PAYLOAD_PROFILE: dict[str, Any] = {
    "name": "primary",
    "trim_applied": False,
    "relevant_limit": 10,
    "context_limit": 3,
    "matched_filter_limit": 4,
    "matched_keywords_limit": 5,
    "snippet_char_limit": 1000,
    "detail_char_limit": 200,
    "profile_context_limit": 8,
}

RETRY_PAYLOAD_PROFILE: dict[str, Any] = {
    "name": "retry_compact",
    "trim_applied": True,
    "relevant_limit": 4,
    "context_limit": 2,
    "matched_filter_limit": 2,
    "matched_keywords_limit": 3,
    "snippet_char_limit": 280,
    "detail_char_limit": 120,
    "profile_context_limit": 4,
}


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return [normalized] if normalized else []
    if isinstance(value, list):
        normalized_values: list[str] = []
        for item in value:
            normalized = " ".join(str(item).split())
            if normalized and normalized not in normalized_values:
                normalized_values.append(normalized)
        return normalized_values
    return []


def _truncate_text(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    if not normalized:
        return None
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3].rstrip() + "..."


def _strip_inline_evidence_ids(value: str) -> tuple[str, bool]:
    """recommendation_reason 본문에 노출된 evidence_id(chunk_id) 토큰을 제거(결정론적 백스톱).

    프롬프트가 본문 노출을 금지해도 모델이 어기는 경우를 대비한다. 토큰을 감싼 빈 괄호와
    잔여 공백/구두점도 정리한다. 반환: (정리된 문자열, 제거 발생 여부).
    """
    stripped = _INLINE_EVIDENCE_ID_PATTERN.sub("", value)
    if stripped == value:
        return value, False
    stripped = re.sub(r"[\(\[（]\s*[\)\]）]", "", stripped)  # 잔여 빈 괄호
    stripped = re.sub(r"\s+([,\.\)\]）])", r"\1", stripped)  # 구두점 앞 공백
    stripped = " ".join(stripped.split())
    return stripped, True


class ReasonedCandidate(BaseModel):
    expert_id: str
    fit: str = FIT_NORMAL
    recommendation_reason: str = ""
    selected_evidence_ids: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    @field_validator("selected_evidence_ids", mode="before")
    @classmethod
    def _normalize_selected_evidence_ids(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)

    @field_validator("risks", mode="before")
    @classmethod
    def _normalize_risks(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)


class ReasonGenerationOutput(BaseModel):
    items: list[ReasonedCandidate] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)

    @field_validator("data_gaps", mode="before")
    @classmethod
    def _normalize_data_gaps(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)


class ReasonGenerator(Protocol):
    async def generate(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        candidates: list[CandidateCard],
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle] | None = None,
        retrieval_score_traces_by_expert_id: dict[str, dict[str, Any]] | None = None,
    ) -> ReasonGenerationOutput: ...


class PassThroughReasonGenerator:
    """LLM 사유 생성을 건너뛰고 기본 정보만 반환하는 폴백 생성기."""

    def __init__(self) -> None:
        self.last_trace: dict[str, Any] = {}

    async def generate(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        candidates: list[CandidateCard],
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle] | None = None,
        retrieval_score_traces_by_expert_id: dict[str, dict[str, Any]] | None = None,
    ) -> ReasonGenerationOutput:
        _ = (query, plan, relevant_evidence_by_expert_id, retrieval_score_traces_by_expert_id)
        output = ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id=candidate.expert_id,
                    fit=FIT_NORMAL,
                    recommendation_reason="",
                    selected_evidence_ids=[],
                    risks=list(candidate.risks),
                )
                for candidate in candidates
            ]
        )
        candidate_ids = [candidate.expert_id for candidate in candidates]
        self.last_trace = {
            "mode": "pass_through",
            "candidate_count": len(candidates),
            "output_count": len(output.items),
            "raw_output_count": len(output.items),
            "returned_ids": candidate_ids,
            "missing_candidate_ids": [],
            "empty_reason_candidate_ids": candidate_ids,
            "empty_selected_evidence_candidate_ids": candidate_ids,
            "retry_count": 0,
            "returned_ratio": 1.0 if candidates else 0.0,
            "prompt_budget_mode": "fallback",
            "trim_applied": True,
        }
        return output


class OpenAICompatReasonGenerator:
    """OpenAI 호환 API로 병렬(Batch) 추천 사유를 생성하는 Reasoner."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fallback = PassThroughReasonGenerator()
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _build_system_prompt(*, use_tools: bool) -> str:
        output_instruction = (
            f"반드시 `{REASON_TOOL_NAME}` 도구를 단 한 번만 호출해야 하며, 도구 호출 외의 다른 텍스트는 출력하지 마세요."
            if use_tools
            else '반드시 `{"items":[...],"data_gaps":[...]}` 형태의 JSON 객체 단 하나만 반환해야 하며, JSON 외의 다른 텍스트는 출력하지 마세요.'
        )
        prompt = f"""
        당신은 R&D 전문가 추천 시스템의 추천 사유 생성기입니다.

        당신의 임무는 이미 추천 랭킹이 확정된 후보자 배치를 받아, 각 후보자가 추천된 객관적인 사유를 생성하는 것입니다.
        입력된 후보자의 전문가 ID(expert_id)를 단 하나도 누락하거나 중복하거나 임의로 생성하지 마세요.

        [배경 및 데이터 활용]
        - 이 후보자들은 시스템에 의해 질의와 관련된 인물로 이미 판별된 상태입니다.
        - **[중요]** 절대 없는 사실을 지어내지 마세요(환각 금지). 반드시 제공된 증거(`relevant_evidence`, `context_evidence`)의 내용에 기반하여 작성해야 합니다.
        - 각 증거는 `type`(paper/patent/project/assessor_activity/specialty)과 `evidence_id`를 갖습니다. `evidence_id`는 내부 참조용이며 **오직 `selected_evidence_ids` 배열에만** 넣고, 추천 사유 본문(`recommendation_reason`)에는 절대 쓰지 마세요.
        - 각 후보는 `matched_concepts`(시스템이 이미 충족으로 판정한 질의 조건)와 `missing_concepts`를 가지며, 각 증거는 `satisfied_concepts`(그 증거가 충족하는 조건)를 가집니다. 조건 id는 영문(예: ai=인공지능, semiconductor=반도체)일 수 있습니다.
        - `profile_context`는 이 후보의 '질의에 직접 매칭되지는 않은' 다른 실적(참고 프로필)입니다. 후보를 과소평가하지 않도록 배경으로만 참고하세요. **`profile_context` 항목은 `selected_evidence_ids`에 넣지 말고, 질의를 직접 충족한 근거인 것처럼 단정하지 마세요.**
        - `counts`(누적 실적 수)는 보조적으로만 활용하세요.

        [출력 규칙]
        - `fit`은 다음 중 하나여야 합니다: {FIT_HIGH}, {FIT_MEDIUM}, {FIT_NORMAL}
        - `recommendation_reason`은 1~2문장의 간결하고 구체적인 한국어 문장으로 작성하며, 320자를 넘지 마세요.
        - **`recommendation_reason` 본문에는 `evidence_id`/chunk_id 같은 내부 식별자(예: `paper_100000045256_c000`)를 절대 쓰지 마세요. 식별자는 오직 `selected_evidence_ids`에만 넣습니다.**
        - **추천 사유는 반드시 제공된 증거의 실적명이나 연구 내용을 언급하여 작성해야 합니다.**
        - 어떤 실적이 질의의 어떤 요구를 뒷받침하는지 자연스러운 도메인 문장으로 녹여 서술하세요 (예: 'OOO 과제로 반도체 설계 역량을, △△△ 논문으로 인공지능 적용 경험을 보여줍니다'). 단, `matched_concepts`에 없는 요구를 충족했다고 주장하지 마세요(환각 금지).
        - 자연스러운 한국어 산문으로 쓰고, 'concept'·'조건'·'핵심 개념을 충족' 같은 시스템 메타 용어나 영문 concept id(예: ai, semiconductor)를 본문에 그대로 노출하지 마세요. 실적명·연구 내용·역할 중심으로 서술하세요.
        - 가능하면 서로 다른 유형의 실적을 2건 이상 엮고(특히 `assessor_activity` 심사 이력이 있으면 함께 언급), 단일 실적만 있으면 그 실적을 구체적으로 서술하세요(없는 실적은 만들지 말 것).
        - 질의에 직접 매칭된 근거가 적더라도 '실적이 부족하다'고 단정하지 마세요. 대신 '질의에 직접 매칭된 근거는 제한적'이라고 표현하고, `profile_context`에 관련 실적이 보이면 '프로필상 관련 실적 보유'를 함께 언급하세요.
        - `selected_evidence_ids`는 인용한 증거의 `evidence_id` 문자열을 그대로 포함시키세요 (최대 {MAX_SELECTED_EVIDENCE_IDS}개).
        - `selected_evidence_ids`에는 제공된 증거의 `evidence_id`(예: `paper_100000045256_c000`)만 넣으세요.
        - 적절한 직접 증거 ID가 없으면 `selected_evidence_ids`는 빈 배열(`[]`)로 두세요.
        - `risks`는 매우 짧고 사실적인 유의사항만 적거나 비워두세요.

        [좋은 예 / 나쁜 예]
        - 좋은 예: "서울시 화재취약지구 화재안전성 개선 연구로 건축소방 분야의 실증 연구를 수행했고, 소방청 자체평가위원회 위원으로 관련 정책 심의 경험을 보유하고 있습니다." (자연스러운 산문 + 다중 실적, 식별자/메타 용어 없음)
        - 나쁜 예: "'…화재안전성능 평가…' 논문(paper_100000435395_c000)으로 건축소방 핵심 개념을 충족했습니다." (← evidence_id 본문 노출 + '핵심 개념을 충족' 같은 기계적 표현, 단일 실적)

        {output_instruction}
        """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _build_reason_tools(allowed_evidence_ids: list[str]) -> list[dict[str, Any]]:
        selected_evidence_item_schema: dict[str, Any] = {"type": "string"}
        if allowed_evidence_ids:
            selected_evidence_item_schema["enum"] = allowed_evidence_ids
        return [
            {
                "type": "function",
                "function": {
                    "name": REASON_TOOL_NAME,
                    "description": "Submit structured recommendation results for the current candidate batch.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "expert_id": {"type": "string"},
                                        "fit": {"type": "string"},
                                        "recommendation_reason": {"type": "string"},
                                        "selected_evidence_ids": {
                                            "type": "array",
                                            "items": selected_evidence_item_schema,
                                            "maxItems": MAX_SELECTED_EVIDENCE_IDS,
                                        },
                                        "risks": {"type": "array", "items": {"type": "string"}},
                                    },
                                    "required": [
                                        "expert_id",
                                        "fit",
                                        "recommendation_reason",
                                        "selected_evidence_ids",
                                        "risks",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "data_gaps": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["items", "data_gaps"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    @classmethod
    def _serialize_relevant(
        cls,
        bundle: RelevantEvidenceBundle,
        *,
        limit: int,
        profile: dict[str, Any],
        chunk_concepts: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """선별된 evidence(doc_type별 top-N)를 평탄화해 직렬화. satisfied_concepts로 어떤 조건을 충족하는지 표기."""
        out: list[dict[str, Any]] = []
        for doc_type, items in bundle.by_doc_type.items():
            for item in items[:limit]:
                out.append(
                    {
                        "evidence_id": item.item_id,
                        "type": doc_type,
                        "title": _truncate_text(item.title, profile["detail_char_limit"]),
                        "date": item.date,
                        "detail": _truncate_text(item.detail, profile["detail_char_limit"]),
                        "snippet": _truncate_text(item.snippet, profile["snippet_char_limit"]),
                        "matched_keywords": list(item.matched_keywords)[: profile["matched_keywords_limit"]],
                        "satisfied_concepts": chunk_concepts.get(item.item_id, []),
                    }
                )
        return out

    @classmethod
    def _serialize_context(
        cls,
        candidate: CandidateCard,
        *,
        limit: int,
        profile: dict[str, Any],
        chunk_concepts: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """후보의 doc_type별 evidence를 간략 맥락으로 직렬화(제목/날짜/충족조건 위주)."""
        out: list[dict[str, Any]] = []
        for doc_type, evidences in candidate.evidence_by_type.items():
            for ev in evidences[:limit]:
                out.append(
                    {
                        "evidence_id": ev.chunk_id,
                        "type": doc_type,
                        "title": _truncate_text(ev.title, profile["detail_char_limit"]),
                        "date": ev.date,
                        "satisfied_concepts": chunk_concepts.get(ev.chunk_id, []),
                    }
                )
        return out

    @classmethod
    def _serialize_profile_context(
        cls, candidate: CandidateCard, *, limit: int, profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """질의에 직접 매칭되지 않은 '참고 프로필'(researcher hydration) — 배경용.

        evidence_id를 부여하지 않는다(인용 금지). 사유가 매칭 근거만 보고 후보를 과소평가하지 않도록
        배경 신호로만 제공한다.
        """
        out: list[dict[str, Any]] = []
        for ev in candidate.profile_evidence[:limit]:
            out.append(
                {
                    "type": ev.doc_type,
                    "title": _truncate_text(ev.title, profile["detail_char_limit"]),
                    "date": ev.date,
                }
            )
        return out

    @staticmethod
    def _compact_retrieval_grounding(trace: dict[str, Any]) -> dict[str, Any]:
        if not trace:
            return {}
        matches = trace.get("matches") or []
        compact = [
            {"doc_type": m.get("doc_type"), "rank": m.get("rank"), "score": m.get("score")}
            for m in matches[:3]
        ]
        return {
            "final_score": trace.get("final_score"),
            "doc_types": trace.get("doc_types"),
            "matches": compact,
        }

    @classmethod
    def _serialize_candidates(
        cls,
        candidates: list[CandidateCard],
        *,
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle] | None,
        retrieval_score_traces_by_expert_id: dict[str, dict[str, Any]] | None,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        relevant_evidence_by_expert_id = relevant_evidence_by_expert_id or {}
        retrieval_score_traces_by_expert_id = retrieval_score_traces_by_expert_id or {}
        for candidate in candidates:
            bundle = relevant_evidence_by_expert_id.get(
                candidate.expert_id, RelevantEvidenceBundle(expert_id=candidate.expert_id)
            )
            # chunk_id -> 충족 concept (top_chunks가 보유한 per-chunk 검색 신호를 evidence에 다시 붙인다).
            chunk_concepts = {
                str(chunk.get("chunk_id")): list(chunk.get("concepts", []))
                for chunk in candidate.top_chunks
                if chunk.get("chunk_id")
            }
            serialized.append(
                {
                    "expert_id": candidate.expert_id,
                    "name": _truncate_text(candidate.name, profile["detail_char_limit"]),
                    "organization": _truncate_text(candidate.organization, profile["detail_char_limit"]),
                    "degree": _truncate_text(candidate.degree, profile["detail_char_limit"]),
                    "rank_score": candidate.rank_score,
                    "shortlist_score": candidate.shortlist_score,
                    "counts": dict(candidate.counts),
                    "doc_types_present": candidate.doc_types_present,
                    "matched_concepts": list(candidate.matched_concepts),
                    "missing_concepts": list(candidate.missing_concepts),
                    "coverage_type": candidate.coverage_type,
                    "matched_filter_summary": list(candidate.matched_filter_summary)[: profile["matched_filter_limit"]],
                    "data_gaps": list(candidate.data_gaps),
                    "retrieval_grounding": cls._compact_retrieval_grounding(
                        retrieval_score_traces_by_expert_id.get(candidate.expert_id, {})
                    ),
                    "relevant_evidence": cls._serialize_relevant(
                        bundle, limit=profile["relevant_limit"], profile=profile,
                        chunk_concepts=chunk_concepts,
                    ),
                    "context_evidence": cls._serialize_context(
                        candidate, limit=profile["context_limit"], profile=profile,
                        chunk_concepts=chunk_concepts,
                    ),
                    "profile_context": cls._serialize_profile_context(
                        candidate, limit=profile.get("profile_context_limit", 8), profile=profile,
                    ),
                }
            )
        return serialized

    @staticmethod
    def _normalize_output(
        output: ReasonGenerationOutput,
        candidates: list[CandidateCard],
    ) -> tuple[ReasonGenerationOutput, dict[str, Any]]:
        by_expert_id = {item.expert_id: item for item in output.items}
        normalized_items: list[ReasonedCandidate] = []
        returned_ids = [item.expert_id for item in output.items]
        missing_candidate_ids: list[str] = []
        empty_reason_candidate_ids: list[str] = []
        empty_selected_evidence_candidate_ids: list[str] = []
        invalid_selected_evidence_candidate_ids: list[str] = []
        invalid_selected_evidence_ids_by_candidate: dict[str, list[str]] = {}
        truncated_reason_candidate_ids: list[str] = []
        leaked_reason_candidate_ids: list[str] = []

        for candidate in candidates:
            item = by_expert_id.get(candidate.expert_id)
            if item is None:
                missing_candidate_ids.append(candidate.expert_id)
                empty_reason_candidate_ids.append(candidate.expert_id)
                empty_selected_evidence_candidate_ids.append(candidate.expert_id)
                normalized_items.append(
                    ReasonedCandidate(
                        expert_id=candidate.expert_id,
                        fit=FIT_NORMAL,
                        recommendation_reason="",
                        selected_evidence_ids=[],
                        risks=list(candidate.risks),
                    )
                )
                continue

            normalized_reason = " ".join(item.recommendation_reason.split())
            normalized_reason, reason_id_stripped = _strip_inline_evidence_ids(normalized_reason)
            if reason_id_stripped:
                leaked_reason_candidate_ids.append(candidate.expert_id)
            if len(normalized_reason) > REASON_MAX_CHARS:
                normalized_reason = _truncate_text(normalized_reason, REASON_MAX_CHARS) or ""
                truncated_reason_candidate_ids.append(candidate.expert_id)
            raw_selected_evidence_ids = [
                " ".join(str(evidence_id).split())
                for evidence_id in list(item.selected_evidence_ids)[:MAX_SELECTED_EVIDENCE_IDS]
            ]
            invalid_selected_evidence_ids = [
                evidence_id
                for evidence_id in raw_selected_evidence_ids
                if evidence_id and not VALID_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
            ]
            normalized_selected_evidence_ids: list[str] = []
            for evidence_id in raw_selected_evidence_ids:
                if (
                    not evidence_id
                    or evidence_id in normalized_selected_evidence_ids
                    or not VALID_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
                ):
                    continue
                normalized_selected_evidence_ids.append(evidence_id)
            normalized_fit = item.fit if item.fit in FIT_VALUES else FIT_NORMAL

            if not normalized_reason:
                empty_reason_candidate_ids.append(candidate.expert_id)
            if not normalized_selected_evidence_ids:
                empty_selected_evidence_candidate_ids.append(candidate.expert_id)
            if invalid_selected_evidence_ids:
                invalid_selected_evidence_candidate_ids.append(candidate.expert_id)
                invalid_selected_evidence_ids_by_candidate[candidate.expert_id] = invalid_selected_evidence_ids

            normalized_items.append(
                ReasonedCandidate(
                    expert_id=candidate.expert_id,
                    fit=normalized_fit,
                    recommendation_reason=normalized_reason,
                    selected_evidence_ids=normalized_selected_evidence_ids,
                    risks=list(item.risks),
                )
            )

        if missing_candidate_ids:
            logger.warning("Reason generator omitted candidates: missing=%s", missing_candidate_ids)
        if empty_reason_candidate_ids:
            logger.warning("Reason generator empty reasons: candidate_ids=%s", empty_reason_candidate_ids)
        if truncated_reason_candidate_ids:
            logger.warning(
                "Reason generator truncated reasons over %d chars: candidate_ids=%s",
                REASON_MAX_CHARS,
                truncated_reason_candidate_ids,
            )
        if leaked_reason_candidate_ids:
            logger.warning(
                "Reason generator leaked evidence ids in prose (scrubbed): candidate_ids=%s",
                leaked_reason_candidate_ids,
            )
        if invalid_selected_evidence_candidate_ids:
            logger.warning(
                "Reason generator invalid evidence ids: candidate_ids=%s invalid=%s",
                invalid_selected_evidence_candidate_ids,
                invalid_selected_evidence_ids_by_candidate,
            )

        return (
            ReasonGenerationOutput(items=normalized_items, data_gaps=output.data_gaps),
            {
                "returned_ids": returned_ids,
                "missing_candidate_ids": missing_candidate_ids,
                "empty_reason_candidate_ids": empty_reason_candidate_ids,
                "empty_selected_evidence_candidate_ids": empty_selected_evidence_candidate_ids,
                "invalid_selected_evidence_candidate_ids": invalid_selected_evidence_candidate_ids,
                "invalid_selected_evidence_ids_by_candidate": invalid_selected_evidence_ids_by_candidate,
                "truncated_reason_candidate_ids": truncated_reason_candidate_ids,
                "leaked_reason_candidate_ids": leaked_reason_candidate_ids,
            },
        )

    @staticmethod
    def _extract_tool_arguments(message: AIMessage) -> Any:
        tool_calls = []
        if isinstance(message.additional_kwargs, dict):
            tool_calls = message.additional_kwargs.get("tool_calls") or []
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            if function.get("name") != REASON_TOOL_NAME:
                continue
            arguments = function.get("arguments")
            if arguments is None:
                continue
            if isinstance(arguments, str):
                return json.loads(arguments)
            return arguments
        raise ValueError("No matching tool call arguments found in model response")

    @classmethod
    def _parse_reason_output(
        cls, message: AIMessage, *, use_tools: bool
    ) -> tuple[ReasonGenerationOutput, str]:
        if use_tools:
            try:
                tool_payload = cls._extract_tool_arguments(message)
                return ReasonGenerationOutput.model_validate(tool_payload), "tool_call"
            except Exception:
                pass
        json_text = _extract_json_object_text(message.content)
        return (
            ReasonGenerationOutput.model_validate_json(json_text),
            "json_fallback" if use_tools else "json_fallback_retry",
        )

    async def _invoke_attempt(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        candidates: list[CandidateCard],
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle] | None,
        retrieval_score_traces_by_expert_id: dict[str, dict[str, Any]] | None,
        seed: int,
        use_tools: bool,
        profile: dict[str, Any],
    ) -> tuple[ReasonGenerationOutput, dict[str, Any]]:
        serialized_candidates = self._serialize_candidates(
            candidates,
            relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
            retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
            profile=profile,
        )
        payload = {
            "query": query,
            "intent_summary": plan.intent_summary,
            "core_keywords": list(plan.core_keywords),
            "task_terms": list(plan.task_terms),
            "candidates": serialized_candidates,
        }
        allowed_evidence_ids = sorted(
            {
                item["evidence_id"]
                for candidate in serialized_candidates
                for key in ("relevant_evidence", "context_evidence")
                for item in candidate.get(key, [])
                if item.get("evidence_id")
            }
        )
        payload_text = json.dumps(payload, ensure_ascii=False)
        invoke_kwargs = build_consistency_invoke_kwargs(
            max_tokens_hint=REASON_GENERATION_MAX_TOKENS,
            seed=seed,
        )
        if use_tools:
            invoke_kwargs["tools"] = self._build_reason_tools(allowed_evidence_ids)
            invoke_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": REASON_TOOL_NAME},
            }
            invoke_kwargs["parallel_tool_calls"] = False

        result = await self.model.ainvoke_non_stream(
            [
                SystemMessage(content=self._build_system_prompt(use_tools=use_tools)),
                HumanMessage(content=payload_text),
            ],
            **invoke_kwargs,
        )

        parsed, parse_mode = self._parse_reason_output(result, use_tools=use_tools)
        normalized, normalization_trace = self._normalize_output(parsed, candidates)
        returned_ids = list(normalization_trace.get("returned_ids", []))
        if not returned_ids:
            raise ValueError("Reason generator returned no matching candidate ids")

        candidate_count = len(candidates)
        returned_ratio = round(len(returned_ids) / candidate_count, 3) if candidate_count else 0.0
        trace = {
            "mode": parse_mode,
            "candidate_count": candidate_count,
            "output_count": len(normalized.items),
            "raw_output_count": len(parsed.items),
            "seed": seed,
            "returned_ratio": returned_ratio,
            "payload_char_count": len(payload_text),
            "payload_token_estimate": max(1, round(len(payload_text) / 4)),
            "prompt_budget_mode": profile["name"],
            "trim_applied": bool(profile["trim_applied"]),
            "allowed_evidence_id_count": len(allowed_evidence_ids),
            **normalization_trace,
        }
        return normalized, trace

    @staticmethod
    def _merge_targeted_retry(
        base: ReasonGenerationOutput,
        retry: ReasonGenerationOutput,
        incomplete_ids: set[str],
    ) -> tuple[ReasonGenerationOutput, list[str]]:
        """targeted 재시도 결과로 base의 빈 사유 후보만 덮어쓴다(다른 후보는 보존)."""
        retry_by_id = {item.expert_id: item for item in retry.items}
        filled: list[str] = []
        merged_items: list[ReasonedCandidate] = []
        for item in base.items:
            if item.expert_id in incomplete_ids and not item.recommendation_reason.strip():
                replacement = retry_by_id.get(item.expert_id)
                if replacement is not None and replacement.recommendation_reason.strip():
                    merged_items.append(replacement)
                    filled.append(item.expert_id)
                    continue
            merged_items.append(item)
        return ReasonGenerationOutput(items=merged_items, data_gaps=base.data_gaps), filled

    async def generate(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        candidates: list[CandidateCard],
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle] | None = None,
        retrieval_score_traces_by_expert_id: dict[str, dict[str, Any]] | None = None,
    ) -> ReasonGenerationOutput:
        if not candidates:
            self.last_trace = {"mode": "tool_call", "candidate_count": 0, "output_count": 0}
            return ReasonGenerationOutput()

        seed = build_deterministic_seed(
            "reason_generation",
            query,
            plan.model_dump(mode="json"),
            [candidate.expert_id for candidate in candidates],
        )

        attempt_specs = [
            {"use_tools": True, "profile": PRIMARY_PAYLOAD_PROFILE},
            {"use_tools": False, "profile": RETRY_PAYLOAD_PROFILE},
        ]
        attempt_history: list[dict[str, Any]] = []
        primary_output: ReasonGenerationOutput | None = None
        primary_trace: dict[str, Any] = {}

        for retry_index, attempt_spec in enumerate(attempt_specs):
            try:
                output, trace = await self._invoke_attempt(
                    query=query,
                    plan=plan,
                    candidates=candidates,
                    relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
                    retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
                    seed=seed,
                    use_tools=attempt_spec["use_tools"],
                    profile=attempt_spec["profile"],
                )
                trace["retry_count"] = retry_index
                trace["attempts"] = [*attempt_history, dict(trace)]
                primary_output, primary_trace = output, trace
                break
            except Exception as exc:
                failed_mode = "tool_call" if attempt_spec["use_tools"] else "json_fallback_retry"
                logger.warning(
                    "Reason generator %s attempt failed: retry=%d reason=%s",
                    failed_mode, retry_index, exc,
                )
                attempt_history.append(
                    {
                        "mode": failed_mode,
                        "retry_index": retry_index,
                        "prompt_budget_mode": attempt_spec["profile"]["name"],
                        "trim_applied": bool(attempt_spec["profile"]["trim_applied"]),
                        "reason": str(exc),
                    }
                )

        if primary_output is None:
            logger.warning("Reason generator fallback activated after retries: attempts=%s", attempt_history)
            fallback_output = await self.fallback.generate(
                query=query,
                plan=plan,
                candidates=candidates,
                relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
                retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
            )
            fallback_trace = dict(self.fallback.last_trace)
            self.last_trace = {
                "mode": "fallback",
                "candidate_count": len(candidates),
                "output_count": len(fallback_output.items),
                "raw_output_count": fallback_trace.get("raw_output_count", len(fallback_output.items)),
                "seed": seed,
                "retry_count": len(attempt_specs),
                "returned_ratio": fallback_trace.get("returned_ratio", 1.0),
                "prompt_budget_mode": "fallback",
                "trim_applied": True,
                "reason": "; ".join(
                    attempt.get("reason", "") for attempt in attempt_history if attempt.get("reason")
                ),
                "returned_ids": list(fallback_trace.get("returned_ids", [])),
                "missing_candidate_ids": list(fallback_trace.get("missing_candidate_ids", [])),
                "empty_reason_candidate_ids": list(fallback_trace.get("empty_reason_candidate_ids", [])),
                "empty_selected_evidence_candidate_ids": list(
                    fallback_trace.get("empty_selected_evidence_candidate_ids", [])
                ),
                "attempts": attempt_history,
            }
            return fallback_output

        # 누락/빈 사유 후보만 targeted 재시도(부분 실패를 서버 fallback 문장으로 묻지 않는다).
        # 재시도 seed를 본 시도와 다르게 둬 동일 출력 재생산을 피한다.
        incomplete_ids = sorted(
            set(primary_trace.get("missing_candidate_ids", []))
            | set(primary_trace.get("empty_reason_candidate_ids", []))
        )
        if incomplete_ids:
            incomplete_set = set(incomplete_ids)
            retry_candidates = [c for c in candidates if c.expert_id in incomplete_set]
            retry_seed = build_deterministic_seed(
                "reason_generation_retry", query, plan.model_dump(mode="json"), incomplete_ids
            )
            try:
                retry_output, retry_trace = await self._invoke_attempt(
                    query=query,
                    plan=plan,
                    candidates=retry_candidates,
                    relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
                    retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
                    seed=retry_seed,
                    use_tools=True,
                    # 재시도된 후보가 배치 동료보다 빈약해지지 않도록 PRIMARY 밀도로 재생성(소수만 재호출 → 토큰 안전).
                    profile=PRIMARY_PAYLOAD_PROFILE,
                )
                primary_output, filled_ids = self._merge_targeted_retry(
                    primary_output, retry_output, incomplete_set
                )
                primary_trace["targeted_retry"] = {
                    "requested_ids": incomplete_ids,
                    "filled_ids": filled_ids,
                    "retry_trace": retry_trace,
                }
            except Exception as exc:
                logger.warning("Targeted reason retry failed: ids=%s reason=%s", incomplete_ids, exc)
                primary_trace["targeted_retry"] = {
                    "requested_ids": incomplete_ids,
                    "filled_ids": [],
                    "error": str(exc),
                }

        self.last_trace = primary_trace
        return primary_output
