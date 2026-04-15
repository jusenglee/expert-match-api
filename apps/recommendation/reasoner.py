from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from apps.core.config import Settings
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text
from apps.core.llm_policies import build_consistency_invoke_kwargs
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.core.utils import build_deterministic_seed
from apps.domain.models import CandidateCard, PlannerOutput
from apps.recommendation.evidence_selector import RelevantEvidenceBundle

logger = logging.getLogger(__name__)


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


class ReasonedCandidate(BaseModel):
    expert_id: str
    fit: str = "보통"
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
        _ = (
            query,
            plan,
            relevant_evidence_by_expert_id,
            retrieval_score_traces_by_expert_id,
        )
        output = ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id=candidate.expert_id,
                    fit="보통",
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
        }
        return output


class OpenAICompatReasonGenerator:
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
    def _build_system_prompt() -> str:
        prompt = """
            당신은 국가 R&D 과제 평가위원 추천 시스템의 '기술 전문 분석관'입니다.
            후보자들의 실적(논문, 특허, 과제)을 심도 있게 분석하여, 왜 이 전문가가 사용자의 질의에 가장 적합한지 기술적으로 설득력 있는 추천 사유를 작성하는 것이 당신의 임무입니다.

            [작성 원칙]
            1. **종합적 분석**: 제공된 `relevant_papers/projects/patents`뿐만 아니라, `all_papers/projects/patents`, `evaluation_activities`(평가위원 활동), `technical_classifications`(기술 분류) 등 모든 컨텍스트를 종합적으로 고려하여 후보자의 전문성을 판단하세요.
            2. **기술적 구체성**: 단순히 "경험이 많음"이라고 하지 마세요. 실적물(Snippet)에 나타난 구체적인 기술 명칭, 방법론, 연구 대상을 언급하며 사용자의 질의와 기술적으로 어떻게 연결되는지 설명하세요.
            3. **증거 인용(Citation)**: 추천 사유 내에서 핵심적인 증거가 된 실적의 제목을 언급하고, 해당 실적이 `relevant_*` 리스트에 있다면 그 `evidence_id`를 `selected_evidence_ids`에 정확히 포함하세요. (최대 4개)
            4. **적합도(Fit) 판단**: 
               - '높음': 질의의 핵심 기술과 실적이 직접적으로 일치하며 최근 성과가 뚜렷함.
               - '중간': 관련 분야의 경험은 풍부하나 질의의 특정 세부 기술과는 약간의 거리가 있음.
               - '보통': 기초적인 관련성은 있으나 직접적인 핵심 증거가 부족함.
            5. **데이터 공백(Data Gaps)**: 만약 핵심 기술에 대한 직접적인 실적이 부족하다면 `data_gaps`에 "OO 기술에 대한 직접적인 논문/과제 이력 확인 필요"와 같이 구체적으로 명시하세요.

            [금기 사항]
            - 제공되지 않은 실적을 지어내거나 과장하지 마세요.
            - "추천합니다"와 같은 상투적인 문구보다는 "XX 기술에 대한 연구 실적을 보유하고 있어 YY 과제 평가에 적합함"과 같은 분석적 어조를 사용하세요.
            - 한국어로 전문적이고 신뢰감 있는 어조를 유지하세요.

            반드시 JSON 형식으로 응답하십시오.
        """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _serialize_evidence_items(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "evidence_id": item.item_id,
                "type": item.type,
                "title": item.title,
                "date": item.date,
                "detail": item.detail,
                "snippet": item.snippet,
                "matched_keywords": list(item.matched_keywords),
            }
            for item in items
        ]

    @staticmethod
    def _serialize_all_publications(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "title": item.publication_title,
                "journal_name": item.journal_name,
                "date": item.publication_year_month,
                "abstract": item.abstract,
                "korean_keywords": list(item.korean_keywords),
                "english_keywords": list(item.english_keywords),
            }
            for item in items
        ]

    @staticmethod
    def _serialize_all_projects(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "title": item.display_title,
                "project_title_korean": item.project_title_korean,
                "project_title_english": item.project_title_english,
                "start_date": item.project_start_date,
                "end_date": item.project_end_date,
                "reference_year": item.reference_year,
                "performing_organization": item.performing_organization,
                "managing_agency": item.managing_agency,
                "research_objective_summary": item.research_objective_summary,
                "research_content_summary": item.research_content_summary,
            }
            for item in items
        ]

    @staticmethod
    def _serialize_all_patents(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "title": item.intellectual_property_title,
                "intellectual_property_type": item.intellectual_property_type,
                "application_registration_type": item.application_registration_type,
                "application_country": item.application_country,
                "application_number": item.application_number,
                "application_date": item.application_date,
                "registration_number": item.registration_number,
                "registration_date": item.registration_date,
            }
            for item in items
        ]

    @staticmethod
    def _serialize_evaluation_activities(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "appoint_org_nm": item.appoint_org_nm,
                "committee_nm": item.committee_nm,
                "appoint_period": item.appoint_period,
                "appoint_dt": item.appoint_dt,
            }
            for item in items
        ]

    @classmethod
    def _serialize_candidates(
        cls,
        candidates: list[CandidateCard],
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle] | None = None,
        retrieval_score_traces_by_expert_id: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        relevant_evidence_by_expert_id = relevant_evidence_by_expert_id or {}
        retrieval_score_traces_by_expert_id = retrieval_score_traces_by_expert_id or {}
        for candidate in candidates:
            relevant_bundle = relevant_evidence_by_expert_id.get(
                candidate.expert_id,
                RelevantEvidenceBundle(expert_id=candidate.expert_id),
            )
            serialized.append(
                {
                    "expert_id": candidate.expert_id,
                    "name": candidate.name,
                    "organization": candidate.organization,
                    "position": candidate.position,
                    "degree": candidate.degree,
                    "major": candidate.major,
                    "rank_score": candidate.rank_score,
                    "shortlist_score": candidate.shortlist_score,
                    "branch_presence_flags": dict(candidate.branch_presence_flags),
                    "counts": dict(candidate.counts),
                    "technical_classifications": list(
                        candidate.technical_classifications
                    ),
                    "evaluation_activity_cnt": candidate.evaluation_activity_cnt,
                    "evaluation_activities": cls._serialize_evaluation_activities(
                        candidate.evaluation_activities
                    ),
                    "matched_filter_summary": list(candidate.matched_filter_summary),
                    "data_gaps": list(candidate.data_gaps),
                    "retrieval_grounding": retrieval_score_traces_by_expert_id.get(
                        candidate.expert_id, {}
                    ),
                    "relevant_papers": cls._serialize_evidence_items(
                        relevant_bundle.papers[:10]
                    ),
                    "relevant_patents": cls._serialize_evidence_items(
                        relevant_bundle.patents[:10]
                    ),
                    "relevant_projects": cls._serialize_evidence_items(
                        relevant_bundle.projects[:10]
                    ),
                    "all_papers": cls._serialize_all_publications(candidate.top_papers[:10]),
                    "all_patents": cls._serialize_all_patents(candidate.top_patents[:10]),
                    "all_projects": cls._serialize_all_projects(candidate.top_projects[:10]),
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
        for candidate in candidates:
            item = by_expert_id.get(candidate.expert_id)
            if item is None:
                missing_candidate_ids.append(candidate.expert_id)
                empty_reason_candidate_ids.append(candidate.expert_id)
                empty_selected_evidence_candidate_ids.append(candidate.expert_id)
                normalized_items.append(
                    ReasonedCandidate(
                        expert_id=candidate.expert_id,
                        fit="보통",
                        recommendation_reason="",
                        selected_evidence_ids=[],
                        risks=list(candidate.risks),
                    )
                )
                continue
            normalized_reason = " ".join(item.recommendation_reason.split())
            normalized_selected_evidence_ids = list(item.selected_evidence_ids)[:4]
            if not normalized_reason:
                empty_reason_candidate_ids.append(candidate.expert_id)
            if not normalized_selected_evidence_ids:
                empty_selected_evidence_candidate_ids.append(candidate.expert_id)
            normalized_items.append(
                ReasonedCandidate(
                    expert_id=candidate.expert_id,
                    fit=item.fit if item.fit in {"높음", "중간", "보통"} else "보통",
                    recommendation_reason=normalized_reason,
                    selected_evidence_ids=normalized_selected_evidence_ids,
                    risks=list(item.risks),
                )
            )
        if missing_candidate_ids:
            logger.warning(
                "Reason generator omitted candidates from output: missing_candidate_ids=%s",
                missing_candidate_ids,
            )
        if empty_reason_candidate_ids:
            logger.warning(
                "Reason generator returned empty recommendation reasons: candidate_ids=%s",
                empty_reason_candidate_ids,
            )
        if empty_selected_evidence_candidate_ids:
            logger.info(
                "Reason generator returned no selected evidence ids: candidate_ids=%s",
                empty_selected_evidence_candidate_ids,
            )
        return (
            ReasonGenerationOutput(
                items=normalized_items,
                data_gaps=list(output.data_gaps),
            ),
            {
                "returned_ids": returned_ids,
                "missing_candidate_ids": missing_candidate_ids,
                "empty_reason_candidate_ids": empty_reason_candidate_ids,
                "empty_selected_evidence_candidate_ids": empty_selected_evidence_candidate_ids,
            },
        )

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
            self.last_trace = {
                "mode": "openai_compat",
                "candidate_count": 0,
                "output_count": 0,
            }
            return ReasonGenerationOutput()

        serialized_candidates = self._serialize_candidates(
            candidates,
            relevant_evidence_by_expert_id,
            retrieval_score_traces_by_expert_id,
        )
        seed = build_deterministic_seed(
            "reason_generation",
            query,
            plan.model_dump(mode="json"),
            [candidate["expert_id"] for candidate in serialized_candidates],
        )
        invoke_kwargs = build_consistency_invoke_kwargs(seed=seed)
        payload = {
            "query": query,
            "intent_summary": plan.intent_summary,
            "core_keywords": list(plan.core_keywords),
            "task_terms": list(plan.task_terms),
            "candidates": serialized_candidates,
        }

        try:
            result = await self.model.ainvoke_non_stream(
                [
                    SystemMessage(content=self._build_system_prompt()),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ],
                **invoke_kwargs,
            )
            json_text = _extract_json_object_text(result.content)
            parsed = ReasonGenerationOutput.model_validate_json(json_text)
            normalized, normalization_trace = self._normalize_output(parsed, candidates)
            self.last_trace = {
                "mode": "openai_compat",
                "candidate_count": len(candidates),
                "output_count": len(normalized.items),
                "raw_output_count": len(parsed.items),
                "seed": seed,
                **normalization_trace,
            }
            return normalized
        except Exception as exc:
            logger.warning("Reason generator fallback activated: reason=%s", exc)
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
                "seed": seed,
                "reason": str(exc),
                "raw_output_count": fallback_trace.get(
                    "raw_output_count", len(fallback_output.items)
                ),
                "returned_ids": list(fallback_trace.get("returned_ids", [])),
                "missing_candidate_ids": list(
                    fallback_trace.get("missing_candidate_ids", [])
                ),
                "empty_reason_candidate_ids": list(
                    fallback_trace.get("empty_reason_candidate_ids", [])
                ),
                "empty_selected_evidence_candidate_ids": list(
                    fallback_trace.get("empty_selected_evidence_candidate_ids", [])
                ),
            }
            return fallback_output
