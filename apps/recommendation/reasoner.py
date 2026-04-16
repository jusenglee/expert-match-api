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

logger = logging.getLogger(__name__)

FIT_HIGH = "\ub192\uc74c"
FIT_MEDIUM = "\uc911\uac04"
FIT_NORMAL = "\ubcf4\ud1b5"
FIT_VALUES = {FIT_HIGH, FIT_MEDIUM, FIT_NORMAL}

MAX_SELECTED_EVIDENCE_IDS = 4
REASON_TOOL_NAME = "submit_recommendation_batch"
REASON_GENERATION_MAX_TOKENS = 1600
VALID_EVIDENCE_ID_PATTERN = re.compile(r"^(paper|project|patent):\d+$")

PRIMARY_PAYLOAD_PROFILE: dict[str, Any] = {
    "name": "primary",
    "trim_applied": False,
    "relevant_limit": 10,
    "all_papers_limit": 3,
    "all_projects_limit": 5,
    "all_patents_limit": 3,
    "evaluation_limit": 3,
    "technical_classifications_limit": 5,
    "matched_filter_limit": 4,
    "matched_keywords_limit": 5,
    "snippet_char_limit": 1000,  # 600에서 1000으로 상향
    "detail_char_limit": 200,
    "abstract_char_limit": 600,
    "project_summary_char_limit": 600,
}

RETRY_PAYLOAD_PROFILE: dict[str, Any] = {
    "name": "retry_compact",
    "trim_applied": True,
    "relevant_limit": 4,
    "all_papers_limit": 1,
    "all_projects_limit": 2,
    "all_patents_limit": 1,
    "evaluation_limit": 1,
    "technical_classifications_limit": 3,
    "matched_filter_limit": 2,
    "matched_keywords_limit": 3,
    "snippet_char_limit": 280,
    "detail_char_limit": 120,
    "abstract_char_limit": 220,
    "project_summary_char_limit": 220,
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
            else (
                '반드시 `{"items":[...],"data_gaps":[...]}` 형태의 JSON 객체 단 하나만 반환해야 하며, JSON 외의 다른 텍스트는 출력하지 마세요.'
            )
        )
        prompt = f"""
        당신은 R&D 전문가 추천 시스템의 추천 사유 생성기입니다.

        당신의 임무는 이미 추천 랭킹이 확정된 후보자 배치를 받아, 각 후보자가 추천된 객관적인 사유를 생성하는 것입니다.
        입력된 후보자의 전문가 ID(expert_id)를 단 하나도 누락하거나 중복하거나 임의로 생성하지 마세요.

        [배경 및 데이터 활용]
        - 이 후보자들은 시스템에 의해 질의와 관련된 인물로 이미 판별된 상태입니다.
        - **[중요]** 절대 없는 사실을 지어내지 마세요(환각 금지). 반드시 제공된 증거(`relevant_papers`, `relevant_projects`, `relevant_patents`)의 내용에 기반하여 작성해야 합니다.
        - `technical_classifications`, `evaluation_activities` 등의 부가 컨텍스트는 보조적으로만 활용하세요.
        - 증거를 인용할 때는 반드시 제공된 증거 ID(`relevant_*`에 있는 ID)만 사용하세요.

        [출력 규칙]
        - `fit`은 다음 중 하나여야 합니다: {FIT_HIGH}, {FIT_MEDIUM}, {FIT_NORMAL}
        - `recommendation_reason`은 1~2문장의 간결하고 구체적인 한국어 문장으로 작성하며, 320자를 넘지 마세요.
        - **추천 사유는 반드시 제공된 증거의 실적명이나 연구 내용을 언급하여 작성해야 합니다.**
        - `selected_evidence_ids`는 제공된 모든 `evidence_id` 문자열을 그대로 포함시키세요 (최대 {MAX_SELECTED_EVIDENCE_IDS}개).
        - `selected_evidence_ids`에는 `paper:<number>`, `project:<number>`, `patent:<number>` 형식의 ID만 넣으세요.
        - 적절한 직접 증거 ID가 없으면 `selected_evidence_ids`는 빈 배열(`[]`)로 두세요.
        - `risks`는 매우 짧고 사실적인 유의사항만 적거나 비워두세요.

        {output_instruction}
        """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _build_reason_tools(
        allowed_evidence_ids: list[str],
    ) -> list[dict[str, Any]]:
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
                                        "risks": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
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
                            "data_gaps": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["items", "data_gaps"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    @classmethod
    def _serialize_evidence_items(
        cls,
        items: list[Any],
        *,
        limit: int,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "evidence_id": item.item_id,
                "type": item.type,
                "title": _truncate_text(item.title, profile["detail_char_limit"]),
                "date": item.date,
                "detail": _truncate_text(item.detail, profile["detail_char_limit"]),
                "snippet": _truncate_text(item.snippet, profile["snippet_char_limit"]),
                "matched_keywords": list(item.matched_keywords)[
                    : profile["matched_keywords_limit"]
                ],
            }
            for item in items[:limit]
        ]

    @classmethod
    def _serialize_all_publications(
        cls,
        items: list[Any],
        *,
        limit: int,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "title": _truncate_text(item.publication_title, profile["detail_char_limit"]),
                "journal_name": _truncate_text(item.journal_name, profile["detail_char_limit"]),
                "date": item.publication_year_month,
                "abstract": _truncate_text(item.abstract, profile["abstract_char_limit"]),
                "korean_keywords": list(item.korean_keywords)[
                    : profile["matched_keywords_limit"]
                ],
                "english_keywords": list(item.english_keywords)[
                    : profile["matched_keywords_limit"]
                ],
            }
            for item in items[:limit]
        ]

    @classmethod
    def _serialize_all_projects(
        cls,
        items: list[Any],
        *,
        limit: int,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "title": _truncate_text(item.display_title, profile["detail_char_limit"]),
                "start_date": item.project_start_date,
                "end_date": item.project_end_date,
                "reference_year": item.reference_year,
                "performing_organization": _truncate_text(
                    item.performing_organization, profile["detail_char_limit"]
                ),
                "managing_agency": _truncate_text(
                    item.managing_agency, profile["detail_char_limit"]
                ),
                "research_objective_summary": _truncate_text(
                    item.research_objective_summary, profile["project_summary_char_limit"]
                ),
                "research_content_summary": _truncate_text(
                    item.research_content_summary, profile["project_summary_char_limit"]
                ),
            }
            for item in items[:limit]
        ]

    @classmethod
    def _serialize_all_patents(
        cls,
        items: list[Any],
        *,
        limit: int,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "title": _truncate_text(
                    item.intellectual_property_title, profile["detail_char_limit"]
                ),
                "intellectual_property_type": _truncate_text(
                    item.intellectual_property_type, profile["detail_char_limit"]
                ),
                "application_registration_type": _truncate_text(
                    item.application_registration_type, profile["detail_char_limit"]
                ),
                "application_country": _truncate_text(
                    item.application_country, profile["detail_char_limit"]
                ),
                "application_date": item.application_date,
                "registration_date": item.registration_date,
            }
            for item in items[:limit]
        ]

    @staticmethod
    def _serialize_evaluation_activities(
        items: list[Any],
        *,
        limit: int,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "appoint_org_nm": _truncate_text(
                    item.appoint_org_nm, profile["detail_char_limit"]
                ),
                "committee_nm": _truncate_text(
                    item.committee_nm, profile["detail_char_limit"]
                ),
                "appoint_period": item.appoint_period,
                "appoint_dt": item.appoint_dt,
            }
            for item in items[:limit]
        ]

    @staticmethod
    def _compact_retrieval_grounding(trace: dict[str, Any]) -> dict[str, Any]:
        if not trace:
            return {}
        branch_matches = trace.get("branch_matches") or []
        compact_matches = [
            {
                "branch": match.get("branch"),
                "rank": match.get("rank"),
                "score": match.get("score"),
            }
            for match in branch_matches[:2]
        ]
        return {
            "primary_branch": trace.get("primary_branch"),
            "final_score": trace.get("final_score"),
            "branch_matches": compact_matches,
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
            relevant_bundle = relevant_evidence_by_expert_id.get(
                candidate.expert_id,
                RelevantEvidenceBundle(expert_id=candidate.expert_id),
            )
            serialized.append(
                {
                    "expert_id": candidate.expert_id,
                    "name": _truncate_text(candidate.name, profile["detail_char_limit"]),
                    "organization": _truncate_text(
                        candidate.organization, profile["detail_char_limit"]
                    ),
                    "position": _truncate_text(
                        candidate.position, profile["detail_char_limit"]
                    ),
                    "degree": _truncate_text(candidate.degree, profile["detail_char_limit"]),
                    "major": _truncate_text(candidate.major, profile["detail_char_limit"]),
                    "rank_score": candidate.rank_score,
                    "shortlist_score": candidate.shortlist_score,
                    "branch_presence_flags": dict(candidate.branch_presence_flags),
                    "counts": dict(candidate.counts),
                    "technical_classifications": list(
                        candidate.technical_classifications
                    )[: profile["technical_classifications_limit"]],
                    "evaluation_activity_cnt": candidate.evaluation_activity_cnt,
                    "evaluation_activities": cls._serialize_evaluation_activities(
                        candidate.evaluation_activities,
                        limit=profile["evaluation_limit"],
                        profile=profile,
                    ),
                    "matched_filter_summary": list(candidate.matched_filter_summary)[
                        : profile["matched_filter_limit"]
                    ],
                    "data_gaps": list(candidate.data_gaps),
                    "retrieval_grounding": cls._compact_retrieval_grounding(
                        retrieval_score_traces_by_expert_id.get(candidate.expert_id, {})
                    ),
                    "relevant_papers": cls._serialize_evidence_items(
                        relevant_bundle.papers,
                        limit=profile["relevant_limit"],
                        profile=profile,
                    ),
                    "relevant_patents": cls._serialize_evidence_items(
                        relevant_bundle.patents,
                        limit=profile["relevant_limit"],
                        profile=profile,
                    ),
                    "relevant_projects": cls._serialize_evidence_items(
                        relevant_bundle.projects,
                        limit=profile["relevant_limit"],
                        profile=profile,
                    ),
                    "all_papers": cls._serialize_all_publications(
                        candidate.top_papers,
                        limit=profile["all_papers_limit"],
                        profile=profile,
                    ),
                    "all_patents": cls._serialize_all_patents(
                        candidate.top_patents,
                        limit=profile["all_patents_limit"],
                        profile=profile,
                    ),
                    "all_projects": cls._serialize_all_projects(
                        candidate.top_projects,
                        limit=profile["all_projects_limit"],
                        profile=profile,
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
            raw_selected_evidence_ids = [
                " ".join(str(evidence_id).split())
                for evidence_id in list(item.selected_evidence_ids)[
                    :MAX_SELECTED_EVIDENCE_IDS
                ]
            ]
            invalid_selected_evidence_ids = [
                evidence_id
                for evidence_id in raw_selected_evidence_ids
                if evidence_id
                and not VALID_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
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
                invalid_selected_evidence_ids_by_candidate[candidate.expert_id] = (
                    invalid_selected_evidence_ids
                )

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
        if invalid_selected_evidence_candidate_ids:
            logger.warning(
                "Reason generator returned invalid selected evidence ids: candidate_ids=%s invalid_ids_by_candidate=%s",
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
        cls,
        message: AIMessage,
        *,
        use_tools: bool,
    ) -> tuple[ReasonGenerationOutput, str]:
        if use_tools:
            try:
                tool_payload = cls._extract_tool_arguments(message)
                return (
                    ReasonGenerationOutput.model_validate(tool_payload),
                    "tool_call",
                )
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
                for key in ("relevant_papers", "relevant_projects", "relevant_patents")
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
                "mode": "tool_call",
                "candidate_count": 0,
                "output_count": 0,
            }
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
                self.last_trace = trace
                return output
            except Exception as exc:
                failed_mode = (
                    "tool_call" if attempt_spec["use_tools"] else "json_fallback_retry"
                )
                logger.warning(
                    "Reason generator %s attempt failed: retry=%d reason=%s",
                    failed_mode,
                    retry_index,
                    exc,
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

        logger.warning(
            "Reason generator fallback activated after retries: attempts=%s",
            attempt_history,
        )
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
            "raw_output_count": fallback_trace.get(
                "raw_output_count", len(fallback_output.items)
            ),
            "seed": seed,
            "retry_count": len(attempt_specs),
            "returned_ratio": fallback_trace.get("returned_ratio", 1.0),
            "prompt_budget_mode": "fallback",
            "trim_applied": True,
            "reason": "; ".join(
                attempt.get("reason", "") for attempt in attempt_history if attempt.get("reason")
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
            "attempts": attempt_history,
        }
        return fallback_output
