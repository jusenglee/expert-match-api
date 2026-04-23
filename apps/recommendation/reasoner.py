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

REASON_TOOL_NAME = "submit_recommendation_batch"
REASON_GENERATION_MAX_TOKENS = 1600
VALID_EVIDENCE_ID_PATTERN = re.compile(r"^(paper|project|patent):\d+$")

PRIMARY_PAYLOAD_PROFILE: dict[str, Any] = {
    "name": "primary",
    "trim_applied": False,
    "selected_limit": 4,
    "matched_keywords_limit": 5,
    "snippet_char_limit": 1000,
    "detail_char_limit": 200,
    "name_char_limit": 80,
}

RETRY_PAYLOAD_PROFILE: dict[str, Any] = {
    "name": "retry_compact",
    "trim_applied": True,
    "selected_limit": 2,
    "matched_keywords_limit": 3,
    "snippet_char_limit": 280,
    "detail_char_limit": 120,
    "name_char_limit": 60,
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
    """
    LLM 추론을 거쳐 평가된 개별 전문가의 추천 사유, 적합도(fit), 그리고 잠재적 리스크를 담는 데이터 모델입니다.
    """

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
    """
    LLM 추론 없이 기본값(보통, 빈 사유)으로 후보자 정보를 통과시키는 폴백(Fallback)용 생성기입니다.
    네트워크 오류나 LLM 제한 발생 시 안전하게 결과를 반환하기 위해 사용됩니다.
    """

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
            "empty_selected_evidence_candidate_ids": [],
            "invalid_selected_evidence_candidate_ids": [],
            "invalid_selected_evidence_ids_by_candidate": {},
            "retry_count": 0,
            "returned_ratio": 1.0 if candidates else 0.0,
            "prompt_budget_mode": "fallback",
            "trim_applied": True,
        }
        return output


class OpenAICompatReasonGenerator:
    """
    OpenAI 호환 LLM API를 사용하여 각 후보자의 추천 사유와 적합도를 생성(추론)하는 클래스입니다.
    검색된 증거(논문, 특허 등)와 검색 엔진 랭킹 정보를 기반으로 다량의 후보자를 한 번에 평가합니다.
    """

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
            f"You must answer via the `{REASON_TOOL_NAME}` tool exactly once."
            if use_tools
            else 'You must return only one JSON object shaped like {"items":[...],"data_gaps":[...]}.'
        )
        prompt = f"""
        You summarize recommendation reasons for a ranked candidate batch.

        Candidate data fields:
        - `selected_evidence`: direct evidence items (papers, patents, projects) matched to the query.
        - `retrieval_grounding`: retrieval ranking signals — `primary_branch` (the index branch
          where this candidate ranked highest), `final_score` (RRF aggregated score), and
          `branch_matches` (top-2 branch-level ranks). Higher score = stronger multi-branch support.

        Your role is summarization only.
        - Do not choose evidence.
        - Do not invent evidence ids.
        - Do not mention any string listed in `do_not_mention`.
        - Base `recommendation_reason` primarily on `selected_evidence` (concrete titles, findings).
        - Use `retrieval_grounding` to inform the `fit` level and to note retrieval breadth
          when evidence alone is insufficient. For example, if branch_matches span multiple
          branches (art + pat + pjt), you may note "다수 연구 영역에서 연관성이 확인됨"
          or similar. Do NOT quote raw numeric scores or rank numbers in the reason text.

        `fit` assignment guide (use retrieval_grounding.final_score as a signal):
        - {FIT_HIGH}: strong direct evidence covering most query aspects AND/OR high final_score
          with matches across 3+ branches.
        - {FIT_MEDIUM}: partial evidence match OR matches concentrated in 1-2 branches.
        - {FIT_NORMAL}: weak or indirect evidence, or limited branch coverage.

        Output rules:
        - `fit` must be one of: {FIT_HIGH}, {FIT_MEDIUM}, {FIT_NORMAL}
        - `recommendation_reason` must be 1-2 sentences and under 320 characters.
        - If direct evidence is weak but retrieval_grounding shows broad multi-branch support,
          write a measured reason acknowledging retrieval breadth without overstating expertise.
        - If both evidence and retrieval_grounding are weak, return a cautious reason or an empty string.
        - `risks` should be short factual caveats only.

        {output_instruction}
        """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _build_reason_tools() -> list[dict[str, Any]]:
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
                                        "risks": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": [
                                        "expert_id",
                                        "fit",
                                        "recommendation_reason",
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
        bundle: RelevantEvidenceBundle,
        *,
        limit: int,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        selected_items = bundle.all_items()[:limit]
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
                "aspect_matches": list(item.aspect_matches),
                "direct_match": item.direct_match,
            }
            for item in selected_items
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
        """
        LLM에게 전달할 후보자 목록을 프롬프트 예산(profile)에 맞게 직렬화(JSON 구조화)합니다.
        불필요한 글자수를 자르거나(Truncate), 검색된 증거(evidence) 중 상위 항목만 선별하여 토큰 초과를 방지합니다.
        """
        serialized: list[dict[str, Any]] = []
        relevant_evidence_by_expert_id = relevant_evidence_by_expert_id or {}
        retrieval_score_traces_by_expert_id = retrieval_score_traces_by_expert_id or {}
        candidate_names = {
            candidate.expert_id: _truncate_text(
                candidate.name, profile["name_char_limit"]
            )
            or candidate.expert_id
            for candidate in candidates
        }
        evidence_ids = sorted(
            {
                item.item_id
                for bundle in relevant_evidence_by_expert_id.values()
                for item in bundle.all_items()
            }
        )

        for candidate in candidates:
            relevant_bundle = relevant_evidence_by_expert_id.get(
                candidate.expert_id,
                RelevantEvidenceBundle(expert_id=candidate.expert_id),
            )
            selected_evidence = cls._serialize_evidence_items(
                relevant_bundle,
                limit=profile["selected_limit"],
                profile=profile,
            )
            do_not_mention = sorted(
                {
                    *[
                        name
                        for expert_id, name in candidate_names.items()
                        if expert_id != candidate.expert_id and name
                    ],
                    *evidence_ids,
                }
            )
            serialized.append(
                {
                    "expert_id": candidate.expert_id,
                    "candidate_name": candidate_names[candidate.expert_id],
                    "organization": _truncate_text(
                        candidate.organization, profile["detail_char_limit"]
                    ),
                    "position": _truncate_text(
                        candidate.position, profile["detail_char_limit"]
                    ),
                    "degree": _truncate_text(
                        candidate.degree, profile["detail_char_limit"]
                    ),
                    "major": _truncate_text(
                        candidate.major, profile["detail_char_limit"]
                    ),
                    "rank_score": candidate.rank_score,
                    "shortlist_score": candidate.shortlist_score,
                    "retrieval_grounding": cls._compact_retrieval_grounding(
                        retrieval_score_traces_by_expert_id.get(candidate.expert_id, {})
                    ),
                    "selected_evidence": selected_evidence,
                    "selected_evidence_summary": {
                        "direct_match_count": relevant_bundle.direct_match_count,
                        "aspect_coverage": relevant_bundle.aspect_coverage,
                        "matched_aspects": list(relevant_bundle.matched_aspects),
                        "generic_only": relevant_bundle.generic_only,
                    },
                    "do_not_mention": do_not_mention,
                }
            )
        return serialized

    @staticmethod
    def _normalize_output(
        output: ReasonGenerationOutput,
        candidates: list[CandidateCard],
    ) -> tuple[ReasonGenerationOutput, dict[str, Any]]:
        """
        LLM이 반환한 응답(ReasonGenerationOutput)을 정규화합니다.
        누락된 후보자가 있거나 잘못된 적합도 값이 들어온 경우 기본값으로 보정합니다.
        """
        by_expert_id = {item.expert_id: item for item in output.items}
        normalized_items: list[ReasonedCandidate] = []
        returned_ids = [item.expert_id for item in output.items]
        missing_candidate_ids: list[str] = []
        empty_reason_candidate_ids: list[str] = []

        for candidate in candidates:
            item = by_expert_id.get(candidate.expert_id)
            if item is None:
                missing_candidate_ids.append(candidate.expert_id)
                empty_reason_candidate_ids.append(candidate.expert_id)
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
            normalized_fit = item.fit if item.fit in FIT_VALUES else FIT_NORMAL
            if not normalized_reason:
                empty_reason_candidate_ids.append(candidate.expert_id)

            normalized_items.append(
                ReasonedCandidate(
                    expert_id=candidate.expert_id,
                    fit=normalized_fit,
                    recommendation_reason=normalized_reason,
                    selected_evidence_ids=[],
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

        return (
            ReasonGenerationOutput(items=normalized_items, data_gaps=output.data_gaps),
            {
                "returned_ids": returned_ids,
                "missing_candidate_ids": missing_candidate_ids,
                "empty_reason_candidate_ids": empty_reason_candidate_ids,
                "empty_selected_evidence_candidate_ids": [],
                "invalid_selected_evidence_candidate_ids": [],
                "invalid_selected_evidence_ids_by_candidate": {},
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

    @staticmethod
    def _build_partial_retry_decision(
        trace: dict[str, Any],
    ) -> tuple[bool, str | None, str | None]:
        missing_candidate_ids = list(trace.get("missing_candidate_ids", []))
        empty_reason_candidate_ids = list(trace.get("empty_reason_candidate_ids", []))
        returned_ratio = float(trace.get("returned_ratio", 0.0) or 0.0)

        if missing_candidate_ids:
            return (
                True,
                "missing_candidate_ids",
                f"missing_candidate_ids={missing_candidate_ids} returned_ratio={returned_ratio:.3f}",
            )
        if empty_reason_candidate_ids:
            return (
                True,
                "empty_reason_candidate_ids",
                f"empty_reason_candidate_ids={empty_reason_candidate_ids} returned_ratio={returned_ratio:.3f}",
            )
        if returned_ratio < 1.0:
            return (
                True,
                "partial_return_ratio",
                f"returned_ratio={returned_ratio:.3f}",
            )
        return False, None, None

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
        """
        주어진 프로필(글자수 제한 등)과 도구(Tool) 사용 여부에 따라 LLM API를 한 번 호출합니다.
        호출 결과를 파싱하고 정규화한 뒤, 추적용 메타데이터(trace)와 함께 반환합니다.
        """
        serialized_candidates = self._serialize_candidates(
            candidates,
            relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
            retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
            profile=profile,
        )
        payload = {
            "query": query,
            "intent_summary": plan.intent_summary,
            "must_aspects": list(plan.must_aspects),
            "generic_terms": list(plan.generic_terms),
            "candidates": serialized_candidates,
        }
        payload_text = json.dumps(payload, ensure_ascii=False)
        invoke_kwargs = build_consistency_invoke_kwargs(
            max_tokens_hint=REASON_GENERATION_MAX_TOKENS,
            seed=seed,
        )
        if use_tools:
            invoke_kwargs["tools"] = self._build_reason_tools()
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
        returned_ratio = (
            round(len(returned_ids) / candidate_count, 3) if candidate_count else 0.0
        )
        selected_evidence_count = sum(
            len(candidate.get("selected_evidence", []))
            for candidate in serialized_candidates
        )
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
            "selected_evidence_count": selected_evidence_count,
            **normalization_trace,
        }
        logger.info(
            "Reason generation attempt completed: mode=%s candidate_count=%d selected_evidence_count=%d returned_ratio=%.3f",
            parse_mode,
            candidate_count,
            selected_evidence_count,
            returned_ratio,
        )
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
        """
        후보자 목록에 대한 추천 사유 생성을 시도합니다.
        기본적으로 풍부한 데이터(PRIMARY_PAYLOAD_PROFILE)로 시도하고,
        일부 응답 누락 시 글자수를 줄인 재시도 프로필(RETRY_PAYLOAD_PROFILE)로 한 번 더 시도합니다.
        최종 실패 시 PassThrough(폴백) 로직을 탑니다.
        """
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
        retry_triggered = False
        retry_trigger: str | None = None
        retry_reason: str | None = None

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
                should_retry, current_retry_trigger, current_retry_reason = (
                    self._build_partial_retry_decision(trace)
                )
                if should_retry and retry_index < len(attempt_specs) - 1:
                    retry_triggered = True
                    retry_trigger = current_retry_trigger
                    retry_reason = current_retry_reason
                    logger.warning(
                        "Reason generator scheduling compact retry: trigger=%s reason=%s",
                        current_retry_trigger,
                        current_retry_reason,
                    )
                    attempt_history.append(
                        {
                            "mode": trace.get("mode", "unknown"),
                            "retry_index": retry_index,
                            "prompt_budget_mode": attempt_spec["profile"]["name"],
                            "trim_applied": bool(
                                attempt_spec["profile"]["trim_applied"]
                            ),
                            "status": "retry_scheduled",
                            "retry_trigger": current_retry_trigger,
                            "reason": current_retry_reason,
                            "returned_ids": list(trace.get("returned_ids", [])),
                            "missing_candidate_ids": list(
                                trace.get("missing_candidate_ids", [])
                            ),
                            "empty_reason_candidate_ids": list(
                                trace.get("empty_reason_candidate_ids", [])
                            ),
                            "returned_ratio": trace.get("returned_ratio", 0.0),
                        }
                    )
                    continue
                trace["retry_count"] = retry_index
                trace["retry_triggered"] = retry_triggered
                trace["retry_trigger"] = retry_trigger
                trace["retry_reason"] = retry_reason
                trace["attempts"] = [
                    *attempt_history,
                    {
                        "mode": trace.get("mode", "unknown"),
                        "retry_index": retry_index,
                        "prompt_budget_mode": attempt_spec["profile"]["name"],
                        "trim_applied": bool(attempt_spec["profile"]["trim_applied"]),
                        "status": "ok",
                        "retry_trigger": retry_trigger,
                        "reason": retry_reason,
                        "returned_ids": list(trace.get("returned_ids", [])),
                        "missing_candidate_ids": list(
                            trace.get("missing_candidate_ids", [])
                        ),
                        "empty_reason_candidate_ids": list(
                            trace.get("empty_reason_candidate_ids", [])
                        ),
                        "returned_ratio": trace.get("returned_ratio", 0.0),
                    },
                ]
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
                        "status": "error",
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
            "retry_triggered": retry_triggered,
            "retry_trigger": retry_trigger,
            "retry_reason": retry_reason,
            "reason": "; ".join(
                attempt.get("reason", "")
                for attempt in attempt_history
                if attempt.get("reason")
            ),
            "returned_ids": list(fallback_trace.get("returned_ids", [])),
            "missing_candidate_ids": list(
                fallback_trace.get("missing_candidate_ids", [])
            ),
            "empty_reason_candidate_ids": list(
                fallback_trace.get("empty_reason_candidate_ids", [])
            ),
            "empty_selected_evidence_candidate_ids": [],
            "invalid_selected_evidence_candidate_ids": [],
            "invalid_selected_evidence_ids_by_candidate": {},
            "attempts": attempt_history,
        }
        return fallback_output
