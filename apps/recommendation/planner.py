"""
Planner implementations for recommendation search intent extraction.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from apps.core.config import Settings
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text
from apps.core.llm_policies import build_consistency_invoke_kwargs
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.core.utils import build_deterministic_seed
from apps.domain.models import PlannerOutput

logger = logging.getLogger(__name__)

MAX_PLANNER_ATTEMPTS = 2


class Planner(Protocol):
    async def plan(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput: ...


def _normalize_string_list(values: list[str]) -> list[str]:
    normalized_values: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split())
        if normalized and normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


class HeuristicPlanner:
    """Deterministic fallback planner used when LLM planning is unavailable."""

    def __init__(self) -> None:
        self.last_trace: dict[str, Any] = {}

    async def plan(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput:
        normalized_query = " ".join(query.split())
        output = PlannerOutput(
            intent_summary=normalized_query,
            hard_filters=dict(filters_override or {}),
            include_orgs=list(include_orgs or []),
            exclude_orgs=list(exclude_orgs or []),
            soft_preferences=[],
            task_terms=[],
            core_keywords=[],
            branch_query_hints={},
            top_k=top_k or 15,
        )
        self.last_trace = {
            "mode": "deterministic_fallback",
            "normalized_query": normalized_query,
            "planner_retry_count": 0,
            "planner_raw_keywords": [],
            "verifier_keywords": [],
            "retrieval_keywords": [],
            "verifier_applied": False,
            "attempts": [],
        }
        return output


class OpenAICompatPlanner:
    """LLM-backed planner with deterministic request shaping."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fallback = HeuristicPlanner()
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _build_planner_prompt() -> str:
        prompt = """
                # 역할
                당신은 동질적인 전문가 코퍼스를 검색하는 전문가 추천 시스템의 R&D 질의 플래너입니다.
                검색에 안전한 도메인 정보만 추출하고, 검색 대상이 아닌 요청어는 분리하세요.
            
                # 출력 목표
                - `core_keywords`: 질문에서 대화형 어미를 제거하고, 논문, 과제, 특허 검색에 적합한 핵심 기술 및 도메인 명사 10개 이내의 나열로 변환
                - `task_terms`: 검색 대상이 아닌 요청, 역할, 또는 행동 용어만 포함
                - `intent_summary`: UI/추적용 짧은 요약 문장
            
                # 규칙
                1. 사용자 질의의 주 언어를 유지하세요. 번역하거나 언어를 섞지 마세요.
                2. 브랜치별 힌트, 바꿔쓰기, 의미 확장 표현을 생성하지 마세요.
                3. `core_keywords`에는 도메인 개념, 기술, 재료, 분야, 시스템, 방법만 포함해야 합니다.
                4. `task_terms`에는 무엇을 검색할지가 아니라, 시스템이 무엇을 해야 하는지를 나타내는 요청/역할/행동 단어를 담아야 합니다.(전문가, 평가위원 추천 등)
                5. 명시적으로 주어진 기관 제약만 `include_orgs`와 `exclude_orgs`에 복사하세요.
                6. 명시적으로 지원되는 구조화 필터만 `hard_filters`에 복사하세요.
                7. 안전한 도메인 키워드가 없으면 `core_keywords`는 빈 리스트로 반환하세요.
                8. JSON만 반환하세요. 마크다운 펜스, 설명문, 숨겨진 추론은 포함하지 마세요.
            
                # 출력 스키마
                {
                  "intent_summary": "string",
                  "core_keywords": ["string"],
                  "task_terms": ["string"],
                  "hard_filters": {},
                  "include_orgs": ["string"],
                  "exclude_orgs": ["string"],
                  "soft_preferences": ["string"],
                  "top_k": 15
                }
            
                # 예시
                Input:
                {
                  "query": "난접근성 화재 진압에서 드론을 접목하려고해. 관련된 전문가를 추천해줘",
                  "filters_override": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5
                }
            
                Output:
                {
                  "intent_summary": "난접근성 화재 진압과 드론 접목 관련 전문가 탐색",
                  "core_keywords": ["난접근성 화재 진압", "드론", "산불 진압 드론", "재난 대응 로봇"],
                  "task_terms": ["전문가", "추천", "평가위원"],
                  "hard_filters": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "soft_preferences": [],
                  "top_k": 5
                }
            """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _build_verifier_prompt() -> str:
        prompt = """
            # Role
            You verify whether a planner output is safe to use for retrieval in a homogeneous expert corpus.

            # Task
            Rewrite the planner output into the same JSON schema while enforcing:
            - `core_keywords` must contain retrieval-safe domain nouns or noun phrases only
            - `task_terms` must contain request-role or action terms only
            - do not translate, paraphrase, or invent new domain keywords
            - move non-retrieval terms out of `core_keywords`
            - if no safe domain keyword remains, return `core_keywords: []`
            - preserve explicit filters and org constraints
            - return JSON only

            # Output Schema
            {
              "intent_summary": "string",
              "core_keywords": ["string"],
              "task_terms": ["string"],
              "hard_filters": {},
              "include_orgs": ["string"],
              "exclude_orgs": ["string"],
              "soft_preferences": ["string"],
              "top_k": 15
            }
        """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _apply_request_constraints(
        *,
        output: PlannerOutput,
        normalized_query: str,
        filters_override: dict[str, Any] | None,
        include_orgs: list[str] | None,
        exclude_orgs: list[str] | None,
        top_k: int | None,
    ) -> PlannerOutput:
        output.intent_summary = " ".join(output.intent_summary.split()) or normalized_query
        output.task_terms = _normalize_string_list(output.task_terms)
        output.core_keywords = _normalize_string_list(output.core_keywords)
        output.branch_query_hints = {}

        if filters_override:
            merged_filters = dict(output.hard_filters)
            merged_filters.update(filters_override)
            output.hard_filters = merged_filters

        if include_orgs:
            for organization in include_orgs:
                if organization not in output.include_orgs:
                    output.include_orgs.append(organization)

        if exclude_orgs:
            for organization in exclude_orgs:
                if organization not in output.exclude_orgs:
                    output.exclude_orgs.append(organization)

        if top_k is not None:
            output.top_k = top_k

        return output

    async def _invoke_json_output(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        seed: int,
    ) -> tuple[PlannerOutput, dict[str, Any], str]:
        invoke_kwargs = build_consistency_invoke_kwargs(seed=seed)
        result = await self.model.ainvoke_non_stream(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ],
            **invoke_kwargs,
        )
        json_text = _extract_json_object_text(result.content)
        parsed_payload = json.loads(json_text)
        output = PlannerOutput.model_validate(parsed_payload)
        return output, parsed_payload, result.content

    async def plan(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput:
        normalized_query = " ".join(query.split())
        payload = {
            "query": normalized_query,
            "filters_override": filters_override or {},
            "include_orgs": include_orgs or [],
            "exclude_orgs": exclude_orgs or [],
            "top_k": top_k or 15,
        }
        attempts: list[dict[str, Any]] = []

        for attempt_index in range(MAX_PLANNER_ATTEMPTS):
            planner_seed = build_deterministic_seed("planner", payload, attempt_index)
            logger.info(
                "LLM planning start: query=%s attempt=%d",
                normalized_query,
                attempt_index + 1,
            )
            try:
                planner_output, planner_parsed, planner_raw_response = (
                    await self._invoke_json_output(
                        system_prompt=self._build_planner_prompt(),
                        payload=payload,
                        seed=planner_seed,
                    )
                )
                planner_output = self._apply_request_constraints(
                    output=planner_output,
                    normalized_query=normalized_query,
                    filters_override=filters_override,
                    include_orgs=include_orgs,
                    exclude_orgs=exclude_orgs,
                    top_k=top_k,
                )
                verifier_seed = build_deterministic_seed(
                    "planner_verifier",
                    payload,
                    attempt_index,
                    planner_output.model_dump(mode="json"),
                )

                verifier_payload = {
                    "query": normalized_query,
                    "planner_output": planner_output.model_dump(mode="json"),
                }
                verified_output, verifier_parsed, verifier_raw_response = (
                    await self._invoke_json_output(
                        system_prompt=self._build_verifier_prompt(),
                        payload=verifier_payload,
                        seed=verifier_seed,
                    )
                )
                verified_output = self._apply_request_constraints(
                    output=verified_output,
                    normalized_query=normalized_query,
                    filters_override=filters_override,
                    include_orgs=include_orgs,
                    exclude_orgs=exclude_orgs,
                    top_k=top_k,
                )

                attempt_trace = {
                    "attempt": attempt_index + 1,
                    "planner_seed": planner_seed,
                    "verifier_seed": verifier_seed,
                    "status": "ok",
                    "planner_raw_response": planner_raw_response,
                    "planner_parsed_json": planner_parsed,
                    "planner_raw_keywords": list(planner_output.core_keywords),
                    "verifier_raw_response": verifier_raw_response,
                    "verifier_parsed_json": verifier_parsed,
                    "verifier_keywords": list(verified_output.core_keywords),
                    "verifier_applied": True,
                }
                attempts.append(attempt_trace)

                if verified_output.core_keywords:
                    self.last_trace = {
                        "mode": "openai_compat",
                        "normalized_query": normalized_query,
                        "planner_retry_count": attempt_index,
                        "planner_raw_keywords": list(planner_output.core_keywords),
                        "verifier_keywords": list(verified_output.core_keywords),
                        "retrieval_keywords": list(verified_output.core_keywords),
                        "verifier_applied": True,
                        "planner_raw_response": planner_raw_response,
                        "planner_parsed_json": planner_parsed,
                        "verifier_raw_response": verifier_raw_response,
                        "verifier_parsed_json": verifier_parsed,
                        "attempts": attempts,
                    }
                    logger.info(
                        "LLM planning success: intent=%r keywords=%d include_orgs=%d exclude_orgs=%d filters=%r",
                        verified_output.intent_summary,
                        len(verified_output.core_keywords),
                        len(verified_output.include_orgs),
                        len(verified_output.exclude_orgs),
                        verified_output.hard_filters,
                    )
                    return verified_output

                attempt_trace["status"] = "empty_keywords"
                attempt_trace["reason"] = "verified_core_keywords_empty"
                logger.warning(
                    "Verifier returned empty core_keywords: query=%s attempt=%d",
                    normalized_query,
                    attempt_index + 1,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "planner_seed": planner_seed,
                        "verifier_seed": None,
                        "status": "error",
                        "reason": str(exc),
                    }
                )
                logger.warning(
                    "Planner/verifier attempt failed: query=%s attempt=%d reason=%s",
                    normalized_query,
                    attempt_index + 1,
                    exc,
                )

        logger.warning(
            "Planner fallback activated after retries: query=%s attempts=%d",
            normalized_query,
            len(attempts),
        )
        fallback_output = await self.fallback.plan(
            query=query,
            filters_override=filters_override,
            include_orgs=include_orgs,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
        )

        last_planner_keywords = []
        last_verifier_keywords = []
        for attempt in reversed(attempts):
            if attempt.get("planner_raw_keywords"):
                last_planner_keywords = list(attempt["planner_raw_keywords"])
                break
        for attempt in reversed(attempts):
            if attempt.get("verifier_keywords"):
                last_verifier_keywords = list(attempt["verifier_keywords"])
                break

        self.last_trace = {
            "mode": "deterministic_fallback",
            "normalized_query": normalized_query,
            "planner_retry_count": max(0, len(attempts) - 1),
            "planner_raw_keywords": last_planner_keywords,
            "verifier_keywords": last_verifier_keywords,
            "retrieval_keywords": [],
            "verifier_applied": bool(attempts),
            "reason": "planner_verifier_retry_exhausted",
            "attempts": attempts,
        }
        return fallback_output
