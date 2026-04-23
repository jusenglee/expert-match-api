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
from apps.search.expansion_lexicon import EXPANSION_LEXICON, get_lexicon_summary

logger = logging.getLogger(__name__)

MAX_PLANNER_ATTEMPTS = 2


PLANNER_VERSION = "v0.4.0"


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


def _normalize_string_list(values: list[str] | None) -> list[str]:
    normalized_values: list[str] = []
    for value in values or []:
        normalized = " ".join(str(value).split())
        if normalized and normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


def _sorted_filter_keys(filters: dict[str, Any] | None) -> list[str]:
    return sorted((filters or {}).keys())


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
        logger.info(
            "플래너 내부 시작: mode=heuristic query_chars=%d top_k=%s include_orgs=%d exclude_orgs=%d filter_keys=%s",
            len(normalized_query),
            top_k,
            len(include_orgs or []),
            len(exclude_orgs or []),
            _sorted_filter_keys(filters_override),
        )
        output = PlannerOutput(
            intent_summary=normalized_query,
            hard_filters=dict(filters_override or {}),
            include_orgs=list(include_orgs or []),
            exclude_orgs=list(exclude_orgs or []),
            task_terms=[],
            core_keywords=[],
            retrieval_core=[],
            role_terms=[],
            action_terms=[],
            top_k=top_k or 15,
        )
        self.last_trace = {
            "mode": "deterministic_fallback",
            "normalized_query": normalized_query,
            "planner_retry_count": 0,
            "planner_keywords": [],
            "retrieval_keywords": [],
            "attempts": [],
        }
        logger.info(
            "플래너 내부 완료: mode=heuristic keywords=0 filters=%s top_k=%d",
            _sorted_filter_keys(output.hard_filters),
            output.top_k,
        )
        return output


class OpenAICompatPlanner:
    """LLM-backed planner that extracts pure retrieval keywords and explicit request terms."""

    def __init__(self, settings: Settings, cache: PlanCache | None = None) -> None:
        self.settings = settings
        self.fallback = HeuristicPlanner()
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self.cache = cache
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _build_system_prompt() -> str:
        lexicon_summary = get_lexicon_summary()
        prompt = f"""
                # 역할
                당신은 ***동질적인 전문가 코퍼스***를 검색하는 전문가 추천 시스템의 R&D 질의 플래너입니다.
                당신은 전문가/평가위원을 모아놓은 qdrant 벡터DB 에 검색 할 쿼리를 만들기 위해, 사용자의 질의를 분석하고 정규화해야 합니다.
            
                # 출력 목표
                - `retrieval_core`: 실제 기술/도메인 매칭에 필요한 핵심 키워드 리스트(Sparse/Keyword 검색용). "평가위원", "전문가" 등 역할어는 제외하세요.
                - `bundle_ids`: 질의의 기술적 맥락을 확장하기 위해 아래 [확장 번들 목록]에서 가장 적합한 ID들을 선택하세요. (없으면 빈 리스트)
                - `semantic_query`: 검색 의도를 담은 자연어 문장(Vector 검색용). 핵심 기술 키워드와 맥락을 포함하세요.
                - `role_terms`: "평가위원", "교수", "전문가" 등 검색 대상의 페르소나/역할 용어 리스트.
                - `action_terms`: "추천", "찾아줘", "선정해줘" 등 사용자가 요청한 행동 용어 리스트.
                - `intent_flags`: 검색 의도에 대한 플래그 (예: "need_experience": true, "prefer_recent": true 등).
                - `intent_summary`: UI/추적용 짧은 요약 문장.
            
                # [확장 번들 목록]
                {lexicon_summary}

                # 규칙
                1. 사용자 질의의 주 언어를 유지하세요. 번역하거나 언어를 섞지 마세요.
                2. `retrieval_core`에는 도메인 개념, 기술, 재료, 분야만 포함해야 합니다. 코퍼스에 공통적으로 나타나는 "전문가", "추천" 등은 여기에 넣지 마세요.
                3. `bundle_ids`는 반드시 위에 제공된 [확장 번들 목록]의 ID 중에서만 선택하세요.
                4. `role_terms`와 `action_terms`는 검색어(Query)가 아니라 제어 신호로 활용됩니다.
                5. 명시적으로 주어진 기관 제약만 `include_orgs`와 `exclude_orgs`에 복사하세요.
                6. 명시적으로 지원되는 구조화 필터만 `hard_filters`에 복사하세요.
                7. 안전한 도메인 키워드가 없으면 `retrieval_core`는 빈 리스트로 반환하세요.
                8. JSON만 반환하세요. 마크다운 펜스, 설명문, 숨겨진 추론은 포함하지 마세요.
                
                # 출력 스키마
                {{
                  "intent_summary": "string",
                  "retrieval_core": ["string"],
                  "bundle_ids": ["string"],
                  "semantic_query": "string",
                  "role_terms": ["string"],
                  "action_terms": ["string"],
                  "intent_flags": {{}},
                  "hard_filters": {{}},
                  "include_orgs": ["string"],
                  "exclude_orgs": ["string"],
                  "top_k": integer
                }}
            
                # 예시
                Input:
                {{
                  "query": "난접근성 화재 진압에서 드론을 접목하려고해. 드론을 화재진압 연구에 사용한 경험이 있는 관련된 전문가를 5명 추천해줘",
                  "filters_override": {{}},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5
                }}
            
                Output:
                {{
                  "intent_summary": "난접근성 화재 진압과 드론 접목 관련 전문가 탐색",
                  "retrieval_core": ["난접근성 화재 진압", "드론"],
                  "bundle_ids": ["uav", "fire_response"],
                  "semantic_query": "난접근성 화재 현장의 화재 진압을 위한 드론 및 무인 로봇 활용 연구 전문가",
                  "role_terms": ["전문가"],
                  "action_terms": ["추천"],
                  "intent_flags": {{ "need_experience": true }},
                  "hard_filters": {{}},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5
                }}
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
        output.retrieval_core = _normalize_string_list(output.retrieval_core)
        output.role_terms = _normalize_string_list(output.role_terms)
        output.action_terms = _normalize_string_list(output.action_terms)
        
        # 번들 ID 유효성 검사 (존재하지 않는 ID는 제거)
        output.bundle_ids = [bid for bid in _normalize_string_list(output.bundle_ids) if bid in EXPANSION_LEXICON]

        # Backward compatibility mapping
        output.core_keywords = list(output.retrieval_core)
        output.task_terms = _normalize_string_list(output.role_terms + output.action_terms)

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
        payload: dict[str, Any],
        seed: int,
    ) -> tuple[PlannerOutput, dict[str, Any], str]:
        invoke_kwargs = build_consistency_invoke_kwargs(seed=seed)
        result = await self.model.ainvoke_non_stream(
            [
                SystemMessage(content=self._build_system_prompt()),
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
        filters = filters_override or {}
        logger.info(
            "플래너 내부 시작: mode=openai_compat query_chars=%d top_k=%s include_orgs=%d exclude_orgs=%d filter_keys=%s",
            len(normalized_query),
            top_k,
            len(include_orgs or []),
            len(exclude_orgs or []),
            _sorted_filter_keys(filters),
        )

        # 1. L1 캐시 조회
        if self.cache:
            cached_output = self.cache.get(normalized_query, filters, PLANNER_VERSION)
            if cached_output:
                logger.info(
                    "플래너 캐시 적중: version=%s query_chars=%d filter_keys=%s",
                    PLANNER_VERSION,
                    len(normalized_query),
                    _sorted_filter_keys(filters),
                )
                output = self._apply_request_constraints(
                    output=cached_output,
                    normalized_query=normalized_query,
                    filters_override=filters_override,
                    include_orgs=include_orgs,
                    exclude_orgs=exclude_orgs,
                    top_k=top_k,
                )
                self.last_trace = {
                    "mode": "cache_hit",
                    "cache": {"canonical_plan": "hit"},
                    "planner_version": PLANNER_VERSION,
                    "normalized_query": normalized_query,
                    "planner_keywords": list(output.retrieval_core),
                    "retrieval_keywords": list(output.retrieval_core),
                    "removed_role_terms": list(output.role_terms + output.action_terms),
                }
                logger.info(
                    "플래너 내부 완료: mode=cache_hit keywords=%d role_terms=%d action_terms=%d filters=%s top_k=%d",
                    len(output.retrieval_core),
                    len(output.role_terms),
                    len(output.action_terms),
                    _sorted_filter_keys(output.hard_filters),
                    output.top_k,
                )
                return output

        # 2. 캐시 미스 시 LLM 호출
        payload = {
            "query": normalized_query,
            "filters_override": filters,
            "include_orgs": include_orgs or [],
            "exclude_orgs": exclude_orgs or [],
            "top_k": top_k or 15,
        }
        attempts: list[dict[str, Any]] = []

        for attempt_index in range(MAX_PLANNER_ATTEMPTS):
            seed = build_deterministic_seed("planner", payload, attempt_index)
            logger.info(
                "플래너 LLM 시도 시작: attempt=%d seed=%d query_chars=%d filter_keys=%s",
                attempt_index + 1,
                seed,
                len(normalized_query),
                _sorted_filter_keys(filters),
            )
            try:
                output, parsed_payload, raw_response = await self._invoke_json_output(
                    payload=payload,
                    seed=seed,
                )
                output = self._apply_request_constraints(
                    output=output,
                    normalized_query=normalized_query,
                    filters_override=filters_override,
                    include_orgs=include_orgs,
                    exclude_orgs=exclude_orgs,
                    top_k=top_k,
                )
                attempt_trace = {
                    "attempt": attempt_index + 1,
                    "seed": seed,
                    "status": "ok",
                    "raw_response": raw_response,
                    "parsed_json": parsed_payload,
                    "planner_keywords": list(output.retrieval_core),
                }
                attempts.append(attempt_trace)
                logger.info(
                    "플래너 LLM 시도 완료: attempt=%d status=ok keywords=%d role_terms=%d action_terms=%d bundles=%d filters=%s",
                    attempt_index + 1,
                    len(output.retrieval_core),
                    len(output.role_terms),
                    len(output.action_terms),
                    len(output.bundle_ids),
                    _sorted_filter_keys(output.hard_filters),
                )

                if output.retrieval_core:
                    self.last_trace = {
                        "mode": "openai_compat",
                        "cache": {"canonical_plan": "miss"},
                        "planner_version": PLANNER_VERSION,
                        "normalized_query": normalized_query,
                        "planner_retry_count": attempt_index,
                        "planner_keywords": list(output.retrieval_core),
                        "retrieval_keywords": list(output.retrieval_core),
                        "removed_role_terms": list(output.role_terms + output.action_terms),
                        "attempts": attempts,
                    }
                    
                    # 결과 캐싱
                    if self.cache:
                        self.cache.set(normalized_query, filters, PLANNER_VERSION, output)

                    logger.info(
                        "플래너 내부 완료: mode=openai_compat intent=%r keywords=%d semantic_query=%s include_orgs=%d exclude_orgs=%d filters=%s top_k=%d",
                        output.intent_summary,
                        len(output.retrieval_core),
                        bool(output.semantic_query),
                        len(output.include_orgs),
                        len(output.exclude_orgs),
                        _sorted_filter_keys(output.hard_filters),
                        output.top_k,
                    )
                    return output

                attempt_trace["status"] = "empty_keywords"
                attempt_trace["reason"] = "planner_retrieval_core_empty"
                logger.warning(
                    "플래너 LLM 시도 결과 키워드 없음: attempt=%d query_chars=%d",
                    attempt_index + 1,
                    len(normalized_query),
                )
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "seed": seed,
                        "status": "error",
                        "reason": str(exc),
                    }
                )
                logger.warning(
                    "플래너 LLM 시도 실패: attempt=%d reason=%s",
                    attempt_index + 1,
                    exc,
                )

        # 3. 모든 시도 실패 시 Fallback 1: 역할어 제거 후 Broad Search
        logger.warning(
            "플래너 fallback 활성화: query_chars=%d attempts=%d",
            len(normalized_query),
            len(attempts),
        )
        
        # 마지막 시도에서 역할어 추출 정보가 있다면 활용
        last_role_terms = []
        last_action_terms = []
        for att in reversed(attempts):
            if "parsed_json" in att:
                last_role_terms = att["parsed_json"].get("role_terms", [])
                last_action_terms = att["parsed_json"].get("action_terms", [])
                break

        fallback_keywords = []
        clean_query = normalized_query
        for term in (last_role_terms + last_action_terms):
            clean_query = clean_query.replace(term, " ")
        
        # 남은 단어들을 키워드로 사용
        fallback_keywords = [kw.strip() for kw in clean_query.split() if len(kw.strip()) > 1]

        fallback_output = PlannerOutput(
            intent_summary=f"[Fallback] {normalized_query}",
            hard_filters=dict(filters),
            include_orgs=list(include_orgs or []),
            exclude_orgs=list(exclude_orgs or []),
            retrieval_core=fallback_keywords,
            core_keywords=fallback_keywords, # 하위 호환
            role_terms=last_role_terms,
            action_terms=last_action_terms,
            top_k=top_k or 15,
        )

        self.last_trace = {
            "mode": "fallback_broad_search",
            "cache": {"canonical_plan": "miss"},
            "normalized_query": normalized_query,
            "planner_retry_count": max(0, len(attempts) - 1),
            "planner_keywords": fallback_keywords,
            "retrieval_keywords": fallback_keywords,
            "removed_role_terms": list(last_role_terms + last_action_terms),
            "reason": "planner_retry_exhausted_or_empty",
            "attempts": attempts,
        }
        logger.info(
            "플래너 내부 완료: mode=fallback_broad_search keywords=%d removed_terms=%d filters=%s top_k=%d",
            len(fallback_keywords),
            len(last_role_terms + last_action_terms),
            _sorted_filter_keys(fallback_output.hard_filters),
            fallback_output.top_k,
        )
        return fallback_output
