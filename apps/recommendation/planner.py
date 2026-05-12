"""
사용자의 자연어 질의에서 검색 의도(Intent)를 분석하고, 최적의 검색 쿼리를 설계하는 Planner 모듈입니다.

[Architecture Overview]
- 사용자가 "화재 관련 컨소시엄 전문가 추천해줘" 라고 입력했을 때, 
  "화재", "컨소시엄"과 같은 기술 도메인 키워드(`retrieval_core`)만을 추출하여 엔진이 노이즈 없이 검색할 수 있게 돕습니다.
- "전문가", "추천해줘"와 같은 역할어/행동어는 `role_terms`, `action_terms`로 분리하여 검색어에서 배제합니다.
- 동의어 확장 번들(Lexicon)을 참조하여, "드론" 이라는 단어가 들어오면 "UAV", "무인기" 등 확장 ID(`bundle_ids`)를 추천합니다.
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


def _filter_summary(filters: dict[str, Any] | None) -> dict[str, Any]:
    return {key: (filters or {}).get(key) for key in _sorted_filter_keys(filters)}


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
            "플래너 내부 완료: mode=heuristic retrieval_core=%s core_keywords=%s role_terms=%s action_terms=%s semantic_query=%r hard_filters=%s top_k=%d",
            output.retrieval_core,
            output.core_keywords,
            output.role_terms,
            output.action_terms,
            output.semantic_query,
            _filter_summary(output.hard_filters),
            output.top_k,
        )
        return output

class OpenAICompatPlanner:
    """
    LLM(OpenAI 호환 API)을 사용하여 사용자의 질의를 깊이 있게 분석하는 플래너입니다.
    Chain-of-Thought(CoT) 및 JSON 포맷팅을 통해 구조화된 계획(PlannerOutput)을 생성합니다.
    """

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
                2-1. **역할/행위어 누수 금지**: 한 단어를 `role_terms` 또는 `action_terms`에 넣었다면 그 단어는 `retrieval_core`에 절대 다시 넣지 마세요. "평가위원", "심사위원", "심사", "평가", "추천", "선정", "찾아줘", "교수", "전문가" 등은 사용자가 *원하는 결과의 역할/행위* 이며, 검색 대상 인물의 실적 텍스트(논문/특허/과제)에는 등장하지 않습니다.
                3. `bundle_ids`는 반드시 위에 제공된 [확장 번들 목록]의 ID 중에서만 선택하세요.
                4. `role_terms`와 `action_terms`는 검색어(Query)가 아니라 제어 신호로 활용됩니다.
                5. `include_orgs`/`exclude_orgs`는 **검색 대상 인물의 소속 기관 제약**일 때만 사용하세요.
                5-1. **대상 기관 vs 소속 기관 구분**: "X에서 수행한 과제를 심사", "X 사업 평가", "X 과제 ~" 처럼 기관 X 가 *심사/평가 대상*으로 등장하는 경우, X 는 `include_orgs`에 넣지 말고 `semantic_query`의 맥락으로만 유지하세요. "X 소속 ~", "X 출신 ~" 처럼 명시적으로 소속을 지정한 경우만 `include_orgs`에 넣습니다.
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

                # 예시 2 (대상 기관 + 역할어 처리)
                Input:
                {{
                  "query": "한국과학기술정보연구원에서 수행한 과제를 심사하기 위한 적절한 평가위원 추천",
                  "filters_override": {{}},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5
                }}

                Output:
                {{
                  "intent_summary": "한국과학기술정보연구원 수행 과제 심사용 평가위원 탐색",
                  "retrieval_core": [],
                  "bundle_ids": [],
                  "semantic_query": "한국과학기술정보연구원에서 수행한 R&D 과제를 심사할 수 있는 동일/유사 도메인 경험을 가진 평가위원",
                  "role_terms": ["평가위원"],
                  "action_terms": ["심사", "추천"],
                  "intent_flags": {{}},
                  "hard_filters": {{}},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5
                }}
                주의: 위 예시에서 "한국과학기술정보연구원"은 *심사 대상 기관* 이므로 `include_orgs`에 넣지 않습니다. "평가위원"은 `role_terms`, "심사"/"추천"은 `action_terms`이며 `retrieval_core`에 중복으로 들어가지 않습니다.
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

        # LLM 이 역할/행위 용어를 retrieval_core 에도 중복으로 흘리는 경우 차감.
        # Why: sparse 인덱스에는 "평가위원"/"심사"/"추천" 같은 *요청 의도* 단어가
        # 적재돼 있지 않아 1차 keyword stage 가 통째로 0건이 되는 사고가 있었음.
        excluded_terms = {
            term.casefold()
            for term in output.role_terms + output.action_terms
            if term
        }
        if excluded_terms:
            output.retrieval_core = [
                term
                for term in output.retrieval_core
                if term.casefold() not in excluded_terms
            ]

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
                    "플래너 내부 완료: mode=cache_hit intent=%r retrieval_core=%s core_keywords=%s role_terms=%s action_terms=%s semantic_query=%r bundle_ids=%s hard_filters=%s top_k=%d",
                    output.intent_summary,
                    output.retrieval_core,
                    output.core_keywords,
                    output.role_terms,
                    output.action_terms,
                    output.semantic_query,
                    output.bundle_ids,
                    _filter_summary(output.hard_filters),
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
                    "플래너 LLM 시도 완료: attempt=%d status=ok retrieval_core=%s core_keywords=%s role_terms=%s action_terms=%s bundle_ids=%s semantic_query=%r hard_filters=%s top_k=%d",
                    attempt_index + 1,
                    output.retrieval_core,
                    output.core_keywords,
                    output.role_terms,
                    output.action_terms,
                    output.bundle_ids,
                    output.semantic_query,
                    _filter_summary(output.hard_filters),
                    output.top_k,
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
                        "플래너 내부 완료: mode=openai_compat intent=%r retrieval_core=%s core_keywords=%s role_terms=%s action_terms=%s semantic_query=%r bundle_ids=%s include_orgs=%s exclude_orgs=%s hard_filters=%s top_k=%d",
                        output.intent_summary,
                        output.retrieval_core,
                        output.core_keywords,
                        output.role_terms,
                        output.action_terms,
                        output.semantic_query,
                        output.bundle_ids,
                        output.include_orgs,
                        output.exclude_orgs,
                        _filter_summary(output.hard_filters),
                        output.top_k,
                    )
                    return output

                attempt_trace["status"] = "empty_keywords"
                attempt_trace["reason"] = "planner_retrieval_core_empty"
                logger.warning(
                    "플래너 LLM 시도 결과 키워드 없음: attempt=%d query_chars=%d parsed_retrieval_core=%s parsed_role_terms=%s parsed_action_terms=%s semantic_query=%r",
                    attempt_index + 1,
                    len(normalized_query),
                    output.retrieval_core,
                    output.role_terms,
                    output.action_terms,
                    output.semantic_query,
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
            "플래너 내부 완료: mode=fallback_broad_search retrieval_core=%s core_keywords=%s removed_terms=%s hard_filters=%s top_k=%d",
            fallback_keywords,
            fallback_output.core_keywords,
            list(last_role_terms + last_action_terms),
            _filter_summary(fallback_output.hard_filters),
            fallback_output.top_k,
        )
        return fallback_output
