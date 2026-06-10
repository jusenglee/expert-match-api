"""
사용자의 자연어 질의에서 검색 의도(Intent)를 분석하고, 최적의 검색 쿼리를 설계하는 Planner 모듈입니다.

[Architecture Overview]
- 사용자가 "화재 관련 컨소시엄 전문가 추천해줘" 라고 입력했을 때, 
  "화재", "컨소시엄"과 같은 기술 도메인 키워드(`retrieval_core`)만을 추출하여 엔진이 노이즈 없이 검색할 수 있게 돕습니다.
- "전문가", "추천해줘"와 같은 역할어/행동어는 `role_terms`, `action_terms`로 분리하여 검색어에서 배제합니다.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import textwrap
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from openai import BadRequestError

from apps.core.cache import PlanCache
from apps.core.config import Settings
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text
from apps.core.llm_policies import build_consistency_invoke_kwargs
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.core.utils import build_deterministic_seed
from apps.domain.models import PlannerOutput

logger = logging.getLogger(__name__)

MAX_PLANNER_ATTEMPTS = 2


PLANNER_VERSION = "v0.5.3"  # concept grounding guard(질의 미근거 concept 제거) + 프롬프트 강화 — 캐시 무효화


# planner 출력을 vLLM guided decoding(JSON 스키마)으로 강제 — prose JSON-only 요청을 구조적으로 보증.
# 미지원 배포에서 BadRequestError가 나면 인스턴스 단위로 자동 비활성 후 prose 폴백(_invoke_json_output 참고).
PLANNER_GUIDED_DECODING = True


@functools.lru_cache(maxsize=1)
def _planner_json_schema() -> dict[str, Any]:
    """PlannerOutput → JSON 스키마(1회 생성 캐시). vLLM extra_body={'guided_json': ...}에 사용."""
    return PlannerOutput.model_json_schema()


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


_HEURISTIC_ROLE_TERMS = (
    "평가위원",
    "심사위원",
    "전문가",
    "연구자",
    "교수",
)
_HEURISTIC_ACTION_TERMS = (
    "추천해줘",
    "찾아줘",
    "선정해줘",
    "추천",
    "심사",
    "평가",
    "선정",
    "발굴",
)


def _heuristic_keywords(
    normalized_query: str,
    role_terms: list[str],
    action_terms: list[str],
) -> list[str]:
    clean_query = normalized_query
    for term in role_terms + action_terms:
        clean_query = clean_query.replace(term, " ")
    keywords: list[str] = []
    for token in clean_query.split():
        normalized = token.strip()
        if len(normalized) > 1 and normalized not in keywords:
            keywords.append(normalized)
    return keywords or ([normalized_query] if normalized_query else [])


# concept grounding(질의 근거) 판정에서 제외하는 일반어 — 이 단어가 질의에 우연히 겹쳐도 concept이
# '근거 있다'고 보지 않는다(예: 'ai' concept의 query_term '시스템'이 '논문투고심사시스템'에 부분일치하는
# 거짓 근거 방지). concept 고유어(label·고유 query/evidence term)로만 grounding을 인정한다.
_GROUNDING_STOPWORDS = frozenset(
    {
        "시스템", "연구", "개발", "기술", "분야", "전문", "산업", "데이터",
        "관리", "서비스", "활용", "기반", "응용", "관련", "방법", "장치",
        "소재", "설계", "분석", "평가", "지원", "구축", "운영", "사업",
        "전문가", "경험", "스마트", "지능형", "고도화", "솔루션",
    }
)


def _term_grounded_in_query(term: str, haystack: str) -> bool:
    """concept term이 질의 haystack(casefold)에 실제 등장하는가. 짧은 영숫자(ai/npu)는 단어경계 강제."""
    normalized = " ".join(term.casefold().split())
    if len(normalized) < 2 or normalized in _GROUNDING_STOPWORDS:
        return False
    if normalized.isascii() and normalized.isalnum() and len(normalized) <= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", haystack) is not None
    return normalized in haystack


def _concept_grounded_in_query(spec: Any, haystack: str) -> bool:
    """concept의 고유어(label·query_terms·evidence_terms·id 토큰) 중 하나라도 질의에 등장하면 근거 있음."""
    needles: list[str] = [getattr(spec, "label", "") or ""]
    needles.extend(getattr(spec, "query_terms", None) or [])
    needles.extend(getattr(spec, "evidence_terms", None) or [])
    needles.extend((getattr(spec, "id", "") or "").split("_"))
    return any(_term_grounded_in_query(n, haystack) for n in needles if n)


def _sanitize_concept_specs(
    specs: list[Any], normalized_query: str, retrieval_core: list[str]
) -> tuple[list[Any], list[str]]:
    """질의에 근거 없는(hallucinated) concept을 제거한다.

    haystack = 사용자 질의 원문 + retrieval_core(planner 추출 키워드). semantic_query는 LLM이 만든
    문장이라(환각 자기-정당화 방지) 근거 판정에서 제외한다. 어떤 고유어도 질의에 없으면 spurious로 보고
    드롭한다 → required gate가 엉뚱한 개념으로 전체 후보를 탈락시키는 0건 사고를 막는다(정밀도 보강).
    """
    if not specs:
        return specs, []
    haystack = " ".join([normalized_query, *(retrieval_core or [])]).casefold()
    kept: list[Any] = []
    dropped: list[str] = []
    for spec in specs:
        if _concept_grounded_in_query(spec, haystack):
            kept.append(spec)
        else:
            dropped.append(getattr(spec, "id", "") or getattr(spec, "label", "") or "?")
    return kept, dropped


class HeuristicPlanner:
    """LLM 없이 동작하는 결정적 플래너. 테스트/오프라인 런타임에서 사용한다."""

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

        role_terms = [
            term for term in _HEURISTIC_ROLE_TERMS if term in normalized_query
        ]
        action_terms = [
            term for term in _HEURISTIC_ACTION_TERMS if term in normalized_query
        ]
        retrieval_core = _heuristic_keywords(
            normalized_query,
            role_terms,
            action_terms,
        )
        output = PlannerOutput(
            intent_summary=normalized_query,
            hard_filters=dict(filters_override or {}),
            include_orgs=list(include_orgs or []),
            exclude_orgs=list(exclude_orgs or []),
            task_terms=_normalize_string_list(role_terms + action_terms),
            core_keywords=list(retrieval_core),
            retrieval_core=list(retrieval_core),
            role_terms=role_terms,
            action_terms=action_terms,
            semantic_query=normalized_query,
            top_k=top_k or 15,
        )
        self.last_trace = {
            "mode": "deterministic_fallback",
            "normalized_query": normalized_query,
            "planner_retry_count": 0,
            "planner_keywords": list(retrieval_core),
            "retrieval_keywords": list(retrieval_core),
            "removed_role_terms": list(role_terms + action_terms),
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
        # LLM 실패 fallback은 plan() 내장 fallback_broad_search 경로를 쓴다.
        # HeuristicPlanner는 llm_backend=='heuristic'일 때 main.py가 직접 주입한다.
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self.cache = cache
        self.last_trace: dict[str, Any] = {}
        # guided_json 지원 여부(미지원 배포에서 BadRequestError 1회 후 prose 폴백으로 고정).
        self._guided_supported: bool = True

    @staticmethod
    def _build_system_prompt() -> str:
        prompt = """
                # 역할
                당신은 ***동질적인 전문가 코퍼스***를 검색하는 전문가 추천 시스템의 R&D 질의 플래너입니다.
                당신은 전문가/평가위원을 모아놓은 qdrant 벡터DB 에 검색 할 쿼리를 만들기 위해, 사용자의 질의를 분석하고 정규화해야 합니다.
                실행 모델은 Solar 102B를 vLLM(OpenAI 호환 API)로 서빙한 모델입니다. 모든 지시와 출력 값은 사용자의 언어를 따르며, 한국어 질의는 한국어로 작성하세요.
            
                # 출력 목표
                - `retrieval_core`: 실제 기술/도메인 매칭에 필요한 핵심 키워드 리스트(Sparse/Keyword 검색용). "평가위원", "전문가" 등 역할어는 제외하세요.
                - `semantic_query`: 검색 의도를 담은 자연어 문장(Vector 검색용). 핵심 기술 키워드와 맥락을 포함하세요.
                - `role_terms`: "평가위원", "교수", "전문가" 등 검색 대상의 페르소나/역할 용어 리스트.
                - `action_terms`: "추천", "찾아줘", "선정해줘" 등 사용자가 요청한 행동 용어 리스트.
                - `intent_flags`: 검색 의도에 대한 플래그 (예: "need_experience": true, "prefer_recent": true 등).
                - `intent_summary`: UI/추적용 짧은 요약 문장.
                - `concept_specs`: 질의의 핵심 기술/도메인 개념 목록(검색·검증 용어 분리). 상세는 아래 [Concept Evidence Plan].

                # 규칙
                1. 사용자 질의의 주 언어를 유지하세요. 번역하거나 언어를 섞지 마세요.
                2. `retrieval_core`에는 도메인 개념, 기술, 재료, 분야만 포함해야 합니다. 코퍼스에 공통적으로 나타나는 "전문가", "추천" 등은 여기에 넣지 마세요.
                2-1. **역할/행위어 누수 금지**: 한 단어를 `role_terms` 또는 `action_terms`에 넣었다면 그 단어는 `retrieval_core`에 절대 다시 넣지 마세요. "평가위원", "심사위원", "심사", "평가", "추천", "선정", "찾아줘", "교수", "전문가" 등은 사용자가 *원하는 결과의 역할/행위* 이며, 검색 대상 인물의 실적 텍스트(논문/특허/과제)에는 등장하지 않습니다.
                3. `role_terms`와 `action_terms`는 검색어(Query)가 아니라 제어 신호로 활용됩니다.
                4. `include_orgs`/`exclude_orgs`는 **검색 대상 인물의 소속 기관 제약**일 때만 사용하세요.
                4-1. **대상 기관 vs 소속 기관 구분**: "X에서 수행한 과제를 심사", "X 사업 평가", "X 과제 ~" 처럼 기관 X 가 *심사/평가 대상*으로 등장하는 경우, X 는 `include_orgs`에 넣지 말고 `semantic_query`의 맥락으로만 유지하세요. "X 소속 ~", "X 출신 ~" 처럼 명시적으로 소속을 지정한 경우만 `include_orgs`에 넣습니다.
                5. 명시적으로 지원되는 구조화 필터만 `hard_filters`에 복사하세요.
                6. 안전한 도메인 키워드가 없으면 `retrieval_core`는 빈 리스트로 반환하세요.
                7. 출력은 JSON 객체 **하나만** 반환하세요. 마크다운 펜스, 설명문, 추론 과정을 출력에 포함하지 마세요. (응답은 구조화 디코딩으로 JSON만 허용됩니다.)

                # [Concept Evidence Plan] (concept_specs 생성 규칙)
                질의에서 핵심 기술/도메인 개념(concept)을 뽑고, concept마다 아래 용어를 **분리**해 생성하세요.
                AI/반도체에 한정하지 마세요 — 배터리/바이오/로봇/양자 등 어떤 도메인이든 동일 규칙으로 만드세요.
                - id: 영문 snake_case 식별자 (예: "ai", "semiconductor", "solid_state_battery").
                - label: 한글 대표어.
                - role: 질의가 반드시 요구하면 "required", 부가/선택이면 "optional".
                - query_terms: 검색(recall)용 대표 검색어 — 최대 8개.
                - evidence_terms: 개념을 '확정'하는 분별력 있는 용어(고유명사·전문용어·약어). 문서에 직접 등장하면 그 개념으로 확정됨 — 최대 12개.
                - weak_terms: 단독으로는 근거가 약한 연관어(확정 근거 아님) — 최대 12개.

                [필수 규칙]
                - evidence_terms와 weak_terms를 반드시 구분하세요.
                  · evidence_terms: 그 개념을 '고유하게' 지시하는 분별력 있는 명사·전문용어·약어. 개념의 핵심 도메인
                    head-noun도 여기 넣어 recall을 확보하세요(예: 배터리, 이차전지, 반도체). 판정 기준 — 문서에 substring으로
                    등장할 때 '같은 개념의 문서에만' 나타나면 evidence입니다. (분별력이 큰 고유어를 evidence 리스트 앞쪽에 두세요.)
                  · weak_terms(확정 근거 아님): 다음 두 부류를 모두 weak로 두세요.
                    (a) 분별력이 없는 일반·기능어: 소재/개발/시스템/지능형/스마트/센서/경험/연구/산업 등.
                    (b) 무관한 2개 이상 도메인의 합성어 부분문자열로 흔히 등장해 거짓 확정을 유발하는 모호어·광역 상위어:
                        "전지"→연료전지·태양전지·축전지, "소자"→반도체소자·광소자·표시소자, "화재"→화재예방·산불.
                - concept id는 가능하면 아래 표준 id를 재사용하세요(런타임 안전망 보강이 자동 적용됩니다):
                  ai, semiconductor, secondary_battery, bio, robot, autonomous_driving, display, hydrogen, quantum, security.
                  표준에 없는 새 도메인만 새 snake_case id를 만드세요.
                - head-noun을 evidence로 올리는 것은 기술/도메인 명사에만 적용합니다. 기관명·역할어·행위어는 concept이 아니므로 evidence 대상이 아닙니다.
                - concept은 최대 5개. 질의에 명시된 기술/도메인만 만드세요. 도메인 개념이 없으면 concept_specs는 빈 배열 [] 로 두세요.
                - 영문 약어(AI/NPU/ADAS 등)는 그 자체로 분별력이 있을 때만 evidence_terms에 넣으세요.
                - **[근거(grounding) 필수 — 환각 금지]** 각 concept은 그 `label` 또는 `query_terms` 중 최소 하나가 **사용자 질의 원문에 실제로 등장**해야 합니다. 질의에 없는 도메인을 추측해서 만들지 마세요. 특히 질의에 인공지능/AI 언급이 전혀 없는데 `ai` concept을 넣는 식의 환각은 절대 금지입니다(예: "논문투고심사시스템 제안평가"는 행정/프로세스 질의이므로 기술 도메인 concept이 없습니다 → `concept_specs: []`). 도메인이 모호하면 `role`을 "optional"로 두거나 `concept_specs`를 비우세요. 런타임은 질의에 근거 없는 concept을 자동 제거합니다.

                # 출력 스키마
                {
                  "intent_summary": "string",
                  "retrieval_core": ["string"],
                  "semantic_query": "string",
                  "role_terms": ["string"],
                  "action_terms": ["string"],
                  "intent_flags": {},
                  "hard_filters": {},
                  "include_orgs": ["string"],
                  "exclude_orgs": ["string"],
                  "top_k": integer,
                  "concept_specs": [
                    {"id": "string", "label": "string", "role": "required|optional",
                     "query_terms": ["string"], "evidence_terms": ["string"], "weak_terms": ["string"]}
                  ]
                }
            
                # 예시
                Input:
                {
                  "query": "난접근성 화재 진압에서 드론을 접목하려고해. 드론을 화재진압 연구에 사용한 경험이 있는 관련된 전문가를 5명 추천해줘",
                  "filters_override": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5
                }
            
                Output:
                {
                  "intent_summary": "난접근성 화재 진압과 드론 접목 관련 전문가 탐색",
                  "retrieval_core": ["난접근성 화재 진압", "드론"],
                  "semantic_query": "난접근성 화재 현장의 화재 진압을 위한 드론 및 무인 로봇 활용 연구 전문가",
                  "role_terms": ["전문가"],
                  "action_terms": ["추천"],
                  "intent_flags": { "need_experience": true },
                  "hard_filters": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5,
                  "concept_specs": [
                    {"id": "fire_suppression", "label": "화재 진압", "role": "required",
                     "query_terms": ["화재 진압", "소방"],
                     "evidence_terms": ["화재 진압", "소방", "화재진압로봇", "난접근성 화재"],
                     "weak_terms": ["화재", "안전", "현장"]},
                    {"id": "drone", "label": "드론", "role": "required",
                     "query_terms": ["드론", "UAV"],
                     "evidence_terms": ["드론", "UAV", "무인기", "무인비행체"],
                     "weak_terms": ["비행", "로봇", "무인"]}
                  ]
                }
                주의: "화재 진압"/"소방"은 화재진압 개념을 고유하게 지시하는 head-noun이라 evidence. 단독 "화재"는 화재예방·산불 등에 공통 매칭되는 광역 상위어라 weak입니다.

                # 예시 2 (대상 기관 + 역할어 처리)
                Input:
                {
                  "query": "한국과학기술정보연구원에서 수행한 과제를 심사하기 위한 적절한 평가위원 추천",
                  "filters_override": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5
                }

                Output:
                {
                  "intent_summary": "한국과학기술정보연구원 수행 과제 심사용 평가위원 탐색",
                  "retrieval_core": [],
                  "semantic_query": "한국과학기술정보연구원에서 수행한 R&D 과제를 심사할 수 있는 동일/유사 도메인 경험을 가진 평가위원",
                  "role_terms": ["평가위원"],
                  "action_terms": ["심사", "추천"],
                  "intent_flags": {},
                  "hard_filters": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5,
                  "concept_specs": []
                }
                주의: 위 예시에서 "한국과학기술정보연구원"은 *심사 대상 기관* 이므로 `include_orgs`에 넣지 않습니다. "평가위원"은 `role_terms`, "심사"/"추천"은 `action_terms`이며 `retrieval_core`에 중복으로 들어가지 않습니다. 기술 도메인이 없으므로 `concept_specs`는 빈 배열입니다.

                # 예시 3 (비-AI 도메인 — concept_specs는 어떤 도메인이든 동일 규칙)
                Input:
                {
                  "query": "전고체 배터리 소재 개발 경험이 있는 연구자를 찾아줘",
                  "filters_override": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 10
                }

                Output:
                {
                  "intent_summary": "전고체 배터리 소재 개발 경험 연구자 탐색",
                  "retrieval_core": ["전고체 배터리", "고체전해질", "소재 개발"],
                  "semantic_query": "전고체 배터리용 고체전해질 등 소재 개발 경험을 가진 연구자",
                  "role_terms": ["연구자"],
                  "action_terms": ["찾아줘"],
                  "intent_flags": { "need_experience": true },
                  "hard_filters": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 10,
                  "concept_specs": [
                    {"id": "secondary_battery", "label": "전고체 배터리", "role": "required",
                     "query_terms": ["전고체전지", "전고체 배터리", "고체전해질"],
                     "evidence_terms": ["전고체전지", "전고체 배터리", "solid-state battery", "고체전해질", "황화물계 전해질", "배터리", "이차전지"],
                     "weak_terms": ["전지", "소재", "개발"]}
                  ]
                }
                주의: concept id는 표준 id "secondary_battery"를 재사용했습니다(전고체전지는 이차전지의 하위 유형이라 registry 안전망 보강이 자동 적용됨). "배터리"는 연료전지/태양전지에 substring으로 나타나지 않아 evidence로 안전하고, "전지"는 그 도메인들에 공통 매칭돼 거짓 확정 위험이므로 weak로 둡니다.

                # 예시 4 (head-noun=evidence 원리는 도메인 불문 — 반도체)
                Input:
                {
                  "query": "시스템반도체 설계 경험이 있는 연구자를 찾아줘",
                  "filters_override": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 10
                }

                Output:
                {
                  "intent_summary": "시스템반도체 설계 경험 연구자 탐색",
                  "retrieval_core": ["시스템반도체", "반도체 설계"],
                  "semantic_query": "시스템반도체/집적회로 설계 경험을 가진 연구자",
                  "role_terms": ["연구자"],
                  "action_terms": ["찾아줘"],
                  "intent_flags": { "need_experience": true },
                  "hard_filters": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 10,
                  "concept_specs": [
                    {"id": "semiconductor", "label": "반도체", "role": "required",
                     "query_terms": ["시스템반도체", "반도체", "집적회로"],
                     "evidence_terms": ["반도체", "시스템반도체", "집적회로", "웨이퍼", "SoC", "파운드리"],
                     "weak_terms": ["소자", "회로", "공정", "시스템"]}
                  ]
                }
                주의: "반도체"는 그 개념을 고유하게 지시하는 head-noun이라 evidence. "소자"는 반도체소자·광소자·표시소자 등 무관 도메인에 공통 매칭되는 모호어라 weak. concept id는 표준 id "semiconductor"를 재사용했습니다.

                # 예시 5 (행정/프로세스 질의 — 기술 도메인 없음 → concept_specs 비움)
                Input:
                {
                  "query": "논문투고심사시스템 제안평가 가능한 전문가를 추천해줘",
                  "filters_override": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5
                }

                Output:
                {
                  "intent_summary": "논문투고심사시스템 제안평가 관련 전문가 탐색",
                  "retrieval_core": ["논문투고심사시스템", "제안평가"],
                  "semantic_query": "논문투고심사시스템 구축/운영 및 제안서 평가 경험이 있는 전문가",
                  "role_terms": ["전문가"],
                  "action_terms": ["추천", "평가"],
                  "intent_flags": {},
                  "hard_filters": {},
                  "include_orgs": [],
                  "exclude_orgs": [],
                  "top_k": 5,
                  "concept_specs": []
                }
                주의: 질의에 인공지능/반도체 등 특정 기술 도메인이 **명시되지 않았습니다**. 시스템·평가·제안 같은 일반 프로세스어만 있으므로 concept을 만들지 않습니다(`concept_specs: []`). 질의에 없는 'ai' 같은 도메인을 추측해 넣으면 안 됩니다.
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

        # v1 branch 기반 lexicon expansion은 제거됐다. 하위호환 필드는 비워 둔다.
        output.bundle_ids = []

        # Backward compatibility mapping
        output.core_keywords = list(output.retrieval_core)
        output.task_terms = _normalize_string_list(output.role_terms + output.action_terms)

        # Concept grounding guard(정밀도 보강, gate 불변):
        # planner(LLM)가 질의에 없는 도메인(예: 비-AI 질의에 'ai' required)을 만들면 required gate가
        # 전체 후보를 탈락시켜 추천 0건이 되는 사고가 있었다. 어떤 concept 고유어도 질의 원문/retrieval_core에
        # 등장하지 않으면 spurious로 보고 제거한다. cache-hit·LLM 두 경로 모두 통과하므로 캐시된 잘못된
        # concept도 읽는 즉시 정화된다(캐시 무효화 불필요).
        if output.concept_specs:
            grounded, dropped = _sanitize_concept_specs(
                output.concept_specs, normalized_query, output.retrieval_core
            )
            if dropped:
                logger.warning(
                    "planner concept grounding: 질의 미근거 concept 제거 dropped=%s kept=%s query=%r",
                    dropped,
                    [getattr(s, "id", "") for s in grounded],
                    normalized_query,
                )
            output.concept_specs = grounded

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
        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        invoke_kwargs = build_consistency_invoke_kwargs(seed=seed)
        if PLANNER_GUIDED_DECODING and self._guided_supported:
            guided_kwargs = {**invoke_kwargs, "extra_body": {"guided_json": _planner_json_schema()}}
            try:
                result = await self.model.ainvoke_non_stream(messages, **guided_kwargs)
            except BadRequestError as exc:
                # 배포가 guided_json을 거부 → 이 인스턴스에서는 비활성 후 prose JSON-only로 폴백.
                self._guided_supported = False
                logger.warning("플래너 guided_json 미지원/거부 — prose 폴백 전환: error=%r", exc)
                result = await self.model.ainvoke_non_stream(messages, **invoke_kwargs)
        else:
            result = await self.model.ainvoke_non_stream(messages, **invoke_kwargs)
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
