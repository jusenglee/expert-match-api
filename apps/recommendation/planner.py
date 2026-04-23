"""
Planner implementations for recommendation search intent extraction.
추천 검색 의도 추출을 위한 Planner 구현체 모음.
"""

from __future__ import annotations

import json
import logging
import re
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
PLANNER_VERSION = "v0.7.2"

# 역할(Role) 관련 쿼리 용어: 전문가, 평가위원 등을 지칭하는 용어 목록
ROLE_QUERY_TERMS = (
    "expert recommendation",
    "reviewer recommendation",
    "evaluation committee",
    "scientific consortium",
    "consortium",
    "committee",
    "expert",
    "reviewer",
    "reviewers",
    "committee member",
    "committee members",
    "evaluation expert",
    "evaluation experts",
    "evaluation reviewer",
    "evaluation reviewers",
    "평가위원 추천",
    "전문가 추천",
    "평가위원",
    "심사위원",
    "전문가",
    "리뷰어",
    "자문위원",
    "컨소시엄",
    "과학 컨소시엄",
    "위원회",
    "협의체",
    "연구단",
    "사업단",
)
# 행동(Action) 관련 쿼리 용어: 추천, 검색, 매칭 등의 행위를 지칭하는 용어 목록
ACTION_QUERY_TERMS = (
    "recommendation",
    "recommend",
    "recommended",
    "find",
    "search",
    "lookup",
    "추천해줘",
    "추천해주세요",
    "추천해",
    "추천",
    "찾아줘",
    "찾아주세요",
    "찾아",
    "매칭",
)
# 일반(Generic) 쿼리 용어: 경험, 역량, 적합성 등 포괄적인 요구사항을 나타내는 용어 목록
GENERIC_QUERY_TERMS = (
    "experience",
    "experiences",
    "capability",
    "capabilities",
    "relevant",
    "suitable",
    "fit",
    "fits",
    "경험",
    "경력",
    "역량",
    "전문성",
    "관련",
    "적합",
    "평가",
)

# 문맥적 평가(Contextual Evaluation) 구문: 특정 평가(예: 과제 평가, 기술 평가)를 나타내는 정규식 패턴 목록
CONTEXTUAL_EVALUATION_PHRASES: tuple[tuple[str, str], ...] = (
    ("과제 평가", r"(?:과제|프로젝트)(?:를|을)?\s*평가"),
    ("기술 평가", r"기술(?:을|를)?\s*평가"),
    ("논문 평가", r"논문(?:을|를)?\s*평가"),
    ("project evaluation", r"\bprojects?\s+evaluation\b|\bevaluate\s+projects?\b"),
    (
        "technology evaluation",
        r"\btechnology\s+evaluation\b|\bevaluate\s+technolog(?:y|ies)\b",
    ),
    ("paper evaluation", r"\bpapers?\s+evaluation\b|\bevaluate\s+papers?\b"),
)

# 반드시 포함되어야 하는 속성(Must Aspects) 중 지나치게 일반적인 단일어(Singleton) 목록
GENERIC_SINGLETON_MUST_ASPECTS = {
    "기술",
    "개발",
    "연구",
    "과제",
    "분석",
    "시스템",
    "플랫폼",
    "컨소시엄",
    "위원회",
    "평가",
    "심사",
    "리뷰",
    "과학",
    "분야",
    "technology",
    "development",
    "research",
    "project",
    "analysis",
    "system",
    "platform",
    "consortium",
    "committee",
    "evaluation",
    "scientific",
    "science",
    "field",
    "area",
}

# 반드시 포함되어야 하는 속성(Must Aspects) 중 지나치게 일반적인 구문(Phrase) 목록
GENERIC_PHRASE_MUST_ASPECTS = {
    "ai 기반",
    "데이터 기반",
    "시스템 개발",
    "기술 개발",
    "과학 컨소시엄",
    "평가 위원회",
    "평가 위원",
    "ai based",
    "data driven",
    "system development",
    "technology development",
    "scientific consortium",
    "evaluation committee",
}


class Planner(Protocol):
    """
    Planner 프로토콜 인터페이스.
    주어진 쿼리를 분석하여 추천 검색에 필요한 의도와 필터 조건을 추출합니다.
    """

    async def plan(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput: ...


def _normalize_text(value: str | None) -> str:
    """문자열의 앞뒤 공백을 제거하고 연속된 공백을 하나로 정규화합니다."""
    return " ".join((value or "").split())


def _normalize_string_list(values: list[str] | None) -> list[str]:
    normalized_values: list[str] = []
    for value in values or []:
        normalized = _normalize_text(str(value))
        if normalized and normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


def _compile_term_pattern(terms: list[str]) -> re.Pattern[str] | None:
    """주어진 용어 목록을 기반으로 정규식 패턴을 컴파일합니다 (길이 역순으로 정렬하여 긴 단어부터 매칭)."""
    normalized_terms = _normalize_string_list(terms)
    if not normalized_terms:
        return None

    # 영문(ASCII)으로만 구성된 단어는 \b (단어 경계)를 추가하여
    # 부분 일치(예: "fitness" 안의 "fit"이 날아가는 현상)를 방지합니다.
    unique_terms = sorted(set(normalized_terms), key=len, reverse=True)
    escaped_terms: list[str] = []
    for term in unique_terms:
        if all(ord(c) < 128 for c in term):
            escaped_terms.append(rf"\b{re.escape(term)}\b")
        else:
            escaped_terms.append(re.escape(term))

    return re.compile("|".join(escaped_terms), flags=re.IGNORECASE)


def _extract_query_terms(query: str, candidates: tuple[str, ...]) -> list[str]:
    """사용자 쿼리에서 후보 용어(candidates)가 포함되어 있는지 검사하고 추출합니다."""
    normalized_query = _normalize_text(query).lower()
    found_terms: list[str] = []
    for candidate in sorted(
        {_normalize_text(term) for term in candidates if term}, key=len, reverse=True
    ):
        if candidate.lower() in normalized_query and candidate not in found_terms:
            found_terms.append(candidate)
    return found_terms


def _strip_terms(value: str, terms: list[str]) -> str:
    """문자열에서 특정 용어 목록(terms)을 찾아 공백으로 치환(제거)합니다."""
    pattern = _compile_term_pattern(terms)
    if pattern is None:
        return _normalize_text(value)
    return _normalize_text(pattern.sub(" ", value))


def _filter_domain_terms(values: list[str], *, forbidden_terms: list[str]) -> list[str]:
    """도메인 용어 목록에서 금지된 용어(메타 용어, 일반 용어 등)를 필터링하여 제거합니다."""
    filtered: list[str] = []
    for value in _normalize_string_list(values):
        cleaned = _strip_terms(value, forbidden_terms)
        if not cleaned:
            continue
        if cleaned not in filtered:
            filtered.append(cleaned)
    return filtered


def _split_fallback_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for part in _normalize_text(value).split():
        if len(part) < 2:
            continue
        if part not in tokens:
            tokens.append(part)
    return tokens


def _extract_contextual_evaluation_terms(query: str) -> list[str]:
    """사용자 쿼리에서 '과제 평가', '기술 평가' 등 특정 평가 문맥을 나타내는 용어를 추출합니다."""
    contextual_terms: list[str] = []
    for canonical_term, pattern_text in CONTEXTUAL_EVALUATION_PHRASES:
        if re.search(pattern_text, query, flags=re.IGNORECASE):
            contextual_terms.append(canonical_term)
    return _normalize_string_list(contextual_terms)


def _is_generic_must_aspect(term: str) -> bool:
    normalized = _normalize_text(term).lower()
    if not normalized:
        return True
    if normalized in GENERIC_PHRASE_MUST_ASPECTS:
        return True
    parts = normalized.split()
    if len(parts) == 1:
        return normalized in GENERIC_SINGLETON_MUST_ASPECTS
    return all(part in GENERIC_SINGLETON_MUST_ASPECTS for part in parts)


def _most_specific_terms(values: list[str]) -> list[str]:
    sorted_values = sorted(
        _normalize_string_list(values),
        key=lambda value: (
            -len(value.replace(" ", "")),
            value.count(" "),
            value,
        ),
    )
    return sorted_values


def _prune_must_aspects(
    raw_values: list[str],
    *,
    contextual_terms: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    추출된 필수 속성(must_aspects)에서 평가 문맥어와 지나치게 일반적인 단어를 가지치기(Pruning)합니다.
    유효한 속성이 모두 제거되면 가장 구체적인 단어를 폴백(fallback)으로 선택합니다.
    """
    contextual_lookup = {
        value.lower() for value in _normalize_string_list(contextual_terms)
    }
    normalized_values = _normalize_string_list(raw_values)
    pruned: list[str] = []
    prune_reasons: list[dict[str, str]] = []

    for value in normalized_values:
        lowered = value.lower()
        if lowered in contextual_lookup:
            prune_reasons.append({"term": value, "reason": "review_context"})
            continue
        if _is_generic_must_aspect(value):
            prune_reasons.append({"term": value, "reason": "generic_phrase"})
            continue
        if value not in pruned:
            pruned.append(value)

    if pruned:
        return pruned, prune_reasons

    fallback_candidates = [
        value
        for value in _most_specific_terms(normalized_values)
        if value.lower() not in contextual_lookup
    ]
    if fallback_candidates:
        fallback_term = fallback_candidates[0]
        prune_reasons.append({"term": fallback_term, "reason": "specificity_fallback"})
        return [fallback_term], prune_reasons
    return [], prune_reasons


def _planner_contract_debug(
    *,
    raw_must_aspects: list[str],
    contextual_terms: list[str],
    forbidden_terms: list[str],
    fallback_terms: list[str] | None = None,
) -> dict[str, Any]:
    normalized_must_aspects = _filter_domain_terms(
        raw_must_aspects,
        forbidden_terms=forbidden_terms,
    )
    if not normalized_must_aspects and fallback_terms:
        normalized_must_aspects = _normalize_string_list(fallback_terms)
    pruned_must_aspects, prune_reasons = _prune_must_aspects(
        normalized_must_aspects,
        contextual_terms=contextual_terms,
    )
    return {
        "raw_must_aspects": _normalize_string_list(raw_must_aspects),
        "normalized_must_aspects": normalized_must_aspects,
        "pruned_must_aspects": pruned_must_aspects,
        "must_aspect_prune_reasons": prune_reasons,
    }


class HeuristicPlanner:
    """
    LLM 플래너를 사용할 수 없거나 실패했을 때 사용되는 결정론적(Deterministic) 폴백 플래너.
    단순 키워드 매칭과 정규화를 기반으로 쿼리를 분석합니다.
    """

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
        normalized_query = _normalize_text(query)
        output = PlannerOutput(
            intent_summary=normalized_query,
            hard_filters=dict(filters_override or {}),
            include_orgs=list(include_orgs or []),
            exclude_orgs=list(exclude_orgs or []),
            task_terms=[],
            core_keywords=[],
            retrieval_core=[],
            must_aspects=[],
            generic_terms=[],
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
            "must_aspects": [],
            "generic_terms": [],
            "removed_meta_terms": [],
            "attempts": [],
        }
        return output


class OpenAICompatPlanner:
    """
    LLM 기반의 플래너.
    사용자 쿼리를 프롬프트로 전달하여 검색에 안전한 도메인 요구사항과 검색 의도를 추출합니다.
    """

    def __init__(self, settings: Settings, cache: Any | None = None) -> None:
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
        You are the query planner for an R&D expert recommendation service.

        Return one JSON object only. Do not return markdown or prose.

        Goal:
        - separate domain requirements from meta request words
        - keep retrieval keywords safe for search
        - expose must-have aspects for downstream evidence gating

        Field guide:
        - `intent_summary`: a short summary of the user's intent.
        - `semantic_query`: one natural sentence for dense retrieval.
        - `bounded_hyde_document`: a hypothetical expert profile document based on the query, restricted to facts in the query.
        - `domain_term_groups`: list of domain term groups (e.g. {{"name": "domain", "mode": "at_least_one", "terms": ["term1", "term2"]}})
        - `soft_preference_groups`: list of preference groups
        - `noise_terms`: terms like "전문가", "평가위원", "연구자" that should not be used for search.
        - `relaxation_policy`: policy to relax filters if candidates are not found.
        - `retrieval_core`: EXACT EXTRACTION ONLY. Korean-only domain terms explicitly present in the query for sparse BM25 retrieval. Do not expand or add sub-topics. No English.
        - `must_aspects`: EXACT EXTRACTION ONLY. Korean-only core aspects explicitly present in the query. Do not infer or hallucinate details. Leave empty if unsure.
        - `evidence_aspects`: EXPANSION ALLOWED. BILINGUAL list of specific terms likely to appear literally in paper
          titles, abstracts, keywords, or project/patent names in this domain. Include both Korean
          and English forms that researchers actually write. These are used for direct evidence
          matching against Korean/English mixed research databases. Aim for 4-8 specific terms.
          DO NOT include any organizational or role-related terms like consortium, committee, or evaluation.
          Example for "의료영상 분석 AI": ["의료영상 분석", "medical image analysis",
          "딥러닝", "deep learning", "영상 진단", "image segmentation", "CT 영상", "MRI"]
        - `generic_terms`: generic request words such as broad capability or experience words.
        - `role_terms`: meta nouns such as expert, reviewer, evaluation committee.
        - `action_terms`: request verbs such as recommend, find, search.
        - `bundle_ids`: optional expansion bundle ids chosen only from the bundle list below.
        - `intent_flags`, `hard_filters`, `include_orgs`, `exclude_orgs`, `top_k`: copy only when explicitly supported by the request.

        Rules:
        1. Preserve the user's language.
        2. Put only Korean domain terms in `retrieval_core`.
        3. Put meta request words in `noise_terms`, `role_terms`, `action_terms`, or `generic_terms`, not in `retrieval_core`.
        4. Do not invent filters, bundle ids, or aspects.
        5. Return empty arrays instead of fabricated terms.
        6. `evidence_aspects` must contain only specific domain terms — no meta words, no generic
           words like "기술", "개발", "연구". Include English equivalents of key Korean terms.
           ABSOLUTELY NO organizational or role terms (like consortium, committee, evaluation) allowed here.
        7. EXTRACT, DO NOT EXPAND. You MUST only extract domain terms that are explicitly present in the user's query for `retrieval_core` and `must_aspects`. Do not infer, hallucinate, or add specific sub-domains, underlying technologies, or related aspects that the user did not directly write.
        8. `bounded_hyde_document` should sound like an actual professional bio, but ONLY use domain constraints explicitly in the query. Do not hallucinate people's names or specific awards.

        Expansion bundles:
        {lexicon_summary}

        Output schema:
        {{
          "intent_summary": "string",
          "semantic_query": "string",
          "bounded_hyde_document": "string",
          "hard_filters": {{}},
          "domain_term_groups": [
            {{
              "name": "string",
              "mode": "string",
              "terms": ["string"]
            }}
          ],
          "soft_preference_groups": [
            {{
              "name": "string",
              "mode": "string",
              "terms": ["string"]
            }}
          ],
          "noise_terms": ["string"],
          "relaxation_policy": {{}},
          "retrieval_core": ["string"],
          "must_aspects": ["string"],
          "evidence_aspects": ["string"],
          "generic_terms": ["string"],
          "bundle_ids": ["string"],
          "role_terms": ["string"],
          "action_terms": ["string"],
          "intent_flags": {{}},
          "include_orgs": ["string"],
          "exclude_orgs": ["string"],
          "top_k": 5
        }}
        """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _extract_meta_terms(
        normalized_query: str,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        role_terms = _extract_query_terms(normalized_query, ROLE_QUERY_TERMS)
        action_terms = _extract_query_terms(normalized_query, ACTION_QUERY_TERMS)
        contextual_terms = _extract_contextual_evaluation_terms(normalized_query)
        generic_terms = _extract_query_terms(normalized_query, GENERIC_QUERY_TERMS)
        if contextual_terms:
            generic_terms = [
                term
                for term in generic_terms
                if term.lower() not in {"평가", "evaluation"}
            ]
        return role_terms, action_terms, generic_terms, contextual_terms

    @classmethod
    def _apply_request_constraints(
        cls,
        *,
        output: PlannerOutput,
        normalized_query: str,
        filters_override: dict[str, Any] | None,
        include_orgs: list[str] | None,
        exclude_orgs: list[str] | None,
        top_k: int | None,
        contextual_terms: list[str] | None = None,
    ) -> PlannerOutput:
        """
        LLM이 생성한 초기 검색 계획(PlannerOutput)에 대해 비즈니스 로직 및 사용자 요청 제약을 엄격하게 적용합니다.
        불필요한 메타 용어를 제거하고, 조건에 맞게 속성(aspects)들을 재조정하며, 필터 조건들을 최종 병합합니다.
        """
        # 1. intent_summary(의도 요약)가 비어있다면 원래 사용자 쿼리로 대체합니다.
        output.intent_summary = (
            _normalize_text(output.intent_summary) or normalized_query
        )

        # 2. 쿼리에서 역할(role), 행동(action), 일반(generic), 문맥적 평가(contextual) 용어들을 추출합니다.
        role_terms, action_terms, generic_terms, query_contextual_terms = (
            cls._extract_meta_terms(normalized_query)
        )
        # 외부에서 전달된 문맥적 용어(contextual_terms)와 쿼리에서 추출한 용어를 병합합니다.
        effective_contextual_terms = _normalize_string_list(
            list(contextual_terms or []) + query_contextual_terms
        )

        # 3. LLM 결과에 쿼리에서 하드코딩으로 추출된 역할/행동/일반 용어들을 추가 병합합니다.
        output.role_terms = _normalize_string_list(output.role_terms + role_terms)
        output.action_terms = _normalize_string_list(output.action_terms + action_terms)
        output.generic_terms = _normalize_string_list(
            output.generic_terms + generic_terms
        )
        if effective_contextual_terms:
            # 평가 문맥이 존재하면 '평가', 'evaluation'과 같은 일반 단어는 generic_terms에서 제외합니다.
            output.generic_terms = [
                term
                for term in output.generic_terms
                if term.lower() not in {"평가", "evaluation"}
            ]

        # 4. 도메인 검색어로 사용해서는 안 될 '금지된 용어(메타/일반 용어)' 목록을 생성합니다.
        forbidden_terms = _normalize_string_list(
            output.role_terms + output.action_terms + output.generic_terms
        )

        # 5. 핵심 검색어(retrieval_core)에서 금지된 용어들을 필터링하여 순수 도메인 용어만 남깁니다.
        raw_retrieval_terms = _normalize_string_list(
            output.retrieval_core or output.core_keywords
        )
        output.retrieval_core = _filter_domain_terms(
            raw_retrieval_terms,
            forbidden_terms=forbidden_terms,
        )

        # 평가 문맥어(예: '기술 평가', '과제 평가')가 핵심 검색어에 포함되지 않도록 한 번 더 제거합니다.
        contextual_lookup = {
            value.lower()
            for value in _normalize_string_list(effective_contextual_terms)
        }
        if contextual_lookup:
            output.retrieval_core = [
                term
                for term in output.retrieval_core
                if term.lower() not in contextual_lookup
            ]
        output.core_keywords = list(output.retrieval_core)
        # 6. 필수 속성(must_aspects) 역시 금지된 용어 필터링을 거치며, 비어있을 경우 핵심 검색어를 재사용합니다.
        raw_must_aspects = _filter_domain_terms(
            output.must_aspects or output.retrieval_core or output.core_keywords,
            forbidden_terms=forbidden_terms,
        )
        if not raw_must_aspects:
            raw_must_aspects = list(output.retrieval_core)

        # 지나치게 일반적인 단어나 문맥어를 제거하는 가지치기(Pruning) 수행
        output.must_aspects, _ = _prune_must_aspects(
            raw_must_aspects,
            contextual_terms=effective_contextual_terms,
        )

        # 7. task_terms 설정 및 인텐트 플래그(intent_flags) 설정 (평가 문맥이 있으면 관련 플래그를 활성화)
        output.task_terms = _normalize_string_list(
            output.role_terms + output.action_terms
        )
        output.intent_flags = dict(output.intent_flags or {})
        if effective_contextual_terms:
            output.intent_flags["review_context"] = True
            output.intent_flags["review_targets"] = effective_contextual_terms
        else:
            output.intent_flags.pop("review_context", None)
            output.intent_flags.pop("review_targets", None)

        # 8. 유효한 확장 번들 아이디(bundle_ids)만 남기고 정리합니다.
        output.bundle_ids = [
            bundle_id
            for bundle_id in _normalize_string_list(output.bundle_ids)
            if bundle_id in EXPANSION_LEXICON
        ]

        # 9. 밀집 검색(Dense Retrieval)용 의미론적 쿼리(semantic_query)가 없다면 핵심 검색어 또는 의도 요약으로 자동 생성합니다.
        if not output.semantic_query:
            output.semantic_query = (
                " ".join(output.retrieval_core)
                if output.retrieval_core
                else output.intent_summary
            )

        # 10. 증거 속성(evidence_aspects) 정제:
        # 증거 텍스트(논문, 특허 등)에 직접 등장할 실제 용어들(한국어+영어 혼합)에서 금지 용어를 우선 제거합니다.
        raw_evidence_aspects = _normalize_string_list(output.evidence_aspects)
        cleaned_evidence_aspects = _filter_domain_terms(
            raw_evidence_aspects,
            forbidden_terms=forbidden_terms,
        )
        # 단순 "기술", "AI 기반" 등 너무 일반적인 단어들은 증거 매칭(evidence matching)에 부적합하므로 제외합니다.
        cleaned_evidence_aspects = [
            term
            for term in cleaned_evidence_aspects
            if not _is_generic_must_aspect(term)
        ]
        if cleaned_evidence_aspects:
            output.evidence_aspects = cleaned_evidence_aspects
        else:
            # LLM이 적절한 evidence_aspects를 생성하지 못했다면, 한국어 중심의 must_aspects를 대체재로 사용합니다.
            # (bilingual 확장은 LLM 재시도 시 이루어짐)
            output.evidence_aspects = list(output.must_aspects)

        # 11. 외부에서 전달받은 강제 필터(filters_override) 및 포함/제외 조직 정보를 병합합니다.
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
        """
        LLM에게 JSON 형태의 출력을 요청하고, 결과를 파싱하여 PlannerOutput 객체로 변환합니다.
        """
        invoke_kwargs = build_consistency_invoke_kwargs(seed=seed)
        result = await self.model.ainvoke_non_stream(
            [
                SystemMessage(content=self._build_system_prompt()),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ],
            **invoke_kwargs,
        )
        # LLM 응답 텍스트에서 순수 JSON 부분만 추출하여 파싱합니다.
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
        """
        사용자 쿼리를 바탕으로 최적의 추천 검색 계획(PlannerOutput)을 수립합니다.
        캐시를 우선 확인하고, 캐시가 없다면 LLM을 통해 기획을 시도하며, 여러 번 실패할 경우 폴백(Fallback) 플래너로 동작합니다.
        """
        normalized_query = _normalize_text(query)
        filters = filters_override or {}

        # 1. 정규식 기반으로 쿼리 내의 역할(role), 행동(action), 일반 용어를 1차 추출합니다.
        (
            detected_role_terms,
            detected_action_terms,
            detected_generic_terms,
            retained_contextual_terms,
        ) = self._extract_meta_terms(normalized_query)
        removed_meta_terms = _normalize_string_list(
            detected_role_terms + detected_action_terms + detected_generic_terms
        )

        # 2. 캐시를 확인합니다. 동일한 쿼리와 필터 조건이 있다면 바로 반환하여 응답 속도를 높입니다.
        if self.cache:
            cached_output = self.cache.get(normalized_query, filters, PLANNER_VERSION)
            if cached_output:
                logger.info("Planner cache hit: query=%r", normalized_query)
                output = self._apply_request_constraints(
                    output=cached_output,
                    normalized_query=normalized_query,
                    filters_override=filters_override,
                    include_orgs=include_orgs,
                    exclude_orgs=exclude_orgs,
                    top_k=top_k,
                    contextual_terms=retained_contextual_terms,
                )
                contract_debug = _planner_contract_debug(
                    raw_must_aspects=list(output.must_aspects or output.retrieval_core),
                    contextual_terms=retained_contextual_terms,
                    forbidden_terms=_normalize_string_list(
                        output.role_terms + output.action_terms + output.generic_terms
                    ),
                    fallback_terms=list(output.retrieval_core),
                )
                self.last_trace = {
                    "mode": "cache_hit",
                    "cache": {"canonical_plan": "hit"},
                    "planner_version": PLANNER_VERSION,
                    "normalized_query": normalized_query,
                    "planner_keywords": list(output.retrieval_core),
                    "retrieval_keywords": list(output.retrieval_core),
                    "must_aspects": list(output.must_aspects),
                    "generic_terms": list(output.generic_terms),
                    "removed_meta_terms": removed_meta_terms,
                    "retained_contextual_terms": retained_contextual_terms,
                    "intent_flags": dict(output.intent_flags),
                    **contract_debug,
                }
                logger.info(
                    "Planner cache applied: query=%r removed_meta_terms=%s retained_contextual_terms=%s retrieval_core=%s must_aspects=%s intent_flags=%s",
                    normalized_query,
                    removed_meta_terms,
                    retained_contextual_terms,
                    output.retrieval_core,
                    output.must_aspects,
                    output.intent_flags,
                )
                return output

        # 3. 캐시가 없다면 최대 MAX_PLANNER_ATTEMPTS 횟수만큼 LLM 플래닝을 시도합니다.
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
                "LLM planning start: query=%r attempt=%d",
                normalized_query,
                attempt_index + 1,
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
                    contextual_terms=retained_contextual_terms,
                )
                attempt_trace = {
                    "attempt": attempt_index + 1,
                    "seed": seed,
                    "status": "ok",
                    "raw_response": raw_response,
                    "parsed_json": parsed_payload,
                    "planner_keywords": list(output.retrieval_core),
                    "must_aspects": list(output.must_aspects),
                    "generic_terms": list(output.generic_terms),
                }
                attempts.append(attempt_trace)

                if output.retrieval_core:
                    contract_debug = _planner_contract_debug(
                        raw_must_aspects=_normalize_string_list(
                            parsed_payload.get("must_aspects")
                            or parsed_payload.get("retrieval_core")
                            or []
                        ),
                        contextual_terms=retained_contextual_terms,
                        forbidden_terms=_normalize_string_list(
                            output.role_terms
                            + output.action_terms
                            + output.generic_terms
                        ),
                        fallback_terms=list(output.retrieval_core),
                    )
                    self.last_trace = {
                        "mode": "openai_compat",
                        "cache": {"canonical_plan": "miss"},
                        "planner_version": PLANNER_VERSION,
                        "normalized_query": normalized_query,
                        "planner_retry_count": attempt_index,
                        "planner_keywords": list(output.retrieval_core),
                        "retrieval_keywords": list(output.retrieval_core),
                        "must_aspects": list(output.must_aspects),
                        "generic_terms": list(output.generic_terms),
                        "removed_meta_terms": removed_meta_terms,
                        "retained_contextual_terms": retained_contextual_terms,
                        "intent_flags": dict(output.intent_flags),
                        **contract_debug,
                        "attempts": attempts,
                    }
                    if self.cache:
                        self.cache.set(
                            normalized_query, filters, PLANNER_VERSION, output
                        )
                    logger.info(
                        "LLM planning success: query=%r removed_meta_terms=%s retained_contextual_terms=%s retrieval_core=%s must_aspects=%s generic_terms=%s intent_flags=%s",
                        normalized_query,
                        removed_meta_terms,
                        retained_contextual_terms,
                        output.retrieval_core,
                        output.must_aspects,
                        output.generic_terms,
                        output.intent_flags,
                    )
                    return output

                attempt_trace["status"] = "empty_keywords"
                attempt_trace["reason"] = (
                    "planner_retrieval_core_empty_after_meta_strip"
                )
                logger.warning(
                    "Planner returned no usable retrieval keywords after meta removal: query=%r attempt=%d removed_meta_terms=%s",
                    normalized_query,
                    attempt_index + 1,
                    removed_meta_terms,
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
                    "Planner attempt failed: query=%r attempt=%d reason=%s",
                    normalized_query,
                    attempt_index + 1,
                    exc,
                )

        # 4. LLM 플래닝이 모두 실패했거나 유효한 검색어가 추출되지 않은 경우, 폴백(Fallback) 방식을 활성화합니다.
        # 쿼리에서 메타 용어만 제거한 단순 키워드 목록을 도메인 검색어로 사용합니다.
        fallback_query = _strip_terms(normalized_query, removed_meta_terms)
        fallback_keywords = _split_fallback_tokens(fallback_query)
        fallback_output = PlannerOutput(
            intent_summary=f"[Fallback] {normalized_query}",
            hard_filters=dict(filters),
            include_orgs=list(include_orgs or []),
            exclude_orgs=list(exclude_orgs or []),
            retrieval_core=fallback_keywords,
            core_keywords=fallback_keywords,
            must_aspects=fallback_keywords,
            generic_terms=detected_generic_terms,
            role_terms=detected_role_terms,
            action_terms=detected_action_terms,
            top_k=top_k or 15,
        )
        fallback_output = self._apply_request_constraints(
            output=fallback_output,
            normalized_query=normalized_query,
            filters_override=filters_override,
            include_orgs=include_orgs,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
            contextual_terms=retained_contextual_terms,
        )
        contract_debug = _planner_contract_debug(
            raw_must_aspects=fallback_keywords,
            contextual_terms=retained_contextual_terms,
            forbidden_terms=_normalize_string_list(
                fallback_output.role_terms
                + fallback_output.action_terms
                + fallback_output.generic_terms
            ),
            fallback_terms=list(fallback_output.retrieval_core),
        )
        self.last_trace = {
            "mode": "fallback_broad_search",
            "cache": {"canonical_plan": "miss"},
            "planner_version": PLANNER_VERSION,
            "normalized_query": normalized_query,
            "planner_retry_count": max(0, len(attempts) - 1),
            "planner_keywords": list(fallback_output.retrieval_core),
            "retrieval_keywords": list(fallback_output.retrieval_core),
            "must_aspects": list(fallback_output.must_aspects),
            "generic_terms": list(fallback_output.generic_terms),
            "removed_meta_terms": removed_meta_terms,
            "retained_contextual_terms": retained_contextual_terms,
            "intent_flags": dict(fallback_output.intent_flags),
            **contract_debug,
            "fallback_terms": list(fallback_output.retrieval_core),
            "reason": "planner_retry_exhausted_or_empty",
            "attempts": attempts,
        }
        logger.warning(
            "Planner fallback activated: query=%r removed_meta_terms=%s retained_contextual_terms=%s fallback_terms=%s intent_flags=%s",
            normalized_query,
            removed_meta_terms,
            retained_contextual_terms,
            fallback_output.retrieval_core,
            fallback_output.intent_flags,
        )
        return fallback_output
