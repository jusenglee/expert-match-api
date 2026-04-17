"""
Planner implementations for recommendation search intent extraction.
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
PLANNER_VERSION = "v0.7.0"

ROLE_QUERY_TERMS = (
    "expert recommendation",
    "reviewer recommendation",
    "evaluation committee",
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
)
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

CONTEXTUAL_EVALUATION_PHRASES: tuple[tuple[str, str], ...] = (
    ("과제 평가", r"(?:과제|프로젝트)(?:를|을)?\s*평가"),
    ("기술 평가", r"기술(?:을|를)?\s*평가"),
    ("논문 평가", r"논문(?:을|를)?\s*평가"),
    ("project evaluation", r"\bprojects?\s+evaluation\b|\bevaluate\s+projects?\b"),
    ("technology evaluation", r"\btechnology\s+evaluation\b|\bevaluate\s+technolog(?:y|ies)\b"),
    ("paper evaluation", r"\bpapers?\s+evaluation\b|\bevaluate\s+papers?\b"),
)

GENERIC_SINGLETON_MUST_ASPECTS = {
    "기술",
    "개발",
    "연구",
    "과제",
    "분석",
    "시스템",
    "플랫폼",
    "technology",
    "development",
    "research",
    "project",
    "analysis",
    "system",
    "platform",
}

GENERIC_PHRASE_MUST_ASPECTS = {
    "ai 기반",
    "데이터 기반",
    "시스템 개발",
    "기술 개발",
    "ai based",
    "data driven",
    "system development",
    "technology development",
}


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


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _normalize_string_list(values: list[str] | None) -> list[str]:
    normalized_values: list[str] = []
    for value in values or []:
        normalized = _normalize_text(str(value))
        if normalized and normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


def _compile_term_pattern(terms: list[str]) -> re.Pattern[str] | None:
    normalized_terms = _normalize_string_list(terms)
    if not normalized_terms:
        return None
    escaped_terms = sorted(
        {re.escape(term) for term in normalized_terms},
        key=len,
        reverse=True,
    )
    return re.compile("|".join(escaped_terms), flags=re.IGNORECASE)


def _extract_query_terms(query: str, candidates: tuple[str, ...]) -> list[str]:
    normalized_query = _normalize_text(query).lower()
    found_terms: list[str] = []
    for candidate in sorted({_normalize_text(term) for term in candidates if term}, key=len, reverse=True):
        if candidate.lower() in normalized_query and candidate not in found_terms:
            found_terms.append(candidate)
    return found_terms


def _strip_terms(value: str, terms: list[str]) -> str:
    pattern = _compile_term_pattern(terms)
    if pattern is None:
        return _normalize_text(value)
    return _normalize_text(pattern.sub(" ", value))


def _filter_domain_terms(values: list[str], *, forbidden_terms: list[str]) -> list[str]:
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
    contextual_lookup = {value.lower() for value in _normalize_string_list(contextual_terms)}
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
    """LLM-backed planner that extracts retrieval-safe domain requirements."""

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
        - `retrieval_core`: Korean-only domain terms for sparse BM25 retrieval. No English.
        - `must_aspects`: Korean-only core aspects for semantic quality description. Leave empty if unsure.
        - `evidence_aspects`: BILINGUAL list of specific terms likely to appear literally in paper
          titles, abstracts, keywords, or project/patent names in this domain. Include both Korean
          and English forms that researchers actually write. These are used for direct evidence
          matching against Korean/English mixed research databases. Aim for 4-8 specific terms.
          Example for "의료영상 분석 AI": ["의료영상 분석", "medical image analysis",
          "딥러닝", "deep learning", "영상 진단", "image segmentation", "CT 영상", "MRI"]
        - `generic_terms`: generic request words such as broad capability or experience words.
        - `role_terms`: meta nouns such as expert, reviewer, evaluation committee.
        - `action_terms`: request verbs such as recommend, find, search.
        - `bundle_ids`: optional expansion bundle ids chosen only from the bundle list below.
        - `semantic_query`: one natural sentence for dense retrieval.
        - `intent_flags`, `hard_filters`, `include_orgs`, `exclude_orgs`, `top_k`: copy only when explicitly supported by the request.

        Rules:
        1. Preserve the user's language.
        2. Put only Korean domain terms in `retrieval_core`.
        3. Put meta request words in `role_terms`, `action_terms`, or `generic_terms`, not in `retrieval_core`.
        4. Do not invent filters, bundle ids, or aspects.
        5. Return empty arrays instead of fabricated terms.
        6. `evidence_aspects` must contain only specific domain terms — no meta words, no generic
           words like "기술", "개발", "연구". Include English equivalents of key Korean terms.

        Expansion bundles:
        {lexicon_summary}

        Output schema:
        {{
          "intent_summary": "string",
          "retrieval_core": ["string"],
          "must_aspects": ["string"],
          "evidence_aspects": ["string"],
          "generic_terms": ["string"],
          "bundle_ids": ["string"],
          "semantic_query": "string",
          "role_terms": ["string"],
          "action_terms": ["string"],
          "intent_flags": {{}},
          "hard_filters": {{}},
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
                term for term in generic_terms if term.lower() not in {"평가", "evaluation"}
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
        output.intent_summary = _normalize_text(output.intent_summary) or normalized_query

        role_terms, action_terms, generic_terms, query_contextual_terms = cls._extract_meta_terms(
            normalized_query
        )
        effective_contextual_terms = _normalize_string_list(
            list(contextual_terms or []) + query_contextual_terms
        )
        output.role_terms = _normalize_string_list(output.role_terms + role_terms)
        output.action_terms = _normalize_string_list(output.action_terms + action_terms)
        output.generic_terms = _normalize_string_list(output.generic_terms + generic_terms)
        if effective_contextual_terms:
            output.generic_terms = [
                term
                for term in output.generic_terms
                if term.lower() not in {"평가", "evaluation"}
            ]

        forbidden_terms = _normalize_string_list(
            output.role_terms + output.action_terms + output.generic_terms
        )

        raw_retrieval_terms = _normalize_string_list(
            output.retrieval_core or output.core_keywords
        )
        output.retrieval_core = _filter_domain_terms(
            raw_retrieval_terms,
            forbidden_terms=forbidden_terms,
        )
        contextual_lookup = {
            value.lower() for value in _normalize_string_list(effective_contextual_terms)
        }
        if contextual_lookup:
            output.retrieval_core = [
                term
                for term in output.retrieval_core
                if term.lower() not in contextual_lookup
            ]
        output.core_keywords = list(output.retrieval_core)
        raw_must_aspects = _filter_domain_terms(
            output.must_aspects or output.retrieval_core or output.core_keywords,
            forbidden_terms=forbidden_terms,
        )
        if not raw_must_aspects:
            raw_must_aspects = list(output.retrieval_core)
        output.must_aspects, _ = _prune_must_aspects(
            raw_must_aspects,
            contextual_terms=effective_contextual_terms,
        )
        output.task_terms = _normalize_string_list(output.role_terms + output.action_terms)
        output.intent_flags = dict(output.intent_flags or {})
        if effective_contextual_terms:
            output.intent_flags["review_context"] = True
            output.intent_flags["review_targets"] = effective_contextual_terms
        else:
            output.intent_flags.pop("review_context", None)
            output.intent_flags.pop("review_targets", None)

        output.bundle_ids = [
            bundle_id
            for bundle_id in _normalize_string_list(output.bundle_ids)
            if bundle_id in EXPANSION_LEXICON
        ]
        if not output.semantic_query:
            output.semantic_query = (
                " ".join(output.retrieval_core) if output.retrieval_core else output.intent_summary
            )

        # evidence_aspects 정제: meta/generic 용어 제거 후 must_aspects 기반 폴백 생성.
        # evidence_aspects는 한국어+영어 혼합으로 실제 evidence text에 등장할 용어만 포함.
        raw_evidence_aspects = _normalize_string_list(output.evidence_aspects)
        cleaned_evidence_aspects = _filter_domain_terms(
            raw_evidence_aspects,
            forbidden_terms=forbidden_terms,
        )
        # generic singleton/phrase 필터 적용 (단순 "기술", "AI 기반" 등 제거)
        cleaned_evidence_aspects = [
            term for term in cleaned_evidence_aspects
            if not _is_generic_must_aspect(term)
        ]
        if cleaned_evidence_aspects:
            output.evidence_aspects = cleaned_evidence_aspects
        else:
            # LLM이 evidence_aspects를 생성하지 않은 경우: must_aspects에서 자동 생성.
            # must_aspects는 한국어 구이므로 그대로 사용(bilingual 확장은 LLM 재시도 시 이루어짐).
            output.evidence_aspects = list(output.must_aspects)

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
        normalized_query = _normalize_text(query)
        filters = filters_override or {}
        (
            detected_role_terms,
            detected_action_terms,
            detected_generic_terms,
            retained_contextual_terms,
        ) = self._extract_meta_terms(normalized_query)
        removed_meta_terms = _normalize_string_list(
            detected_role_terms + detected_action_terms + detected_generic_terms
        )

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
                            output.role_terms + output.action_terms + output.generic_terms
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
                        self.cache.set(normalized_query, filters, PLANNER_VERSION, output)
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
                attempt_trace["reason"] = "planner_retrieval_core_empty_after_meta_strip"
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
