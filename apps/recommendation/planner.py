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


def _normalize_string_list(values: list[str] | None) -> list[str]:
    normalized_values: list[str] = []
    for value in values or []:
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
            task_terms=[],
            core_keywords=[],
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
        return output


class OpenAICompatPlanner:
    """LLM-backed planner that extracts pure retrieval keywords and explicit request terms."""

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
    def _build_system_prompt() -> str:
        prompt = """
            Role:
            You extract retrieval-safe domain keywords for an expert recommendation search system.

            Goal:
            Separate pure domain keywords from request-role terms.

            Rules:
            - Return JSON only.
            - `core_keywords` must contain domain nouns or noun phrases only.
            - `task_terms` must contain request-role or action terms only.
            - Do not translate, paraphrase, or invent new domain keywords.
            - Do not add branch hints, retrieval views, or rewritten search sentences.
            - Preserve explicit filters and organization constraints exactly when they are provided.
            - If no safe domain keywords remain, return `core_keywords: []`.

            Output schema:
            {
              "intent_summary": "string",
              "core_keywords": ["string"],
              "task_terms": ["string"],
              "hard_filters": {},
              "include_orgs": ["string"],
              "exclude_orgs": ["string"],
              "top_k": 15
            }

            Example:
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
              "intent_summary": "난접근성 화재 진압과 드론 접목 전문가 탐색",
              "core_keywords": ["난접근성 화재 진압", "드론"],
              "task_terms": ["전문가 추천"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
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
        payload = {
            "query": normalized_query,
            "filters_override": filters_override or {},
            "include_orgs": include_orgs or [],
            "exclude_orgs": exclude_orgs or [],
            "top_k": top_k or 15,
        }
        attempts: list[dict[str, Any]] = []

        for attempt_index in range(MAX_PLANNER_ATTEMPTS):
            seed = build_deterministic_seed("planner", payload, attempt_index)
            logger.info(
                "LLM planning start: query=%s attempt=%d",
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
                )
                attempt_trace = {
                    "attempt": attempt_index + 1,
                    "seed": seed,
                    "status": "ok",
                    "raw_response": raw_response,
                    "parsed_json": parsed_payload,
                    "planner_keywords": list(output.core_keywords),
                }
                attempts.append(attempt_trace)

                if output.core_keywords:
                    self.last_trace = {
                        "mode": "openai_compat",
                        "normalized_query": normalized_query,
                        "planner_retry_count": attempt_index,
                        "planner_keywords": list(output.core_keywords),
                        "retrieval_keywords": list(output.core_keywords),
                        "attempts": attempts,
                    }
                    logger.info(
                        "LLM planning success: intent=%r keywords=%d include_orgs=%d exclude_orgs=%d filters=%r",
                        output.intent_summary,
                        len(output.core_keywords),
                        len(output.include_orgs),
                        len(output.exclude_orgs),
                        output.hard_filters,
                    )
                    return output

                attempt_trace["status"] = "empty_keywords"
                attempt_trace["reason"] = "planner_core_keywords_empty"
                logger.warning(
                    "Planner returned empty core_keywords: query=%s attempt=%d",
                    normalized_query,
                    attempt_index + 1,
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
                    "Planner attempt failed: query=%s attempt=%d reason=%s",
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

        last_planner_keywords: list[str] = []
        for attempt in reversed(attempts):
            if attempt.get("planner_keywords"):
                last_planner_keywords = list(attempt["planner_keywords"])
                break

        self.last_trace = {
            "mode": "deterministic_fallback",
            "normalized_query": normalized_query,
            "planner_retry_count": max(0, len(attempts) - 1),
            "planner_keywords": last_planner_keywords,
            "retrieval_keywords": [],
            "reason": "planner_retry_exhausted",
            "attempts": attempts,
        }
        return fallback_output
