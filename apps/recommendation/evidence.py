from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from apps.core.config import Settings
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text
from apps.core.llm_policies import build_consistency_invoke_kwargs
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.core.utils import build_deterministic_seed
from apps.domain.models import EvidenceItem, ExpertPayload, PlannerOutput, RecommendationDecision

logger = logging.getLogger(__name__)


def _normalize_string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        normalized = " ".join(values.split())
        return [normalized] if normalized else []
    if isinstance(values, list):
        normalized_values: list[str] = []
        for value in values:
            normalized = " ".join(str(value).split())
            if normalized and normalized not in normalized_values:
                normalized_values.append(normalized)
        return normalized_values
    return []


class _EvidenceResolverOutput(BaseModel):
    alignment_passed: bool = False
    selected_option_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("selected_option_ids", "notes", mode="before")
    @classmethod
    def _normalize_lists(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)


@dataclass(slots=True)
class EvidenceResolutionResult:
    evidence: list[EvidenceItem]
    alignment_passed: bool
    source_option_ids: list[str]
    notes: list[str]
    status: str
    resolver_mode: str


class EvidenceResolver(Protocol):
    async def resolve(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        recommendation: RecommendationDecision,
        payload: ExpertPayload | None,
    ) -> EvidenceResolutionResult: ...


class PassThroughEvidenceResolver:
    async def resolve(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        recommendation: RecommendationDecision,
        payload: ExpertPayload | None,
    ) -> EvidenceResolutionResult:
        _ = (query, plan, payload)
        evidence = list(recommendation.evidence)
        return EvidenceResolutionResult(
            evidence=evidence,
            alignment_passed=bool(evidence),
            source_option_ids=[],
            notes=[],
            status="pass_through",
            resolver_mode="pass_through",
        )


class OpenAICompatEvidenceResolver:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self._semaphore = asyncio.Semaphore(max(1, settings.llm_judge_max_concurrency))

    @staticmethod
    def _build_system_prompt() -> str:
        prompt = """
            Role:
            You verify which evidence items can safely be shown in the UI for one expert recommendation.

            Task:
            Select only the evidence options that directly support both:
            1. the user query
            2. the recommendation reasons

            Rules:
            - Use only the provided `option_id` values.
            - Do not invent or rewrite evidence.
            - Do not translate or paraphrase titles.
            - If no option directly supports the query and reasons, return `alignment_passed: false`.
            - Prefer the smallest supporting set.
            - Return JSON only.

            Output schema:
            {
              "alignment_passed": true,
              "selected_option_ids": ["paper:0", "project:1"],
              "notes": ["string"]
            }
        """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _profile_detail(payload: ExpertPayload) -> str:
        parts = [
            payload.basic_info.affiliated_organization or "unknown organization",
            payload.researcher_profile.highest_degree or "unknown degree",
            payload.researcher_profile.major_field or "unknown major",
        ]
        return " / ".join(parts)

    @classmethod
    def _build_evidence_options(
        cls, payload: ExpertPayload
    ) -> tuple[dict[str, EvidenceItem], list[dict[str, Any]]]:
        option_map: dict[str, EvidenceItem] = {}
        serialized_options: list[dict[str, Any]] = []

        profile_option_id = "profile:0"
        profile_item = EvidenceItem(
            type="profile",
            title=payload.basic_info.researcher_name,
            detail=cls._profile_detail(payload),
        )
        option_map[profile_option_id] = profile_item
        serialized_options.append(
            {
                "option_id": profile_option_id,
                "type": "profile",
                "title": profile_item.title,
                "detail": profile_item.detail,
                "organization": payload.basic_info.affiliated_organization,
                "position": payload.basic_info.position_title,
            }
        )

        for index, paper in enumerate(payload.publications):
            option_id = f"paper:{index}"
            item = EvidenceItem(
                type="paper",
                title=paper.publication_title,
                date=paper.publication_year_month,
                detail=paper.journal_name,
            )
            option_map[option_id] = item
            serialized_options.append(
                {
                    "option_id": option_id,
                    "type": "paper",
                    "title": paper.publication_title,
                    "date": paper.publication_year_month,
                    "detail": paper.journal_name,
                    "korean_keywords": list(paper.korean_keywords),
                    "english_keywords": list(paper.english_keywords),
                    "abstract": paper.abstract,
                }
            )

        for index, patent in enumerate(payload.intellectual_properties):
            option_id = f"patent:{index}"
            item = EvidenceItem(
                type="patent",
                title=patent.intellectual_property_title,
                date=patent.registration_date or patent.application_date,
                detail=patent.application_registration_type or patent.application_country,
            )
            option_map[option_id] = item
            serialized_options.append(
                {
                    "option_id": option_id,
                    "type": "patent",
                    "title": patent.intellectual_property_title,
                    "date": patent.registration_date or patent.application_date,
                    "detail": patent.application_registration_type
                    or patent.application_country,
                    "application_country": patent.application_country,
                    "application_number": patent.application_number,
                    "registration_number": patent.registration_number,
                }
            )

        for index, project in enumerate(payload.research_projects):
            option_id = f"project:{index}"
            item = EvidenceItem(
                type="project",
                title=project.display_title,
                date=project.project_end_date or project.project_start_date,
                detail=project.managing_agency or project.performing_organization,
            )
            option_map[option_id] = item
            serialized_options.append(
                {
                    "option_id": option_id,
                    "type": "project",
                    "title": project.display_title,
                    "date": project.project_end_date or project.project_start_date,
                    "detail": project.managing_agency or project.performing_organization,
                    "performing_organization": project.performing_organization,
                    "managing_agency": project.managing_agency,
                    "research_objective_summary": project.research_objective_summary,
                    "research_content_summary": project.research_content_summary,
                }
            )

        return option_map, serialized_options

    @staticmethod
    def _compute_seed(
        *,
        query: str,
        plan: PlannerOutput,
        recommendation: RecommendationDecision,
        serialized_options: list[dict[str, Any]],
    ) -> int:
        recommendation_payload = recommendation.model_dump(
            mode="json",
            exclude={"evidence"},
        )
        return build_deterministic_seed(
            "evidence_resolver",
            query,
            plan.model_dump(mode="json"),
            recommendation_payload,
            serialized_options,
        )

    async def resolve(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        recommendation: RecommendationDecision,
        payload: ExpertPayload | None,
    ) -> EvidenceResolutionResult:
        if payload is None:
            return EvidenceResolutionResult(
                evidence=[],
                alignment_passed=False,
                source_option_ids=[],
                notes=["Expert payload was missing during evidence resolution."],
                status="missing_payload",
                resolver_mode="openai_compat",
            )

        option_map, serialized_options = self._build_evidence_options(payload)
        seed = self._compute_seed(
            query=query,
            plan=plan,
            recommendation=recommendation,
            serialized_options=serialized_options,
        )
        invoke_kwargs = build_consistency_invoke_kwargs(seed=seed)
        resolver_payload = {
            "query": query,
            "core_keywords": list(plan.core_keywords),
            "recommendation": recommendation.model_dump(mode="json", exclude={"evidence"}),
            "evidence_options": serialized_options,
        }

        try:
            async with self._semaphore:
                result = await self.model.ainvoke_non_stream(
                    [
                        SystemMessage(content=self._build_system_prompt()),
                        HumanMessage(content=json.dumps(resolver_payload, ensure_ascii=False)),
                    ],
                    **invoke_kwargs,
                )
            json_text = _extract_json_object_text(result.content)
            parsed_output = _EvidenceResolverOutput.model_validate_json(json_text)
        except Exception as exc:
            logger.warning(
                "Evidence resolver failed: expert_id=%s reason=%s",
                recommendation.expert_id,
                exc,
            )
            return EvidenceResolutionResult(
                evidence=[],
                alignment_passed=False,
                source_option_ids=[],
                notes=[f"Evidence resolver failed: {exc}"],
                status="error",
                resolver_mode="openai_compat",
            )

        selected_option_ids: list[str] = []
        resolved_evidence: list[EvidenceItem] = []
        for option_id in parsed_output.selected_option_ids:
            if option_id in option_map and option_id not in selected_option_ids:
                selected_option_ids.append(option_id)
                resolved_evidence.append(option_map[option_id])

        alignment_passed = bool(parsed_output.alignment_passed and resolved_evidence)
        status = "resolved" if alignment_passed else "unaligned"
        if parsed_output.alignment_passed and not resolved_evidence:
            parsed_output.notes.append("Resolver selected only unknown option ids.")
            status = "invalid_selection"

        return EvidenceResolutionResult(
            evidence=resolved_evidence if alignment_passed else [],
            alignment_passed=alignment_passed,
            source_option_ids=selected_option_ids,
            notes=list(parsed_output.notes),
            status=status,
            resolver_mode="openai_compat",
        )
