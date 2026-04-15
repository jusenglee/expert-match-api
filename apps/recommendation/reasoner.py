from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from apps.core.config import Settings
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text
from apps.core.llm_policies import build_consistency_invoke_kwargs
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.core.utils import build_deterministic_seed
from apps.domain.models import CandidateCard, PlannerOutput
from apps.recommendation.evidence_selector import RelevantEvidenceBundle

logger = logging.getLogger(__name__)


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


class ReasonedCandidate(BaseModel):
    expert_id: str
    fit: str = "보통"
    recommendation_reason: str = ""
    risks: list[str] = Field(default_factory=list)

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
    ) -> ReasonGenerationOutput:
        _ = (query, plan, relevant_evidence_by_expert_id)
        output = ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id=candidate.expert_id,
                    fit="보통",
                    recommendation_reason="",
                    risks=list(candidate.risks),
                )
                for candidate in candidates
            ]
        )
        self.last_trace = {
            "mode": "pass_through",
            "candidate_count": len(candidates),
            "output_count": len(output.items),
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
    def _build_system_prompt() -> str:
        prompt = """
            Role:
            You generate recommendation reasons for already-ranked expert candidates.

            Task:
            - Do not reorder candidates.
            - Do not drop candidates.
            - Do not create new expert ids.
            - For each candidate, write one grounded recommendation reason based only on the provided data.
            - Use only the provided relevant_papers, relevant_projects, and relevant_patents as direct evidence.
            - Do not claim direct query evidence from counts, profile fields, or missing evidence arrays alone.
            - If relevant evidence arrays are empty, explicitly say direct evidence matching the query was not found and stay cautious.
            - Do not hallucinate evidence that is not present in the payload.
            - Keep reasons concise and factual.
            - Risks are optional.
            - Return JSON only.

            Output schema:
            {
              "items": [
                {
                  "expert_id": "string",
                  "fit": "높음|중간|보통",
                  "recommendation_reason": "string",
                  "risks": ["string"]
                }
              ],
              "data_gaps": ["string"]
            }
        """
        return textwrap.dedent(prompt).strip()

    @staticmethod
    def _serialize_evidence_items(items: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "title": item.title,
                "date": item.date,
                "detail": item.detail,
                "snippet": item.snippet,
                "matched_keywords": list(item.matched_keywords),
            }
            for item in items
        ]

    @classmethod
    def _serialize_candidates(
        cls,
        candidates: list[CandidateCard],
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle] | None = None,
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        relevant_evidence_by_expert_id = relevant_evidence_by_expert_id or {}
        for candidate in candidates:
            relevant_bundle = relevant_evidence_by_expert_id.get(
                candidate.expert_id,
                RelevantEvidenceBundle(expert_id=candidate.expert_id),
            )
            serialized.append(
                {
                    "expert_id": candidate.expert_id,
                    "name": candidate.name,
                    "organization": candidate.organization,
                    "position": candidate.position,
                    "degree": candidate.degree,
                    "major": candidate.major,
                    "rank_score": candidate.rank_score,
                    "counts": dict(candidate.counts),
                    "matched_filter_summary": list(candidate.matched_filter_summary),
                    "data_gaps": list(candidate.data_gaps),
                    "relevant_papers": cls._serialize_evidence_items(
                        relevant_bundle.papers
                    ),
                    "relevant_patents": cls._serialize_evidence_items(
                        relevant_bundle.patents
                    ),
                    "relevant_projects": cls._serialize_evidence_items(
                        relevant_bundle.projects
                    ),
                }
            )
        return serialized

    @staticmethod
    def _normalize_output(
        output: ReasonGenerationOutput,
        candidates: list[CandidateCard],
    ) -> ReasonGenerationOutput:
        by_expert_id = {item.expert_id: item for item in output.items}
        normalized_items: list[ReasonedCandidate] = []
        for candidate in candidates:
            item = by_expert_id.get(candidate.expert_id)
            if item is None:
                normalized_items.append(
                    ReasonedCandidate(
                        expert_id=candidate.expert_id,
                        fit="보통",
                        recommendation_reason="",
                        risks=list(candidate.risks),
                    )
                )
                continue
            normalized_items.append(
                ReasonedCandidate(
                    expert_id=candidate.expert_id,
                    fit=item.fit if item.fit in {"높음", "중간", "보통"} else "보통",
                    recommendation_reason=" ".join(item.recommendation_reason.split()),
                    risks=list(item.risks),
                )
            )
        return ReasonGenerationOutput(
            items=normalized_items,
            data_gaps=list(output.data_gaps),
        )

    async def generate(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        candidates: list[CandidateCard],
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle] | None = None,
    ) -> ReasonGenerationOutput:
        if not candidates:
            self.last_trace = {
                "mode": "openai_compat",
                "candidate_count": 0,
                "output_count": 0,
            }
            return ReasonGenerationOutput()

        serialized_candidates = self._serialize_candidates(
            candidates,
            relevant_evidence_by_expert_id,
        )
        seed = build_deterministic_seed(
            "reason_generation",
            query,
            plan.model_dump(mode="json"),
            [candidate["expert_id"] for candidate in serialized_candidates],
        )
        invoke_kwargs = build_consistency_invoke_kwargs(seed=seed)
        payload = {
            "query": query,
            "core_keywords": list(plan.core_keywords),
            "task_terms": list(plan.task_terms),
            "candidates": serialized_candidates,
        }

        try:
            result = await self.model.ainvoke_non_stream(
                [
                    SystemMessage(content=self._build_system_prompt()),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ],
                **invoke_kwargs,
            )
            json_text = _extract_json_object_text(result.content)
            parsed = ReasonGenerationOutput.model_validate_json(json_text)
            normalized = self._normalize_output(parsed, candidates)
            self.last_trace = {
                "mode": "openai_compat",
                "candidate_count": len(candidates),
                "output_count": len(normalized.items),
                "seed": seed,
            }
            return normalized
        except Exception as exc:
            logger.warning("Reason generator fallback activated: reason=%s", exc)
            fallback_output = await self.fallback.generate(
                query=query,
                plan=plan,
                candidates=candidates,
                relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
            )
            self.last_trace = {
                "mode": "fallback",
                "candidate_count": len(candidates),
                "output_count": len(fallback_output.items),
                "seed": seed,
                "reason": str(exc),
            }
            return fallback_output
