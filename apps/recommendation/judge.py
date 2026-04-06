from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from apps.core.config import Settings
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.domain.models import CandidateCard, EvidenceItem, JudgeOutput, PlannerOutput, RecommendationDecision

logger = logging.getLogger(__name__)


def _extract_json_object_text(content: Any) -> str:
    """Extract the first JSON object from an LLM response payload."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        text = "".join(parts)
    else:
        text = str(content)

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


def _normalize_string_list(value: Any) -> list[str] | Any:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        normalized_items: list[str] = []
        for item in value:
            if item is None:
                continue
            normalized = item.strip() if isinstance(item, str) else str(item).strip()
            if normalized:
                normalized_items.append(normalized)
        return normalized_items
    return value


def _normalize_rank(value: Any, fallback_rank: int) -> int | Any:
    if value is None:
        return fallback_rank
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback_rank
        try:
            return int(stripped)
        except ValueError:
            return value
    return value


def _normalize_judge_payload(payload: Any, shortlist: list[CandidateCard]) -> tuple[Any, bool]:
    if not isinstance(payload, dict):
        return payload, False

    normalized_payload = dict(payload)
    normalized_applied = False
    name_to_expert_ids: dict[str, list[str]] = {}
    for card in shortlist:
        normalized_name = card.name.strip()
        name_to_expert_ids.setdefault(normalized_name, []).append(card.expert_id)

    for key in ("not_selected_reasons", "data_gaps"):
        normalized_value = _normalize_string_list(normalized_payload.get(key))
        if normalized_value != normalized_payload.get(key):
            normalized_payload[key] = normalized_value
            normalized_applied = True

    recommended = normalized_payload.get("recommended")
    if recommended is None:
        normalized_payload["recommended"] = []
        return normalized_payload, True
    if not isinstance(recommended, list):
        return normalized_payload, normalized_applied

    normalized_recommended: list[Any] = []
    for fallback_rank, item in enumerate(recommended, start=1):
        if not isinstance(item, dict):
            return normalized_payload, normalized_applied

        normalized_item = dict(item)

        rank = _normalize_rank(normalized_item.get("rank"), fallback_rank)
        if rank != normalized_item.get("rank"):
            normalized_item["rank"] = rank
            normalized_applied = True

        for key in ("reasons", "risks"):
            normalized_value = _normalize_string_list(normalized_item.get(key))
            if normalized_value != normalized_item.get(key):
                normalized_item[key] = normalized_value
                normalized_applied = True

        name = normalized_item.get("name")
        if isinstance(name, str):
            stripped_name = name.strip()
            if stripped_name != name:
                normalized_item["name"] = stripped_name
                normalized_applied = True
            name = stripped_name

        expert_id = normalized_item.get("expert_id")
        if isinstance(expert_id, str):
            stripped_expert_id = expert_id.strip()
            if stripped_expert_id != expert_id:
                normalized_item["expert_id"] = stripped_expert_id
                normalized_applied = True
            expert_id = stripped_expert_id

        if not expert_id and isinstance(name, str):
            matches = name_to_expert_ids.get(name, [])
            if len(matches) == 1:
                normalized_item["expert_id"] = matches[0]
                normalized_applied = True

        fit = normalized_item.get("fit")
        if isinstance(fit, str):
            stripped_fit = fit.strip()
            if stripped_fit != fit:
                normalized_item["fit"] = stripped_fit
                normalized_applied = True

        normalized_recommended.append(normalized_item)

    normalized_payload["recommended"] = normalized_recommended
    return normalized_payload, normalized_applied


class Judge(Protocol):
    async def judge(self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]) -> JudgeOutput:
        ...


class HeuristicJudge:
    async def judge(self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]) -> JudgeOutput:
        ranked_cards = sorted(shortlist, key=lambda card: card.shortlist_score, reverse=True)
        recommendations: list[RecommendationDecision] = []
        global_data_gaps: list[str] = []

        for rank, card in enumerate(ranked_cards[: max(plan.top_k, 3)], start=1):
            reasons: list[str] = []
            evidence: list[EvidenceItem] = []

            if card.branch_coverage.get("art"):
                reasons.append("논문 근거가 확인됩니다.")
                if card.top_papers:
                    evidence.append(
                        EvidenceItem(
                            type="paper",
                            title=card.top_papers[0].publication_title,
                            date=card.top_papers[0].publication_year_month,
                            detail=card.top_papers[0].journal_name,
                        )
                    )

            if card.branch_coverage.get("pat"):
                reasons.append("특허 근거가 확인됩니다.")
                if card.top_patents:
                    evidence.append(
                        EvidenceItem(
                            type="patent",
                            title=card.top_patents[0].intellectual_property_title,
                            date=card.top_patents[0].registration_date or card.top_patents[0].application_date,
                            detail=card.top_patents[0].application_registration_type,
                        )
                    )

            if card.branch_coverage.get("pjt"):
                reasons.append("과제 수행 근거가 확인됩니다.")
                if card.top_projects:
                    evidence.append(
                        EvidenceItem(
                            type="project",
                            title=card.top_projects[0].display_title,
                            date=card.top_projects[0].project_end_date or card.top_projects[0].project_start_date,
                            detail=card.top_projects[0].managing_agency,
                        )
                    )

            if card.branch_coverage.get("basic"):
                reasons.append("전문가 프로필 적합성이 확인됩니다.")
                evidence.append(
                    EvidenceItem(
                        type="profile",
                        title=f"{card.name} / {card.organization or '소속 미상'}",
                        detail=f"{card.degree or '학위 미상'} / {card.major or '전공 미상'}",
                    )
                )

            if card.data_gaps:
                global_data_gaps.extend(card.data_gaps)

            fit = "높음" if card.shortlist_score >= 20 else "중간" if card.shortlist_score >= 10 else "보통"
            recommendations.append(
                RecommendationDecision(
                    rank=rank,
                    expert_id=card.expert_id,
                    name=card.name,
                    fit=fit,
                    reasons=reasons[:3],
                    evidence=evidence[:4],
                    risks=card.risks + card.data_gaps[:1],
                )
            )

        unique_gaps = list(dict.fromkeys(global_data_gaps))
        not_selected_reasons: list[str] = []
        if len(ranked_cards) > len(recommendations):
            not_selected_reasons.append("상위 추천 대비 근거 다양성이나 최근성이 상대적으로 낮습니다.")

        return JudgeOutput(
            recommended=recommendations[:5],
            not_selected_reasons=not_selected_reasons,
            data_gaps=unique_gaps,
        )


class OpenAICompatJudge:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fallback = HeuristicJudge()
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

    async def judge(self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]) -> JudgeOutput:
        system_prompt = """
            You are the senior recommendation judge for a Korean R&D evaluator recommendation system.
            Return exactly one JSON object that matches this schema:
            {
              "recommended": [
                {
                  "rank": 1,
                  "expert_id": "shortlist expert_id",
                  "name": "shortlist name",
                  "fit": "높음|중간|보통",
                  "reasons": ["reason"],
                  "evidence": [
                    {
                      "type": "paper|patent|project|profile",
                      "title": "exact shortlist evidence title",
                      "date": "optional date",
                      "detail": "optional detail"
                    }
                  ],
                  "risks": ["risk"]
                }
              ],
              "not_selected_reasons": ["reason"],
              "data_gaps": ["gap"]
            }

            Rules:
            - Copy `expert_id` exactly from the shortlist input.
            - Every recommendation must include `rank`, `expert_id`, `name`, `fit`, `reasons`, `evidence`, and `risks`.
            - `reasons`, `risks`, `not_selected_reasons`, and `data_gaps` must always be arrays of strings, never a single string.
            - Only use evidence that already exists in `top_papers`, `top_patents`, `top_projects`, or the profile fields of the shortlist input.
            - If no candidate should be recommended, return `recommended=[]` and explain why in `not_selected_reasons` and/or `data_gaps`.
            - Do not return markdown, code fences, or any prose outside the JSON object.
        """
        user_payload = {
            "query": query,
            "plan": plan.model_dump(mode="json"),
            "shortlist": [card.model_dump(mode="json") for card in shortlist],
        }

        normalized_recommendation_count = 0
        try:
            result = await self.model.ainvoke_non_stream(
                [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(user_payload, ensure_ascii=False))]
            )
            raw_payload = json.loads(_extract_json_object_text(result.content))
            normalized_payload, normalized_applied = _normalize_judge_payload(raw_payload, shortlist)
            if isinstance(normalized_payload, dict):
                recommended = normalized_payload.get("recommended", [])
                normalized_recommendation_count = len(recommended) if isinstance(recommended, list) else 0
            if normalized_applied:
                logger.info(
                    "Judge response normalized before validation: shortlist_count=%d normalized_recommendation_count=%d",
                    len(shortlist),
                    normalized_recommendation_count,
                )
            return JudgeOutput.model_validate(normalized_payload)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Judge fallback activated: reason=%s shortlist_count=%d normalized_recommendation_count=%d fallback_used=true",
                exc,
                len(shortlist),
                normalized_recommendation_count,
            )
            return await self.fallback.judge(query=query, plan=plan, shortlist=shortlist)
