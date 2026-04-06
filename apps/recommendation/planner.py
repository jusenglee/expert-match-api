from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from apps.core.config import Settings
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.domain.models import PlannerOutput

logger = logging.getLogger(__name__)


def _extract_json_object_text(content: Any) -> str:
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


class Planner(Protocol):
    async def plan(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput:
        ...


class HeuristicPlanner:
    async def plan(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput:
        # 테스트 전용 planner다.
        # 운영 경로에서는 사용하지 않고, API/검색 파이프라인 계약을 빠르게 검증할 때만 쓴다.
        hard_filters = dict(filters_override or {})
        exclude_list = list(exclude_orgs or [])
        branch_hints = {
            "basic": "전문가 프로필 적합성 중심으로 질의를 정제한다.",
            "art": "논문과 최근 연구실적 중심으로 질의를 정제한다.",
            "pat": "특허와 사업화 키워드 중심으로 질의를 정제한다.",
            "pjt": "과제수행과 전문기관 경험 중심으로 질의를 정제한다.",
        }
        soft_preferences: list[str] = []

        recent_match = re.search(r"최근\s*(\d+)\s*년", query)
        if recent_match:
            years = int(recent_match.group(1))
            hard_filters.setdefault("art_recent_years", years)
            hard_filters.setdefault("pjt_recent_years", years)
            soft_preferences.append(f"최근 {years}년 실적")

        if "SCIE" in query.upper():
            hard_filters.setdefault("art_sci_slct_nm", "SCIE")
            soft_preferences.append("SCIE 논문 선호")

        if "박사" in query:
            hard_filters.setdefault("degree_slct_nm", "박사")
        elif "석사" in query:
            hard_filters.setdefault("degree_slct_nm", ["석사", "박사"])

        if "특허" in query:
            hard_filters.setdefault("patent_cnt_min", 1)
            branch_hints["pat"] = "발명명, 등록 특허, 사업화 가능성 중심으로 정제한다."
            soft_preferences.append("특허 근거 선호")

        if "논문" in query:
            hard_filters.setdefault("article_cnt_min", 1)
            branch_hints["art"] = "논문명, 키워드, 초록, 학술지 중심으로 정제한다."

        if "과제" in query or "연구수행" in query:
            hard_filters.setdefault("project_cnt_min", 1)
            branch_hints["pjt"] = "과제명, 목표, 내용, 수행기관 중심으로 정제한다."

        exclude_patterns = re.findall(r"([A-Za-z가-힣0-9()주식회사㈜]+)\s*제외", query)
        for item in exclude_patterns:
            if item not in exclude_list:
                exclude_list.append(item)

        return PlannerOutput(
            intent_summary=query.strip(),
            hard_filters=hard_filters,
            exclude_orgs=exclude_list,
            soft_preferences=soft_preferences,
            branch_query_hints=branch_hints,
            top_k=top_k or 5,
        )


class OpenAICompatPlanner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fallback = HeuristicPlanner()
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

    async def plan(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput:
        # LLM planner는 branch on/off를 하지 않고,
        # hard filter와 branch별 질의 보정 힌트만 생성한다.
        system_prompt = (
            "You are a planner for NTIS evaluator recommendation. "
            "Return exactly one JSON object and no prose. Do not use markdown, code fences, or extra keys. "
            "The object must contain these keys: "
            "intent_summary(string), hard_filters(object), exclude_orgs(array of strings), "
            "soft_preferences(array of strings), branch_query_hints(object), top_k(integer). "
            "branch_query_hints must be an object with exactly these keys: basic, art, pat, pjt. "
            "Each branch_query_hints value must be a string. Never return branch_query_hints as a list. "
            "Do not decide branch on/off; only populate branch_query_hints. "
            "hard_filters must contain only deterministic filter conditions."
        )
        user_prompt = json.dumps(
            {
                "query": query,
                "filters_override": filters_override or {},
                "exclude_orgs": exclude_orgs or [],
                "top_k": top_k or 5,
            },
            ensure_ascii=False,
        )
        try:
            result = await self.model.ainvoke_non_stream(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            payload = json.loads(_extract_json_object_text(result.content))
            return PlannerOutput.model_validate(payload)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Planner fallback activated: %s", exc)
            return await self.fallback.plan(
                query=query,
                filters_override=filters_override,
                exclude_orgs=exclude_orgs,
                top_k=top_k,
            )
