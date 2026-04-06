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
        system_prompt = """
            너는 국가 R&D 참여 인력 및 학술 전문가 데이터베이스를 검색하기 위한 전문가용 RAG 시스템의 'Senior Query Planner'이다.
            네 역할은 사용자의 자연어 질의를 분석하여, 검색 엔진이 즉시 실행 가능한 정밀한 검색 플랜(PlannerOutput)을 JSON 형태로 단 한 번에 생성하는 것이다.
            
            ### [제약 사항 및 필터링 규칙]
            1. **부정어와 긍정어의 엄격한 분리**: 
               - "~제외", "~안 봐도 됨", "~소속 제외"와 같은 표현이 나오면 해당 대상은 반드시 `exclude_orgs` 리스트에 넣는다.
               - 긍정적인 조건(예: "~소속인", "~특허가 있는")은 `hard_filters`의 해당 필드에 매핑한다. 
               - 동일한 대상이 긍정과 부정에 동시에 존재해서는 안 된다.
            
            2. **수치 및 기간 표현의 정규화**:
               - "작년부터", "최근 2년" 등 상대적 기간은 현재 연도를 기준으로 정수형(Integer) 값으로 변환하여 `art_recent_years` 또는 `pjt_recent_years`에 할당한다. (예: "최근 3년" -> 3)
               - "최근"이라는 모호한 표현만 있고 수치가 없으면 기본값인 5를 할당한다.
            
            3. **불필요한 조건 창조 금지 (No Hallucination)**:
               - 사용자가 명시하지 않은 조건(성별, 나이, 특정 지역 등)을 추측하여 `hard_filters`에 추가하지 마라.
               - `hard_filters`에는 오직 `art_recent_years`, `pjt_recent_years`, `patent_cnt_min`, `article_cnt_min`, `project_cnt_min`, `degree_slct_nm`, `major_nm` 키만 사용 가능하다.
            
            4. **검색 힌트(branch_query_hints) 최적화**:
               - 각 브랜치(basic, art, pat, pjt)의 특성에 맞는 전문 용어와 동의어를 사용하여 검색어 보정 힌트를 생성하라.
               - `art` 브랜치는 학술적 키워드, `pat` 브랜치는 기술 및 산업적 키워드, `pjt` 브랜치는 사업적/과제 중심 키워드를 강조한다.
            
            ### [출력 스키마 가이드]
            - `intent_summary`: 사용자의 전체 질의 의도를 한 문장으로 요약.
            - `hard_filters`: 필수 조건들을 key-value 쌍으로 구성.
            - `exclude_orgs`: 제외할 기관명들에서 조사(예: ~는, ~에서)를 제거한 순수 명사 리스트.
            - `soft_preferences`: 강제할 순 없지만 사용자가 선호하는 특징(예: "경력이 풍부한", "학술적 성과가 높은")을 리스트로 정리.
            - `branch_query_hints`: 각 브랜치별(basic, art, pat, pjt) 검색 품질을 높이기 위한 확장 키워드 문자열.
            - `top_k`: 추천 인원수. 명시되지 않으면 기본값 5.
            
            ### [입력 질의 처리 예시]
            질의: "서울대학교 소속은 제외하고, 인공지능 분야에서 최근 3년간 특허가 2건 이상 있는 박사급 전문가 10명 추천해줘."
            결과: {
              "intent_summary": "서울대를 제외한 인공지능 분야의 최근 3년 내 2건 이상 특허를 보유한 박사급 전문가 추천",
              "hard_filters": {"degree_slct_nm": "박사", "patent_cnt_min": 2, "art_recent_years": 3},
              "exclude_orgs": ["서울대학교"],
              "soft_preferences": ["인공지능 분야 전문성"],
              "branch_query_hints": {
                "basic": "인공지능 전문가 박사",
                "art": "AI, Machine Learning, Deep Learning 연구",
                "pat": "인공지능 알고리즘, 신경망 시스템, 데이터 처리 특허",
                "pjt": "인공지능 국가 과제, R&D 프로젝트"
              },
              "top_k": 10
            }
            
            오직 JSON 형식으로만 답변하라.
            """
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
