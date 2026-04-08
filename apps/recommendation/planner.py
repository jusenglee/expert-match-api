"""
사용자의 자연어 질의를 분석하여 검색 계획(Plan)을 수립하는 모듈입니다.
LLM을 사용하여 의도를 파악하고 적절한 검색 필터와 힌트를 생성하거나,
LLM 사용이 불가능할 경우 규칙 기반(Heuristic)으로 대안을 제시합니다.
"""

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
    """
    LLM 응답 텍스트에서 JSON 객체 부분만 추출합니다.
    마크다운 코드 블록(```json ... ```)이나 불필요한 앞뒤 텍스트를 제거합니다.
    """
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
    # 마크다운 코드 블록 제거
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    # 가장 바깥쪽 중괄호({}) 찾기
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return text


class Planner(Protocol):
    """검색 계획을 수립하는 플래너의 인터페이스 정의입니다."""
    async def plan(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput:
        """자연어 질의를 분석하여 검색 파라미터를 생성합니다."""
        ...


class HeuristicPlanner:
    """
    규칙 기반(정규표현식 등)으로 검색 계획을 수립하는 플래너입니다.
    주로 테스트용이거나 LLM 장애 시 Fallback용으로 사용됩니다.
    """
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
        # 기본 가중치는 모두 1.0
        branch_weights = {
            "basic": 1.0,
            "art": 1.0,
            "pat": 1.0,
            "pjt": 1.0,
        }
        soft_preferences: list[str] = []

        # '최근 N년' 패턴 매칭
        recent_match = re.search(r"최근\s*(\d+)\s*년", query)
        if recent_match:
            years = int(recent_match.group(1))
            hard_filters.setdefault("art_recent_years", years)
            hard_filters.setdefault("pat_recent_years", years)  # 특허도 명시적으로 포함
            hard_filters.setdefault("pjt_recent_years", years)
            soft_preferences.append(f"최근 {years}년 실적")

        # 'SCIE' 등 특정 논문 조건 매칭
        if "SCIE" in query.upper():
            hard_filters.setdefault("art_sci_slct_nm", "SCIE")
            soft_preferences.append("SCIE 논문 선호")

        # 학위 조건 매칭
        if "박사" in query:
            hard_filters.setdefault("degree_slct_nm", "박사")
        elif "석사" in query:
            hard_filters.setdefault("degree_slct_nm", ["석사", "박사"])

        # 특정 실적 존재 여부 매칭 및 브랜치 가중치 조정
        if "특허" in query:
            hard_filters.setdefault("patent_cnt_min", 1)
            branch_hints["pat"] = "발명명, 등록 특허, 사업화 가능성 중심으로 정제한다."
            branch_weights["pat"] = 1.0
            branch_weights["art"] = 0.5  # 특허 강조 시 논문 비중 축소
            soft_preferences.append("특허 근거 선호")

        if "논문" in query:
            hard_filters.setdefault("article_cnt_min", 1)
            branch_hints["art"] = "논문명, 키워드, 초록, 학술지 중심으로 정제한다."
            branch_weights["art"] = 1.0

        if "과제" in query or "연구수행" in query:
            hard_filters.setdefault("project_cnt_min", 1)
            branch_hints["pjt"] = "과제명, 목표, 내용, 수행기관 중심으로 정제한다."
            branch_weights["pjt"] = 1.0

        # '~제외' 패턴을 통한 제외 기관 추출 (강한 배제)
        exclude_patterns = re.findall(r"([A-Za-z가-힣0-9()주식회사㈜]+)\s*제외", query)
        for item in exclude_patterns:
            if item not in exclude_list:
                exclude_list.append(item)

        # 간단한 핵심 키워드 추출 (Heuristic)
        core_keywords: list[str] = []
        for kw in ["자율주행", "인체", "에너지", "나노", "양자", "반도체", "인공지능", "배터리"]:
            if kw in query:
                core_keywords.append(kw)

        return PlannerOutput(
            intent_summary=query.strip(),
            core_keywords=core_keywords,
            hard_filters=hard_filters,
            exclude_orgs=exclude_list,
            soft_preferences=soft_preferences,
            branch_weights=branch_weights,
            branch_query_hints=branch_hints,
            top_k=top_k or 5,
        )


class OpenAICompatPlanner:
    """
    OpenAI 호환 API(LLM)를 사용하여 정교한 검색 계획을 수립하는 플래너입니다.
    사용자의 복잡한 의도를 파악하여 최적의 검색 필터와 보정 힌트를 생성합니다.
    """
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
        """LLM을 호출하여 질의를 분석하고 계획을 수립합니다."""
        # LLM planner는 branch on/off를 하지 않고,
        # hard filter와 branch별 질의 보정 힌트만 생성한다.
        system_prompt = """
        [역할 및 목적]
        당신은 국가 R&D 전문가 추천 모듈의 'Senior Query Planner'입니다.
        사용자의 자연어 질의를 분석해 가장 효과적인 하이브리드(Vector + Metadata) 검색 계획을 수립해야 합니다.

        [Harnessing: 출력 스키마 및 정확한 값 매핑 규칙 (필독)]
        다음 JSON 스키마를 엄격하게 준수하며, 값은 허용된(Allowed) 형식만 사용해야 합니다.

        1. `intent_summary` (String): 검색 의도를 명확하고 간결하게 1-2문장으로 요약.
        2. `core_keywords` (Array of Strings): 기술 키워드 뿐 아니라 검색에 필요한 도메인 핵심어 (예: "유물", "문화재", "고려시대").
        3. `hard_filters` (Object): 메타데이터 필터 조건.
           [허용 키와 값 규칙]
           - `degree_slct_nm`: 학위 조건. 질의에 "박사"만 있으면 `"박사"`. "박사 이상" 등 모호한 단어는 절대 허용 불가, 오직 `"박사"`, `"석사"`, 또는 `["석사", "박사"]`만 허용.
           - `art_recent_years`, `pat_recent_years`, `pjt_recent_years` (Integer): "최근 X년" 조건이 있을 때 X를 정수로 할당.
           - `article_cnt_min`, `patent_cnt_min`, `project_cnt_min` (Integer): 해당 실적이 반드시 있어야 함을 명시할 경우 1로 설정.
           - **주의**: 전공이나 연구 분야 단어("한국사", "인공지능")는 절대로 `hard_filters`에 넣지 말고 `core_keywords`나 `branch_query_hints`에 포함시킬 것.

        4. `exclude_orgs` (Array of Strings): 명시적으로 제외해야 할 기관명 (예: "A대학교 제외").
        5. `branch_weights` (Object): 브랜치 가중치. (basic, art, pat, pjt).
        6. `branch_query_hints` (Object): 각 브랜치에서 벡터 유사도 매칭을 고도화할 수 있도록 검색용 문구를 작성.
        7. `top_k` (Integer): 추천을 원하는 전문가 수.

        [Harnessing: 생각의 사슬(Chain-of-Thought) 처리 단계]
        당신은 최종 결과물(JSON)을 출력하기 전에 분석 과정을 `<thinking>` 태그에 작성해야 합니다.
        단계 1. 질의의 맥락 요약.
        단계 2. 제외 기관 식별.
        단계 3. 메타데이터 필터(hard_filters) 대상 추출 (반드시 허용된 Enum과 Integer 매핑 확인, 예: "박사 이상" -> "박사").
        단계 4. 핵심 키워드 정리.
        단계 5. 최종 JSON 도출.

        [최종 출력 포맷]
        <thinking>
        (여기에 분석 과정을 작성)
        </thinking>
        ```json
        {
          "intent_summary": "...",
          "core_keywords": [...],
          "hard_filters": {...},
          "exclude_orgs": [...],
          "branch_weights": {...},
          "branch_query_hints": {...},
          "top_k": 5
        }
        ```
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
            # 모델 호출
            logger.info("LLM 질의 분석(Planning) 시작: 질의=%s", query)
            result = await self.model.ainvoke_non_stream(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            # 결과 텍스트에서 JSON 추출 및 파싱
            json_text = _extract_json_object_text(result.content)
            logger.debug("LLM 응답 텍스트(추출됨): %s", json_text)
            
            payload = json.loads(json_text)
            output = PlannerOutput.model_validate(payload)
            
            logger.info("질의 분석 성공: 의도=%r 제외기관=%d개 필터=%r", 
                        output.intent_summary, len(output.exclude_orgs), output.hard_filters)
            return output
        except Exception as exc:
            # LLM 장애 또는 파싱 에러 시 HeuristicPlanner로 대체
            logger.warning("플래너 Fallback 활성화: 사유=%s", exc)
            return await self.fallback.plan(
                query=query,
                filters_override=filters_override,
                exclude_orgs=exclude_orgs,
                top_k=top_k,
            )
