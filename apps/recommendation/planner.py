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
from apps.core.json_utils import extract_json_object_text as _extract_json_object_text
from apps.core.llm_policies import build_consistency_invoke_kwargs
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.domain.models import PlannerOutput

logger = logging.getLogger(__name__)


# _extract_json_object_text 는 apps.core.json_utils 로 이전됨 (import 위 참조)


class Planner(Protocol):
    """검색 계획을 수립하는 플래너의 인터페이스 정의입니다."""
    async def plan(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        include_orgs: list[str] | None = None,
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
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput:
        # 테스트 전용 planner다.
        # 운영 경로에서는 사용하지 않고, API/검색 파이프라인 계약을 빠르게 검증할 때만 쓴다.
        hard_filters = dict(filters_override or {})
        include_list = list(include_orgs or [])
        exclude_list = list(exclude_orgs or [])
        branch_hints = {
            "basic": "전문가 프로필 적합성 중심으로 질의를 정제한다.",
            "art": "논문과 최근 연구실적 중심으로 질의를 정제한다.",
            "pat": "특허와 사업화 키워드 중심으로 질의를 정제한다.",
            "pjt": "과제수행과 전문기관 경험 중심으로 질의를 정제한다.",
        }
        # 기본 가중치는 모두 1.0
        # branch_weights = {
        #     "basic": 1.0,
        #     "art": 1.0,
        #     "pat": 1.0,
        #     "pjt": 1.0,
        # }
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
            # branch_weights["pat"] = 1.0
            # branch_weights["art"] = 0.5  # 특허 강조 시 논문 비중 축소
            soft_preferences.append("특허 근거 선호")

        if "논문" in query:
            hard_filters.setdefault("article_cnt_min", 1)
            branch_hints["art"] = "논문명, 키워드, 초록, 학술지 중심으로 정제한다."
            # branch_weights["art"] = 1.0

        if "과제" in query or "연구수행" in query:
            hard_filters.setdefault("project_cnt_min", 1)
            branch_hints["pjt"] = "과제명, 목표, 내용, 수행기관 중심으로 정제한다."
            # branch_weights["pjt"] = 1.0

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
            include_orgs=include_list,
            exclude_orgs=exclude_list,
            soft_preferences=soft_preferences,
            # branch_weights=branch_weights,
            branch_query_hints=branch_hints,
            top_k=top_k or 15,
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
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> PlannerOutput:
        """LLM을 호출하여 질의를 분석하고 계획을 수립합니다."""
        # LLM planner는 branch on/off를 하지 않고,
        # hard filter와 branch별 질의 보정 힌트만 생성한다.
        system_prompt = (
            "당신은 국가 R&D 전문가 추천 시스템의 Query Planner입니다.\n"
            "사용자의 자연어 질의를 분석하여 하이브리드 검색 계획을 JSON 하나로 반환합니다.\n\n"
            "## 출력 규칙\n"
            "- 반드시 JSON 객체 하나만 출력하세요.\n"
            "- JSON 앞뒤에 설명, 마크다운, 코드 펜스(```)를 절대 붙이지 마세요.\n"
            "- 첫 글자는 반드시 '{'이고 마지막 글자는 반드시 '}'여야 합니다.\n\n"
            "## JSON 스키마\n"
            '{\n'
            '  "intent_summary": "검색 의도를 1-2문장으로 요약",\n'
            '  "core_keywords": ["기술 도메인 핵심어"],\n'
            '  "hard_filters": { 허용 키만 사용 },\n'
            '  "include_orgs": ["검색 대상 기관명 (소속 제한)"],\n'
            '  "exclude_orgs": ["제외할 기관명"],\n'
            '  "soft_preferences": ["선호 조건 설명"],\n'
            # '  "branch_weights": {"basic":1.0,"art":1.0,"pat":1.0,"pjt":1.0},\n'
            '  "branch_query_hints": {"basic":"...","art":"...","pat":"...","pjt":"..."},\n'
            '  "top_k": 15\n'
            '}\n\n'
            "## 기관 필터 규칙 (매우 중요 — 혼동 금지)\n"
            "- include_orgs: 추천 대상을 특정 기관 소속 연구자로 **제한**할 때 사용.\n"
            "  → 패턴 예시: 'X 소속 연구자 중', 'X 소속에서 추천', 'X 연구원 소속 전문가'\n"
            "  → 이 경우 X를 include_orgs에 넣고 exclude_orgs는 비워야 함.\n"
            "- exclude_orgs: 특정 기관을 결과에서 **제외**할 때 사용.\n"
            "  → 패턴 예시: 'X 제외', 'X 빼고', 'X 소속 제외'\n"
            "- 절대 혼동 금지: '소속 연구자 중'은 포함(include)이지 제외(exclude)가 아님.\n"
            "- API로 전달된 include_orgs/exclude_orgs가 있으면 질의에서 추출한 값과 합산(union)하세요.\n\n"
            "## hard_filters 허용 키 및 값 규칙 (이 외의 키는 사용 금지)\n"
            '- "degree_slct_nm": 학위 조건. 허용 값: "박사", "석사", 또는 ["석사","박사"]. '
            '"박사 이상" 같은 표현은 "박사"로 변환.\n'
            '- "art_recent_years" / "pat_recent_years" / "pjt_recent_years" (정수): "최근 X년" → X\n'
            '- "article_cnt_min" / "patent_cnt_min" / "project_cnt_min" (정수): 해당 실적 필수 시 1\n'
            '- "art_sci_slct_nm": SCIE 논문 요구 시 "SCIE"\n'
            "- 주의: 전공·연구 분야 단어는 hard_filters에 넣지 말고 core_keywords 또는 branch_query_hints에 포함\n\n"
            # "## branch_weights 설명\n"
            # "- basic: 전문가 프로필(전공, 학위, 소속)\n"
            # "- art: 논문(학술지, 키워드, 초록)\n"
            # "- pat: 특허(발명명, 등록, 출원)\n"
            # "- pjt: 연구과제(과제명, 수행기관, 연구목표)\n"
            # "- 질의와 관련 높은 브랜치는 1.0, 낮은 브랜치는 0.3~0.7로 설정\n\n"
            "## branch_query_hints 작성 규칙\n"
            "- 각 브랜치의 벡터 유사도를 높이기 위한 검색 보조 문구를 한국어로 작성\n"
            "- 원본 질의의 핵심 의도를 해당 브랜치 관점으로 재구성\n\n"
            "## 예시 1\n"
            "질의: \"자율주행 센서 융합 분야 박사급 전문가 5명, 최근 5년 논문실적, A대학교 제외\"\n"
            '{"intent_summary":"자율주행 센서 융합 분야에서 최근 5년간 논문실적이 있는 박사급 전문가를 추천합니다.",'
            '"core_keywords":["자율주행","센서 융합","LiDAR","카메라 융합","ADAS"],'
            '"hard_filters":{"degree_slct_nm":"박사","art_recent_years":5,"article_cnt_min":1},'
            '"include_orgs":[],'
            '"exclude_orgs":["A대학교"],'
            '"soft_preferences":["최근 5년 논문실적 보유","센서 융합 관련 연구경험"],'
            #'"branch_weights":{"basic":0.5,"art":1.0,"pat":0.7,"pjt":0.8},'
            '"branch_query_hints":{"basic":"자율주행 센서 융합 전공 박사 연구자 프로필",'
            '"art":"자율주행 센서 융합 LiDAR 카메라 융합 인지 알고리즘 논문 키워드 초록",'
            '"pat":"자율주행 센서 융합 인지 시스템 특허 발명 등록",'
            '"pjt":"자율주행 센서 융합 ADAS 연구과제 수행 목표 내용"},'
            '"top_k":15}\n\n'
            "## 예시 2\n"
            "질의: \"반도체 공정 장비 관련 특허 보유 전문가\"\n"
            '{"intent_summary":"반도체 공정 장비 관련 특허를 보유한 전문가를 추천합니다.",'
            '"core_keywords":["반도체","공정 장비","웨이퍼","식각","증착"],'
            '"hard_filters":{"patent_cnt_min":1},'
            '"include_orgs":[],'
            '"exclude_orgs":[],'
            '"soft_preferences":["반도체 공정 장비 관련 등록 특허 보유"],'
            #'"branch_weights":{"basic":0.5,"art":0.5,"pat":1.0,"pjt":0.7},'
            '"branch_query_hints":{"basic":"반도체 공정 장비 전공 연구자 프로필",'
            '"art":"반도체 공정 장비 식각 증착 관련 논문",'
            '"pat":"반도체 공정 장비 웨이퍼 식각 증착 특허 발명명 등록",'
            '"pjt":"반도체 공정 장비 관련 연구과제 수행 경험"},'
            '"top_k":15}\n\n'
            "## 예시 3\n"
            "질의: \"한국과학기술정보연구원 소속 연구자 중 국가 R&D 성과물 과제 관련 평가위원 추천\"\n"
            '{"intent_summary":"한국과학기술정보연구원 소속 연구자 중 국가 R&D 성과물 과제 관련 평가위원을 추천합니다.",'
            '"core_keywords":["국가 R&D 성과물","연구성과"],'
            '"hard_filters":{"project_cnt_min":1},'
            '"include_orgs":["한국과학기술정보연구원"],'
            '"exclude_orgs":[],'
            '"soft_preferences":["국가 R&D 성과물 평가 경험","성과물 기반 연구과제 수행"],'
            #'"branch_weights":{"basic":1.0,"art":0.7,"pat":0.5,"pjt":1.0},'
            '"branch_query_hints":{"basic":"한국과학기술정보연구원 소속 연구자 프로필",'
            '"art":"국가 R&D 성과물 관련 논문 키워드 초록",'
            '"pat":"국가 R&D 성과물 관련 특허 발명명",'
            '"pjt":"국가 R&D 성과물 기반 연구과제 수행 목표 내용"},'
            '"top_k":15}\n'
        )
        system_prompt += (
            "\n## Consistency constraints\n"
            "- Only use constraints that are explicit in the user query or override inputs.\n"
            "- Do not invent organizations, date ranges, degrees, minimum counts, or branch preferences.\n"
            "- If a condition is ambiguous or not stated, leave it out of hard_filters and soft_preferences.\n"
        )
        user_prompt = json.dumps(
            {
                "query": query,
                "filters_override": filters_override or {},
                "include_orgs": include_orgs or [],
                "exclude_orgs": exclude_orgs or [],
                "top_k": top_k or 15,
            },
            ensure_ascii=False,
        )
        try:
            # 모델 호출
            logger.info("LLM 질의 분석(Planning) 시작: 질의=%s", query)
            invoke_kwargs = build_consistency_invoke_kwargs()
            result = await self.model.ainvoke_non_stream(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
                **invoke_kwargs,
            )
            # 결과 텍스트에서 JSON 추출 및 파싱
            json_text = _extract_json_object_text(result.content)
            logger.debug("LLM 응답 텍스트(추출됨): %s", json_text)

            payload = json.loads(json_text)
            output = PlannerOutput.model_validate(payload)

            # API 레벨 include_orgs 병합: LLM이 추출한 것 + API로 직접 전달된 것 합산
            if include_orgs:
                for org in include_orgs:
                    if org not in output.include_orgs:
                        output.include_orgs.append(org)

            # API 레벨 exclude_orgs 병합: LLM이 추출한 것 + API로 직접 전달된 것 합산
            if exclude_orgs:
                for org in exclude_orgs:
                    if org not in output.exclude_orgs:
                        output.exclude_orgs.append(org)

            logger.info(
                "질의 분석 성공: 의도=%r 포함기관=%d개 제외기관=%d개 필터=%r",
                output.intent_summary,
                len(output.include_orgs),
                len(output.exclude_orgs),
                output.hard_filters,
            )
            return output
        except Exception as exc:
            # LLM 장애 또는 파싱 에러 시 HeuristicPlanner로 대체
            logger.warning("플래너 Fallback 활성화: 사유=%s", exc)
            return await self.fallback.plan(
                query=query,
                filters_override=filters_override,
                include_orgs=include_orgs,
                exclude_orgs=exclude_orgs,
                top_k=top_k,
            )
