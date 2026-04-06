"""
추천 후보자들을 상세 평가하고 최종 순위를 결정하는 모듈입니다.
검색된 전문가 후보군(Shortlist)의 실적과 데이터를 분석하여
질의에 가장 적합한 전문가를 선별하고 추천 근거를 생성합니다.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from apps.core.config import Settings
from apps.core.openai_compat_llm import OpenAICompatChatModel
from apps.domain.models import CandidateCard, EvidenceItem, JudgeOutput, PlannerOutput, RecommendationDecision

logger = logging.getLogger(__name__)


class Judge(Protocol):
    """전문가 적합성을 평가하고 순위를 매기는 심사역 인터페이스입니다."""
    async def judge(self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]) -> JudgeOutput:
        """후보자 명단을 심사하여 최종 추천 결과를 반환합니다."""
        ...


class HeuristicJudge:
    """
    규칙 기반(Heuristic)으로 후보자를 평가하는 심사역입니다.
    주로 테스트용이거나 LLM 장애 시 Fallback용으로 사용되며, 점수 기반으로 순위를 매깁니다.
    """
    async def judge(self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]) -> JudgeOutput:
        # 테스트 전용 judge다.
        # 운영 경로에서는 실제 LLM judge를 사용하고, 이 구현은 계약 테스트용으로만 남긴다.
        
        # 1. 숏리스트 점수(shortlist_score) 기준으로 내림차순 정렬
        ranked_cards = sorted(shortlist, key=lambda card: card.shortlist_score, reverse=True)
        recommendations: list[RecommendationDecision] = []
        global_data_gaps: list[str] = []

        # 2. 상위 N명(기본 3명)에 대해 추천 근거와 증거(Evidence) 생성
        for rank, card in enumerate(ranked_cards[: max(plan.top_k, 3)], start=1):
            reasons = []
            evidence = []
            
            # 논문 실적 확인
            if card.branch_coverage.get("art"):
                reasons.append("논문 근거가 확인됨")
                if card.top_papers:
                    evidence.append(
                        EvidenceItem(
                            type="paper",
                            title=card.top_papers[0].publication_title,
                            date=card.top_papers[0].publication_year_month,
                            detail=card.top_papers[0].journal_name,
                        )
                    )
            
            # 특허 실적 확인
            if card.branch_coverage.get("pat"):
                reasons.append("특허 근거가 확인됨")
                if card.top_patents:
                    evidence.append(
                        EvidenceItem(
                            type="patent",
                            title=card.top_patents[0].intellectual_property_title,
                            date=card.top_patents[0].registration_date or card.top_patents[0].application_date,
                            detail=card.top_patents[0].application_registration_type,
                        )
                    )
            
            # 과제 실적 확인
            if card.branch_coverage.get("pjt"):
                reasons.append("과제 수행 근거가 확인됨")
                if card.top_projects:
                    evidence.append(
                        EvidenceItem(
                            type="project",
                            title=card.top_projects[0].display_title,
                            date=card.top_projects[0].project_end_date or card.top_projects[0].project_start_date,
                            detail=card.top_projects[0].managing_agency,
                        )
                    )
            
            # 기본 프로필 정보 확인
            if card.branch_coverage.get("basic"):
                reasons.append("전문가 프로필 적합성이 확인됨")
                evidence.append(
                    EvidenceItem(
                        type="profile",
                        title=f"{card.name} / {card.organization or '소속미상'}",
                        detail=f"{card.degree or '학위미상'} / {card.major or '전공미상'}",
                    )
                )

            # 데이터 결측치 수집
            if card.data_gaps:
                global_data_gaps.extend(card.data_gaps)

            # 최종 결정 객체 생성
            recommendations.append(
                RecommendationDecision(
                    rank=rank,
                    expert_id=card.expert_id,
                    name=card.name,
                    fit="높음" if card.shortlist_score >= 20 else "중간",
                    reasons=reasons[:3],
                    evidence=evidence[:4],
                    risks=card.risks + card.data_gaps[:1],
                )
            )

        # 3. 중복 제거된 전체 데이터 결측치 취합
        unique_gaps = list(dict.fromkeys(global_data_gaps))
        not_selected = []
        if len(ranked_cards) > len(recommendations):
            not_selected.append("상위 추천 대비 근거 다양성 또는 최근성이 상대적으로 약함")

        return JudgeOutput(
            recommended=recommendations[:5],
            not_selected_reasons=not_selected,
            data_gaps=unique_gaps,
        )


class OpenAICompatJudge:
    """
    OpenAI 호환 API(LLM)를 사용하여 후보자를 정형/비정형으로 평가하는 심사역입니다.
    전문가 개개인의 성과를 질의의 맥락에 맞춰 비교 분석합니다.
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fallback = HeuristicJudge()
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

    async def judge(self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]) -> JudgeOutput:
        """LLM을 호출하여 전문가 적합성을 다각도로 심사합니다."""
        # hard filter는 이미 검색 단계에서 보장되었다는 전제 아래,
        # 이 단계는 후보 비교와 추천 서술만 담당한다.
        system_prompt = """
            당신은 국가 R&D 평가위원 추천 시스템의 '수석 심사관(Senior Recommendation Judge)'입니다.
            검색 엔진이 1차로 추려낸 전문가 후보군(Shortlist)을 평가하여, 사용자의 질의와 플랜에 가장 부합하는 최종 추천 명단과 그 근거를 JSON 형태로 반환해야 합니다.
            
            [심사 가이드라인 및 엄격한 제약사항]
            1. 완벽한 근거 기반 (No Hallucination):
               - 제공된 후보군의 `top_papers`, `top_patents`, `top_projects` 배열에 존재하는 데이터만 근거(`evidence`)로 사용하십시오.
               - 절대로 존재하지 않는 논문, 특허, 과제를 지어내어 평가해서는 안 됩니다.
            
            2. 적합도(Fit) 및 순위(Rank) 산정:
               - 사용자의 `query` 및 `plan.intent_summary`와 후보의 실적이 얼마나 일치하는지 분석하여 '높음', '중간', '보통' 중 하나로 평가하십시오.
               - 가장 적합도가 높고 근거가 탄탄한 순서대로 순위(`rank`)를 1부터 매겨 `recommended` 배열에 정렬하십시오.
            
            3. 추천 사유(Reasons)와 근거(Evidence) 작성 규칙:
               - `reasons`: 왜 이 후보가 추천되었는지, 다른 후보 대비 어떤 강점이 있는지 구체적으로 서술하십시오. (예: "주어진 쿼리의 AI 분야 논문 실적이 가장 풍부함")
               - `evidence`: 사유를 뒷받침하는 구체적인 실적을 추출하십시오. 
                 * type은 "paper", "patent", "project", "profile" 중 하나여야 합니다.
                 * title, date, detail 필드에 제공된 데이터의 값을 정확히 복사하여 넣으십시오.
            
            4. 리스크 및 데이터 공백(Risks & Data Gaps):
               - 후보 개인의 약점(예: "최근 3년 내 관련 논문 없음")은 해당 후보의 `risks` 배열에 추가하십시오.
               - 후보군 전체의 공통적인 문제나 검색된 데이터의 한계(예: "특허 실적을 보유한 후보가 전반적으로 부족함")는 최상위 `data_gaps`에 추가하십시오.
               - 추천하지 않은 후보가 있다면 `not_selected_reasons`에 그 사유를 명시하십시오.
        """
        user_payload = {
            "query": query,
            "plan": plan.model_dump(mode="json"),
            "shortlist": [card.model_dump(mode="json") for card in shortlist],
        }
        try:
            # LLM 호출
            result = await self.model.ainvoke_non_stream(
                [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(user_payload, ensure_ascii=False))]
            )
            return JudgeOutput.model_validate(json.loads(result.content))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            # API 장애나 데이터 검증 실패 시 HeuristicJudge로 Fallback
            logger.warning("Judge fallback activated: %s", exc)
            return await self.fallback.judge(query=query, plan=plan, shortlist=shortlist)
