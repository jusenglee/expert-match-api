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
    async def judge(self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]) -> JudgeOutput:
        ...


class HeuristicJudge:
    async def judge(self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]) -> JudgeOutput:
        # 테스트 전용 judge다.
        # 운영 경로에서는 실제 LLM judge를 사용하고, 이 구현은 계약 테스트용으로만 남긴다.
        ranked_cards = sorted(shortlist, key=lambda card: card.shortlist_score, reverse=True)
        recommendations: list[RecommendationDecision] = []
        global_data_gaps: list[str] = []

        for rank, card in enumerate(ranked_cards[: max(plan.top_k, 3)], start=1):
            reasons = []
            evidence = []
            if card.branch_coverage.get("art"):
                reasons.append("논문 근거가 확인됨")
                if card.top_papers:
                    evidence.append(
                        EvidenceItem(
                            type="paper",
                            title=card.top_papers[0].paper_nm,
                            date=card.top_papers[0].jrnl_pub_dt,
                            detail=card.top_papers[0].jrnl_nm,
                        )
                    )
            if card.branch_coverage.get("pat"):
                reasons.append("특허 근거가 확인됨")
                if card.top_patents:
                    evidence.append(
                        EvidenceItem(
                            type="patent",
                            title=card.top_patents[0].ipr_invention_nm,
                            date=card.top_patents[0].regist_dt or card.top_patents[0].aply_dt,
                            detail=card.top_patents[0].ipr_regist_type_nm,
                        )
                    )
            if card.branch_coverage.get("pjt"):
                reasons.append("과제 수행 근거가 확인됨")
                if card.top_projects:
                    evidence.append(
                        EvidenceItem(
                            type="project",
                            title=card.top_projects[0].display_title,
                            date=card.top_projects[0].end_dt or card.top_projects[0].start_dt,
                            detail=card.top_projects[0].rsch_mgnt_org_nm,
                        )
                    )
            if card.branch_coverage.get("basic"):
                reasons.append("전문가 프로필 적합성이 확인됨")
                evidence.append(
                    EvidenceItem(
                        type="profile",
                        title=f"{card.name} / {card.organization or '소속미상'}",
                        detail=f"{card.degree or '학위미상'} / {card.major or '전공미상'}",
                    )
                )

            if card.data_gaps:
                global_data_gaps.extend(card.data_gaps)

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
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fallback = HeuristicJudge()
        self.model = OpenAICompatChatModel(
            model_name=settings.llm_model_name,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

    async def judge(self, *, query: str, plan: PlannerOutput, shortlist: list[CandidateCard]) -> JudgeOutput:
        # hard filter는 이미 검색 단계에서 보장되었다는 전제 아래,
        # 이 단계는 후보 비교와 추천 서술만 담당한다.
        system_prompt = (
            "평가위원 추천 judge 역할이다. hard filter는 이미 보장되었다. "
            "반드시 JSON만 출력하고 recommended/not_selected_reasons/data_gaps만 포함한다."
        )
        user_payload = {
            "query": query,
            "plan": plan.model_dump(mode="json"),
            "shortlist": [card.model_dump(mode="json") for card in shortlist],
        }
        try:
            result = await self.model.ainvoke_non_stream(
                [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(user_payload, ensure_ascii=False))]
            )
            return JudgeOutput.model_validate(json.loads(result.content))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Judge fallback activated: %s", exc)
            return await self.fallback.judge(query=query, plan=plan, shortlist=shortlist)
