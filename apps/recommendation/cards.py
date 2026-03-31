from __future__ import annotations

from datetime import datetime

from apps.domain.models import CandidateCard, ExpertPayload, SearchHit


def _date_key(value: str | None) -> str:
    return value or "0000-00-00"


class CandidateCardBuilder:
    def build_small_cards(self, hits: list[SearchHit], hard_filters: dict[str, object]) -> list[CandidateCard]:
        cards = [self._build_card(hit, hard_filters) for hit in hits]
        for card in cards:
            branch_bonus = sum(1 for value in card.branch_coverage.values() if value) * 10
            quantity_bonus = card.counts.get("scie_cnt", 0) * 3 + card.counts.get("project_cnt", 0) * 2
            card.shortlist_score = float(branch_bonus + quantity_bonus + card.counts.get("patent_cnt", 0) * 2)
        return sorted(cards, key=lambda item: item.shortlist_score, reverse=True)

    def shortlist(self, cards: list[CandidateCard], limit: int) -> list[CandidateCard]:
        return cards[:limit]

    def _build_card(self, hit: SearchHit, hard_filters: dict[str, object]) -> CandidateCard:
        payload = hit.payload
        top_papers = sorted(payload.art, key=lambda item: _date_key(item.jrnl_pub_dt), reverse=True)[:2]
        top_patents = sorted(payload.pat, key=lambda item: _date_key(item.regist_dt or item.aply_dt), reverse=True)[:1]
        top_projects = sorted(payload.pjt, key=lambda item: _date_key(item.end_dt or item.start_dt), reverse=True)[:2]

        matched_filter_summary = []
        if hard_filters.get("degree_slct_nm"):
            matched_filter_summary.append(f"학위 조건 충족: {payload.degree_slct_nm or '미확인'}")
        if hard_filters.get("art_sci_slct_nm") == "SCIE":
            matched_filter_summary.append(f"SCIE 수: {payload.scie_cnt}")
        if hard_filters.get("project_cnt_min") is not None:
            matched_filter_summary.append(f"과제 수: {payload.project_cnt}")

        risks = []
        data_gaps = []
        if not payload.art:
            data_gaps.append("논문 근거 부족")
        if not payload.pat:
            data_gaps.append("특허 근거 부족")
        if not payload.pjt:
            data_gaps.append("과제 근거 부족")
        if len(data_gaps) >= 2:
            risks.append("근거 영역이 편중되어 있음")

        return CandidateCard(
            expert_id=payload.doc_id,
            name=payload.hm_nm,
            organization=payload.blng_org_nm,
            position=payload.position_nm,
            degree=payload.degree_slct_nm,
            major=payload.major_slct_nm,
            branch_coverage=hit.branch_coverage,
            counts={
                "article_cnt": payload.article_cnt,
                "scie_cnt": payload.scie_cnt,
                "patent_cnt": payload.patent_cnt,
                "project_cnt": payload.project_cnt,
            },
            top_papers=top_papers,
            top_patents=top_patents,
            top_projects=top_projects,
            matched_filter_summary=matched_filter_summary,
            risks=risks,
            data_gaps=data_gaps,
        )

