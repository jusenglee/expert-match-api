"""
검색된 결과를 전문가 카드(CandidateCard) 형태로 구조화하고 초기 점수를 산정하는 모듈입니다.
Raw 데이터를 사용자에게 보여주기 적합한 형태로 가공하고, 숏리스트 선정을 위한 가중치 기반 점수를 계산합니다.
"""

from __future__ import annotations

from datetime import datetime

from apps.domain.models import CandidateCard, ExpertPayload, SearchHit


def _date_key(value: str | None) -> str:
    """날짜 문자열 정렬을 위한 키 생성 함수입니다. (None인 경우 가장 과거 날짜로 취급)"""
    return value or "0000-00-00"


class CandidateCardBuilder:
    """
    검색 히트(SearchHit) 데이터를 분석하여 전문가 카드를 생성하고 점수를 부여하는 클래스입니다.
    """
    def build_small_cards(self, hits: list[SearchHit], hard_filters: dict[str, object]) -> list[CandidateCard]:
        """
        검색된 히트 리스트를 카드 리스트로 변환하고, 가중치 기반 점수를 부여한 후 정렬합니다.
        """
        cards = [self._build_card(hit, hard_filters) for hit in hits]
        
        # 각 카드별로 초기 점수(shortlist_score) 산정
        for card in cards:
            # 1. 브랜치 커버리지 보너스: 논문, 특허 등 다양한 소스에서 데이터가 발견될수록 가산점
            branch_bonus = sum(1 for value in card.branch_coverage.values() if value) * 10
            
            # 2. 실적 수량 보너스: SCIE 논문(3점), 과제 수행(2점), 특허(2점) 가중치 부여
            quantity_bonus = (
                card.counts.get("scie_cnt", 0) * 3 + 
                card.counts.get("project_cnt", 0) * 2 +
                card.counts.get("patent_cnt", 0) * 2
            )
            
            card.shortlist_score = float(branch_bonus + quantity_bonus)
            
        # 점수 높은 순으로 정렬
        return sorted(cards, key=lambda item: item.shortlist_score, reverse=True)

    def shortlist(self, cards: list[CandidateCard], limit: int) -> list[CandidateCard]:
        """정렬된 카드 목록에서 상위 N명을 추출하여 숏리스트를 구성합니다."""
        return cards[:limit]

    def _build_card(self, hit: SearchHit, hard_filters: dict[str, object]) -> CandidateCard:
        """개별 검색 히트를 분석하여 카드 객체를 생성합니다."""
        payload = hit.payload
        
        # 1. 각 분야별 주요 실적 정렬 (최신순) 및 상위 N개 추출
        top_papers = sorted(payload.publications, key=lambda item: _date_key(item.publication_year_month), reverse=True)[:2]
        top_patents = sorted(payload.intellectual_properties, key=lambda item: _date_key(item.registration_date or item.application_date), reverse=True)[:1]
        top_projects = sorted(payload.research_projects, key=lambda item: _date_key(item.project_end_date or item.project_start_date), reverse=True)[:2]

        # 2. 적용된 필터 조건 충족 여부 요약
        matched_filter_summary = []
        if hard_filters.get("degree_slct_nm"):
            matched_filter_summary.append(f"학위 조건 충족: {payload.researcher_profile.highest_degree or '미확인'}")
        if hard_filters.get("art_sci_slct_nm") == "SCIE":
            matched_filter_summary.append(f"SCIE 수: {payload.researcher_profile.scie_publication_count}")
        if hard_filters.get("project_cnt_min") is not None:
            matched_filter_summary.append(f"과제 수: {payload.researcher_profile.research_project_count}")

        # 3. 데이터 결측치 및 잠재적 리스크 분석
        risks = []
        data_gaps = []
        if not payload.publications:
            data_gaps.append("논문 근거 부족")
        if not payload.intellectual_properties:
            data_gaps.append("특허 근거 부족")
        if not payload.research_projects:
            data_gaps.append("과제 근거 부족")
        
        if len(data_gaps) >= 2:
            risks.append("근거 영역이 편중되어 있음")

        # 4. 카드 객체 조립
        return CandidateCard(
            expert_id=payload.basic_info.researcher_id,
            name=payload.basic_info.researcher_name,
            organization=payload.basic_info.affiliated_organization,
            position=payload.basic_info.position_title,
            degree=payload.researcher_profile.highest_degree,
            major=payload.researcher_profile.major_field,
            branch_coverage=hit.branch_coverage,
            counts={
                "article_cnt": payload.researcher_profile.publication_count,
                "scie_cnt": payload.researcher_profile.scie_publication_count,
                "patent_cnt": payload.researcher_profile.intellectual_property_count,
                "project_cnt": payload.researcher_profile.research_project_count,
            },
            top_papers=top_papers,
            top_patents=top_patents,
            top_projects=top_projects,
            matched_filter_summary=matched_filter_summary,
            risks=risks,
            data_gaps=data_gaps,
        )

