"""
검색된 결과를 전문가 카드(CandidateCard) 형태로 구조화하고 초기 점수를 산정하는 모듈입니다.
Raw 데이터를 사용자에게 보여주기 적합한 형태로 가공하고, 숏리스트 선정을 위한 가중치 기반 점수를 계산합니다.
"""

from __future__ import annotations

from typing import Any

from apps.domain.models import CandidateCard, PlannerOutput, SearchHit


def _date_key(value: str | None) -> str:
    """날짜 문자열 정렬을 위한 키 생성 함수입니다. (None인 경우 가장 과거 날짜로 취급)"""
    return value or "0000-00-00"


class CandidateCardBuilder:
    """
    검색 히트(SearchHit) 데이터를 분석하여 전문가 카드를 생성하고 점수를 부여하는 클래스입니다.
    """
    def build_small_cards(self, hits: list[SearchHit], plan: PlannerOutput) -> list[CandidateCard]:
        """
        검색된 히트 리스트를 카드 리스트로 변환하고, 적합도 점수(relevance_score)를 정규화하여 부여합니다.
        """
        if not hits:
            return []

        cards = [self._build_card(hit, plan) for hit in hits]
        
        # 1. 적합도 정규화 (최대 점수 = 100점)
        max_rrf_score = max(hit.score for hit in hits) if hits else 0
        
        for i, card in enumerate(cards):
            # Qdrant RRF 점수를 100점 만점으로 환산
            raw_score = hits[i].score
            normalized_score = (raw_score / max_rrf_score * 100) if max_rrf_score > 0 else 0
            
            card.relevance_score = round(float(normalized_score), 1)
            # 현재는 적합도만 제공하기로 했으므로 shortlist_score도 적합도로 통일
            card.shortlist_score = card.relevance_score
            
        # 적합도 점수 높은 순으로 정렬
        return sorted(cards, key=lambda item: item.relevance_score, reverse=True)

    def shortlist(self, cards: list[CandidateCard], limit: int) -> list[CandidateCard]:
        """정렬된 카드 목록에서 상위 N명을 추출하여 숏리스트를 구성합니다."""
        return cards[:limit]

    def _build_card(self, hit: SearchHit, plan: PlannerOutput) -> CandidateCard:
        """개별 검색 히트를 분석하여 카드 객체를 생성합니다."""
        payload = hit.payload
        keywords = [k.lower() for k in plan.core_keywords]
        
        # 1. 분야별 실적 점수 산정 및 정렬 (매칭 점수 + 최신순)
        def score_and_sort(items: list[Any], title_attr: str, content_attrs: list[str]) -> tuple[list[Any], dict[str, int]]:
            scored_items = []
            matched_stats = {k: 0 for k in plan.core_keywords}
            
            for item in items:
                score = 0
                title = str(getattr(item, title_attr, "")).lower()
                
                # 가중치 1순위: 제목 매칭 (+2)
                for kw in keywords:
                    if kw in title:
                        score += 2
                        matched_stats[plan.core_keywords[keywords.index(kw)]] += 1
                
                # 가중치 2순위: 내용 매칭 (+1)
                for attr in content_attrs:
                    content = str(getattr(item, attr, "")).lower()
                    for kw in keywords:
                        if kw in content:
                            score += 1
                            # 이미 제목에서 카운트했더라도 통계에는 합산 (혹은 중복 제거 전략 선택 가능)
                            # 여기서는 단순히 해당 기술 키워드가 얼마나 언급되는지 빈도로 집계
                
                # 날짜 키 추출
                date_val = ""
                if hasattr(item, "publication_year_month"):
                    date_val = item.publication_year_month
                elif hasattr(item, "registration_date"):
                    date_val = item.registration_date or item.application_date or ""
                elif hasattr(item, "project_end_date"):
                    date_val = item.project_end_date or item.project_start_date or ""
                
                scored_items.append((score, _date_key(date_val), item))
            
            # (매칭 점수 DESC, 날짜 DESC) 순으로 정렬
            sorted_items = [x[2] for x in sorted(scored_items, key=lambda x: (x[0], x[1]), reverse=True)]
            return sorted_items, matched_stats

        # 각 분야별 상위 실적 추출
        sorted_papers, paper_matched = score_and_sort(payload.publications, "publication_title", ["abstract", "korean_keywords"])
        sorted_patents, patent_matched = score_and_sort(payload.intellectual_properties, "intellectual_property_title", ["intellectual_property_type"])
        sorted_projects, project_matched = score_and_sort(payload.research_projects, "display_title", ["research_objective_summary", "research_content_summary"])

        # 전체 키워드 매칭 통계 합산
        total_matched_counts = {}
        for kw in plan.core_keywords:
            total_matched_counts[kw] = paper_matched[kw] + patent_matched[kw] + project_matched[kw]

        # 2. 적용된 필터 조건 충족 여부 요약
        matched_filter_summary = []
        hard_filters = plan.hard_filters
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
            keyword_matched_counts=total_matched_counts,
            top_papers=sorted_papers[:2],
            top_patents=sorted_patents[:1],
            top_projects=sorted_projects[:2],
            matched_filter_summary=matched_filter_summary,
            risks=risks,
            data_gaps=data_gaps,
        )

