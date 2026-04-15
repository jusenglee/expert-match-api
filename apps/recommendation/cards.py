"""
Build candidate cards from retrieved expert payloads.
"""

from __future__ import annotations

from apps.domain.models import CandidateCard, PlannerOutput, SearchHit


def _date_key(value: str | None) -> str:
    return value or "0000-00-00"


class CandidateCardBuilder:
    def build_small_cards(
        self, hits: list[SearchHit], plan: PlannerOutput
    ) -> list[CandidateCard]:
        if not hits:
            return []

        cards = [self._build_card(hit, plan) for hit in hits]
        max_rrf_score = max(hit.score for hit in hits) if hits else 0

        for index, card in enumerate(cards):
            raw_score = hits[index].score
            normalized_score = (raw_score / max_rrf_score * 100) if max_rrf_score else 0
            card.rank_score = round(float(normalized_score), 1)
            card.shortlist_score = card.rank_score

        return cards

    def shortlist(self, cards: list[CandidateCard], limit: int) -> list[CandidateCard]:
        return cards[:limit]

    def _build_card(self, hit: SearchHit, plan: PlannerOutput) -> CandidateCard:
        payload = hit.payload
        sorted_papers = sorted(
            payload.publications,
            key=lambda item: (_date_key(item.publication_year_month), item.publication_title),
            reverse=True,
        )
        sorted_patents = sorted(
            payload.intellectual_properties,
            key=lambda item: (
                _date_key(item.registration_date or item.application_date),
                item.intellectual_property_title,
            ),
            reverse=True,
        )
        sorted_projects = sorted(
            payload.research_projects,
            key=lambda item: (
                _date_key(item.project_end_date or item.project_start_date),
                item.display_title,
            ),
            reverse=True,
        )

        matched_filter_summary: list[str] = []
        hard_filters = plan.hard_filters
        if hard_filters.get("degree_slct_nm"):
            matched_filter_summary.append(
                f"Degree filter matched: {payload.researcher_profile.highest_degree or 'unknown'}"
            )
        if hard_filters.get("art_sci_slct_nm") == "SCIE":
            matched_filter_summary.append(
                f"SCIE count: {payload.researcher_profile.scie_publication_count}"
            )
        if hard_filters.get("project_cnt_min") is not None:
            matched_filter_summary.append(
                f"Project count: {payload.researcher_profile.research_project_count}"
            )

        risks: list[str] = []
        data_gaps: list[str] = []
        if not payload.publications:
            data_gaps.append("Publication evidence is missing.")
        if not payload.intellectual_properties:
            data_gaps.append("Patent evidence is missing.")
        if not payload.research_projects:
            data_gaps.append("Project evidence is missing.")
        if len(data_gaps) >= 2:
            risks.append("Evidence coverage is limited.")

        return CandidateCard(
            expert_id=payload.basic_info.researcher_id,
            name=payload.basic_info.researcher_name,
            organization=payload.basic_info.affiliated_organization,
            position=payload.basic_info.position_title,
            degree=payload.researcher_profile.highest_degree,
            major=payload.researcher_profile.major_field,
            branch_presence_flags=hit.data_presence_flags,
            counts={
                "article_cnt": payload.researcher_profile.publication_count,
                "scie_cnt": payload.researcher_profile.scie_publication_count,
                "patent_cnt": payload.researcher_profile.intellectual_property_count,
                "project_cnt": payload.researcher_profile.research_project_count,
            },
            technical_classifications=list(payload.technical_classifications),
            evaluation_activity_cnt=payload.evaluation_activity_cnt,
            evaluation_activities=list(payload.evaluation_activities),
            top_papers=sorted_papers,
            top_patents=sorted_patents,
            top_projects=sorted_projects,
            matched_filter_summary=matched_filter_summary,
            risks=risks,
            data_gaps=data_gaps,
        )
