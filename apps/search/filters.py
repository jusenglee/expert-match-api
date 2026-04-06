"""
사용자 질의 분석 결과(Hard Filters)를 Qdrant 검색 엔진이 이해할 수 있는
실행 가능한 필터 조건(models.Filter)으로 변환하는 모듈입니다.
학위, 실적 건수, 최근 연도, 제외 기관 등 다양한 조건을 처리합니다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from qdrant_client import models

from apps.search.text_utils import normalize_org_name


class QdrantFilterCompiler:
    """
    고수준의 검색 조건을 저수준의 Qdrant 필터 객체로 컴파일하는 클래스입니다.
    """
    def compile(self, hard_filters: dict[str, Any], exclude_orgs: list[str]) -> models.Filter | None:
        """
        주어진 하드 필터와 제외 기관 목록을 바탕으로 Qdrant 필터 객체를 생성합니다.
        
        - 'must': 모든 조건을 만족해야 함 (AND)
        - 'must_not': 해당 조건에 맞으면 제외함 (NOT)
        """
        # planner는 질의 의도 위주로 조건을 뽑고, 이 컴파일러는 이를 실제 DB 스키마 필드와 매핑하여 확정한다.
        must: list[models.Condition] = []
        must_not: list[models.Condition] = []

        # 1. 학위 조건 처리 (최고 학위 필드 매핑)
        if degree := hard_filters.get("degree_slct_nm"):
            values = degree if isinstance(degree, list) else [degree]
            must.append(
                models.FieldCondition(
                    key="researcher_profile.highest_degree",
                    match=models.MatchAny(any=values),
                )
            )

        # 2. 통계성 실적 건수 조건 처리 (최소 건수 이상인 전문가만 선별)
        counts_mapping = {
            "article_cnt": "researcher_profile.publication_count",
            "scie_cnt": "researcher_profile.scie_publication_count",
            "patent_cnt": "researcher_profile.intellectual_property_count",
            "project_cnt": "researcher_profile.research_project_count",
        }
        for field_name, backend_key in counts_mapping.items():
            min_value = hard_filters.get(f"{field_name}_min")
            if min_value is not None:
                must.append(
                    models.FieldCondition(
                        key=backend_key,
                        range=models.Range(gte=min_value),
                    )
                )

        # 3. 논문 관련 상세 조건 (최근 연도 및 SCIE 여부 - Nested 구조 지원)
        recent_years = hard_filters.get("art_recent_years")
        scie_required = hard_filters.get("art_sci_slct_nm")
        if recent_years or scie_required:
            art_must: list[models.Condition] = []
            if scie_required:
                art_must.append(
                    models.FieldCondition(key="journal_index_type", match=models.MatchValue(value=scie_required))
                )
            if recent_years:
                cutoff = f"{datetime.now(UTC).year - int(recent_years)}-01-01"
                art_must.append(
                    models.FieldCondition(
                        key="publication_year_month",
                        range=models.DatetimeRange(gte=cutoff),
                    )
                )
            # 전문가 하위의 'publications' 리스트 내부에 조건 적용
            must.append(
                models.NestedCondition(
                    nested=models.Nested(
                        key="publications",
                        filter=models.Filter(must=art_must),
                    )
                )
            )

        # 4. 특허 관련 상세 조건 (등록 유형 및 최근 연도 - Nested 구조 지원)
        pat_regist_type = hard_filters.get("pat_ipr_regist_type_nm")
        pat_recent_years = hard_filters.get("pat_recent_years")
        if pat_regist_type or pat_recent_years:
            pat_must: list[models.Condition] = []
            if pat_regist_type:
                pat_must.append(
                    models.FieldCondition(
                        key="application_registration_type",
                        match=models.MatchValue(value=pat_regist_type),
                    )
                )
            if pat_recent_years:
                cutoff = f"{datetime.now(UTC).year - int(pat_recent_years)}-01-01"
                pat_must.append(
                    models.FieldCondition(
                        key="application_date",
                        range=models.DatetimeRange(gte=cutoff),
                    )
                )
            must.append(
                models.NestedCondition(
                    nested=models.Nested(
                        key="intellectual_properties",
                        filter=models.Filter(must=pat_must),
                    )
                )
            )

        # 5. 과제 관련 최근성 조건 (Nested 구조 지원)
        pjt_recent_years = hard_filters.get("pjt_recent_years")
        if pjt_recent_years:
            cutoff = f"{datetime.now(UTC).year - int(pjt_recent_years)}-01-01"
            must.append(
                models.NestedCondition(
                    nested=models.Nested(
                        key="research_projects",
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="project_end_date",
                                    range=models.DatetimeRange(gte=cutoff),
                                )
                            ]
                        ),
                    )
                )
            )

        # 6. 제외 기관 명단 처리 (정확한 기관명 매칭을 통한 제외)
        for org in exclude_orgs:
            normalized = normalize_org_name(org)
            if normalized:
                must_not.append(
                    models.FieldCondition(
                        key="basic_info.affiliated_organization_exact",
                        match=models.MatchValue(value=normalized),
                    )
                )

        # 설정된 조건이 하나도 없으면 필터 미적용
        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)
