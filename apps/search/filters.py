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
    고수준의 검색 조건(Planner가 추출한 hard_filters 등)을 
    실제 Qdrant 벡터 데이터베이스가 이해할 수 있는 저수준의 필터 구조(models.Filter)로 컴파일하는 클래스입니다.
    """
    def compile(
        self,
        hard_filters: dict[str, Any],
        exclude_orgs: list[str],
        include_orgs: list[str] | None = None,
    ) -> models.Filter | None:
        """
        주어진 하드 필터(hard_filters) 및 소속 기관 포함/제외 명단을 분석하여 단일 Qdrant 필터 객체를 생성합니다.

        [Qdrant 필터 논리 구조]
        - 'must': 배열 내의 모든 조건을 동시에 만족해야 함 (AND 조건)
        - 'must_not': 배열 내의 조건 중 하나라도 맞으면 결과에서 제외함 (NOT 조건)
        - 'min_should': 배열 내 조건 중 최소 N개 이상을 만족해야 함 (OR 조건, min_count 설정)
        """
        # planner는 질의 의도 위주로 조건을 뽑고, 이 컴파일러는 이를 실제 DB 스키마 필드와 매핑하여 확정한다.
        must: list[models.Condition] = []
        must_not: list[models.Condition] = []

        # 0. 소속 기관 포함(include) 조건 처리
        # "X 소속 연구자 중" 패턴: X 기관 소속 연구자만 결과에 포함 (must 조건)
        # exclude_orgs(must_not)와 반대 방향. 동일한 정규화 함수 사용.
        for org in (include_orgs or []):
            normalized = normalize_org_name(org)
            if normalized:
                must.append(
                    models.FieldCondition(
                        key="basic_info.affiliated_organization_exact",
                        match=models.MatchValue(value=normalized),
                    )
                )

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
        # '활동 지속성' 의도로 생성된 순수 최근연도 조건은 아래 recent_activity_conditions에
        # 모아서 OR(should) 로 처리한다. SCIE 같은 한정자가 함께 있으면 hard 조건(must) 유지.
        recent_activity_conditions: list[models.Condition] = []

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
            art_condition = models.NestedCondition(
                nested=models.Nested(
                    key="publications",
                    filter=models.Filter(must=art_must),
                )
            )
            if recent_years and not scie_required:
                # 순수 활동 최근성 조건 → OR 풀에 합류
                recent_activity_conditions.append(art_condition)
            else:
                # SCIE 등 한정자가 포함된 경우 → 독립 hard 조건 유지
                must.append(art_condition)

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
            pat_condition = models.NestedCondition(
                nested=models.Nested(
                    key="intellectual_properties",
                    filter=models.Filter(must=pat_must),
                )
            )
            if pat_recent_years and not pat_regist_type:
                # 순수 활동 최근성 조건 → OR 풀에 합류
                recent_activity_conditions.append(pat_condition)
            else:
                # 등록 유형 한정자가 포함된 경우 → 독립 hard 조건 유지
                must.append(pat_condition)

        # 5. 과제 관련 최근성 조건 (Nested 구조 지원)
        pjt_recent_years = hard_filters.get("pjt_recent_years")
        if pjt_recent_years:
            cutoff = f"{datetime.now(UTC).year - int(pjt_recent_years)}-01-01"
            pjt_condition = models.NestedCondition(
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
            # 과제 최근성은 항상 OR 풀에 합류
            recent_activity_conditions.append(pjt_condition)

        # 활동 최근성 조건 취합:
        # 2개 이상이면 min_should(OR, min_count=1)로 묶어 단일 must 요소로 삽입.
        # 이를 통해 논문·특허·과제 중 하나라도 최근 활동 이력이 있으면 통과하도록 함.
        # (기존: 개별 must → AND 교집합으로 사실상 결과 0건 발생)
        if len(recent_activity_conditions) >= 2:
            must.append(
                models.Filter(
                    min_should=models.MinShould(
                        conditions=recent_activity_conditions,
                        min_count=1,
                    ),
                )
            )
        elif len(recent_activity_conditions) == 1:
            must.append(recent_activity_conditions[0])

        # 6. 전공(major_nm) 필터 처리 (부분 일치)
        if major := hard_filters.get("major_nm"):
            must.append(
                models.FieldCondition(
                    key="researcher_profile.major_field",
                    match=models.MatchText(text=major),
                )
            )

        # 7. 제외 기관 명단 처리 (정확한 기관명 매칭을 통한 제외)
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
