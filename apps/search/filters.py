from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from qdrant_client import models

from apps.search.text_utils import normalize_org_name


class QdrantFilterCompiler:
    def compile(self, hard_filters: dict[str, Any], exclude_orgs: list[str]) -> models.Filter | None:
        # hard filter는 LLM이 바꾸지 못하게 하고, 이 컴파일러가 Qdrant 조건으로 확정한다.
        # 즉, planner가 뽑은 조건을 검색 친화적 문장 수준이 아니라 실행 가능한 DB 필터로 낮춘다.
        must: list[models.Condition] = []
        must_not: list[models.Condition] = []

        if degree := hard_filters.get("degree_slct_nm"):
            values = degree if isinstance(degree, list) else [degree]
            must.append(
                models.FieldCondition(
                    key="degree_slct_nm",
                    match=models.MatchAny(any=values),
                )
            )

        for field_name in ("article_cnt", "scie_cnt", "patent_cnt", "project_cnt"):
            min_value = hard_filters.get(f"{field_name}_min")
            if min_value is not None:
                must.append(
                    models.FieldCondition(
                        key=field_name,
                        range=models.Range(gte=min_value),
                    )
                )

        recent_years = hard_filters.get("art_recent_years")
        scie_required = hard_filters.get("art_sci_slct_nm")
        if recent_years or scie_required:
            art_must: list[models.Condition] = []
            # 논문 조건은 같은 art[] 객체에 동시에 걸려야 한다.
            # 그래야 "SCIE인 논문"과 "최근 논문"이 서로 다른 항목인 경우를 막을 수 있다.
            if scie_required:
                art_must.append(
                    models.FieldCondition(key="sci_slct_nm", match=models.MatchValue(value=scie_required))
                )
            if recent_years:
                cutoff = f"{datetime.now(UTC).year - int(recent_years)}-01-01"
                art_must.append(
                    models.FieldCondition(
                        key="jrnl_pub_dt",
                        range=models.DatetimeRange(gte=cutoff),
                    )
                )
            must.append(
                models.NestedCondition(
                    nested=models.Nested(
                        key="art",
                        filter=models.Filter(must=art_must),
                    )
                )
            )

        pat_regist_type = hard_filters.get("pat_ipr_regist_type_nm")
        pat_recent_years = hard_filters.get("pat_recent_years")
        if pat_regist_type or pat_recent_years:
            pat_must: list[models.Condition] = []
            if pat_regist_type:
                pat_must.append(
                    models.FieldCondition(
                        key="ipr_regist_type_nm",
                        match=models.MatchValue(value=pat_regist_type),
                    )
                )
            if pat_recent_years:
                cutoff = f"{datetime.now(UTC).year - int(pat_recent_years)}-01-01"
                pat_must.append(
                    models.FieldCondition(
                        key="aply_dt",
                        range=models.DatetimeRange(gte=cutoff),
                    )
                )
            must.append(
                models.NestedCondition(
                    nested=models.Nested(
                        key="pat",
                        filter=models.Filter(must=pat_must),
                    )
                )
            )

        pjt_recent_years = hard_filters.get("pjt_recent_years")
        if pjt_recent_years:
            cutoff = f"{datetime.now(UTC).year - int(pjt_recent_years)}-01-01"
            must.append(
                models.NestedCondition(
                    nested=models.Nested(
                        key="pjt",
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="end_dt",
                                    range=models.DatetimeRange(gte=cutoff),
                                )
                            ]
                        ),
                    )
                )
            )

        for org in exclude_orgs:
            normalized = normalize_org_name(org)
            if normalized:
                # 기관 제외는 exact 필터로 처리해야 운영자가 예측 가능하다.
                must_not.append(
                    models.FieldCondition(
                        key="blng_org_nm_exact",
                        match=models.MatchValue(value=normalized),
                    )
                )

        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)
