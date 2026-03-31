from __future__ import annotations

from dataclasses import dataclass


BRANCHES: tuple[str, str, str, str] = ("basic", "art", "pat", "pjt")

DENSE_VECTOR_BY_BRANCH = {
    "basic": "basic_vector_e5i",
    "art": "art_vector_e5i",
    "pat": "pat_vector_e5i",
    "pjt": "pjt_vector_e5i",
}

SPARSE_VECTOR_BY_BRANCH = {
    "basic": "basic_vector_bm25",
    "art": "art_vector_bm25",
    "pat": "pat_vector_bm25",
    "pjt": "pjt_vector_bm25",
}

DATE_FIELD_CORRECTIONS = {
    "TOT_RSCH_START_DT": "start_dt",
    "TOT_RSCH_END_DT": "end_dt",
    "STAN_YR": "stan_yr",
}

FILTERABLE_ROOT_FIELDS = {
    "doc_id",
    "blng_org_nm_exact",
    "degree_slct_nm",
    "article_cnt",
    "scie_cnt",
    "patent_cnt",
    "project_cnt",
}

FILTERABLE_NESTED_FIELDS = {
    "art": {"sci_slct_nm", "jrnl_pub_dt"},
    "pat": {"ipr_regist_type_nm", "ipr_regist_nat_nm", "aply_dt", "regist_dt"},
    "pjt": {"start_dt", "end_dt", "stan_yr", "pjt_prfrm_org_nm", "rsch_mgnt_org_nm"},
}

PAYLOAD_INDEX_FIELDS: tuple[tuple[str, str], ...] = (
    ("doc_id", "keyword"),
    ("blng_org_nm_exact", "keyword"),
    ("degree_slct_nm", "keyword"),
    ("article_cnt", "integer"),
    ("scie_cnt", "integer"),
    ("patent_cnt", "integer"),
    ("project_cnt", "integer"),
    ("art[].sci_slct_nm", "keyword"),
    ("art[].jrnl_pub_dt", "datetime"),
    ("pat[].ipr_regist_type_nm", "keyword"),
    ("pat[].ipr_regist_nat_nm", "keyword"),
    ("pat[].aply_dt", "datetime"),
    ("pat[].regist_dt", "datetime"),
    ("pjt[].start_dt", "datetime"),
    ("pjt[].end_dt", "datetime"),
    ("pjt[].stan_yr", "integer"),
    ("pjt[].pjt_prfrm_org_nm", "keyword"),
    ("pjt[].rsch_mgnt_org_nm", "keyword"),
)


@dataclass(frozen=True, slots=True)
class SearchSchemaRegistry:
    dense_vector_by_branch: dict[str, str]
    sparse_vector_by_branch: dict[str, str]

    @classmethod
    def default(cls) -> "SearchSchemaRegistry":
        return cls(
            dense_vector_by_branch=DENSE_VECTOR_BY_BRANCH,
            sparse_vector_by_branch=SPARSE_VECTOR_BY_BRANCH,
        )

