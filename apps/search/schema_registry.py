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
    "basic_info.researcher_id",
    "basic_info.affiliated_organization_exact",
    "researcher_profile.highest_degree",
    "researcher_profile.publication_count",
    "researcher_profile.scie_publication_count",
    "researcher_profile.intellectual_property_count",
    "researcher_profile.research_project_count",
}

FILTERABLE_NESTED_FIELDS = {
    "publications": {"journal_index_type", "publication_year_month"},
    "intellectual_properties": {"application_registration_type", "application_country", "application_date", "registration_date"},
    "research_projects": {"project_start_date", "project_end_date", "reference_year", "performing_organization", "managing_agency"},
}

PAYLOAD_INDEX_FIELDS: tuple[tuple[str, str], ...] = (
    ("basic_info.researcher_id", "keyword"),
    ("basic_info.affiliated_organization_exact", "keyword"),
    ("researcher_profile.highest_degree", "keyword"),
    ("researcher_profile.publication_count", "integer"),
    ("researcher_profile.scie_publication_count", "integer"),
    ("researcher_profile.intellectual_property_count", "integer"),
    ("researcher_profile.research_project_count", "integer"),
    ("publications[].journal_index_type", "keyword"),
    ("publications[].publication_year_month", "datetime"),
    ("intellectual_properties[].application_registration_type", "keyword"),
    ("intellectual_properties[].application_country", "keyword"),
    ("intellectual_properties[].application_date", "datetime"),
    ("intellectual_properties[].registration_date", "datetime"),
    ("research_projects[].project_start_date", "datetime"),
    ("research_projects[].project_end_date", "datetime"),
    ("research_projects[].reference_year", "integer"),
    ("research_projects[].performing_organization", "keyword"),
    ("research_projects[].managing_agency", "keyword"),
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

