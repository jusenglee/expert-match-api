"""flat chunk schema_registry 상수(단일 벡터명·인덱스 세트·family) 검증 (v2.1)."""
from __future__ import annotations

from apps.search import schema_registry as sr
from apps.search.doc_types import DOC_TYPE_TO_FAMILY, DOC_TYPES


def test_single_named_vectors():
    assert sr.DENSE_VECTOR_NAME == "vector_e5i"
    assert sr.SPARSE_VECTOR_NAME == "vector_splade"


def test_dense_vector_geometry():
    assert sr.DENSE_VECTOR_SIZE == 1024
    assert sr.DENSE_DISTANCE == "Cosine"


def test_doc_types_reexported_exactly_five():
    assert sr.DOC_TYPES == DOC_TYPES
    assert set(sr.DOC_TYPES) == {
        "paper",
        "patent",
        "project",
        "assessor_activity",
        "specialty",
    }


def test_families_four_values():
    assert set(sr.FAMILIES) == {"identity", "achievement", "assessment", "expertise"}


def test_doc_type_to_family_map_covers_all_doc_types():
    assert set(sr.DOC_TYPE_TO_FAMILY_MAP) == set(DOC_TYPES)
    assert sr.DOC_TYPE_TO_FAMILY_MAP == dict(DOC_TYPE_TO_FAMILY)
    # achievement = {paper, patent, project}
    assert sr.DOC_TYPE_TO_FAMILY_MAP["paper"] == "achievement"
    assert sr.DOC_TYPE_TO_FAMILY_MAP["patent"] == "achievement"
    assert sr.DOC_TYPE_TO_FAMILY_MAP["project"] == "achievement"
    assert sr.DOC_TYPE_TO_FAMILY_MAP["assessor_activity"] == "assessment"
    assert sr.DOC_TYPE_TO_FAMILY_MAP["specialty"] == "expertise"
    # identity has no doc_type member
    assert "identity" not in sr.DOC_TYPE_TO_FAMILY_MAP.values()


def test_filterable_fields_are_flat_keys():
    # flat root keys present
    for field in (
        "researcher_id",
        "doc_type",
        "doc_date",
        "affiliated_organization",
        "highest_degree",
        "publication_count",
        "scie_publication_count",
        "intellectual_property_count",
        "research_project_count",
        "researcher_assessor_activity_count",
    ):
        assert field in sr.FILTERABLE_FIELDS
    # doc_attrs.* keys present
    assert "doc_attrs.indexing_database" in sr.FILTERABLE_FIELDS
    assert "doc_attrs.is_scie" in sr.FILTERABLE_FIELDS
    # no v1.x nested[] residue, no v2.0-doc'd event_year/researcher_meta keys
    for field in sr.FILTERABLE_FIELDS:
        assert "[]" not in field
    assert "event_year" not in sr.FILTERABLE_FIELDS
    assert "event_date" not in sr.FILTERABLE_FIELDS
    assert not any(f.startswith("researcher_meta.") for f in sr.FILTERABLE_FIELDS)


def test_payload_index_fields_shape():
    paths = [path for path, _ in sr.PAYLOAD_INDEX_FIELDS]
    # core keyword indexes
    assert ("researcher_id", "keyword") in sr.PAYLOAD_INDEX_FIELDS
    assert ("doc_type", "keyword") in sr.PAYLOAD_INDEX_FIELDS
    assert ("affiliated_organization", "keyword") in sr.PAYLOAD_INDEX_FIELDS
    assert ("highest_degree", "keyword") in sr.PAYLOAD_INDEX_FIELDS
    # 5 flat count fields are integer indexes
    for count in (
        "publication_count",
        "scie_publication_count",
        "intellectual_property_count",
        "research_project_count",
        "researcher_assessor_activity_count",
    ):
        assert (count, "integer") in sr.PAYLOAD_INDEX_FIELDS
    # recency is a flat datetime field on doc_date (not event_date/event_year)
    assert ("doc_date", "datetime") in sr.PAYLOAD_INDEX_FIELDS
    assert ("event_date", "datetime") not in sr.PAYLOAD_INDEX_FIELDS
    assert ("event_year", "integer") not in sr.PAYLOAD_INDEX_FIELDS
    # no nested(v1.x) residue, no researcher_meta nesting
    assert not any("[]" in path for path in paths)
    assert not any(path.startswith("researcher_meta.") for path in paths)


def test_payload_index_subset_of_filterable():
    paths = {path for path, _ in sr.PAYLOAD_INDEX_FIELDS}
    assert paths <= sr.FILTERABLE_FIELDS


def test_v1x_branch_constants_removed():
    for removed in (
        "BRANCHES",
        "DENSE_VECTOR_BY_BRANCH",
        "SPARSE_VECTOR_BY_BRANCH",
        "SearchSchemaRegistry",
        "DOC_TYPES_V2",
        "DOC_TYPE_TO_FAMILY_V2",
        "FILTERABLE_FIELDS_V2",
        "PAYLOAD_INDEX_FIELDS_V2",
    ):
        assert not hasattr(sr, removed), f"{removed} should be removed from schema_registry"
