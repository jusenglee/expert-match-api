from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from apps.core.config import Settings
from apps.core.runtime_validation import validate_runtime_settings
from apps.search.live_validator import LiveContractValidator
from apps.search.schema_registry import BRANCHES, PAYLOAD_INDEX_FIELDS, SearchSchemaRegistry


def test_strict_runtime_settings_reject_fallback_backends():
    settings = Settings(
        llm_backend="heuristic",
        embedding_backend="hashing",
        strict_runtime_validation=True,
    )

    with pytest.raises(RuntimeError):
        validate_runtime_settings(settings)


class FakeDependencyValidator:
    def validate_backends(self):
        from apps.core.runtime_validation import BackendCheckResult

        return [
            BackendCheckResult(name="llm_backend", ok=True, detail="ok"),
            BackendCheckResult(name="embedding_backend", ok=True, detail="ok"),
        ]


class FakeQdrantClient:
    def __init__(self, *, missing_vector: bool = False, missing_index: bool = False, no_points: bool = False):
        dense_vectors = {
            SearchSchemaRegistry.default().dense_vector_by_branch[branch]: SimpleNamespace(size=1024)
            for branch in BRANCHES
        }
        sparse_vectors = {
            SearchSchemaRegistry.default().sparse_vector_by_branch[branch]: SimpleNamespace(modifier=models.Modifier.IDF)
            for branch in BRANCHES
        }
        if missing_vector:
            dense_vectors.pop("art_vector_e5i")
        payload_schema = {
            field_name: SimpleNamespace(data_type=schema_name)
            for field_name, schema_name in PAYLOAD_INDEX_FIELDS
        }
        if missing_index:
            payload_schema.pop("research_projects[].reference_year")
        self.collection_info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=dense_vectors, sparse_vectors=sparse_vectors)),
            payload_schema=payload_schema,
        )
        self.no_points = no_points

    def get_collection(self, collection_name):
        return self.collection_info

    def scroll(self, **kwargs):
        if self.no_points:
            return [], None
        payload = {
            "basic_info": {"researcher_id": "11008395", "affiliated_organization_exact": "주식회사미소테크"},
            "researcher_profile": {},
            "publications": [{"publication_title": "테스트 논문", "publication_year_month": "2024-09-01", "journal_index_type": "SCIE"}],
            "intellectual_properties": [{"intellectual_property_title": "테스트 특허", "application_date": "2024-06-01"}],
            "research_projects": [{"project_title_korean": "테스트 과제", "project_start_date": "2019-10-07", "project_end_date": "2020-04-06", "reference_year": 2019}],
        }
        return [SimpleNamespace(id="11008395", payload=payload)], None


def test_live_validator_reports_ready_when_contract_is_satisfied():
    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(),
        settings=settings,
        registry=SearchSchemaRegistry.default(),
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is True
    assert report.checks["collection_exists"] is True
    assert report.sample_point_id == "11008395"


def test_live_validator_reports_missing_vector_and_index():
    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(missing_vector=True, missing_index=True),
        settings=settings,
        registry=SearchSchemaRegistry.default(),
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is False
    assert report.checks["dense_vectors_present"] is False
    assert report.checks["payload_indexes_present"] is False
