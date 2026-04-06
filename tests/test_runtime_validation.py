from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from apps.core.config import Settings
from apps.core.runtime_validation import validate_runtime_settings
from apps.search.live_validator import LiveContractValidator
from apps.search.schema_registry import BRANCHES, PAYLOAD_INDEX_FIELDS, SearchSchemaRegistry


def build_valid_sample_payload() -> dict[str, object]:
    return {
        "basic_info": {"researcher_id": "11008395", "affiliated_organization_exact": "Test Organization"},
        "researcher_profile": {},
        "publications": [
            {"publication_title": "Test paper", "publication_year_month": "2024-09-01", "journal_index_type": "SCIE"}
        ],
        "intellectual_properties": [
            {"intellectual_property_title": "Test patent", "application_date": "2024-06-01"}
        ],
        "research_projects": [
            {
                "project_title_korean": "Test project",
                "project_start_date": "2019-10-07",
                "project_end_date": "2020-04-06",
                "reference_year": 2019,
            }
        ],
    }


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
    def __init__(
        self,
        *,
        missing_vector: bool = False,
        missing_index: bool = False,
        no_points: bool = False,
        malformed_payload: bool = False,
        sparse_modifier: object = models.Modifier.IDF,
        sample_payloads: list[object] | None = None,
    ):
        dense_vectors = {
            SearchSchemaRegistry.default().dense_vector_by_branch[branch]: SimpleNamespace(size=1024)
            for branch in BRANCHES
        }
        sparse_vectors = {
            SearchSchemaRegistry.default().sparse_vector_by_branch[branch]: SimpleNamespace(modifier=sparse_modifier)
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
        self.malformed_payload = malformed_payload
        payloads = sample_payloads or [build_valid_sample_payload()]
        self.sample_records = [
            SimpleNamespace(id=str(11008395 + index), payload=payload)
            for index, payload in enumerate(payloads)
        ]

    def get_collection(self, collection_name):
        return self.collection_info

    def scroll(self, **kwargs):
        if self.no_points:
            return [], None
        if self.malformed_payload:
            return [SimpleNamespace(id="11008395", payload=["not", "an", "object"])], None

        offset = kwargs.get("offset")
        limit = kwargs.get("limit", 1)
        start = int(offset) if offset is not None else 0
        end = start + limit
        records = self.sample_records[start:end]
        next_offset = end if end < len(self.sample_records) else None
        return records, next_offset


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
    assert report.checks["sample_payload_valid"] is True
    assert report.sample_point_id == "11008395"


def test_live_validator_scans_for_representative_sample_point():
    settings = Settings()
    incomplete_payload = build_valid_sample_payload()
    incomplete_payload["publications"] = []
    validator = LiveContractValidator(
        client=FakeQdrantClient(sample_payloads=[incomplete_payload, build_valid_sample_payload()]),
        settings=settings,
        registry=SearchSchemaRegistry.default(),
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is True
    assert report.sample_point_id == "11008396"
    assert report.checks["sample_art_present"] is True


def test_live_validator_allows_missing_patent_evidence_in_sample():
    settings = Settings()
    payload_without_patents = build_valid_sample_payload()
    payload_without_patents["intellectual_properties"] = []
    validator = LiveContractValidator(
        client=FakeQdrantClient(sample_payloads=[payload_without_patents]),
        settings=settings,
        registry=SearchSchemaRegistry.default(),
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is True
    assert report.sample_point_id == "11008395"
    assert report.checks["sample_pat_present"] is False
    assert "Sample point is missing intellectual_properties[]." not in report.issues


def test_live_validator_accepts_modifier_like_object_with_idf_name():
    class ModifierLike:
        name = "IDF"

    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(sparse_modifier=ModifierLike()),
        settings=settings,
        registry=SearchSchemaRegistry.default(),
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is True
    assert report.checks["sparse_vectors_idf"] is True


def test_settings_default_local_embedding_model_path_points_to_repo_bundle():
    settings = Settings()

    assert settings.embedding_model_name.endswith("multilingual-e5-large-instruct")
    assert settings.embedding_backend == "local"


def test_settings_default_local_embedding_bundle_contains_required_modules():
    settings = Settings()
    model_path = Path(settings.embedding_model_name)

    assert (model_path / "modules.json").is_file()
    assert (model_path / "1_Pooling" / "config.json").is_file()
    assert (model_path / "2_Normalize").is_dir()


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


def test_live_validator_reports_malformed_sample_payload_without_raising():
    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(malformed_payload=True),
        settings=settings,
        registry=SearchSchemaRegistry.default(),
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is False
    assert report.checks["sample_payload_valid"] is False
    assert any("Sample point payload inspection failed" in issue for issue in report.issues)
