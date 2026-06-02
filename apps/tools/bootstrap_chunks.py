"""ntis_researcher_chunks 컬렉션 부트스트랩 + 스모크 CLI (VPN 경유 실행).

서브커맨드:
  ensure   flat 단일 벡터 컬렉션(vector_e5i + vector_splade) + payload 인덱스 생성. (Qdrant 필요)
  inspect  컬렉션 vectors/sparse/payload_schema 출력 + 레거시 컬렉션 Point 수(보존 확인).
  smoke    표본 flat chunk upsert + query_points(dense/sparse/doc_type 필터/OR recency) + 멱등 확인.

스모크는 임베딩 서버 없이 스키마·필터 동작만 검증하도록 pseudo 벡터(stable_unit_vector + 소형
sparse map)를 사용한다. Point ID는 flat chunk_id 코덱 그대로(<doc_type>_<숫자doc_id>_c<NNN>).

⚠️ VPN 필요: NTIS_QDRANT_URL(기본 http://203.250.234.159:8005) 접근이 VPN 경유로만 가능.
실행 예) ! NTIS_QDRANT_COLLECTION_NAME=ntis_researcher_chunks python -m apps.tools.bootstrap_chunks smoke
"""
from __future__ import annotations

import argparse

from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.domain.models import ChunkPayload
from apps.search.doc_types import DocType, build_chunk_id
from apps.search.qdrant_bootstrap import LEGACY_V1X_COLLECTIONS, QdrantBootstrapper
from apps.search.schema_registry import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from apps.search.text_utils import stable_unit_vector

_SMOKE_RID = "SMOKE001"
_SMOKE_NAME = "스모크연구자"
_COMMON_ROOT = {
    "researcher_id": _SMOKE_RID,
    "researcher_name": _SMOKE_NAME,
    "affiliated_organization": "스모크기관",
    "highest_degree": "박사",
    "publication_count": 3,
    "scie_publication_count": 1,
    "intellectual_property_count": 1,
    "research_project_count": 2,
    "researcher_assessor_activity_count": 1,
}

# doc_type별 최소 표본 (flat root + doc_attrs + doc_date)
_SAMPLE_SPECS = [
    (DocType.PAPER, "100000000001", "2024-05-01",
     {"main_language_title": "스모크 논문", "journal_name": "스모크지", "indexing_database": "SCIE",
      "is_scie": "Y", "publication_year_month": "2024-05", "keywords": "스모크"}),
    (DocType.PATENT, "100000000002", "2023-08-01",
     {"intellectual_property_title": "스모크 특허", "intellectual_property_type": "특허권",
      "application_registration_type": "등록", "application_country": "대한민국", "application_date": "2023-08-01"}),
    (DocType.PROJECT, "100000000003", "2023-03-01",
     {"project_title_korean": "스모크 과제", "performing_organization": "스모크기관",
      "managing_agency": "스모크청", "project_period": "2023-03-01 ~ 2024-02-28"}),
    (DocType.ASSESSOR_ACTIVITY, "100000000004", "2024-06-01", {}),
    (DocType.SPECIALTY, "100000000005", "NONE", {}),
]


def _client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30.0)


def _bootstrapper(settings: Settings) -> QdrantBootstrapper:
    return QdrantBootstrapper(client=_client(settings), settings=settings, sparse_runtime=None)


def _sample_chunks() -> list[ChunkPayload]:
    chunks: list[ChunkPayload] = []
    for doc_type, doc_num, doc_date, attrs in _SAMPLE_SPECS:
        chunk_id = build_chunk_id(doc_type, doc_num, 0)
        chunks.append(
            ChunkPayload(
                doc_type=doc_type,
                doc_id=f"{doc_type}_{doc_num}",
                chunk_id=chunk_id,
                chunk_text=f"{doc_type} 스모크 본문",
                doc_date=doc_date,
                doc_attrs=attrs,
                **_COMMON_ROOT,
            )
        )
    return chunks


def _pseudo_point(settings: Settings, chunk: ChunkPayload) -> models.PointStruct:
    dense = stable_unit_vector(chunk.chunk_text, settings.embedding_vector_size)
    sparse = models.SparseVector(indices=[1, 7, 42], values=[1.0, 0.5, 0.25])
    return models.PointStruct(
        id=chunk.chunk_id,
        vector={DENSE_VECTOR_NAME: dense, SPARSE_VECTOR_NAME: sparse},
        payload=chunk.model_dump(),
    )


def cmd_ensure(settings: Settings) -> int:
    _bootstrapper(settings).ensure_collection(recreate=False)
    print(f"[ensure] collection={settings.qdrant_collection_name} ensured (single-vector flat schema + indexes)")
    return 0


def cmd_inspect(settings: Settings) -> int:
    client = _client(settings)
    info = client.get_collection(settings.qdrant_collection_name)
    params = info.config.params
    print(f"[inspect] {settings.qdrant_collection_name}")
    print("  vectors:", getattr(params, "vectors", None))
    print("  sparse_vectors:", getattr(params, "sparse_vectors", None))
    schema = getattr(info, "payload_schema", None)
    print("  payload_schema keys:", sorted((schema or {}).keys()))
    for legacy in LEGACY_V1X_COLLECTIONS:
        try:
            print(f"  [preservation] legacy {legacy} count =", client.count(legacy).count)
        except Exception as exc:  # noqa: BLE001
            print(f"  [preservation] legacy {legacy}: {exc}")
    return 0


def cmd_smoke(settings: Settings) -> int:
    collection = settings.qdrant_collection_name
    client = _client(settings)
    _bootstrapper(settings).ensure_collection(recreate=False)

    chunks = _sample_chunks()
    points = [_pseudo_point(settings, c) for c in chunks]
    client.upsert(collection_name=collection, points=points)
    count_after_first = client.count(collection).count
    client.upsert(collection_name=collection, points=points)
    count_after_second = client.count(collection).count
    print(f"[smoke] upserted {len(points)} chunks; count {count_after_first} -> {count_after_second} (idempotent={count_after_first == count_after_second})")

    dense_q = stable_unit_vector(chunks[0].chunk_text, settings.embedding_vector_size)
    dense_hits = client.query_points(collection_name=collection, query=dense_q, using=DENSE_VECTOR_NAME, limit=5).points
    print(f"[smoke] dense hits = {len(dense_hits)}")

    sparse_hits = client.query_points(
        collection_name=collection,
        query=models.SparseVector(indices=[1, 7, 42], values=[1.0, 0.5, 0.25]),
        using=SPARSE_VECTOR_NAME,
        limit=5,
    ).points
    print(f"[smoke] sparse hits = {len(sparse_hits)}")

    paper_hits = client.query_points(
        collection_name=collection,
        query=dense_q,
        using=DENSE_VECTOR_NAME,
        query_filter=models.Filter(must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="paper"))]),
        limit=5,
    ).points
    print(f"[smoke] doc_type=paper filtered hits = {len(paper_hits)}")

    # OR recency 가드(HARD): doc_date 기준 2개 doc_type OR → 0건이 아님
    or_recency = models.Filter(
        must=[
            models.Filter(
                min_should=models.MinShould(
                    conditions=[
                        models.Filter(must=[
                            models.FieldCondition(key="doc_type", match=models.MatchValue(value="paper")),
                            models.FieldCondition(key="doc_date", range=models.DatetimeRange(gte="2020-01-01T00:00:00Z")),
                        ]),
                        models.Filter(must=[
                            models.FieldCondition(key="doc_type", match=models.MatchValue(value="project")),
                            models.FieldCondition(key="doc_date", range=models.DatetimeRange(gte="2020-01-01T00:00:00Z")),
                        ]),
                    ],
                    min_count=1,
                )
            )
        ]
    )
    or_hits = client.query_points(collection_name=collection, query=dense_q, using=DENSE_VECTOR_NAME, query_filter=or_recency, limit=5).points
    print(f"[smoke] OR-recency(paper|project, doc_date>=2020) hits = {len(or_hits)} (must be >=1)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ntis_researcher_chunks 부트스트랩/스모크 (VPN 필요)")
    parser.add_argument("command", choices=["ensure", "inspect", "smoke"])
    args = parser.parse_args(argv)
    settings = Settings()
    return {"ensure": cmd_ensure, "inspect": cmd_inspect, "smoke": cmd_smoke}[args.command](settings)


if __name__ == "__main__":
    raise SystemExit(main())
