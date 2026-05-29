"""WO-B: ntis_researcher_chunks 컬렉션 부트스트랩 + B-4 스모크 CLI (VPN 경유 실행).

서브커맨드:
  ensure   v2.0 컬렉션(단일 dense_e5i + sparse_splade) + payload 인덱스 생성. (Qdrant 필요)
  inspect  컬렉션 vectors/sparse/payload_schema 출력 + 레거시 컬렉션 Point 수(보존 확인).
  smoke    표본 chunk upsert + query_points(dense/sparse/doc_type 필터/OR recency) + 멱등 확인.

스모크는 임베딩 서버 없이 스키마·필터 동작만 검증하도록 **pseudo 벡터**(stable_unit_vector + 소형
sparse map)를 사용한다. 실제 임베딩 적재는 WO-A/WO-D. Point ID는 WO-0 chunk_id 코덱 그대로.

⚠️ VPN 필요: NTIS_QDRANT_URL(기본 http://203.250.234.159:8005) 접근이 VPN 경유로만 가능.
실행 예) ! NTIS_QDRANT_COLLECTION_NAME=ntis_researcher_chunks python -m apps.tools.bootstrap_chunks smoke
"""
from __future__ import annotations

import argparse
import sys

from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.domain.models import ResearcherMeta
from apps.ingest.converters import build_chunk
from apps.search.doc_types import DocType
from apps.search.qdrant_bootstrap import LEGACY_V1X_COLLECTIONS, QdrantBootstrapper
from apps.search.schema_registry import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    SearchSchemaRegistry,
)
from apps.search.text_utils import stable_unit_vector

_SAMPLE_META = ResearcherMeta(
    affiliated_organization="스모크기관",
    highest_degree="박사",
    publication_count=3,
    research_project_count=2,
    researcher_assessor_count=1,
)

# doc_type별 최소 표본 domain_attrs (B-4: profile + publication + research_project + researcher_assessor)
_SAMPLE_ATTRS = {
    DocType.PROFILE: {"position_title": "책임연구원", "highest_degree": "박사", "major_field": "기계공학",
                      "publication_count": 3, "research_project_count": 2, "researcher_assessor_count": 1},
    DocType.PUBLICATION: {"title_primary": "스모크 논문", "journal_name": "스모크지", "journal_class": "SCIE",
                          "publication_year_month": "2024-05", "abstract": "스모크 초록", "keywords": ["스모크"]},
    DocType.RESEARCH_PROJECT: {"project_title_korean": "스모크 과제", "performing_organization": "스모크기관",
                               "managing_agency": "스모크청", "research_summary_korean": "스모크 요약",
                               "research_period_start": "2023-03-01", "research_period_end": "2024-02-28"},
    DocType.RESEARCHER_ASSESSOR: {"appointing_organization": "스모크재단", "evaluation_committee_name": "스모크위원회",
                                  "appointing_organization_type": "정부(공공)", "appointment_period_type": "임기제",
                                  "appointment_date": "2024-06-01"},
}


def _client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30.0)


def _bootstrapper(settings: Settings) -> QdrantBootstrapper:
    # sparse_runtime=None → QdrantBootstrapper가 model_requires_idf_modifier(sparse_model_name)로
    # modifier를 도출(WO-B B-3 폴백). 기본 PIXIE-Splade → IDF 없음. (실제 backend 정합은 readiness가 점검)
    return QdrantBootstrapper(
        client=_client(settings),
        settings=settings,
        registry=SearchSchemaRegistry.default(),
        sparse_runtime=None,
    )


def _sample_chunks(settings: Settings):
    chunks = []
    for index, (doc_type, attrs) in enumerate(_SAMPLE_ATTRS.items(), start=1):
        chunks.append(
            build_chunk(
                doc_type,
                researcher_id="SMOKE001",
                researcher_name="스모크연구자",
                doc_seq=index,
                domain_attrs=attrs,
                researcher_meta=_SAMPLE_META,
            )
        )
    return chunks


def _pseudo_point(settings: Settings, chunk):
    dense = stable_unit_vector(chunk.chunk_text, settings.embedding_vector_size)
    sparse = models.SparseVector(indices=[1, 7, 42], values=[1.0, 0.5, 0.25])
    return models.PointStruct(
        id=chunk.chunk_id,
        vector={DENSE_VECTOR_NAME: dense, SPARSE_VECTOR_NAME: sparse},
        payload=chunk.model_dump(),
    )


def cmd_ensure(settings: Settings) -> int:
    _bootstrapper(settings).ensure_collection(recreate=False)
    print(f"[ensure] collection={settings.qdrant_collection_name} ensured (v2 single-vector schema + indexes)")
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

    chunks = _sample_chunks(settings)
    points = [_pseudo_point(settings, c) for c in chunks]
    client.upsert(collection_name=collection, points=points)
    count_after_first = client.count(collection).count
    # 멱등: 동일 chunk_id 재upsert → Point 수 불변
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

    pub_hits = client.query_points(
        collection_name=collection,
        query=dense_q,
        using=DENSE_VECTOR_NAME,
        query_filter=models.Filter(must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="publication"))]),
        limit=5,
    ).points
    print(f"[smoke] doc_type=publication filtered hits = {len(pub_hits)}")

    # OR recency 가드(HARD-4) 사전 확인: 2개 doc_type event_year>=cutoff OR → 0건이 아님
    or_recency = models.Filter(
        must=[
            models.Filter(
                min_should=models.MinShould(
                    conditions=[
                        models.Filter(must=[
                            models.FieldCondition(key="doc_type", match=models.MatchValue(value="publication")),
                            models.FieldCondition(key="event_year", range=models.Range(gte=2020)),
                        ]),
                        models.Filter(must=[
                            models.FieldCondition(key="doc_type", match=models.MatchValue(value="research_project")),
                            models.FieldCondition(key="event_year", range=models.Range(gte=2020)),
                        ]),
                    ],
                    min_count=1,
                )
            )
        ]
    )
    or_hits = client.query_points(collection_name=collection, query=dense_q, using=DENSE_VECTOR_NAME, query_filter=or_recency, limit=5).points
    print(f"[smoke] OR-recency(publication|research_project, event_year>=2020) hits = {len(or_hits)} (must be >=1)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WO-B ntis_researcher_chunks 부트스트랩/스모크 (VPN 필요)")
    parser.add_argument("command", choices=["ensure", "inspect", "smoke"])
    args = parser.parse_args(argv)
    settings = Settings()
    if not args.command:
        parser.print_help()
        return 2
    return {"ensure": cmd_ensure, "inspect": cmd_inspect, "smoke": cmd_smoke}[args.command](settings)


if __name__ == "__main__":
    raise SystemExit(main())
