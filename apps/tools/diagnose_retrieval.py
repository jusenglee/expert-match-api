r"""Live retrieval diagnostics for dense/sparse/hybrid/grouped Qdrant search.

This tool is read-only. It compares where a target chunk appears across:
- sparse SPLADE/BM25 search
- dense vector similarity search
- chunk-level hybrid RRF search
- grouped hybrid RRF search by researcher_id

Run directly, for example:

    python D:\Project\python_project\Ntis_person_API\apps\tools\diagnose_retrieval.py

The default case is query="인공지능 반도체" and
target_chunk_id="paper_100003957045_c000". Override them with CLI flags when needed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Allow direct execution:
#   python D:\Project\python_project\Ntis_person_API\apps\tools\diagnose_retrieval.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.domain.models import PlannerOutput
from apps.search.query_builder import QueryTextBuilder
from apps.search.relevance import (
    CONCEPT_VIEW_PREFIX,
    ConceptPlan,
    chunk_concept_signals,
    tag_chunk_concepts,
)
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from apps.search.sparse_runtime import (
    SparseRuntimeConfig,
    prepare_sparse_runtime_environment,
    resolve_sparse_runtime,
)


TITLE_KEYS = (
    "main_language_title",
    "sub_language_title",
    "project_title_korean",
    "intellectual_property_title",
    "specific_specialty_name",
)

DEFAULT_QUERY = "자동차 관련 컨소시엄에 적합한 전기차 배터리 분야 전문가를 추천"
DEFAULT_TARGET_CHUNK_ID = "paper_100003957045_c000"
DIAGNOSTIC_LLM_REASONING_EFFORT = "high"
DIAGNOSTIC_LLM_INCLUDE_REASONING = False
DIAGNOSTIC_LLM_DISABLE_THINKING = False


def _apply_diagnostic_llm_policy() -> tuple[str, bool, bool]:
    """Use high-effort thinking for any LLM call made from this diagnostic process."""
    from apps.core import llm_policies

    previous = (
        llm_policies.CONSISTENCY_REASONING_EFFORT,
        llm_policies.CONSISTENCY_INCLUDE_REASONING,
        llm_policies.CONSISTENCY_DISABLE_THINKING,
    )
    llm_policies.CONSISTENCY_REASONING_EFFORT = DIAGNOSTIC_LLM_REASONING_EFFORT
    llm_policies.CONSISTENCY_INCLUDE_REASONING = DIAGNOSTIC_LLM_INCLUDE_REASONING
    llm_policies.CONSISTENCY_DISABLE_THINKING = DIAGNOSTIC_LLM_DISABLE_THINKING
    return previous


def _restore_llm_policy(snapshot: tuple[str, bool, bool]) -> None:
    from apps.core import llm_policies

    (
        llm_policies.CONSISTENCY_REASONING_EFFORT,
        llm_policies.CONSISTENCY_INCLUDE_REASONING,
        llm_policies.CONSISTENCY_DISABLE_THINKING,
    ) = snapshot


@dataclass(slots=True)
class Runtime:
    settings: Settings
    client: QdrantClient
    dense_encoder: Any
    sparse_runtime: SparseRuntimeConfig
    sparse_encoder: Any | None


@dataclass(slots=True)
class SparseToken:
    token_id: int
    token: str
    weight: float


@dataclass(slots=True)
class SparseOverlap:
    available: bool
    reason: str | None = None
    overlap_token_count: int = 0
    overlap_score: float = 0.0
    top_tokens: list[dict[str, Any]] = field(default_factory=list)


def _build_dense_encoder(settings: Settings) -> Any:
    from apps.search.encoders import HashingDenseEncoder, OpenAIEmbeddingEncoder

    if settings.embedding_backend == "openai":
        return OpenAIEmbeddingEncoder(
            model_name=settings.embedding_model_name,
            vector_size=settings.embedding_vector_size,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
        )
    if settings.embedding_backend == "local":
        from apps.search.encoders import LocalSentenceTransformerEncoder

        return LocalSentenceTransformerEncoder(
            model_name=settings.embedding_model_name,
            vector_size=settings.embedding_vector_size,
        )
    return HashingDenseEncoder(
        model_name=settings.embedding_model_name,
        vector_size=settings.embedding_vector_size,
    )


def _build_runtime(settings: Settings, *, timeout: float) -> Runtime:
    from apps.search.encoders import SpladeSparseEncoder

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        cloud_inference=settings.qdrant_cloud_inference,
        timeout=timeout,
        check_compatibility=False,
    )
    cache_dir = prepare_sparse_runtime_environment(settings)
    dense_encoder = _build_dense_encoder(settings)
    sparse_runtime, sparse_encoder = resolve_sparse_runtime(
        client=client,
        settings=settings,
        cache_dir=cache_dir,
        sparse_encoder_factory=SpladeSparseEncoder,
    )
    return Runtime(
        settings=settings,
        client=client,
        dense_encoder=dense_encoder,
        sparse_runtime=sparse_runtime,
        sparse_encoder=sparse_encoder,
    )


# ---------------------------------------------------------------------------
# Offline mode (--offline): no VPN/Qdrant. In-memory flat corpus + fake client so
# the full diagnostic (including the production multiview path) runs as a smoke test.
# ---------------------------------------------------------------------------
def _offline_payload(
    researcher_id: str,
    name: str,
    *,
    doc_type: str,
    doc_num: str,
    chunk_index: int,
    text: str,
    organization: str = "오프라인대학교",
    doc_attrs: dict | None = None,
) -> dict[str, Any]:
    return {
        "researcher_id": researcher_id,
        "researcher_name": name,
        "doc_type": doc_type,
        "doc_id": f"{doc_type}_{doc_num}",
        "chunk_id": f"{doc_type}_{doc_num}_c{chunk_index:03d}",
        "chunk_text": text,
        "doc_date": "2024-01-01",
        "affiliated_organization": organization,
        "highest_degree": "박사",
        "publication_count": 5,
        "scie_publication_count": 2,
        "intellectual_property_count": 1,
        "research_project_count": 3,
        "researcher_assessor_activity_count": 1,
        "doc_attrs": doc_attrs or {},
    }


def _offline_corpus() -> list[dict[str, Any]]:
    """대표 시나리오: joint / 균형 / 한쪽몰빵 / 반도체-only / AI-only(활동) / off-topic."""
    return [
        # R1 — joint(한 chunk가 AI+반도체 동시), 균형. main tier 최상위 기대.
        _offline_payload("M0001", "이상헌", doc_type="paper", doc_num="100000000001",
                         chunk_index=0, text="인공지능 반도체 가속기 설계 및 딥러닝 추론 최적화"),
        _offline_payload("M0001", "이상헌", doc_type="project", doc_num="200000000001",
                         chunk_index=0, text="반도체 공정 데이터 기반 머신러닝 수율 예측 시스템"),
        _offline_payload("M0001", "이상헌", doc_type="specialty", doc_num="M0001",
                         chunk_index=0, text="인공지능 반도체 시스템반도체 설계"),
        # R2 — AI/반도체 분리 균형. main tier(separate) 기대.
        # patent의 '반도체' 신호는 chunk_text가 아니라 doc_attrs에만 존재 → production의
        # doc_attrs 기반 concept 태깅 경로(relevance._payload_text)를 오프라인에서 실제로 검증.
        _offline_payload("M0002", "권영수", doc_type="paper", doc_num="100000000002",
                         chunk_index=0, text="딥러닝 신경망 기반 영상 인식 인공지능 연구"),
        _offline_payload("M0002", "권영수", doc_type="patent", doc_num="300000000002",
                         chunk_index=0, text="지식재산권명: 저전력 신호처리 회로 설계",
                         doc_attrs={"intellectual_property_title": "시스템반도체 집적회로 저전력 설계",
                                    "keywords": "반도체;집적회로;저전력"}),
        # R3 — AI 강, 반도체 약(한쪽 몰빵). balance가 R1/R2 아래로 눌러야 함.
        _offline_payload("M0003", "김편중", doc_type="paper", doc_num="100000000003",
                         chunk_index=0, text="생성형 인공지능 거대언어모델 딥러닝 학습 기법"),
        _offline_payload("M0003", "김편중", doc_type="paper", doc_num="100000000004",
                         chunk_index=0, text="인공지능 강화학습 정책 최적화 신경망"),
        _offline_payload("M0003", "김편중", doc_type="assessor_activity", doc_num="M0003",
                         chunk_index=0, text="평가위원회명: 반도체 소부장 평가위원"),
        # R4 — 반도체 only(AI 없음). required 미충족 → partial → fallback tier 기대.
        _offline_payload("M0004", "박반도", doc_type="paper", doc_num="100000000005",
                         chunk_index=0, text="전력반도체 MOSFET 소자 신뢰성 분석"),
        _offline_payload("M0004", "박반도", doc_type="patent", doc_num="300000000005",
                         chunk_index=0, text="반도체 패키징 열 방출 구조 특허"),
        # R5 — off-topic. 어떤 view에도 안 잡혀 결과에서 제외 기대.
        _offline_payload("M0005", "최무관", doc_type="paper", doc_num="100000000006",
                         chunk_index=0, text="조선시대 행정 제도와 모호성 가설의 재검토"),
        # R6 — AI-only(평가활동). partial → fallback 기대.
        _offline_payload("M0006", "정평가", doc_type="assessor_activity", doc_num="M0006",
                         chunk_index=0, text="임명기관: 한국연구재단 평가위원회명: 인공지능 미래기술"),
    ]


class _OfflineDenseEncoder:
    """dense 쿼리를 텍스트 그대로 통과(오프라인 fake client가 키워드 스코어링에 사용)."""

    model_name = "offline-hash"
    vector_size = 8

    def embed(self, text: str) -> str:
        return text


def _offline_searchable(payload: dict[str, Any]) -> str:
    return _search_text(payload).casefold()


def _offline_rank(query_text: str, corpus: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    """질의 토큰 substring 중첩 수로 점수화(>0만). 결정적·로컬. dense/sparse/concept view 공통."""
    tokens = {t for t in re.split(r"\s+", (query_text or "").casefold()) if len(t) >= 2}
    scored: list[tuple[float, dict[str, Any]]] = []
    for payload in corpus:
        haystack = _offline_searchable(payload)
        overlap = sum(1 for t in tokens if t in haystack)
        if overlap > 0:
            scored.append((float(overlap), payload))
    scored.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
    return scored


class _OfflineFakeClient:
    """query_points/query_points_groups/scroll를 in-memory corpus로 흉내(read-only)."""

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self.corpus = corpus
        self._by_chunk = {c["chunk_id"]: c for c in corpus}

    @staticmethod
    def _route(query: Any, using: Any, prefetch: Any) -> str | None:
        if using == DENSE_VECTOR_NAME:
            return query if isinstance(query, str) else None
        if using == SPARSE_VECTOR_NAME:
            return getattr(query, "text", None)
        if prefetch:  # FusionQuery 하이브리드: 첫 prefetch(dense) 텍스트로 근사.
            head = prefetch[0].query
            return head if isinstance(head, str) else getattr(head, "text", None)
        return None

    def _points(self, query_text: str | None, limit: int) -> Any:
        ranked = _offline_rank(query_text, self.corpus) if query_text is not None else []
        points = [
            SimpleNamespace(id=payload["chunk_id"], score=score, payload=payload)
            for score, payload in ranked[: max(0, limit)]
        ]
        return SimpleNamespace(points=points)

    def query_points(self, *, query=None, using=None, prefetch=None, limit=10, **_kwargs) -> Any:
        return self._points(self._route(query, using, prefetch), limit)

    def query_points_groups(self, *, prefetch=None, query=None, group_by="researcher_id",
                            group_size=10, limit=80, **_kwargs) -> Any:
        ranked = _offline_rank(self._route(query, None, prefetch), self.corpus)
        groups_by_key: dict[str, list[Any]] = {}
        for score, payload in ranked:
            key = str(payload.get(group_by) or "")
            bucket = groups_by_key.setdefault(key, [])
            if len(bucket) < group_size:
                bucket.append(SimpleNamespace(id=payload["chunk_id"], score=score, payload=payload))
        groups = [SimpleNamespace(id=key, hits=hits) for key, hits in groups_by_key.items()][:limit]
        return SimpleNamespace(groups=groups)

    def scroll(self, *, scroll_filter=None, limit=2, **_kwargs):
        chunk_id = None
        try:
            chunk_id = scroll_filter.must[0].match.value  # type: ignore[union-attr]
        except (AttributeError, IndexError, TypeError):
            chunk_id = None
        payload = self._by_chunk.get(str(chunk_id)) if chunk_id is not None else None
        records = [SimpleNamespace(id=payload["chunk_id"], payload=payload)] if payload else []
        return records[:limit], None


def _build_offline_runtime(settings: Settings) -> Runtime:
    return Runtime(
        settings=settings,
        client=_OfflineFakeClient(_offline_corpus()),
        dense_encoder=_OfflineDenseEncoder(),
        sparse_runtime=SparseRuntimeConfig(
            backend="offline_fake",
            active_model_name="offline-splade",
            requires_idf_modifier=False,
            used_fallback=False,
        ),
        sparse_encoder=None,
    )


# NOTE: dense/sparse 쿼리 구성은 진단툴에서 자체 정의하지 않고 production retriever의
# `_dense_prompt`/`_build_dense_query`/`_build_sparse_query`를 그대로 재사용한다(임베딩 단일 출처).
# 과거의 로컬 _dense_query_text(한국어 instruct 프리픽스)는 production(영문 프리픽스)과 어긋나
# raw dense/hybrid 기준선을 왜곡했기에 제거함.


def _point_payload(point: Any) -> dict[str, Any]:
    return getattr(point, "payload", None) or (
        point.get("payload", {}) if isinstance(point, dict) else {}
    ) or {}


def _point_id(point: Any) -> str:
    return str(getattr(point, "id", "") or (point.get("id", "") if isinstance(point, dict) else ""))


def _point_score(point: Any) -> float | None:
    if hasattr(point, "score"):
        score = getattr(point, "score")
    elif isinstance(point, dict):
        score = point.get("score")
    else:
        score = None
    return float(score) if score is not None else None


def _title(payload: dict[str, Any]) -> str:
    attrs = payload.get("doc_attrs") or {}
    for key in TITLE_KEYS:
        value = attrs.get(key)
        if isinstance(value, str) and value.strip() and value.strip().upper() != "NONE":
            return value.strip()
    text = payload.get("chunk_text") or ""
    return " ".join(str(text).split())[:80]


def _search_text(payload: dict[str, Any]) -> str:
    attrs = payload.get("doc_attrs") or {}
    parts: list[str] = [
        str(payload.get("chunk_text") or ""),
        str(payload.get("doc_type") or ""),
        str(payload.get("researcher_name") or ""),
        str(payload.get("affiliated_organization") or ""),
    ]
    for value in attrs.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value)
    return " ".join(parts)


def _surface_query_term_hits(payload: dict[str, Any], keywords: list[str]) -> list[str]:
    haystack = _search_text(payload).casefold()
    return [keyword for keyword in keywords if keyword.casefold() in haystack]


def _sparse_vector_items(vector: Any) -> dict[int, float]:
    if vector is None:
        return {}
    if isinstance(vector, models.SparseVector):
        return {int(idx): float(val) for idx, val in zip(vector.indices, vector.values)}
    if hasattr(vector, "indices") and hasattr(vector, "values"):
        return {
            int(idx): float(val)
            for idx, val in zip(getattr(vector, "indices") or [], getattr(vector, "values") or [])
        }
    if isinstance(vector, dict):
        if "indices" in vector and "values" in vector:
            return {
                int(idx): float(val)
                for idx, val in zip(vector.get("indices") or [], vector.get("values") or [])
            }
        output: dict[int, float] = {}
        for key, value in vector.items():
            try:
                output[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return output
    return {}


def _point_vector(point: Any, vector_name: str) -> Any | None:
    vectors = getattr(point, "vector", None)
    if vectors is None and isinstance(point, dict):
        vectors = point.get("vector") or point.get("vectors")
    if isinstance(vectors, dict):
        return vectors.get(vector_name)
    if hasattr(vectors, vector_name):
        return getattr(vectors, vector_name)
    return vectors if vector_name == "" else None


def _decode_sparse_token(runtime: Runtime, token_id: int) -> str:
    tokenizer = getattr(runtime.sparse_encoder, "_tokenizer", None)
    if tokenizer is None:
        return f"#{token_id}"
    try:
        return str(tokenizer.convert_ids_to_tokens(int(token_id)))
    except Exception:  # noqa: BLE001 - diagnostic output should not fail retrieval.
        return f"#{token_id}"


def _top_sparse_tokens(
    runtime: Runtime,
    sparse_map: dict[int, float],
    *,
    limit: int,
    weights: dict[int, float] | None = None,
) -> list[SparseToken]:
    weighted_items = []
    for token_id, value in sparse_map.items():
        sort_value = weights[token_id] if weights and token_id in weights else value
        weighted_items.append((token_id, value, sort_value))
    weighted_items.sort(key=lambda item: (-item[2], item[0]))
    return [
        SparseToken(
            token_id=token_id,
            token=_decode_sparse_token(runtime, token_id),
            weight=float(value),
        )
        for token_id, value, _sort_value in weighted_items[:limit]
    ]


def _sparse_overlap(
    point: Any,
    *,
    query_sparse_map: dict[int, float],
    runtime: Runtime,
    limit: int,
) -> SparseOverlap:
    if not query_sparse_map:
        return SparseOverlap(available=False, reason="query_sparse_vector_unavailable")
    doc_vector = _point_vector(point, SPARSE_VECTOR_NAME)
    doc_sparse_map = _sparse_vector_items(doc_vector)
    if not doc_sparse_map:
        return SparseOverlap(available=False, reason="document_sparse_vector_not_returned")
    overlap_ids = sorted(set(query_sparse_map) & set(doc_sparse_map))
    products = {token_id: query_sparse_map[token_id] * doc_sparse_map[token_id] for token_id in overlap_ids}
    top_overlap_ids = sorted(overlap_ids, key=lambda token_id: (-products[token_id], token_id))[:limit]
    return SparseOverlap(
        available=True,
        overlap_token_count=len(overlap_ids),
        overlap_score=sum(products.values()),
        top_tokens=[
            {
                "token_id": token_id,
                "token": _decode_sparse_token(runtime, token_id),
                "query_weight": round(query_sparse_map[token_id], 6),
                "doc_weight": round(doc_sparse_map[token_id], 6),
                "product": round(products[token_id], 6),
            }
            for token_id in top_overlap_ids
        ],
    )


def _sparse_token_payload(tokens: list[SparseToken]) -> list[dict[str, Any]]:
    return [
        {"token_id": token.token_id, "token": token.token, "weight": round(token.weight, 6)}
        for token in tokens
    ]


def _rank_by_chunk(points: list[Any]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, point in enumerate(points, start=1):
        chunk_id = str(_point_payload(point).get("chunk_id") or "")
        if chunk_id and chunk_id not in ranks:
            ranks[chunk_id] = rank
    return ranks


def _format_point(
    *,
    rank: int,
    point: Any,
    target_chunk_id: str,
    target_researcher_id: str | None,
    keywords: list[str],
    retriever: QdrantHybridRetriever | None = None,
    concept_plan: ConceptPlan | None = None,
    runtime: Runtime | None = None,
    query_sparse_map: dict[int, float] | None = None,
    sparse_overlap_limit: int = 5,
    source_ranks: dict[str, dict[str, int]] | None = None,
) -> str:
    payload = _point_payload(point)
    score = _point_score(point)
    chunk_id = str(payload.get("chunk_id") or "")
    researcher_id = str(payload.get("researcher_id") or "")
    flags: list[str] = []
    if chunk_id == target_chunk_id:
        flags.append("TARGET_CHUNK")
    if target_researcher_id and researcher_id == target_researcher_id:
        flags.append("TARGET_RESEARCHER")
    flag_text = f" flags={','.join(flags)}" if flags else ""
    score_text = f"{score:.6f}" if score is not None else "-"
    surface_hits = _surface_query_term_hits(payload, keywords)
    concepts = _relevance_tags(retriever, payload, concept_plan)
    extra_parts = [
        f"surface_query_term_hits={surface_hits}",
        f"concept_hits={concepts}",
    ]
    if runtime is not None and query_sparse_map is not None:
        overlap = _sparse_overlap(
            point,
            query_sparse_map=query_sparse_map,
            runtime=runtime,
            limit=sparse_overlap_limit,
        )
        if overlap.available:
            extra_parts.append(
                "sparse_overlap="
                + json.dumps(
                    {
                        "tokens": overlap.overlap_token_count,
                        "score": round(overlap.overlap_score, 6),
                        "top": overlap.top_tokens,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            extra_parts.append(f"sparse_overlap_unavailable={overlap.reason}")
    if source_ranks:
        ranks = {
            name: ranks_by_chunk[chunk_id]
            for name, ranks_by_chunk in source_ranks.items()
            if chunk_id in ranks_by_chunk
        }
        if ranks:
            extra_parts.append(f"source_ranks={ranks}")
    return (
        f"{rank:>3}. score={score_text} point_id={_point_id(point)} "
        f"chunk_id={chunk_id} researcher_id={researcher_id} "
        f"name={payload.get('researcher_name')!r} doc_type={payload.get('doc_type')!r} "
        f"doc_date={payload.get('doc_date')!r} title={_title(payload)!r} "
        f"{' '.join(extra_parts)}{flag_text}"
    )


def _points_from_query_response(response: Any) -> list[Any]:
    if hasattr(response, "points"):
        return list(response.points or [])
    if isinstance(response, dict):
        return list(response.get("points") or [])
    return []


def _groups_from_response(response: Any) -> list[Any]:
    if hasattr(response, "groups"):
        return list(response.groups or [])
    if isinstance(response, dict):
        return list(response.get("groups") or [])
    return []


def _group_hits(group: Any) -> list[Any]:
    if hasattr(group, "hits"):
        return list(group.hits or [])
    if isinstance(group, dict):
        return list(group.get("hits") or [])
    return []


def _group_id(group: Any) -> str:
    if hasattr(group, "id"):
        return str(group.id)
    if isinstance(group, dict):
        return str(group.get("id") or "")
    return ""


def _doc_type_filter(settings: Settings) -> models.Filter | None:
    whitelist = settings.retrieval_doc_types
    if whitelist:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchAny(any=list(whitelist)),
                )
            ]
        )
    return None


def _print_dense_query_section(
    *,
    runtime: Runtime,
    raw_query: str,
    dense_query_text: str,
    dense_vector: list[float],
) -> None:
    print("\n[Dense Query]")
    print(f"model={getattr(runtime.dense_encoder, 'model_name', '-')}")
    print(f"raw_query={raw_query!r}")
    print(f"encoded_query={dense_query_text!r}")
    dim = len(dense_vector) if isinstance(dense_vector, (list, tuple)) else "text-passthrough(offline)"
    print(f"vector_dim={dim}")


def _print_sparse_query_section(
    *,
    runtime: Runtime,
    sparse_query_text: str,
    sparse_query: models.Document | models.SparseVector,
    query_sparse_map: dict[int, float],
    token_limit: int,
) -> None:
    print("\n[Sparse Query]")
    print(
        "backend="
        f"{runtime.sparse_runtime.backend} model={runtime.sparse_runtime.active_model_name} "
        f"fallback={runtime.sparse_runtime.used_fallback}"
    )
    print(f"query_text={sparse_query_text!r}")
    if not isinstance(sparse_query, models.SparseVector):
        print("local_sparse_vector=False reason=fastembed_builtin_or_remote_document_query")
        return
    print(f"local_sparse_vector=True vector_terms={len(query_sparse_map)}")
    print(
        "top_tokens="
        + json.dumps(
            _sparse_token_payload(
                _top_sparse_tokens(runtime, query_sparse_map, limit=token_limit)
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _sparse_section_title(runtime: Runtime) -> str:
    if runtime.sparse_runtime.backend == "fastembed_builtin":
        return "Sparse BM25 Fallback Search"
    return "Sparse SPLADE Search"


def _find_target_point(
    client: QdrantClient,
    *,
    collection_name: str,
    target_chunk_id: str,
) -> Any | None:
    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchValue(value=target_chunk_id),
                )
            ]
        ),
        limit=2,
        with_payload=True,
        with_vectors=False,
    )
    return records[0] if records else None


def _relevance_tags(
    retriever: QdrantHybridRetriever | None,
    payload: dict[str, Any],
    concept_plan: ConceptPlan | None,
) -> list[str]:
    """production relevance.tag_chunk_concepts 기준 태깅(alias 신호만; view hit는 검색 시 결합)."""
    if retriever is None or concept_plan is None or not concept_plan.all_concepts:
        return []
    chunk_payload = retriever._validate_chunk(payload)
    if chunk_payload is None:
        return []
    return tag_chunk_concepts(chunk_payload, view_concept_hits=set(), concept_plan=concept_plan)


def _print_target(
    target_point: Any | None,
    *,
    target_chunk_id: str,
    keywords: list[str],
    retriever: QdrantHybridRetriever | None = None,
    concept_plan: ConceptPlan | None = None,
) -> str | None:
    print("\n[Target Payload]")
    if target_point is None:
        print(f"not_found chunk_id={target_chunk_id!r}")
        return None
    payload = _point_payload(target_point)
    point_id = _point_id(target_point)
    researcher_id = str(payload.get("researcher_id") or "")
    print(f"point_id={point_id}")
    print(f"payload.chunk_id={payload.get('chunk_id')!r}")
    if point_id != str(payload.get("chunk_id") or ""):
        print("id_note=payload.chunk_id_is_authoritative point_id differs")
    print(f"researcher_id={researcher_id}")
    print(f"researcher_name={payload.get('researcher_name')!r}")
    print(f"doc_type={payload.get('doc_type')!r}")
    print(f"doc_date={payload.get('doc_date')!r}")
    print(f"title={_title(payload)!r}")
    print(f"chunk_text={payload.get('chunk_text')!r}")
    print(f"surface_query_term_hits={_surface_query_term_hits(payload, keywords)}")
    print(f"relevance_tags(production)={_relevance_tags(retriever, payload, concept_plan)}")
    print("doc_attrs=" + json.dumps(payload.get("doc_attrs") or {}, ensure_ascii=False, sort_keys=True))
    return researcher_id


def _print_points_section(
    title: str,
    points: list[Any],
    *,
    target_chunk_id: str,
    target_researcher_id: str | None,
    keywords: list[str],
    retriever: QdrantHybridRetriever | None = None,
    concept_plan: ConceptPlan | None = None,
    runtime: Runtime | None = None,
    query_sparse_map: dict[int, float] | None = None,
    sparse_overlap_limit: int = 5,
    source_ranks: dict[str, dict[str, int]] | None = None,
) -> None:
    print(f"\n[{title}]")
    target_hit = any((_point_payload(point).get("chunk_id") or "") == target_chunk_id for point in points)
    target_researcher_hit = any(
        target_researcher_id
        and str(_point_payload(point).get("researcher_id") or "") == target_researcher_id
        for point in points
    )
    print(f"count={len(points)} target_chunk_hit={target_hit} target_researcher_hit={target_researcher_hit}")
    for rank, point in enumerate(points, start=1):
        print(
            _format_point(
                rank=rank,
                point=point,
                target_chunk_id=target_chunk_id,
                target_researcher_id=target_researcher_id,
                keywords=keywords,
                retriever=retriever,
                concept_plan=concept_plan,
                runtime=runtime,
                query_sparse_map=query_sparse_map,
                sparse_overlap_limit=sparse_overlap_limit,
                source_ranks=source_ranks,
            )
        )


def _print_grouped_section(
    groups: list[Any],
    *,
    target_chunk_id: str,
    target_researcher_id: str | None,
    keywords: list[str],
    retriever: QdrantHybridRetriever | None = None,
    concept_plan: ConceptPlan | None = None,
    source_ranks: dict[str, dict[str, int]] | None = None,
) -> None:
    print("\n[Grouped Hybrid RRF Search]")
    target_chunk_group_rank: int | None = None
    target_researcher_group_rank: int | None = None
    for index, group in enumerate(groups, start=1):
        hits = _group_hits(group)
        group_payloads = [_point_payload(point) for point in hits]
        if any((payload.get("chunk_id") or "") == target_chunk_id for payload in group_payloads):
            target_chunk_group_rank = target_chunk_group_rank or index
        if target_researcher_id and any(
            str(payload.get("researcher_id") or "") == target_researcher_id
            for payload in group_payloads
        ):
            target_researcher_group_rank = target_researcher_group_rank or index

    print(
        "group_count="
        f"{len(groups)} target_chunk_group_rank={target_chunk_group_rank} "
        f"target_researcher_group_rank={target_researcher_group_rank}"
    )
    for group_rank, group in enumerate(groups, start=1):
        hits = _group_hits(group)
        if not hits:
            continue
        first_payload = _point_payload(hits[0])
        group_id = _group_id(group)
        group_flags: list[str] = []
        if target_researcher_id and any(
            str(_point_payload(point).get("researcher_id") or "") == target_researcher_id
            for point in hits
        ):
            group_flags.append("TARGET_RESEARCHER_GROUP")
        if any((_point_payload(point).get("chunk_id") or "") == target_chunk_id for point in hits):
            group_flags.append("TARGET_CHUNK_GROUP")
        flag_text = f" flags={','.join(group_flags)}" if group_flags else ""
        print(
            f"\nGroup {group_rank}: id={group_id} researcher_id={first_payload.get('researcher_id')!r} "
            f"name={first_payload.get('researcher_name')!r} hits={len(hits)}{flag_text}"
        )
        for hit_rank, point in enumerate(hits, start=1):
            print(
                "  "
                + _format_point(
                    rank=hit_rank,
                    point=point,
                    target_chunk_id=target_chunk_id,
                    target_researcher_id=target_researcher_id,
                    keywords=keywords,
                    retriever=retriever,
                    concept_plan=concept_plan,
                    source_ranks=source_ranks,
                )
            )


def _print_post_group_gate_section(
    groups: list[Any],
    *,
    retriever: QdrantHybridRetriever,
    concept_plan: ConceptPlan,
    target_researcher_id: str | None,
) -> None:
    print("\n[Post-Group Relevance Gate]")
    print(f"active_concepts={sorted(concept_plan.required)}")
    if not concept_plan.required:
        print("gate_active=False")
        return

    survivor_count = 0
    filtered_count = 0
    kept_chunk_count = 0
    dropped_chunk_count = 0
    target_after_gate = False
    target_gate_record: dict[str, Any] | None = None
    filtered_examples: list[dict[str, Any]] = []

    for group_rank, group in enumerate(groups, start=1):
        aggregation = retriever._candidate_from_group(group, concept_plan=concept_plan)
        kept_chunk_count += aggregation.kept_chunks
        dropped_chunk_count += aggregation.dropped_chunks
        if aggregation.candidate is not None:
            survivor_count += 1
            if target_researcher_id and aggregation.candidate.researcher_id == target_researcher_id:
                target_after_gate = True
                target_gate_record = {
                    "group_rank": group_rank,
                    "status": "survived",
                    "matched_concepts": sorted(aggregation.matched_concepts),
                    "kept_chunks": aggregation.kept_chunks,
                    "dropped_chunks": aggregation.dropped_chunks,
                }
            continue

        if aggregation.drop_reason == "relevance_concepts_missing":
            filtered_count += 1
            record = {
                "group_rank": group_rank,
                "researcher_id": aggregation.researcher_id,
                "name": aggregation.researcher_name,
                "reason": aggregation.drop_reason,
                "matched_concepts": sorted(aggregation.matched_concepts),
                "missing_concepts": sorted(aggregation.missing_concepts),
                "kept_chunks": aggregation.kept_chunks,
                "dropped_chunks": aggregation.dropped_chunks,
            }
            if len(filtered_examples) < 10:
                filtered_examples.append(record)
            if target_researcher_id and aggregation.researcher_id == target_researcher_id:
                target_gate_record = record

    print(
        "survivor_count="
        f"{survivor_count} filtered_count={filtered_count} kept_chunks={kept_chunk_count} "
        f"dropped_chunks={dropped_chunk_count} target_researcher_after_gate={target_after_gate}"
    )
    if target_gate_record is not None:
        print("target_researcher_gate=" + json.dumps(target_gate_record, ensure_ascii=False, sort_keys=True))
    if filtered_examples:
        print("filtered_examples=" + json.dumps(filtered_examples, ensure_ascii=False, sort_keys=True))


def _print_concept_evidence_plan(concept_plan: ConceptPlan) -> None:
    """동적 Concept Evidence Plan 상세(planner 산출 → registry 보강/감지). 진단용."""
    print("\n[Concept Evidence Plan]")
    print(f"source={concept_plan.source} required={concept_plan.required} optional={concept_plan.optional}")
    for spec in concept_plan.specs:
        print(
            f"  - {spec.id} ({spec.role}, src={spec.source}, conf={spec.confidence})"
            f"\n      query={spec.query_terms}"
            f"\n      evidence(strong)={spec.evidence_terms}"
            f"\n      weak={spec.weak_terms}"
        )


def _print_multiview_relevance_section(
    retriever: QdrantHybridRetriever,
    *,
    query: str,
    plan: PlannerOutput,
    concept_plan: ConceptPlan,
    query_filter: models.Filter | None,
    target_researcher_id: str | None,
    top_chunks: int,
) -> None:
    """production `search()`(멀티뷰 flat + 관련도 재점수)를 그대로 실행해 최종 랭킹을 출력.

    이 섹션이 실제 서비스 정렬과 동일하다(이하 grouped 섹션은 진단/AB 비교용 legacy).
    chunk마다 concept 신호(confirmed=evidence 확정 / weak_only / view_only)를 함께 표기해
    '왜 이 chunk가 확정 또는 약신호인지'를 추적할 수 있게 한다.
    """
    print("\n[Multiview Flat Relevance - PRODUCTION search()]")
    result = asyncio.run(retriever.search(query=query, plan=plan, query_filter=query_filter))
    qp = result.query_payload
    concept_plan_trace = qp.get("concept_plan", {})
    print(
        "concept_plan="
        + json.dumps(
            {
                "required": concept_plan_trace.get("required"),
                "optional": concept_plan_trace.get("optional"),
                "source": concept_plan_trace.get("source"),
            },
            ensure_ascii=False,
        )
    )
    print(f"view_sources={concept_plan_trace.get('view_sources')}")
    print(f"view_counts={qp.get('view_counts')}")
    print(
        f"gate_enabled={qp.get('relevance_gate_enabled')} "
        f"active_concepts={qp.get('relevance_gate_active_concepts')}"
    )
    print(
        f"merged_chunks={qp.get('merged_chunk_count')} main_count={qp.get('main_count')} "
        f"fallback_count={qp.get('fallback_count')} org_filtered={qp.get('org_filtered_count')} "
        f"final={qp.get('final_hit_count')}"
    )
    main_count = int(qp.get("main_count") or 0)
    target_rank: int | None = None
    for rank, hit in enumerate(result.hits, start=1):
        tier = "MAIN" if rank <= main_count else "FALLBACK"
        is_target = bool(target_researcher_id) and hit.researcher_id == target_researcher_id
        if is_target:
            target_rank = rank
        flags = " flags=TARGET_RESEARCHER" if is_target else ""
        breakdown = {key: round(value, 6) for key, value in hit.score_breakdown.items()}
        print(
            f"\n{rank:>3}. [{tier}] researcher_id={hit.researcher_id} name={hit.researcher_name!r} "
            f"score={hit.group_score:.6f} coverage={hit.coverage_type!r} "
            f"matched_concepts={hit.matched_concepts}{flags}"
        )
        print(f"     breakdown={json.dumps(breakdown, ensure_ascii=False, sort_keys=True)}")
        for chunk in hit.chunks[: max(0, top_chunks)]:
            title = _title(chunk.payload.model_dump())
            view_hits = {
                src[len(CONCEPT_VIEW_PREFIX):]
                for src in chunk.sources
                if src.startswith(CONCEPT_VIEW_PREFIX)
            }
            sig = chunk_concept_signals(chunk.payload, concept_plan, view_concept_hits=view_hits)
            signals = {
                "confirmed": sorted(sig["confirmed"]),
                "weak_only": sorted(sig["weak_only"]),
                "view_only": sorted(sig["view_only"]),
            }
            print(
                f"       - fused={chunk.score:.6f} doc_type={chunk.doc_type} "
                f"chunk_id={chunk.chunk_id} concepts={chunk.concepts} sources={chunk.sources} "
                f"signals={signals} title={title!r}"
            )
    print(f"\ntarget_researcher_rank={target_rank}")
    if result.filtered_out_candidates:
        print(
            "filtered_out="
            + json.dumps(result.filtered_out_candidates, ensure_ascii=False, sort_keys=True)
        )


def _diagnose(args: argparse.Namespace) -> int:
    settings = Settings(
        qdrant_collection_name=args.collection,
        strict_runtime_validation=False,
    ) if args.collection else Settings(strict_runtime_validation=False)
    runtime = (
        _build_offline_runtime(settings)
        if args.offline
        else _build_runtime(settings, timeout=args.timeout)
    )
    collection = runtime.settings.qdrant_collection_name
    keywords = args.keywords or args.query.split()
    query_builder = QueryTextBuilder()
    planner_for_diagnostics = PlannerOutput(
        intent_summary=args.query,
        retrieval_core=keywords,
        core_keywords=keywords,
    )
    # 기본 동작: online이면 실제 planner LLM(vLLM/OpenAI 호환)을 호출해 concept_specs를 생성한다(라이브
    # source=planner 경로와 동일 진단). 실패하거나 --offline이면 위 키워드 기반 PlannerOutput으로 폴백.
    if not args.offline:
        from apps.recommendation.planner import PLANNER_VERSION, OpenAICompatPlanner

        # planner 모듈 내부 로그(플래너 내부 시작/LLM 시도 완료/내부 완료 등)를 stdout로 surface.
        planner_logger = logging.getLogger("apps.recommendation.planner")
        if not planner_logger.handlers:
            _planner_handler = logging.StreamHandler(sys.stdout)
            _planner_handler.setFormatter(logging.Formatter("    [LOG %(levelname)s] %(name)s: %(message)s"))
            planner_logger.addHandler(_planner_handler)
            planner_logger.setLevel(logging.INFO)
            planner_logger.propagate = False

        print("\n[Planner LLM]")
        print(f"backend=openai_compat(vLLM) model={runtime.settings.llm_model_name!r} planner_version={PLANNER_VERSION}")
        print(f"base_url={runtime.settings.llm_base_url}")
        print(f"query={args.query!r}")
        live_planner = OpenAICompatPlanner(runtime.settings)
        _planner_t0 = time.perf_counter()
        try:
            planner_for_diagnostics = asyncio.run(live_planner.plan(query=args.query))
            _planner_ms = (time.perf_counter() - _planner_t0) * 1000.0
            planner_trace = getattr(live_planner, "last_trace", {}) or {}
            print(
                f"status=ok elapsed_ms={_planner_ms:.1f} mode={planner_trace.get('mode')} "
                f"retry={planner_trace.get('planner_retry_count')}"
            )
            print(f"retrieval_core={planner_for_diagnostics.retrieval_core}")
            print(f"semantic_query={planner_for_diagnostics.semantic_query!r}")
            print(f"concept_specs={len(planner_for_diagnostics.concept_specs)}개")
            for spec in planner_for_diagnostics.concept_specs:
                print(
                    f"  - {spec.id} ({spec.role}, src={spec.source}) "
                    f"query={spec.query_terms} evidence={spec.evidence_terms} weak={spec.weak_terms}"
                )
            for idx, attempt in enumerate(planner_trace.get("attempts", []) or []):
                print(f"  attempt[{idx}] status={attempt.get('status')} reason={attempt.get('reason')}")
        except Exception as exc:  # noqa: BLE001
            _planner_ms = (time.perf_counter() - _planner_t0) * 1000.0
            print(f"status=error elapsed_ms={_planner_ms:.1f} error={exc!r}")
            print("→ 키워드 기반 fallback PlannerOutput 사용")
            if args.debug:
                traceback.print_exc()
    # --plan-json: 실제/임의 planner 산출(concept_specs 포함)을 주입해 source=planner 경로 진단(LLM보다 우선).
    if args.plan_json:
        overrides = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
        merged = planner_for_diagnostics.model_dump()
        merged.update(overrides)
        planner_for_diagnostics = PlannerOutput.model_validate(merged)
    diagnostic_retriever = QdrantHybridRetriever(
        client=runtime.client,
        settings=runtime.settings,
        dense_encoder=runtime.dense_encoder,
        query_builder=query_builder,
        sparse_encoder=runtime.sparse_encoder,
        sparse_runtime=runtime.sparse_runtime,
    )
    # production search()와 동일: 동적 Concept Evidence Plan 먼저 확정 → build_search_query_plan에 주입.
    production_concept_plan = diagnostic_retriever._resolve_concept_plan(
        args.query, planner_for_diagnostics
    )
    search_query_plan = query_builder.build_search_query_plan(
        args.query, planner_for_diagnostics, production_concept_plan
    )
    sparse_text = args.sparse_query.strip() if args.sparse_query else search_query_plan.sparse_joint_query
    if args.sparse_query:
        search_query_plan.sparse_joint_query = sparse_text
    # production retriever와 동일한 임베딩 구성을 재사용(raw dense/hybrid 기준선의 충실도 보장).
    dense_query_text = diagnostic_retriever._dense_prompt(search_query_plan.dense_query)
    dense_query = diagnostic_retriever._build_dense_query(search_query_plan.dense_query)
    sparse_query = diagnostic_retriever._build_sparse_query(sparse_text)
    query_sparse_map = _sparse_vector_items(sparse_query)
    print("[Runtime]")
    print(f"offline={args.offline}")
    print(f"collection={collection}")
    print(f"qdrant_url={runtime.settings.qdrant_url}")
    print(f"dense_model={getattr(runtime.dense_encoder, 'model_name', '-')}")
    print(
        f"sparse_backend={runtime.sparse_runtime.backend} "
        f"sparse_model={runtime.sparse_runtime.active_model_name} "
        f"fallback={runtime.sparse_runtime.used_fallback}"
    )
    print(f"query={args.query!r}")
    print(f"dense_query={search_query_plan.dense_query!r}")
    print(f"sparse_joint_query={sparse_text!r}")
    print(f"sparse_concept_queries={search_query_plan.sparse_concept_queries}")
    print(f"keywords={keywords}")
    print(f"optional_concepts={search_query_plan.optional_concepts}")
    # concept·검색문은 동적 Concept Evidence Plan(planner 산출 → registry 보강/감지) 단일 출처.
    print(
        f"production_required_concepts={production_concept_plan.required} "
        f"(source={production_concept_plan.source}) optional={production_concept_plan.optional}"
    )
    print(f"grouped_gate_active_concepts={sorted(production_concept_plan.required)}")

    _print_dense_query_section(
        runtime=runtime,
        raw_query=args.query,
        dense_query_text=dense_query_text,
        dense_vector=dense_query,
    )
    if args.show_sparse_tokens:
        _print_sparse_query_section(
            runtime=runtime,
            sparse_query_text=sparse_text,
            sparse_query=sparse_query,
            query_sparse_map=query_sparse_map,
            token_limit=args.sparse_token_limit,
        )

    target_point = _find_target_point(
        runtime.client,
        collection_name=collection,
        target_chunk_id=args.target_chunk_id,
    )
    target_researcher_id = _print_target(
        target_point,
        target_chunk_id=args.target_chunk_id,
        keywords=keywords,
        retriever=diagnostic_retriever,
        concept_plan=production_concept_plan,
    )

    query_filter = _doc_type_filter(runtime.settings)

    # 동적 Concept Evidence Plan 상세(planner 산출 → registry 보강/감지).
    _print_concept_evidence_plan(production_concept_plan)

    # PRODUCTION 경로(실제 서비스 정렬과 동일). 이하 raw/grouped 섹션은 심층 진단/AB 비교용.
    _print_multiview_relevance_section(
        diagnostic_retriever,
        query=args.query,
        plan=planner_for_diagnostics,
        concept_plan=production_concept_plan,
        query_filter=query_filter,
        target_researcher_id=target_researcher_id,
        top_chunks=args.top_chunks,
    )

    sparse_response = runtime.client.query_points(
        collection_name=collection,
        query=sparse_query,
        using=SPARSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=args.limit,
        with_payload=True,
        with_vectors=[SPARSE_VECTOR_NAME],
    )
    sparse_points = _points_from_query_response(sparse_response)
    _print_points_section(
        _sparse_section_title(runtime),
        sparse_points,
        target_chunk_id=args.target_chunk_id,
        target_researcher_id=target_researcher_id,
        keywords=keywords,
        retriever=diagnostic_retriever,
        concept_plan=production_concept_plan,
        runtime=runtime,
        query_sparse_map=query_sparse_map,
        sparse_overlap_limit=args.sparse_overlap_limit,
    )

    dense_response = runtime.client.query_points(
        collection_name=collection,
        query=dense_query,
        using=DENSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
    )
    dense_points = _points_from_query_response(dense_response)
    _print_points_section(
        "Dense Similarity Search",
        dense_points,
        target_chunk_id=args.target_chunk_id,
        target_researcher_id=target_researcher_id,
        keywords=keywords,
        retriever=diagnostic_retriever,
        concept_plan=production_concept_plan,
    )
    source_ranks = {
        "dense": _rank_by_chunk(dense_points),
        "sparse": _rank_by_chunk(sparse_points),
    }

    sparse_prefetches = [
        models.Prefetch(
            query=diagnostic_retriever._build_sparse_query(sparse_query_text),
            using=SPARSE_VECTOR_NAME,
            limit=args.prefetch_limit,
            filter=query_filter,
        )
        for sparse_query_text in search_query_plan.sparse_queries().values()
    ]
    prefetch = [
        models.Prefetch(
            query=dense_query,
            using=DENSE_VECTOR_NAME,
            limit=args.prefetch_limit,
            filter=query_filter,
        ),
        *sparse_prefetches,
    ]
    hybrid_response = runtime.client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
    )
    hybrid_points = _points_from_query_response(hybrid_response)
    _print_points_section(
        "Hybrid RRF Chunk Search",
        hybrid_points,
        target_chunk_id=args.target_chunk_id,
        target_researcher_id=target_researcher_id,
        keywords=keywords,
        retriever=diagnostic_retriever,
        concept_plan=production_concept_plan,
        source_ranks=source_ranks,
    )

    print("\n[Legacy Grouped Path - diagnostic/AB only; NOT the production ranking]")
    grouped_response = runtime.client.query_points_groups(
        collection_name=collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        group_by="researcher_id",
        group_size=args.group_size,
        limit=args.group_limit,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )
    _print_grouped_section(
        _groups_from_response(grouped_response),
        target_chunk_id=args.target_chunk_id,
        target_researcher_id=target_researcher_id,
        keywords=keywords,
        retriever=diagnostic_retriever,
        concept_plan=production_concept_plan,
        source_ranks=source_ranks,
    )
    _print_post_group_gate_section(
        _groups_from_response(grouped_response),
        retriever=diagnostic_retriever,
        concept_plan=production_concept_plan,
        target_researcher_id=target_researcher_id,
    )

    print("\n[Interpretation]")
    print("- [Multiview Flat Relevance] is the PRODUCTION ranking (search()): per-view flat 회수 →")
    print("  payload.chunk_id 병합 → normalized rank 융합 → concept 태깅 → capped evidence 재점수.")
    print("- coverage=joint/separate는 required 전부 충족(main tier), partial은 fallback tier로 강등.")
    print("- breakdown(joint/balance/concept/support)로 점수 기여를 분해; balance는 한쪽 몰빵을 억제.")
    print("- raw sparse/dense/hybrid 섹션과 Legacy Grouped 섹션은 심층 진단/AB 비교용(최종 정렬 아님).")
    print("- surface_query_term_hits is only a payload substring check; it is not SPLADE scoring evidence.")
    print("- sparse_overlap is query-vector/document-vector token overlap when Qdrant returns document sparse vectors.")
    print("- payload.chunk_id is the authoritative evidence id; point_id may differ in this collection.")
    print("- --offline은 in-memory 코퍼스 + fake client로 VPN 없이 전체 경로를 약식 검증한다.")
    return 0


def diagnose(args: argparse.Namespace) -> int:
    snapshot = _apply_diagnostic_llm_policy()
    try:
        return _diagnose(args)
    finally:
        _restore_llm_policy(snapshot)


def main(argv: list[str] | None = None) -> int:
    # 한글/유니코드 출력이 콘솔 기본 인코딩(cp949 등)에서 깨지지 않도록 UTF-8로 재설정.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Diagnose live Qdrant retrieval paths.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--sparse-query", default=None, help="override the text sent to SPLADE/BM25")
    parser.add_argument("--target-chunk-id", default=DEFAULT_TARGET_CHUNK_ID)
    parser.add_argument("--keywords", nargs="*", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="VPN/Qdrant 없이 in-memory 코퍼스+fake client로 약식 실행(스모크).",
    )
    parser.add_argument(
        "--plan-json",
        default=None,
        help="planner 산출 JSON(concept_specs 포함 PlannerOutput 일부) 주입 → source=planner 경로 진단.",
    )
    parser.add_argument(
        "--top-chunks", type=int, default=5,
        help="멀티뷰 production 섹션에서 후보별 표시할 evidence chunk 수.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--prefetch-limit", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--group-limit", type=int, default=80)
    parser.add_argument("--show-sparse-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sparse-token-limit", type=int, default=30)
    parser.add_argument("--sparse-overlap-limit", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--debug", action="store_true", help="print full traceback on errors")
    args = parser.parse_args(argv)
    if args.collection:
        args.collection = args.collection.strip()
    if args.keywords is not None:
        args.keywords = [item for item in args.keywords if item.strip()]
    # 오프라인 기본 타깃은 in-memory 코퍼스의 joint chunk(이상헌)로 교체.
    if args.offline and args.target_chunk_id == DEFAULT_TARGET_CHUNK_ID:
        args.target_chunk_id = "paper_100000000001_c000"
    try:
        return diagnose(args)
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic should summarize failures.
        print("\n[Error]")
        print(f"type={type(exc).__name__}")
        print(f"reason={exc}")
        print("hint=Check VPN/Qdrant reachability, collection name, and --timeout.")
        if args.debug:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
