"""SPLADE sparse stopword 진단 (오프라인 분석 — 운영 코드/인덱스 미변경).

목적: '상세/detail' 같은 코퍼스 편재(stopword) 토큰을 *데이터로* 자동 식별한다(도메인 사전 X).
  1) Qdrant에 적재된 sparse 벡터를 scroll → token_id별 문서빈도(DF) 집계.
  2) --query를 SPLADE로 인코딩 → 질의가 만든 토큰 각각의 DF를 교차표시(어느 확장이 stopword인가).
  3) df_ratio >= threshold 토큰을 mask 후보로 출력(JSON + id 리스트) → query-side 마스킹/IDF에 사용.
  4) --inspect: 상위 후보가 chunk_text/doc_attrs에 리터럴로 박혔는지 확인(boilerplate vs 모델 artifact 판별).

원리: sparse 점수 = Σ q_w[t] × d_w[t]. 편재 토큰은 변별력이 없는데 q_w가 높아 무관 문서에 점수를 준다.
      query 가중치를 0으로(또는 ×IDF) 누르면 기여가 사라진다 → 재색인 불필요.

실행 예:
  python apps/tools/sparse_stopwords.py --query "전기차 배터리" --sample 50000 --threshold 0.3 --inspect
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from typing import Any

from qdrant_client import QdrantClient

from apps.core.config import Settings
from apps.search.encoders import SpladeSparseEncoder
from apps.search.schema_registry import SPARSE_VECTOR_NAME
from apps.search.sparse_runtime import (
    prepare_sparse_runtime_environment,
    resolve_sparse_runtime,
)


def _utf8_stdout() -> None:
    """한글 토큰이 cp949 콘솔에서 깨지지 않도록 stdout을 utf-8로."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _build(settings: Settings, timeout: float) -> tuple[QdrantClient, Any, Any]:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        cloud_inference=settings.qdrant_cloud_inference,
        timeout=timeout,
        check_compatibility=False,
    )
    sparse_encoder = None
    try:
        cache_dir = prepare_sparse_runtime_environment(settings)
        _, sparse_encoder = resolve_sparse_runtime(
            client=client,
            settings=settings,
            cache_dir=cache_dir,
            sparse_encoder_factory=SpladeSparseEncoder,
        )
    except Exception as exc:  # noqa: BLE001 - 진단 스크립트는 인코더 없이도 DF는 돈다.
        print(f"[warn] sparse 인코더 로드 실패({exc}); --query 교차표시 비활성, 토크나이저 폴백 시도")

    tokenizer = getattr(sparse_encoder, "_tokenizer", None)
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(settings.sparse_model_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 토크나이저 폴백 실패({exc}); token_id만 출력")
    return client, sparse_encoder, tokenizer


def _sparse_indices(point: Any) -> list[int] | None:
    """named 벡터 point에서 sparse indices 추출(dict/obj 모두 방어)."""
    v = getattr(point, "vector", None)
    sv = v.get(SPARSE_VECTOR_NAME) if isinstance(v, dict) else v
    return getattr(sv, "indices", None)


def _scan_df(client: QdrantClient, collection: str, sample: int, batch: int) -> tuple[Counter, int]:
    df: Counter = Counter()
    n = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            with_payload=False,
            with_vectors=[SPARSE_VECTOR_NAME],
            limit=batch,
            offset=offset,
        )
        for p in points:
            idx = _sparse_indices(p)
            if not idx:
                continue
            n += 1
            df.update(int(i) for i in idx)
        if offset is None or (sample and n >= sample):
            break
    return df, n


def _inspect_payload(
    client: QdrantClient, collection: str, candidates: list[dict], batch: int, sample: int = 500
) -> None:
    """상위 stopword 후보 토큰이 chunk_text/doc_attrs에 리터럴로 들어있는 비율 → boilerplate 신호."""
    texts: list[str] = []
    offset = None
    while len(texts) < sample:
        points, offset = client.scroll(
            collection_name=collection,
            with_payload=["chunk_text", "doc_attrs"],
            with_vectors=False,
            limit=batch,
            offset=offset,
        )
        for p in points:
            pl = p.payload or {}
            parts = [str(pl.get("chunk_text", ""))]
            da = pl.get("doc_attrs") or {}
            if isinstance(da, dict):
                parts.extend(str(val) for val in da.values())
            texts.append(" ".join(parts).casefold())
        if offset is None:
            break
    print(f"\n[Boilerplate check] chunk_text/doc_attrs {len(texts)}건 샘플 — 토큰 리터럴 포함률")
    print("  (포함률 높음 = 적재 텍스트에 박힌 boilerplate → 색인 정제가 근본; 낮음 = 모델 확장 artifact → query mask)")
    for c in candidates:
        tok = c["token"].lstrip("#").casefold()
        if len(tok) < 2:
            print(f"  {c['token']:<16} (subword/단문자 — 스킵)")
            continue
        hits = sum(1 for t in texts if tok in t)
        frac = hits / len(texts) if texts else 0.0
        kind = "BOILERPLATE(텍스트에 박힘)" if frac >= 0.3 else "모델확장 artifact 가능"
        print(f"  {c['token']:<16} 리터럴포함={frac:.2f}  → {kind}")


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="SPLADE sparse stopword DF 진단(오프라인 분석)")
    ap.add_argument("--query", default="전기차 배터리", help="교차표시할 질의(SPLADE 인코딩)")
    ap.add_argument("--sample", type=int, default=50000, help="스캔할 최대 문서 수(0=전체)")
    ap.add_argument("--batch", type=int, default=1000, help="scroll 배치 크기")
    ap.add_argument("--top", type=int, default=40, help="DF 상위 N 토큰 출력")
    ap.add_argument("--threshold", type=float, default=0.3, help="mask 후보 df_ratio 컷")
    ap.add_argument("--timeout", type=float, default=60.0, help="Qdrant 타임아웃(초)")
    ap.add_argument("--inspect", action="store_true", help="상위 후보의 chunk_text 리터럴 포함률(boilerplate 판별)")
    ap.add_argument("--dump-idf", default=None, help="{token_id: idf} JSON 산출 경로(query-side IDF 보정용 사전)")
    ap.add_argument("--idf-ref", type=float, default=3.5, help="downweight 계수 기준 idf(이상이면 계수 1.0)")
    ap.add_argument("--idf-hard-floor", type=float, default=0.0, help="idf<=이 값이면 계수 0(하드 마스크). 0=비활성")
    args = ap.parse_args(argv)

    settings = Settings()
    client, sparse_encoder, tokenizer = _build(settings, args.timeout)
    collection = settings.qdrant_collection_name

    def decode(tid: int) -> str:
        if tokenizer is None:
            return f"#{tid}"
        try:
            return str(tokenizer.convert_ids_to_tokens(int(tid)))
        except Exception:  # noqa: BLE001
            return f"#{tid}"

    print(f"[Sparse DF] collection={collection} url={settings.qdrant_url} sample={args.sample or 'ALL'}")
    df, n = _scan_df(client, collection, args.sample, args.batch)
    if n == 0:
        print("문서 0건 — 컬렉션/연결/VPN 확인 필요.")
        return 1
    ratio = {tid: c / n for tid, c in df.items()}
    print(f"scanned_docs={n} distinct_tokens={len(df)}\n")

    # 2) query 교차표시: 질의가 만든 토큰 각각의 DF
    if sparse_encoder is not None and args.query:
        qmap = sparse_encoder.embed(args.query)
        print(f"[Query cross-ref] query='{args.query}' tokens={len(qmap)} "
              f"(q_weight desc 상위 30; downweight-only ref={args.idf_ref} floor={args.idf_hard_floor})")
        print(f"  {'token':<16}{'id':>8}{'q_w':>8}{'df':>8}{'idf':>7}{'factor':>8}{'adj_w':>8}")
        for tid, w in sorted(qmap.items(), key=lambda x: -x[1])[:30]:
            tid_i = int(tid)
            r = ratio.get(tid_i, 0.0)
            idf = math.log((n + 1) / (df.get(tid_i, 0) + 1))
            factor = 0.0 if idf <= args.idf_hard_floor else min(1.0, idf / args.idf_ref)
            print(f"  {decode(tid):<16}{tid_i:>8}{w:>8.3f}{r:>8.3f}{idf:>7.2f}{factor:>8.2f}{w * factor:>8.3f}")
        print()

    # 3) 전역 DF 상위 = stopword 후보
    top = sorted(ratio.items(), key=lambda x: -x[1])[: args.top]
    print(f"[Top-{args.top} by DF] (코퍼스 편재 = stopword 후보)")
    print(f"  {'token':<16}{'id':>8}{'df_ratio':>10}")
    for tid, r in top:
        print(f"  {decode(tid):<16}{tid:>8}{r:>10.3f}")
    print()

    # mask 후보(JSON + 코드용 id 리스트)
    mask = [
        {"id": tid, "token": decode(tid), "df_ratio": round(r, 4)}
        for tid, r in sorted(ratio.items(), key=lambda x: -x[1])
        if r >= args.threshold
    ]
    print(f"[Mask candidates] df_ratio>={args.threshold}: {len(mask)}개")
    print(json.dumps(mask, ensure_ascii=False, indent=2))
    print("\n# 코드/설정용 token_id 리스트:")
    print(sorted(m["id"] for m in mask))

    # idf 사전 산출(설정+검색이 함께 쓰는 생성물 — 손으로 안 씀). idf = ln((N+1)/(df+1)).
    if args.dump_idf:
        default_idf = math.log((n + 1) / 1)  # 코퍼스 미등장 토큰 기본값(희귀=높은 idf; 점수 기여는 0)
        idf = {str(tid): round(math.log((n + 1) / (c + 1)), 6) for tid, c in df.items()}
        payload = {"n_docs": n, "default_idf": round(default_idf, 6), "idf": idf}
        with open(args.dump_idf, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False)
        print(f"\n[dump-idf] {len(idf)} tokens -> {args.dump_idf} (default_idf={default_idf:.3f})")

    # 4) boilerplate 판별(옵션)
    if args.inspect and mask:
        _inspect_payload(client, collection, mask[:10], args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
