# 마이그레이션 계획 — v1.x → v2.0 (chunk 재설계)

**작성일:** 2026-05-28
**범위:** 데이터 적재, Qdrant 컬렉션, `apps/` 코드, 외부 계약
**제약:** 외부 의존 서버(LLM/Qdrant/Embedding)는 VPN 환경에서만 호출 가능 — 통합 검증은 VPN 필요 구간 별도 표시.
**선행 결정:** [`../architecture/ADR/0002-chunk-level-point-model.md`](../architecture/ADR/0002-chunk-level-point-model.md) 외 ADR 0003~0005. 데이터 모델: [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md).

> ⚠️ **2026-06-02 flat 정렬로 일부 supersede:** 이 로드맵은 v2.0 **설계 시점(2026-05-28)** 의 계획이다. 실제 적재 데이터가 **flat chunk payload**(연구자 메타 root 평탄화, `doc_attrs`, 단일 `doc_date`, doc_type 5종)로 확정되면서 아래 세부의 일부 가정이 바뀌었다. **현행 정설은 [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md)(v2.1) 와 메모리 flat-payload-contract**이며, 본 문서는 전환 이력으로만 참조한다. 특히 다음은 더 이상 유효하지 않다:
> - **Phase A(적재)는 외부 제공자 소관**이 되었다 — 구 `apps/ingest`는 `legacy_v1x/ingest/`로 격리됨(본 레포 책임 아님).
> - 벡터명은 `dense_e5i`/`sparse_splade`가 아니라 **`vector_e5i`/`vector_splade`**.
> - payload는 3층(`researcher_meta`/`domain_attrs`)이 아니라 **flat**(root 메타 + `doc_attrs`).
> - recency 키는 `event_year`/`event_date`가 아니라 단일 **`doc_date`**(문자열, `"NONE"` 가능).
> - doc_type은 11종이 아니라 **5종**, 연구자 count는 8개가 아니라 **5개**(assessor count 단일 병합).

> 이 문서는 **설계·전환 로드맵**이다. 본 작업 세트(2026-05-28)의 산출물은 문서 재작성까지이며, 아래 코드/적재 변경은 후속 단계다.

---

## 0. 전환 원칙

- **신규 컬렉션 병행(blue/green):** 기존 `researcher_recommend_proto`를 건드리지 않고 신규 `ntis_researcher_chunks`를 따로 만들어 검증 후 컷오버. 롤백은 컬렉션 이름 환경변수 되돌리기.
- **단계 분리:** "검색/집계 변경"과 "사유 생성 변경"과 "계약/문서 변경"을 섞지 않는다(v1.x 교훈).
- 둘 이상의 단계(planner/retrieval/reasoner/evidence_selector)가 같이 바뀌면 [`../architecture/SERVICE_FLOW.md`](../architecture/SERVICE_FLOW.md)와 [`../api/DATA_CONTRACT.md`](../api/DATA_CONTRACT.md)를 같은 커밋에서 갱신.

---

## 1. Phase A — 데이터 적재 (ingestion) — ⚠️ 외부 제공자 소관 (격리됨)

> **2026-06-02 변경:** 적재는 **외부 제공자가 flat chunk payload로 직접 적재**한다. 본 레포의 구 적재 코드(`apps/ingest`)는 `legacy_v1x/ingest/`로 격리됐고, 아래 표는 **flat 계약 기준의 적재 산출물 명세**(외부 제공자 책임)로만 남긴다.

| 작업 | 내용 | VPN |
|---|---|---|
| A-1 | 원천 → flat chunk payload 변환: doc_type별 `chunk_text` 직렬화(요청 어투 배제), 단일 `doc_date`(문자열, 결측은 `"NONE"`) | 불필요 |
| A-2 | 연구자 공통 메타 root 평탄화: 연구자별 5개 count(`publication_count`/`scie_publication_count`/`intellectual_property_count`/`research_project_count`/`researcher_assessor_activity_count`) + 식별 메타를 그 연구자의 모든 chunk root에 동일 주입 | 불필요 |
| A-3 | dense(`vector_e5i`)/sparse(`vector_splade`) 임베딩 생성: 입력은 `chunk_text` 단일 | **필요**(임베딩 서버) |
| A-4 | 적재 불변식 검증([`DATA_MODEL.md §6`](../architecture/DATA_MODEL.md)) 자동 점검 스크립트 | 불필요(로컬 표본) |

산출물(외부): flat chunk payload. 원천 규격은 [`../전체 샘플 payload (도메인별).txt`](../전체%20샘플%20payload%20(도메인별).txt).

## 2. Phase B — Qdrant 컬렉션

| 작업 | 내용 | VPN |
|---|---|---|
| B-1 | `ntis_researcher_chunks` 생성: vectors=`vector_e5i`(1024,Cosine)+`vector_splade`, Point ID=`chunk_id` | **필요** |
| B-2 | payload 인덱스 생성([`DATA_MODEL.md §5`](../architecture/DATA_MODEL.md)) | **필요** |
| B-3 | sparse modifier 정합(SPLADE=none / bm25 fallback=IDF) 부트스트랩 확인 | **필요** |
| B-4 | 표본 upsert → `query_points` 스모크(검색·집계 동작) | **필요** |

## 3. Phase C — 코드 변경 로드맵 (`apps/`)

> 후속 작업. 변경 지점만 명시한다(구현은 별도).

| 모듈 | 변경 | 비고 |
|---|---|---|
| `apps/core/config.py` | 컬렉션 기본값, doc_type/집계/evidence/후보리랭커 설정 추가, branch 한정 설정 제거 | [`../operation/ENVIRONMENT.md`](../operation/ENVIRONMENT.md) |
| `apps/search/schema_registry.py` | `DENSE/SPARSE_VECTOR_BY_BRANCH` → 단일 벡터(`vector_e5i`/`vector_splade`) + `doc_type` 필터 정의, family 매핑, flat 인덱스 필드 | 핵심 |
| `apps/search/retriever.py` | 2단계 검색을 doc_type 경로로, **연구자 집계(RRF 누적+chunk_cap+prior+dedupe)** 신설, 정렬 유지 | 핵심 |
| `apps/search/filters.py` | hard filter를 flat root 메타/`doc_date` 기준으로, recency OR(min_should) 유지 | recency 0건 회귀 방지 |
| `apps/recommendation/planner.py` | `hard_filters` 허용 키 재정의(`recent_years`/`*_count_min` 등), intent flag(prior 힌트) | [`../api/DATA_CONTRACT.md`](../api/DATA_CONTRACT.md) |
| `apps/recommendation/evidence_selector.py` | chunk-native: 후보 매칭 chunk을 family별 재랭크, `CrossEncoderEvidenceSelector` 기본 + lexical 강등 | |
| `apps/recommendation/reasoner.py` | `selected_evidence_ids`=chunk_id, 입력 풀을 chunk 기반으로 | [`../api/REASONER_RUNTIME_POLICY.md`](../api/REASONER_RUNTIME_POLICY.md) |
| `apps/recommendation/cards.py` | 후보 카드를 flat root 메타/`doc_type_coverage`/`matched_doc_types`로 | |
| `apps/api/schemas.py` | `searched_doc_types`, `doc_type_coverage`, `evidence[*].chunk_id`, `counts` 명칭 | BREAKING |
| `apps/search/live_validator.py` / `apps/tools/validate_live.py` | 신규 벡터/인덱스/샘플 구조 점검 | readiness |

## 4. Phase D — 컷오버 & 검증

1. 스테이징에서 `NTIS_QDRANT_COLLECTION_NAME=ntis_researcher_chunks`로 `/health/ready` 그린 확인.
2. [`../operation/GOLDEN_TESTS.md`](../operation/GOLDEN_TESTS.md) 시나리오(특히 7~16, chunk/집계/OR recency/chunk_id evidence) 통과.
3. 동일 질의 셋으로 v1.x 대비 후보·근거 비교(품질 회귀 점검).
4. 프로덕션 환경변수 전환 → 컷오버. 문제 시 컬렉션명 롤백.

## 5. 롤백

- 신규 컬렉션은 구 컬렉션과 독립 → 환경변수만 되돌리면 즉시 v1.x 동작 복귀(데이터 손실 없음).
- 외부 소비자에는 v2.0 필드 변경(BREAKING)을 사전 공지([`../api/EXTERNAL_API_CHANGELOG.md`](../api/EXTERNAL_API_CHANGELOG.md)).
