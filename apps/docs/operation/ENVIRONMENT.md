# 환경 변수 (Environment Variables) — flat chunk 모델

**문서 버전:** v2.1 (flat payload 정렬, 2026-06-02)

모든 환경 변수는 `NTIS_` 접두사를 사용한다. 데이터 모델은 [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md) 참조.

## 핵심 설정 (Core)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_APP_NAME` | `NTIS Evaluator Recommendation API` | FastAPI 앱 제목 |
| `NTIS_APP_ENV` | `prod` | 런타임 환경 |
| `NTIS_APP_HOST` | `0.0.0.0` | 바인딩 주소 |
| `NTIS_APP_PORT` | `8011` | 수신 포트 |
| `NTIS_STRICT_RUNTIME_VALIDATION` | `true` | 필수 설정 미충족 시 추천 서비스 비활성화 |
| `NTIS_RUNTIME_DIR` | `runtime` | 런타임 출력 디렉터리 |
| `NTIS_FEEDBACK_DB_PATH` | `runtime/feedback.db` | 피드백 SQLite 경로 |
| `NTIS_FEEDBACK_TABLE` | `feedback_events` | 피드백 테이블 |

## Qdrant 설정

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_QDRANT_URL` | `http://203.250.234.159:8005` | Qdrant URL |
| `NTIS_QDRANT_API_KEY` | (없음) | API 키 |
| `NTIS_QDRANT_COLLECTION_NAME` | `researcher_recommend_v1` | **chunk 컬렉션** (구 `researcher_recommend_proto` 폐기) |
| `NTIS_QDRANT_CLOUD_INFERENCE` | `false` | 클라우드 추론 활성화 |

> chunk 컬렉션은 Point 1개 = chunk 1개이며 payload root `chunk_id`가 authoritative evidence id다. Point ID는 `chunk_id` 권장이지만 운영 컬렉션이 UUID를 쓰더라도 런타임은 `payload.chunk_id`를 기준으로 evidence를 resolve한다. named vector는 `vector_e5i`(dense) + `vector_splade`(sparse) 단일 쌍이다. doc_type은 named vector가 아니라 **payload 필터**다. 적재(ingestion)는 외부 제공자 소관(구 `apps/ingest`는 `legacy_v1x/ingest/`로 격리).

## LLM 백엔드

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_LLM_BACKEND` | `openai_compat` | 플래너/리즈너 LLM 백엔드 |
| `NTIS_LLM_BASE_URL` | `http://203.250.234.159:8010/v1` | OpenAI 호환 URL |
| `NTIS_LLM_API_KEY` | `EMPTY` | API 키 |
| `NTIS_LLM_MODEL_NAME` | `/model` | 모델 이름 |
| `NTIS_USE_MAP_REDUCE_JUDGING` | `true` | 사유 생성 시 내부 배치 라운드 사용 여부(후보 순위에 영향 없음) |
| `NTIS_LLM_JUDGE_BATCH_SIZE` | `10` | 내부 병렬 심사 배치 크기 |
| `NTIS_LLM_JUDGE_MAX_CONCURRENCY` | `10` | LLM 동시 호출 상한 |

## 임베딩 백엔드 (Dense)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_EMBEDDING_BACKEND` | `local` | 임베딩 백엔드 |
| `NTIS_EMBEDDING_BASE_URL` | `http://203.250.234.159:8011/v1` | OpenAI 호환 URL |
| `NTIS_EMBEDDING_API_KEY` | `EMPTY` | API 키 |
| `NTIS_EMBEDDING_MODEL_NAME` | `<repo>/multilingual-e5-large-instruct` | 로컬 번들/원격 모델 |
| `NTIS_EMBEDDING_VECTOR_SIZE` | `1024` | Dense 벡터 크기(`vector_e5i`) |

로컬 번들 교체 시 `modules.json`, `1_Pooling/config.json`, `2_Normalize/` 구조를 유지해야 한다. chunk 모델에서 dense 입력은 `chunk_text` 단일 필드다.

## Sparse 및 오프라인 설정

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_SPARSE_MODEL_NAME` | `<repo>/models/PIXIE-Splade-v1.0` | Sparse 모델(`vector_splade`) |
| `NTIS_SPARSE_CACHE_DIR` | `<repo>/models` | 캐시 디렉터리 |
| `NTIS_SPARSE_LOCAL_FILES_ONLY` | `false` | 로컬 파일만 사용 |
| `NTIS_HF_HUB_OFFLINE` | `false` | HF Hub 오프라인 강제 |
| `NTIS_SPARSE_IDF_PATH` | `<repo>/models/PIXIE-Splade-v1.0/sparse_idf.json` | query-side SPLADE IDF 보정 사전(존재 시 자동 ON, `""`면 비활성) |
| `NTIS_SPARSE_IDF_REF` | `3.5` | 이 idf 이상이면 계수 1.0(빈출 토큰 억제 강도) |
| `NTIS_SPARSE_IDF_HARD_FLOOR` | `0.0` | idf≤이 값이면 계수 0(하드 마스크). 0=soft only |

Sparse fallback 체인: `로컬 PIXIE-Splade-v1.0 → online telepix/PIXIE-Splade-v1.0 → Qdrant/bm25`. PIXIE/SPLADE면 sparse vector modifier가 없어야 하고, `Qdrant/bm25` fallback이면 modifier가 `IDF`여야 한다. `NTIS_HF_HUB_OFFLINE=true` 또는 `NTIS_SPARSE_LOCAL_FILES_ONLY=true`면 online PIXIE를 건너뛴다. `ntis-validate-live`/`/health/ready`도 동일 resolver를 사용한다.

> **query-side IDF 보정(downweight-only):** `_build_sparse_query`가 query 토큰 가중치에 `0 if idf≤hard_floor else min(1, idf/ref)` 계수를 곱해 코퍼스 편재 토큰(SPLADE 확장 artifact)을 억제한다. 계수는 1.0을 넘지 않아(boost 금지) 희귀 토큰은 보존하고 빈출 토큰만 누른다. 쿼리 측에만 적용되며 재색인이 필요 없다.

## 검색·집계 제어 (Retrieval & Aggregation)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_RETRIEVAL_DOC_TYPES` | (전체) | 검색 대상 doc_type 화이트리스트(미설정=5종 전체: `paper`/`patent`/`project`/`assessor_activity`/`specialty`). 운영 축소용 |
| `NTIS_PREFETCH_LIMIT` | `256` | 뷰별 `query_points`가 가져오는 chunk 풀 크기(모든 검색 모드 공통) |
| `NTIS_GROUP_SIZE` | `10` | (진단 grouped 경로 전용) researcher 그룹당 회수할 최대 chunk 수 |
| `NTIS_DOC_TYPE_CHUNK_CAP` | `3` | 연구자 집계 시 doc_type별 기여 chunk 상한(다작 독식 방지) |
| `NTIS_RETRIEVAL_LIMIT` | `80` | (진단 grouped 경로 전용) 연구자 집계 후 후보 상한 |
| `NTIS_DOC_TYPE_PRIORS` | (미설정=equal) | family/doc_type 집계 가중. 예: `assessment:1.3,achievement:1.0`. **앱단 랭크 누적 가중**이며 Qdrant weighted RRF가 아님 |
| `NTIS_FINAL_RECOMMENDATION_MIN` | `1` | 최소 최종 추천 수 |
| `NTIS_FINAL_RECOMMENDATION_MAX` | `15` | 최대 최종 추천 수(사용자 노출 결과 상한) |

> `NTIS_DOC_TYPE_PRIORS`는 [`../architecture/DESIGN_GUIDELINES.md §6.2`](../architecture/DESIGN_GUIDELINES.md)의 옵트인 레버다. 기본 equal을 권장하며, intent가 명확할 때만 조정한다.
> `NTIS_RETRIEVAL_LIMIT`/`NTIS_GROUP_SIZE`는 진단용 `search_grouped_diagnostic` 경로에서만 쓰이며, production `/recommend`·`/search/candidates`의 multiview/hybrid/keyword_similarity 경로는 사용하지 않는다.

## 사용자 선택형 검색 모드 (`search_mode`)

요청 본문 `search_mode`로 검색 전략을 고른다(기본 `multiview`). 자세한 동작은 [`../architecture/DESIGN_GUIDELINES.md §6.5`](../architecture/DESIGN_GUIDELINES.md)·[`../api/DATA_CONTRACT.md §3`](../api/DATA_CONTRACT.md) 참조.

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_HYBRID_VIEW_WEIGHTS` | `{"dense_full": 1.0, "sparse_raw": 1.0}` | `hybrid` 모드의 view 가중(균등 RRF) |
| `NTIS_KEYWORD_FIRST_STAGE_LIMIT` | `512` | `keyword_similarity` 모드 1차(SPLADE) 후보 풀 상한 |

## Evidence 리랭커 (chunk 근거 선별)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_EVIDENCE_RERANKER_BACKEND` | `lexical` | `cross_encoder` / `lexical`. 모델 로드 실패 시 자동 `lexical` 강등 |
| `NTIS_CROSS_ENCODER_MODEL_NAME` | (없음) | evidence 재랭크용 cross-encoder 모델. `cross_encoder` 백엔드 사용 시 지정 필요(미설정이면 lexical 강등) |
| `NTIS_CROSS_ENCODER_BASE_URL` | (없음) | cross-encoder 서빙 URL(원격 서빙 시) |
| `NTIS_EVIDENCE_FAMILY_CAP` | `achievement:10,assessment:6,expertise:6,identity:1` | family별 LLM 입력 chunk 상한 |

> evidence 리랭커는 **후보(연구자) 순위에 영향을 주지 않는다**. 목적은 토큰 절감·grounding 품질·정규화. [`../api/REASONER_RUNTIME_POLICY.md`](../api/REASONER_RUNTIME_POLICY.md).

## 후보 리랭커 (기본 OFF, 옵트인 실험)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_CANDIDATE_RERANKER` | `off` | `off` / `band`. `band`는 score 동률 밴드 내 재배열만 허용(탈락·생성 금지) |

> 기본 추천 순위 척추는 RRF다. [`../architecture/DESIGN_GUIDELINES.md §6.3`](../architecture/DESIGN_GUIDELINES.md) 결정 참조.

## 시드 설정

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_SEED_ON_STARTUP` | `false` | 시작 시 개발용 시드 적재 |
| `NTIS_SEED_ALLOW_RECREATE_COLLECTION` | `false` | 시드 시 기존 컬렉션 삭제 후 재생성 |

## 운영 참고

- 권장 라이브 설정: `NTIS_STRICT_RUNTIME_VALIDATION=true`, `NTIS_LLM_BACKEND=openai_compat`, `NTIS_EMBEDDING_BACKEND=local`(또는 `openai`), `NTIS_CANDIDATE_RERANKER=off`, `NTIS_SEED_ON_STARTUP=false`. evidence 리랭커를 `cross_encoder`로 쓰려면 `NTIS_EVIDENCE_RERANKER_BACKEND=cross_encoder`와 함께 `NTIS_CROSS_ENCODER_MODEL_NAME`을 반드시 지정한다(미지정 시 자동 lexical 강등).
- 엄격 검증이 켜진 경우에도 플래너/리즈너는 LLM 호출 실패 시 휴리스틱·결정론적 fallback으로 동작한다.
- 트래픽 전 반드시 `GET /health/ready`로 컬렉션·벡터·인덱스 준비를 확인한다.

## v1.x 대비 폐기/변경된 환경 변수

| 구 변수 | 처리 |
|---|---|
| `NTIS_BRANCH_PREFETCH_LIMIT` / `NTIS_BRANCH_OUTPUT_LIMIT` | → `NTIS_PREFETCH_LIMIT`(뷰별 단일 limit). branch별 분리 prefetch/output limit은 폐기 |
| `NTIS_SHORTLIST_LIMIT` / `NTIS_DOC_TYPE_PREFETCH_LIMIT` / `NTIS_DOC_TYPE_OUTPUT_LIMIT` | 폐기(미지원). 현행은 `NTIS_PREFETCH_LIMIT`/`NTIS_GROUP_SIZE`/`NTIS_RETRIEVAL_LIMIT` |
| branch별 벡터명 상수(`*_vector_e5i`, `*_vector_splade`) | 단일 `vector_e5i`/`vector_splade`로 대체(코드 상수) |
| `NTIS_QDRANT_COLLECTION_NAME` 기본값 `researcher_recommend_proto` | `researcher_recommend_v1` |
