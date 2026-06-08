# 환경 변수 (Environment Variables) — flat chunk 모델

**문서 버전:** v2.1 (flat payload 정렬, 2026-06-02)

모든 환경 변수는 `NTIS_` 접두사를 사용한다. 데이터 모델은 [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md) 참조.

## 핵심 설정 (Core)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_APP_NAME` | `NTIS 전문가 추천 API` | FastAPI 앱 제목 |
| `NTIS_APP_ENV` | `prod` | 런타임 환경 |
| `NTIS_APP_HOST` | `0.0.0.0` | 바인딩 주소 |
| `NTIS_APP_PORT` | `8011` | 수신 포트 |
| `NTIS_API_PREFIX` | `""` | API 접두사 |
| `NTIS_STRICT_RUNTIME_VALIDATION` | `true` | 필수 설정 미충족 시 추천 서비스 비활성화 |
| `NTIS_RUNTIME_DIR` | `runtime` | 런타임 출력 디렉터리 |
| `NTIS_FEEDBACK_DB_PATH` | `runtime/feedback.db` | 피드백 SQLite 경로 |
| `NTIS_FEEDBACK_TABLE` | `feedback_events` | 피드백 테이블 |

## Qdrant 설정

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_QDRANT_URL` | `http://203.250.234.159:8005` | Qdrant URL |
| `NTIS_QDRANT_API_KEY` | (없음) | API 키 |
| `NTIS_QDRANT_COLLECTION_NAME` | `ntis_researcher_chunks` | **chunk 컬렉션** (구 `researcher_recommend_proto` 폐기) |
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

Sparse fallback 체인: `로컬 PIXIE-Splade-v1.0 → online telepix/PIXIE-Splade-v1.0 → Qdrant/bm25`. PIXIE/SPLADE면 sparse vector modifier가 없어야 하고, `Qdrant/bm25` fallback이면 modifier가 `IDF`여야 한다. `NTIS_HF_HUB_OFFLINE=true` 또는 `NTIS_SPARSE_LOCAL_FILES_ONLY=true`면 online PIXIE를 건너뛴다. `ntis-validate-live`/`/health/ready`도 동일 resolver를 사용한다.

## 검색·집계 제어 (Retrieval & Aggregation)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_RETRIEVAL_DOC_TYPES` | (전체) | 검색 대상 doc_type 화이트리스트(미설정=5종 전체: `paper`/`patent`/`project`/`assessor_activity`/`specialty`). 운영 축소용 |
| `NTIS_DOC_TYPE_PREFETCH_LIMIT` | `100` | doc_type 경로별 prefetch(1차/2차) 제한 |
| `NTIS_DOC_TYPE_OUTPUT_LIMIT` | `50` | doc_type 경로별 융합 출력 제한 |
| `NTIS_DOC_TYPE_CHUNK_CAP` | `3` | 연구자 집계 시 doc_type별 기여 chunk 상한(다작 독식 방지) |
| `NTIS_RETRIEVAL_LIMIT` | `80` | 연구자 집계 후 후보 상한 |
| `NTIS_SHORTLIST_LIMIT` | `40` | 사유 생성 단계로 넘길 숏리스트 크기 |
| `NTIS_DOC_TYPE_PRIORS` | (미설정=equal) | family/doc_type 집계 가중. 예: `assessment:1.3,achievement:1.0`. **앱단 랭크 누적 가중**이며 Qdrant weighted RRF가 아님 |
| `NTIS_FINAL_RECOMMENDATION_MIN` | `1` | 최소 최종 추천 수 |
| `NTIS_FINAL_RECOMMENDATION_MAX` | `15` | 최대 최종 추천 수 |

> `NTIS_DOC_TYPE_PRIORS`는 [`../architecture/DESIGN_GUIDELINES.md §6.2`](../architecture/DESIGN_GUIDELINES.md)의 옵트인 레버다. 기본 equal을 권장하며, intent가 명확할 때만 조정한다.

## Evidence 리랭커 (chunk 근거 선별)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_EVIDENCE_RERANKER_BACKEND` | `cross_encoder` | `cross_encoder` / `lexical`. 모델 로드 실패 시 자동 `lexical` 강등 |
| `NTIS_CROSS_ENCODER_MODEL` | `Dongjin-kr/ko-reranker` | evidence 재랭크용 cross-encoder(로컬 번들 권장) |
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

- 권장 라이브 설정: `NTIS_STRICT_RUNTIME_VALIDATION=true`, `NTIS_LLM_BACKEND=openai_compat`, `NTIS_EMBEDDING_BACKEND=local`(또는 `openai`), `NTIS_EVIDENCE_RERANKER_BACKEND=cross_encoder`, `NTIS_CANDIDATE_RERANKER=off`, `NTIS_SEED_ON_STARTUP=false`.
- 엄격 검증이 켜진 경우에도 플래너/리즈너는 LLM 호출 실패 시 휴리스틱·결정론적 fallback으로 동작한다.
- 트래픽 전 반드시 `GET /health/ready`로 컬렉션·벡터·인덱스 준비를 확인한다.

## v1.x 대비 폐기/변경된 환경 변수

| 구 변수 | 처리 |
|---|---|
| `NTIS_BRANCH_PREFETCH_LIMIT` / `NTIS_BRANCH_OUTPUT_LIMIT` | → `NTIS_DOC_TYPE_PREFETCH_LIMIT` / `NTIS_DOC_TYPE_OUTPUT_LIMIT` |
| branch별 벡터명 상수(`*_vector_e5i`, `*_vector_splade`) | 단일 `vector_e5i`/`vector_splade`로 대체(코드 상수) |
| `NTIS_QDRANT_COLLECTION_NAME` 기본값 `researcher_recommend_proto` | `ntis_researcher_chunks` |
