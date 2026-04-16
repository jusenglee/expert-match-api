# 환경 변수 (Environment Variables)

모든 환경 변수는 `NTIS_` 접두사를 사용합니다.

## 핵심 설정 (Core)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_APP_NAME` | `NTIS 전문가 추천 API` | FastAPI 앱 제목 |
| `NTIS_APP_ENV` | `prod` | 런타임 환경 (prod, dev 등) |
| `NTIS_APP_HOST` | `0.0.0.0` | FastAPI 서비스 바인딩 주소 |
| `NTIS_APP_PORT` | `8011` | FastAPI 수신 포트 |
| `NTIS_API_PREFIX` | `""` | 예약된 API 접두사 설정 |
| `NTIS_STRICT_RUNTIME_VALIDATION` | `true` | 운영 환경에서 필요한 설정들이 충족되지 않으면 추천 서비스를 비활성화함 |
| `NTIS_RUNTIME_DIR` | `runtime` | 런타임 출력물 저장 디렉토리 |
| `NTIS_FEEDBACK_DB_PATH` | `runtime/feedback.db` | 피드백 저장용 SQLite 경로 |
| `NTIS_FEEDBACK_TABLE` | `feedback_events` | 피드백 테이블 이름 |

## Qdrant 설정

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_QDRANT_URL` | `http://203.250.234.159:8005` | Qdrant 기본 URL |
| `NTIS_QDRANT_API_KEY` | (설정 안 됨) | Qdrant API 키 |
| `NTIS_QDRANT_COLLECTION_NAME` | `researcher_recommend_proto` | 사용할 컬렉션 이름 |
| `NTIS_QDRANT_CLOUD_INFERENCE` | `false` | Qdrant 클라우드 추론 활성화 여부 |

## LLM 백엔드 (LLM Backend)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_LLM_BACKEND` | `openai_compat` | 플래너/판정기용 LLM 백엔드 종류 |
| `NTIS_LLM_BASE_URL` | `http://203.250.234.159:8010/v1` | OpenAI 호환 LLM 서버 URL |
| `NTIS_LLM_API_KEY` | `EMPTY` | OpenAI 호환 API 키 |
| `NTIS_LLM_MODEL_NAME` | `/model` | 플래너/판정기용 모델 이름 |
| `NTIS_USE_MAP_REDUCE_JUDGING` | `true` | OpenAICompatJudge가 shortlist를 내부 배치 라운드로 심사할지 여부 |
| `NTIS_LLM_JUDGE_BATCH_SIZE` | `10` | Judge 내부 병렬 심사 배치 크기 |
| `NTIS_LLM_JUDGE_MAX_CONCURRENCY` | `10` | Judge 내부 LLM 동시 호출 상한 |

## 임베딩 백엔드 (Embedding Backend)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_EMBEDDING_BACKEND` | `local` | 임베딩 추출용 백엔드 종류 |
| `NTIS_EMBEDDING_BASE_URL` | `http://203.250.234.159:8011/v1` | OpenAI 호환 임베딩 서버 URL |
| `NTIS_EMBEDDING_API_KEY` | `EMPTY` | OpenAI 호환 API 키 |
| `NTIS_EMBEDDING_MODEL_NAME` | `<repo>/multilingual-e5-large-instruct` | 로컬 번들 경로 또는 원격 모델 이름 |
| `NTIS_EMBEDDING_VECTOR_SIZE` | `1024` | Dense 벡터 크기 |

로컬 임베딩 번들 경로는 기본적으로 리포지토리 번들인 `multilingual-e5-large-instruct`를 가리킵니다. 해당 번들을 교체할 때 `modules.json`, `1_Pooling/config.json`, `2_Normalize/` 구조를 그대로 유지해야 서버 시작 시 오류가 발생하지 않습니다.

## Sparse 및 오프라인 설정 (Sparse & Offline)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_SPARSE_MODEL_NAME` | `<repo>/models/PIXIE-Splade-v1.0` | Sparse 벡터 생성용 모델 이름 (로컬 번들 경로 또는 Hugging Face repo id) |
| `NTIS_SPARSE_CACHE_DIR` | `<repo>/models` | Sparse 모델 캐시 디렉토리 |
| `NTIS_SPARSE_LOCAL_FILES_ONLY` | `false` | Sparse 모델 로드 시 로컬 파일만 사용 여부 |
| `NTIS_HF_HUB_OFFLINE` | `false` | HuggingFace Hub 오프라인 모드 강제 여부 |

기본 sparse fallback 체인은 `로컬 PIXIE-Splade-v1.0 -> online telepix/PIXIE-Splade-v1.0 -> Qdrant/bm25` 순서입니다. 로컬 또는 online PIXIE 가 성공하면 커스텀 `SpladeSparseEncoder` 를 사용하고, 둘 다 실패하면 `Qdrant/bm25` 를 FastEmbed/Qdrant builtin sparse 모델로 로드합니다.

기본 `NTIS_SPARSE_MODEL_NAME` 은 저장소 내부의 `models/PIXIE-Splade-v1.0` 로컬 번들을 가리킵니다. 경로가 실제로 존재하면 tokenizer/model 을 `local_files_only=True` 로 로드하고, 경로가 없으면 local PIXIE 시도가 실패한 것으로 기록한 뒤 다음 fallback 단계로 넘어갑니다. `NTIS_HF_HUB_OFFLINE=true` 또는 `NTIS_SPARSE_LOCAL_FILES_ONLY=true` 이면 online PIXIE 단계는 건너뛰고 바로 `Qdrant/bm25` fallback 을 시도합니다.

`Qdrant/bm25` 는 커스텀 SPLADE 인코더가 아니라 FastEmbed builtin sparse 경로로 초기화됩니다. 이 모드에서는 Qdrant sparse vector modifier 가 `IDF` 여야 하며, PIXIE/SPLADE 모드에서는 modifier 가 없어야 합니다.

`ntis-validate-live` 와 `/health/ready` 도 동일한 sparse runtime resolver 를 사용하므로, 실제 active backend 가 `Qdrant/bm25` 로 선택된 경우 sparse vector modifier 기대값은 항상 `IDF` 입니다.

## 검색 제어 (Retrieval Controls)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_BRANCH_PREFETCH_LIMIT` | `100` | 각 데이터 브랜치별 프리페치(Prefetch) 제한 |
| `NTIS_BRANCH_OUTPUT_LIMIT` | `50` | 브랜치 통합(Fusion) 출력 제한 |
| `NTIS_RETRIEVAL_LIMIT` | `80` | 최종 검색 결과 통합 제한 |
| `NTIS_SHORTLIST_LIMIT` | `40` | 판정(Judge) 단계로 넘길 숏리스트 크기 |
| `NTIS_FINAL_RECOMMENDATION_MIN` | `1` | 최소 요구 최종 추천 수 |
| `NTIS_FINAL_RECOMMENDATION_MAX` | `20` | 최대 허용 최종 추천 수 |

## 시드 설정 (Seed Controls)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_SEED_ON_STARTUP` | `false` | 서버 시작 시 개발용 시드 데이터를 컬렉션에 로드할지 여부 |
| `NTIS_SEED_ALLOW_RECREATE_COLLECTION` | `false` | 시드 수행 시 기존 컬렉션을 삭제 후 재생성할지 여부 |

## 운영 참고 사항 (Operational Notes)

- 권장되는 라이브 설정:
  - `NTIS_STRICT_RUNTIME_VALIDATION=true`
  - `NTIS_LLM_BACKEND=openai_compat`
  - `NTIS_USE_MAP_REDUCE_JUDGING=true`
  - `NTIS_LLM_JUDGE_BATCH_SIZE=10`
  - `NTIS_LLM_JUDGE_MAX_CONCURRENCY=10`
  - `NTIS_EMBEDDING_BACKEND=openai` 또는 `local`
  - `NTIS_SEED_ON_STARTUP=false`
- 엄격한 런타임 검증이 활성화된 경우 라이브 설정은 위와 같이 구성되어야 하며, `NTIS_SEED_ON_STARTUP=false`여야 합니다. 그럼에도 불구하고 플래너와 판정기는 실제 실행 중에 AI 모델 호출이 실패하면 내부적으로 휴리스틱 폴백을 수행합니다.
- 엄격한 검증이나 의존성 초기화에 실패하더라도 프로세스는 시작될 수 있으나, 실제 추천 트래픽을 보내기 전에 반드시 `GET /health/ready` 엔드포인트의 응답을 확인하시기 바랍니다.
