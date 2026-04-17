# 환경 변수 (Environment Variables)

모든 환경 변수는 `NTIS_` 접두사를 사용한다.

## Core

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_APP_NAME` | `NTIS Evaluator Recommendation API` | FastAPI 앱 이름 |
| `NTIS_APP_ENV` | `prod` | 실행 환경 |
| `NTIS_APP_HOST` | `0.0.0.0` | 바인드 주소 |
| `NTIS_APP_PORT` | `8011` | 서비스 포트 |
| `NTIS_STRICT_RUNTIME_VALIDATION` | `true` | readiness 실패 시 추천 런타임 비활성화 여부 |
| `NTIS_RUNTIME_DIR` | `runtime` | 런타임 출력 디렉터리 |
| `NTIS_FEEDBACK_DB_PATH` | `runtime/feedback.db` | 피드백 SQLite 경로 |
| `NTIS_FEEDBACK_TABLE` | `feedback_events` | 피드백 테이블 이름 |

## Qdrant

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_QDRANT_URL` | `http://203.250.234.159:8005` | Qdrant URL |
| `NTIS_QDRANT_API_KEY` | unset | Qdrant API key |
| `NTIS_QDRANT_COLLECTION_NAME` | `researcher_recommend_proto` | collection 이름 |
| `NTIS_QDRANT_CLOUD_INFERENCE` | `false` | Qdrant cloud inference 사용 여부 |

## LLM

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_LLM_BACKEND` | `openai_compat` | planner / reasoner LLM backend |
| `NTIS_LLM_BASE_URL` | `http://203.250.234.159:8010/v1` | OpenAI-compatible endpoint |
| `NTIS_LLM_API_KEY` | `EMPTY` | LLM API key |
| `NTIS_LLM_MODEL_NAME` | `/model` | 모델 이름 |

## Dense Embedding

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_EMBEDDING_BACKEND` | `local` | dense embedding backend |
| `NTIS_EMBEDDING_BASE_URL` | `http://203.250.234.159:8011/v1` | remote embedding endpoint |
| `NTIS_EMBEDDING_API_KEY` | `EMPTY` | embedding API key |
| `NTIS_EMBEDDING_MODEL_NAME` | `<repo>/multilingual-e5-large-instruct` | 로컬 bundle 또는 모델 이름 |
| `NTIS_EMBEDDING_VECTOR_SIZE` | `1024` | dense vector size |

## Sparse / Offline

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_SPARSE_MODEL_NAME` | `<repo>/models/PIXIE-Splade-v1.0` | 기본 sparse 모델 경로 또는 HF repo id |
| `NTIS_SPARSE_CACHE_DIR` | `<repo>/models` | sparse cache dir |
| `NTIS_SPARSE_LOCAL_FILES_ONLY` | `false` | sparse 로딩 시 local files only 강제 |
| `NTIS_HF_HUB_OFFLINE` | `false` | Hugging Face offline 모드 |

현재 sparse fallback 체인은 다음과 같다.

1. 로컬 `models/PIXIE-Splade-v1.0`
2. online `telepix/PIXIE-Splade-v1.0`
3. `Qdrant/bm25` FastEmbed builtin

동작 규칙:

- 로컬 PIXIE가 있으면 `SpladeSparseEncoder`를 local path로 로드한다.
- 로컬 PIXIE 실패 후 online 단계가 허용되면 HF repo id를 시도한다.
- 둘 다 실패하면 `Qdrant/bm25`를 builtin sparse backend로 사용한다.
- `NTIS_HF_HUB_OFFLINE=true` 또는 `NTIS_SPARSE_LOCAL_FILES_ONLY=true`면 online PIXIE 단계는 건너뛴다.

sparse modifier 기대값:

- PIXIE / custom SPLADE: `None`
- `Qdrant/bm25`: `IDF`

## Retrieval / Recommendation

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_BRANCH_PREFETCH_LIMIT` | `100` | branch prefetch limit |
| `NTIS_BRANCH_OUTPUT_LIMIT` | `50` | branch output limit |
| `NTIS_RETRIEVAL_LIMIT` | `80` | retrieval limit |
| `NTIS_FINAL_RECOMMENDATION_MIN` | `1` | 최소 추천 수 |
| `NTIS_FINAL_RECOMMENDATION_MAX` | `20` | 최대 추천 수 |

주의:

- 현재 reason generation batch 크기는 코드 상수 `5`로 고정되어 있다.
- shortlist gate는 env 가중치로 조정하지 않는다.

## Legacy Compatibility

아래 설정은 `Settings`에는 남아 있지만 현재 recommendation runtime의 핵심 경로에서는 직접 사용하지 않는다.

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_LLM_JUDGE_BATCH_SIZE` | `10` | legacy judge batch size |
| `NTIS_LLM_JUDGE_MAX_CONCURRENCY` | `10` | legacy judge concurrency |
| `NTIS_USE_MAP_REDUCE_JUDGING` | `true` | legacy judge toggle |

현재 활성 경로는 judge/map-reduce가 아니라 `evidence selector -> shortlist gate -> reasoner -> validator`다.

## Seed

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_SEED_ON_STARTUP` | `false` | startup seed 수행 여부 |
| `NTIS_SEED_ALLOW_RECREATE_COLLECTION` | `false` | seed 시 collection 재생성 허용 여부 |

## 운영 메모

- production에서는 `GET /health/ready`와 `python -m apps.tools.validate_live` 결과를 함께 확인한다.
- `Qdrant/bm25` fallback은 정상 운영 모드다. 이 경우 readiness 실패로 보지 않는다.
- 메타어 제거, shortlist gate, validator fallback 상태는 trace와 로그에서 확인한다.
## 2026-04-17 retrieval notes

- Retrieval ranking uses equal-weight RRF. There is no active environment variable for branch/path weighting.
- Expanded-path execution is runtime-driven by query text difference, not by a per-branch weight toggle.
- Future-dated project evidence remains allowed; observability is handled through selector/service trace rather than an environment switch.
