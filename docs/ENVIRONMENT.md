# 환경 변수

모든 환경 변수는 `NTIS_` prefix를 쓴다.

## 공통

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_APP_ENV` | `prod` | 실행 환경 |
| `NTIS_STRICT_RUNTIME_VALIDATION` | `true` | 운영전용 fail-fast 사용 여부 |
| `NTIS_QDRANT_URL` | `http://localhost:6333` | Qdrant URL |
| `NTIS_QDRANT_API_KEY` | 없음 | Qdrant API key |
| `NTIS_QDRANT_COLLECTION_NAME` | `expert_master` | 컬렉션명 |
| `NTIS_QDRANT_CLOUD_INFERENCE` | `false` | BM25 inference 사용 환경 |

## LLM backend

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_LLM_BACKEND` | `openai_compat` | 운영 경로 backend |
| `NTIS_LLM_BASE_URL` | `http://localhost:8010/v1` | OpenAI 호환 LLM URL |
| `NTIS_LLM_API_KEY` | `EMPTY` | OpenAI 호환 키 |
| `NTIS_LLM_MODEL_NAME` | `/model` | planner/judge용 모델명 |

## Embedding backend

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_EMBEDDING_BACKEND` | `openai` | 운영 경로 embedding backend |
| `NTIS_EMBEDDING_BASE_URL` | `http://localhost:8011/v1` | OpenAI 호환 embedding URL |
| `NTIS_EMBEDDING_API_KEY` | `EMPTY` | OpenAI 호환 키 |
| `NTIS_EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-large-instruct` | dense model 계약명 |
| `NTIS_EMBEDDING_VECTOR_SIZE` | `1024` | dense vector 크기 |

## 검색 파라미터

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_BRANCH_PREFETCH_LIMIT` | `80` | branch prefetch limit |
| `NTIS_BRANCH_OUTPUT_LIMIT` | `50` | branch fusion output limit |
| `NTIS_RETRIEVAL_LIMIT` | `40` | final retrieval limit |
| `NTIS_SHORTLIST_LIMIT` | `10` | shortlist 크기 |
| `NTIS_FINAL_RECOMMENDATION_MIN` | `3` | 최종 추천 최소 인원 |
| `NTIS_FINAL_RECOMMENDATION_MAX` | `5` | 최종 추천 최대 인원 |

## 기타

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_SEED_ON_STARTUP` | `false` | 로컬 개발 seed 수행 여부 |
| `NTIS_SEED_ALLOW_RECREATE_COLLECTION` | `false` | seed 전에 컬렉션 재생성 여부 |
| `NTIS_FEEDBACK_DB_PATH` | `runtime/feedback.db` | feedback sqlite 경로 |

## 운영 기본값

운영 경로에서는 아래를 권장한다.

- `NTIS_STRICT_RUNTIME_VALIDATION=true`
- `NTIS_LLM_BACKEND=openai_compat`
- `NTIS_EMBEDDING_BACKEND=openai`
- `NTIS_SEED_ON_STARTUP=false`

운영전용 모드에서는 heuristic/hash fallback과 seed startup을 허용하지 않는다.
