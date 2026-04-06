# Environment Variables

All environment variables use the `NTIS_` prefix.

## Core

| Variable | Default | Description |
|---|---|---|
| `NTIS_APP_NAME` | `NTIS Evaluator Recommendation API` | FastAPI app title |
| `NTIS_APP_ENV` | `prod` | runtime environment |
| `NTIS_API_PREFIX` | `""` | reserved API prefix setting |
| `NTIS_STRICT_RUNTIME_VALIDATION` | `true` | require production-safe runtime settings before the recommendation runtime becomes available |
| `NTIS_RUNTIME_DIR` | `runtime` | runtime output directory |
| `NTIS_FEEDBACK_DB_PATH` | `runtime/feedback.db` | feedback SQLite path |
| `NTIS_FEEDBACK_TABLE` | `feedback_events` | feedback table name |

## Qdrant

| Variable | Default | Description |
|---|---|---|
| `NTIS_QDRANT_URL` | `http://203.250.234.159:8005` | Qdrant base URL |
| `NTIS_QDRANT_API_KEY` | unset | Qdrant API key |
| `NTIS_QDRANT_COLLECTION_NAME` | `expert_master` | collection name |
| `NTIS_QDRANT_CLOUD_INFERENCE` | `false` | Qdrant cloud inference toggle |

## LLM Backend

| Variable | Default | Description |
|---|---|---|
| `NTIS_LLM_BACKEND` | `openai_compat` | planner/judge backend |
| `NTIS_LLM_BASE_URL` | `http://203.250.234.159:8010/v1` | OpenAI-compatible LLM URL |
| `NTIS_LLM_API_KEY` | `EMPTY` | OpenAI-compatible API key |
| `NTIS_LLM_MODEL_NAME` | `/model` | planner/judge model name |

## Embedding Backend

| Variable | Default | Description |
|---|---|---|
| `NTIS_EMBEDDING_BACKEND` | `local` | embedding backend |
| `NTIS_EMBEDDING_BASE_URL` | `http://203.250.234.159:8011/v1` | OpenAI-compatible embedding URL |
| `NTIS_EMBEDDING_API_KEY` | `EMPTY` | OpenAI-compatible API key |
| `NTIS_EMBEDDING_MODEL_NAME` | `<repo>/multilingual-e5-large-instruct` | local bundle path or remote model name |
| `NTIS_EMBEDDING_VECTOR_SIZE` | `1024` | dense vector size |

The default local embedding path points at the repository bundle `multilingual-e5-large-instruct`. When replacing that bundle, keep `modules.json`, `1_Pooling/config.json`, and `2_Normalize/` intact or startup will fail.

## Retrieval Controls

| Variable | Default | Description |
|---|---|---|
| `NTIS_BRANCH_PREFETCH_LIMIT` | `80` | branch prefetch limit |
| `NTIS_BRANCH_OUTPUT_LIMIT` | `50` | branch fusion output limit |
| `NTIS_RETRIEVAL_LIMIT` | `40` | final retrieval limit |
| `NTIS_SHORTLIST_LIMIT` | `10` | shortlist size |
| `NTIS_FINAL_RECOMMENDATION_MIN` | `3` | minimum final recommendations |
| `NTIS_FINAL_RECOMMENDATION_MAX` | `5` | maximum final recommendations |

## Seed Controls

| Variable | Default | Description |
|---|---|---|
| `NTIS_SEED_ON_STARTUP` | `false` | run the development seed path during startup |
| `NTIS_SEED_ALLOW_RECREATE_COLLECTION` | `false` | allow seed flow to recreate the collection |

## Operational Notes

- Recommended live settings:
  - `NTIS_STRICT_RUNTIME_VALIDATION=true`
  - `NTIS_LLM_BACKEND=openai_compat`
  - `NTIS_EMBEDDING_BACKEND=openai` or `local`
  - `NTIS_SEED_ON_STARTUP=false`
- With strict runtime validation enabled, heuristic/hash fallbacks are not allowed to serve live traffic.
- If strict validation or dependency initialization fails, the process may still start in degraded mode. Inspect `GET /health/ready` before sending recommendation traffic.
