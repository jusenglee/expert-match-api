# Environment Variables

All runtime variables use the `NTIS_` prefix.

## Core

| Variable | Default | Meaning |
|---|---|---|
| `NTIS_APP_NAME` | `NTIS Evaluator Recommendation API` | FastAPI app name |
| `NTIS_APP_ENV` | `prod` | runtime environment |
| `NTIS_APP_HOST` | `0.0.0.0` | bind host |
| `NTIS_APP_PORT` | `8011` | bind port |
| `NTIS_STRICT_RUNTIME_VALIDATION` | `true` | fail readiness on runtime validation errors |

## Qdrant

| Variable | Default | Meaning |
|---|---|---|
| `NTIS_QDRANT_URL` | `http://203.250.234.159:8005` | Qdrant URL |
| `NTIS_QDRANT_API_KEY` | unset | Qdrant API key |
| `NTIS_QDRANT_COLLECTION_NAME` | `researcher_recommend_proto` | collection name |

## LLM

| Variable | Default | Meaning |
|---|---|---|
| `NTIS_LLM_BACKEND` | `openai_compat` | planner / reasoner backend |
| `NTIS_LLM_BASE_URL` | `http://203.250.234.159:8010/v1` | OpenAI-compatible endpoint |
| `NTIS_LLM_API_KEY` | `EMPTY` | LLM API key |
| `NTIS_LLM_MODEL_NAME` | `/model` | model name |

## Dense Embedding

| Variable | Default | Meaning |
|---|---|---|
| `NTIS_EMBEDDING_BACKEND` | `local` | dense embedding backend |
| `NTIS_EMBEDDING_MODEL_NAME` | `<repo>/multilingual-e5-large-instruct` | dense bundle or model name |
| `NTIS_EMBEDDING_VECTOR_SIZE` | `1024` | dense vector size |

## Sparse / Offline

| Variable | Default | Meaning |
|---|---|---|
| `NTIS_SPARSE_MODEL_NAME` | `<repo>/models/PIXIE-Splade-v1.0` | sparse local path or HF repo id |
| `NTIS_SPARSE_CACHE_DIR` | `<repo>/models` | sparse cache dir |
| `NTIS_SPARSE_LOCAL_FILES_ONLY` | `false` | force local-only sparse loading |
| `NTIS_HF_HUB_OFFLINE` | `false` | Hugging Face offline mode |

Sparse fallback chain:

1. local `models/PIXIE-Splade-v1.0`
2. online `telepix/PIXIE-Splade-v1.0`
3. builtin `Qdrant/bm25`

Sparse modifier expectation:

- custom PIXIE / SPLADE: `None`
- `Qdrant/bm25`: `IDF`

## Retrieval / Recommendation

| Variable | Default | Meaning |
|---|---|---|
| `NTIS_BRANCH_PREFETCH_LIMIT` | `100` | per-branch prefetch limit |
| `NTIS_BRANCH_OUTPUT_LIMIT` | `50` | per-branch output limit |
| `NTIS_RETRIEVAL_LIMIT` | `80` | retrieval limit |
| `NTIS_FINAL_RECOMMENDATION_MIN` | `1` | minimum recommendation count |
| `NTIS_FINAL_RECOMMENDATION_MAX` | `20` | maximum recommendation count |

## Operational Notes

- Retrieval ranking uses equal-weight RRF.
- Dense query text may differ from sparse query text because `semantic_query` is dense-only.
- The runtime still uses two paths only: `stable` and `expanded`.
- Contextual review/evaluation phrases are runtime intent flags, not environment-driven gates.
- Future-dated project evidence remains allowed.

## Cache Notes

The planner cache version and retrieval query payload schema changed with the dense/sparse split and `must_aspects` semantics update.

**Planner v0.7.0 (2026-04-17):** Added `evidence_aspects` bilingual field. Cached `PlannerOutput` from before this version will have `evidence_aspects = []` and fall back to `must_aspects` in the evidence selector. Cache auto-repopulates as queries are re-run; for immediate effect, manually invalidate planner cache artifacts.

If you roll out this change into a long-lived environment, invalidate or rotate:

- planner cache artifacts (schema version: `v0.7.0`)
- retrieval result cache artifacts (schema version: `v3_branch_instruct_prefix`)
