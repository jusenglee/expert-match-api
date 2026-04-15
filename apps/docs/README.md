# NTIS Expert Recommendation Service

Current request flow:

1. Planner extracts pure `core_keywords` and non-retrieval `task_terms`.
2. Query builder joins `core_keywords` into one cleaned retrieval query.
3. Retriever runs branch-level dense+sparse hybrid search and RRF.
4. Final search order is fixed as `RRF score desc -> researcher name asc -> expert_id asc`.
5. `/search/candidates` returns candidates in that search order.
6. `/recommend` takes only ordered Top-k candidates, asks the LLM to generate reasons, and returns them in the same order.

## Endpoint Behavior

- `POST /search/candidates`
  - Returns the search-ordered candidate list.
  - Uses explicit `top_k` only as a response limit when the caller provides it.
- `POST /recommend`
  - Uses explicit `top_k` first.
  - If explicit `top_k` is missing, falls back to planner-extracted `top_k`.
  - Sends only ordered Top-k candidates to the LLM for reason generation.
  - Does not rerank after LLM reasoning.

## Retrieval Rules

- Raw natural-language query does not enter retrieval text.
- Retrieval text is built only from planner `core_keywords`.
- All four branches receive the same cleaned query text.
- Branch fusion uses RRF.
- Final application-level sort uses score first and name only as the tie-breaker.

## Trace Keys

API trace currently includes:

- `planner`
- `planner_trace`
- `reason_generation_trace` (`/recommend` only)
- `raw_query`
- `planner_keywords`
- `retrieval_keywords`
- `planner_retry_count`
- `retrieval_skipped_reason`
- `branch_queries`
- `final_sort_policy`
- `candidate_ids`
- `recommendation_ids` (`/recommend` only)
- `top_k_used` (`/recommend` only)
- `query_payload`
- `timers`

## Docs

- `apps/docs/CONTRACT.md`: runtime contract for planner, retrieval, reason generation, and trace
- `apps/docs/SERVICE_FLOW.md`: request-time behavior and ordering semantics
- `apps/docs/GOLDEN_TESTS.md`: regression scenarios and acceptance criteria
- `apps/docs/RUNBOOK.md`: operations guidance
- `apps/docs/ENVIRONMENT.md`: environment settings
