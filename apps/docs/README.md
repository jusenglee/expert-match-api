# NTIS Expert Recommendation Service

Current request flow:

1. Planner extracts pure `core_keywords` and non-retrieval `task_terms`.
2. Query builder joins `core_keywords` into one cleaned retrieval query.
3. Retriever runs branch-level dense+sparse hybrid search and RRF.
4. Final search order is fixed as `RRF score desc -> researcher name asc -> expert_id asc`, and trace keeps per-hit branch provenance in `retrieval_score_traces`.
5. `/search/candidates` returns candidates in that search order.
6. `/recommend` takes only ordered Top-k candidates, re-ranks each candidate's internal evidence against planner `core_keywords`, asks the LLM to generate reasons and evidence selections from only that relevant evidence pool, and returns them in the same order.
7. `reason_generation` runs in sequential batches of up to 5 candidates, and the server fills any empty reason with a conservative evidence-based fallback sentence.
8. The LLM prompt also includes broad raw candidate context for those Top-k candidates, including retrieval grounding, full paper/project/patent lists, and evaluation activities.

## Endpoint Behavior

- `POST /search/candidates`
  - Returns the search-ordered candidate list.
  - Uses explicit `top_k` only as a response limit when the caller provides it.
- `POST /recommend`
  - Uses explicit `top_k` first.
  - If explicit `top_k` is missing, falls back to planner-extracted `top_k`.
  - Re-ranks each candidate's papers/projects/patents against planner `core_keywords` before reason generation.
  - Builds an LLM evidence pool with up to `paper=4`, `project=4`, `patent=4` per candidate.
  - Sends only ordered Top-k candidates plus that evidence pool to the LLM in sequential batches of up to 5 candidates.
  - Also sends broad raw candidate context for those Top-k candidates, including retrieval grounding and full payload-backed paper/project/patent summaries.
  - Uses the LLM-selected evidence ids to populate final `recommendation.evidence`.
  - Generates a server-side conservative fallback reason when the LLM omits a candidate or returns an empty reason.
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
- `reason_generation_trace.batches` (`/recommend` only, nested)
- `reason_generation_trace.selected_evidence` (`/recommend` only, nested)
- `reason_generation_trace.server_fallback_reasons` (`/recommend` only, nested)
- `raw_query`
- `planner_keywords`
- `retrieval_keywords`
- `planner_retry_count`
- `retrieval_skipped_reason`
- `branch_queries`
- `retrieval_score_traces`
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
