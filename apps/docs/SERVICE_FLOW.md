# Service Flow

## Runtime Flow

### 1. Planner

`RecommendationService.search_candidates()` starts with planner execution.

Planner responsibilities:

- normalize the incoming query
- extract pure `core_keywords`
- separate request-role language into `task_terms`
- preserve explicit filters, include-orgs, exclude-orgs, and explicit `top_k`

Planner does not:

- build rewritten retrieval sentences
- emit retrieval views
- emit branch hints

### 2. Retrieval

`QueryTextBuilder` joins planner `core_keywords` into one cleaned retrieval query.

`QdrantHybridRetriever` then:

- runs dense + sparse search per branch
- fuses each branch with RRF
- fuses branch outputs with RRF
- records per-hit branch provenance in `retrieval_score_traces`
- applies deterministic final sorting:
  - `score desc`
  - `name asc` on score ties
  - `expert_id asc` as the final stable tie-break

### 3. Candidate Return

`/search/candidates` returns the ordered candidate list directly.

Behavior:

- if explicit `top_k` is provided on the endpoint, it limits the returned candidate count
- otherwise the endpoint returns the full retrieval result set

### 4. Recommendation Return

`/recommend` reuses the same search path and then:

- selects ordered Top-k candidates
- re-ranks each candidate's internal evidence against planner `core_keywords`
- builds an LLM evidence pool of up to `paper=4`, `project=4`, `patent=4` per candidate
- sends only those candidates and that evidence pool to the LLM in sequential batches of up to 5 candidates
- sends additional raw candidate context for those same Top-k candidates, including retrieval grounding, evaluation activities, technical classifications, and full paper/project/patent summaries
- receives `fit`, `recommendation_reason`, and selected evidence ids grounded on the provided pool
- keeps the original retrieval order
- resolves final `recommendation.evidence` from the LLM-selected evidence ids with deterministic fallback
- generates a conservative server fallback reason when the LLM omits a candidate or returns an empty reason

The LLM does not:

- rerank
- filter
- change candidate order

### 5. Empty or Failed Retrieval

If planner returns empty `core_keywords` after retry:

- retrieval is skipped
- `/search/candidates` returns zero candidates
- `/recommend` returns zero recommendations with a structured reason

## Trace Behavior

Current active trace fields:

- `planner`
- `planner_trace`
- `reason_generation_trace`
- `reason_generation_trace.batches`
- `reason_generation_trace.evidence_selection`
- `reason_generation_trace.selected_evidence`
- `reason_generation_trace.server_fallback_reasons`
- `raw_query`
- `planner_keywords`
- `retrieval_keywords`
- `planner_retry_count`
- `retrieval_skipped_reason`
- `branch_queries`
- `retrieval_score_traces`
- `final_sort_policy`
- `candidate_ids`
- `recommendation_ids`
- `top_k_used`
- `query_payload`
- `timers`

Removed from the active runtime path:

- verifier stage
- retrieval views
- branch query hints
- judge map-reduce
- evidence resolver alignment stage
