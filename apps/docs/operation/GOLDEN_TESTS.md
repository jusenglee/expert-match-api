# Golden Tests

## Scenarios

### 1. Pure keyword extraction

- Input: a natural-language recommendation query with request-role wording
- Expected:
  - planner extracts only domain `core_keywords`
  - request-role wording stays out of retrieval text

### 2. Explicit Top-k priority

- Input: a query plus explicit `top_k`
- Expected:
  - explicit `top_k` wins over any count implied in natural language
  - `/recommend` returns exactly that Top-k count

### 3. Search ordering stability

- Input: a query that produces equal RRF scores for at least two hits
- Expected:
  - final order uses `score desc`
  - score ties are broken by `name asc`

### 4. Search vs recommend split

- Input: a query without explicit candidate limit
- Expected:
  - `/search/candidates` returns the full retrieval-ordered candidate set
  - `/recommend` returns only ordered Top-k

### 5. Reason generation does not rerank

- Input: a query with several strong candidates
- Expected:
  - only ordered Top-k candidates are sent to the LLM
  - returned recommendations keep the exact retrieval order

### 6. Retrieval skipped on empty planner keywords

- Input: a query whose planner cannot produce safe `core_keywords`
- Expected:
  - Qdrant retrieval is skipped
  - `retrieval_skipped_reason` is present in trace
  - `recommendations=[]`

### 7. Payload-backed evidence

- Input: a query whose recommended candidate has publications or projects
- Expected:
  - final evidence comes from payload-backed preview data
  - evidence is deterministic and does not depend on LLM post-selection

### 8. Relevant evidence selection for reasons

- Input: a query whose recommended candidate has both newer irrelevant evidence and older relevant evidence
- Expected:
  - `/recommend` still preserves candidate retrieval order
  - reason generation receives only query-relevant evidence
  - newer but irrelevant evidence is not used as direct grounding for `recommendation_reason`

### 9. LLM-selected evidence alignment

- Input: a query whose recommended candidate has multiple relevant evidence items
- Expected:
  - the LLM receives a bounded per-candidate evidence pool
  - the LLM returns `selected_evidence_ids` from that pool
  - final `recommendation.evidence` resolves from those selected ids instead of latest preview data
  - invalid or empty evidence selections fall back deterministically

### 10. Batched reason generation and server fallback

- Input: a query whose `/recommend` Top-k is larger than 5 and whose LLM output omits or empties some candidates
- Expected:
  - reason generation runs in sequential batches of up to 5 candidates
  - returned recommendations still preserve the original retrieval order
  - omitted or empty reason candidates receive a conservative server fallback reason
  - trace exposes per-batch candidate ids and server fallback targets

### 11. Broad raw candidate context in LLM prompt

- Input: a query with returned recommendation candidates
- Expected:
  - the LLM payload includes the bounded relevant evidence pool
  - the same payload also includes broad raw candidate context such as retrieval grounding, evaluation activities, and full paper/project/patent summaries for the Top-k candidates
  - external API response shape remains unchanged

### 12. Retrieval score provenance trace

- Input: a query with returned candidates
- Expected:
  - trace exposes `retrieval_score_traces`
  - each trace item identifies the returned expert and matched branch list
  - playground can display the primary retrieval branch and branch-local match ranks

### 13. Tool-calling reason generation with retry

- Input: a `/recommend` query whose first reason-generation attempt fails to return usable structured output
- Expected:
  - the service first attempts tool-calling structured output
  - one smaller-payload retry is attempted before server fallback
  - trace exposes per-batch `mode`, `retry_count`, `returned_ratio`, `prompt_budget_mode`, `trim_applied`, and `attempts`

### 14. Relevant evidence pool policy

- Input: a candidate with many matching papers, projects, and patents
- Expected:
  - evidence selection caps the relevant pool at `10` papers, `10` projects, and `10` patents
  - final `recommendation.evidence` still resolves from the selected ids or deterministic fallback

### 15. Evidence id contract and resolve trace

- Input: a recommendation response where the model returns invalid-format ids or ids not present in the candidate pool
- Expected:
  - invalid-format ids are recorded separately from unresolved-but-well-formed ids
  - trace exposes both the ids returned by the LLM and the ids actually available to the resolver
  - profile fallback is explainable from trace without relying on server logs

### 16. Keyword pool then hybrid retrieval

- Input: a query whose sparse keyword stage returns a narrower candidate pool than a direct hybrid search would return
- Expected:
  - sparse keyword retrieval runs before hybrid retrieval on every request
  - hybrid retrieval is restricted to the keyword-stage `basic_info.researcher_id` pool
  - candidates outside the keyword-stage pool are not returned even if they appear in the hybrid response
  - trace exposes `query_payload.retrieval_mode="keyword_pool_then_hybrid"` and `keyword_stage_candidate_count`

### 17. Step-by-step retrieval logging

- Input: a normal `/recommend` or `/search/candidates` request
- Expected:
  - `trace.server_logs` includes user-query receipt, planner start/completion, retrieval start/completion, and candidate card build logs
  - retriever logs include 1st-stage keyword search start/completion and 2nd-stage hybrid search start/completion
  - `trace.query_payload` exposes branch/path counts and support pass/filter counts without logging vectors or full payloads

## Acceptance Criteria

- Sparse keyword retrieval text is built only from planner `retrieval_core`/`core_keywords`; hybrid retrieval may use planner `semantic_query` inside that keyword candidate pool.
- Retrieval always uses the fixed `keyword_pool_then_hybrid` flow: sparse keyword candidate pool first, then hybrid RRF inside that pool.
- `/search/candidates` preserves retrieval order.
- `/recommend` preserves retrieval order for returned items.
- `/recommend` sends only Top-k candidates to the LLM.
- `/recommend` re-ranks candidate-internal evidence against planner `core_keywords` before LLM reason generation.
- `/recommend` builds a relevant evidence pool capped at `10/10/10` before LLM reason generation.
- `/recommend` batches reason generation in groups of up to 5 candidates.
- `/recommend` uses tool calling first, then one compact JSON retry, then deterministic server fallback.
- `/recommend` resolves final `recommendation.evidence` from the LLM-selected relevant evidence ids.
- `/recommend` generates a conservative fallback reason when the LLM omits a candidate or returns an empty reason.
- Trace exposes `planner_keywords`, `retrieval_keywords`, `planner_retry_count`, `retrieval_skipped_reason`, `retrieval_score_traces`, `final_sort_policy`, `top_k_used`, `query_payload.retrieval_mode`, `query_payload.keyword_stage_candidate_count`, `query_payload.hybrid_stage_raw_branch_counts`, `query_payload.aggregated_candidate_count`, `query_payload.support_pass_count`, `query_payload.support_filtered_count`, `server_logs`, `reason_generation_trace.batches`, `reason_generation_trace.reason_generation_failed`, `reason_generation_trace.server_fallback_reasons`, and candidate-level evidence resolution details needed to explain fallback.
- Server logs must summarize user query, planner, 1st-stage keyword retrieval, and 2nd-stage hybrid retrieval without printing LLM raw responses, dense vectors, or full Qdrant payloads.
- Legacy verifier, multi-view retrieval, judge, and evidence-resolver traces are no longer part of the active contract.
