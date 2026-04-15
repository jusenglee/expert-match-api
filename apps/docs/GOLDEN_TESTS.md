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

## Acceptance Criteria

- Retrieval text is built only from planner `core_keywords`.
- `/search/candidates` preserves retrieval order.
- `/recommend` preserves retrieval order for returned items.
- `/recommend` sends only Top-k candidates to the LLM.
- `/recommend` re-ranks candidate-internal evidence against planner `core_keywords` before LLM reason generation.
- `/recommend` batches reason generation in groups of up to 5 candidates.
- `/recommend` resolves final `recommendation.evidence` from the LLM-selected relevant evidence ids.
- `/recommend` generates a conservative fallback reason when the LLM omits a candidate or returns an empty reason.
- Trace exposes `planner_keywords`, `retrieval_keywords`, `planner_retry_count`, `retrieval_skipped_reason`, `retrieval_score_traces`, `final_sort_policy`, `top_k_used`, `reason_generation_trace.batches`, and `reason_generation_trace.server_fallback_reasons`.
- Legacy verifier, multi-view retrieval, judge, and evidence-resolver traces are no longer part of the active contract.
