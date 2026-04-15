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

## Acceptance Criteria

- Retrieval text is built only from planner `core_keywords`.
- `/search/candidates` preserves retrieval order.
- `/recommend` preserves retrieval order for returned items.
- `/recommend` sends only Top-k candidates to the LLM.
- Trace exposes `planner_keywords`, `retrieval_keywords`, `planner_retry_count`, `retrieval_skipped_reason`, `final_sort_policy`, and `top_k_used`.
- Legacy verifier, multi-view retrieval, judge, and evidence-resolver traces are no longer part of the active contract.
