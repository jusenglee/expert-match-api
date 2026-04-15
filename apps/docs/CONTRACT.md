# Contract

## Planner Output

`PlannerOutput` is the structured contract between planner and retrieval.

```json
{
  "intent_summary": "Find drone-fire-response experts",
  "hard_filters": {},
  "include_orgs": [],
  "exclude_orgs": [],
  "task_terms": ["expert recommendation"],
  "core_keywords": ["fire suppression", "drone"],
  "top_k": 5
}
```

Planner rules:

- Planner returns one JSON object only.
- `core_keywords` contains retrieval-safe domain nouns or noun phrases only.
- `task_terms` contains request-role or action terms only.
- Explicit request parameters override natural-language extraction.
- Planner retries once when output is invalid or `core_keywords` is empty.
- Fallback planner returns normalized intent, empty `core_keywords`, copied request filters, copied org constraints, and explicit `top_k` when present.

## Query Builder Contract

`QueryTextBuilder` builds a single retrieval query from planner output.

Current behavior:

- raw user query is not used as retrieval text
- `intent_summary` is not used as retrieval text
- `task_terms` is not used as retrieval text
- retrieval text is the newline-joined `core_keywords`
- all four branches receive the same retrieval text

## Retrieval Contract

`QdrantHybridRetriever` always searches the four branches `basic`, `art`, `pat`, `pjt`.

For each branch:

- dense search runs on the branch dense vector
- sparse BM25 search runs on the branch sparse vector
- branch-local RRF merges dense and sparse
- top-level RRF merges all branch results

Current retrieval behavior:

- invalid payloads are skipped
- `exclude_orgs` is applied after payload validation
- if planner still returns empty `core_keywords` after retry, retrieval is skipped
- final application-level sort is `score desc -> researcher_name asc -> expert_id asc`

## Candidate Card Contract

`CandidateCard` is the internal search result contract used by both `/search/candidates` and `/recommend`.

Current behavior:

- card order follows retrieval order exactly
- `rank_score` is derived from retrieval score normalization
- `shortlist_score` equals `rank_score`
- `top_papers`, `top_patents`, and `top_projects` are deterministic payload previews
- `/recommend` may derive query-relevant evidence from those previews before LLM reason generation

## Reason Generation Contract

`OpenAICompatReasonGenerator` receives only ordered Top-k candidates.

Output schema:

```json
{
  "items": [
    {
      "expert_id": "11008395",
      "fit": "높음",
      "recommendation_reason": "Fire-response publications and projects are present.",
      "reasons": ["Fire-response publications and projects are present."],
      "risks": []
    }
  ],
  "data_gaps": []
}
```

Reason generation rules:

- the LLM does not rerank candidates
- the LLM does not drop candidates
- the LLM does not create new expert ids
- candidate-internal evidence is re-ranked against planner `core_keywords` before serialization
- only selected relevant evidence is provided as direct grounding for `recommendation_reason`
- if no relevant evidence is selected for a candidate, the LLM must not hallucinate direct query evidence
- output is normalized back to the original candidate order
- fallback reason generation keeps the same candidate order and does not inject a new ranking policy

## Response Assembly Contract

`POST /recommend` builds final `RecommendationDecision` items by:

- taking the search-ordered Top-k candidates
- inserting LLM-generated `fit` and `recommendation_reason`
- exposing `reasons` as a backward-compatible single-item alias of `recommendation_reason`
- building payload-backed `evidence` deterministically from the candidate preview data
- returning the same order received from retrieval

`POST /search/candidates` returns the search-ordered candidate list without LLM reason generation.

## API Trace Contract

Trace keys currently emitted:

- `planner`
- `planner_trace`
- `reason_generation_trace` (`/recommend` only)
- `reason_generation_trace.evidence_selection` (`/recommend` only, nested)
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

## Response Semantics

- `POST /search/candidates` returns the retrieval order as-is.
- `POST /recommend` returns only ordered Top-k items.
- Explicit request parameters take priority over values inferred from natural-language query.
- `POST /recommend` returns `200` even when no recommendations survive because retrieval was skipped or no candidates were found.
