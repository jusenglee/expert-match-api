# Contract

## Planner Output

`PlannerOutput` is the structured contract between planner and retrieval.

```json
{
  "intent_summary": "난접근성 화재 진압과 드론 접목 관련 전문가 탐색",
  "hard_filters": {},
  "include_orgs": [],
  "exclude_orgs": [],
  "soft_preferences": [],
  "task_terms": ["전문가", "추천"],
  "core_keywords": ["난접근성 화재 진압", "드론"],
  "branch_query_hints": {},
  "top_k": 5
}
```

Planner rules:

- Planner must return a single JSON object only.
- Invocation uses low-variance settings: `temperature=0.0`, `top_p=0.2`, `reasoning_effort=low`, `include_reasoning=false`, `disable_thinking=true`.
- A deterministic `seed` is derived from normalized query text, request overrides, and attempt index.
- `core_keywords` is the only retrieval-semantic output. It must contain domain nouns or noun phrases only.
- `task_terms` stores non-retrieval request, role, or action terms. Retrieval does not consume it.
- `intent_summary` is retained for UI and trace only. Retrieval does not consume it.
- `include_orgs` and `exclude_orgs` are preserved exactly from request input when present.
- Planner is followed by a verifier LLM call that rewrites the planner JSON into a retrieval-safe plan.
- Planner retries once when planner or verifier output is invalid, or when verified `core_keywords` is empty.
- Fallback planner returns normalized intent, empty `core_keywords`, empty `task_terms`, copied request filters, and copied org constraints.
- `branch_weights` remains on the Pydantic model for backward compatibility, but retrieval does not consume it.
- `branch_query_hints` is deprecated and retained only for backward compatibility. Retrieval does not consume it.

## Query Builder Contract

`QueryTextBuilder` builds every branch query from `core_keywords` only.

Current behavior:

- raw user query is not used as retrieval text
- planner `intent_summary` is not used as retrieval text
- `task_terms` is not used as retrieval text
- `branch_query_hints` is ignored
- all four branches receive the same cleaned query text
- keyword normalization is limited to whitespace collapse and de-duplication

## Retrieval Contract

`QdrantHybridRetriever` always searches the four branches `basic`, `art`, `pat`, `pjt`.

For each branch:

- dense search runs on the branch dense vector
- sparse BM25 search runs on the branch sparse vector
- branch-local RRF merges dense and sparse
- top branch results are merged again with top-level RRF

Current retrieval behavior:

- planner branch weights are ignored
- dense and sparse both use the same verifier-approved `core_keywords` query
- `exclude_orgs` is applied after payload validation
- invalid payloads are skipped and do not fail the request
- if verifier still returns empty `core_keywords` after retry, retrieval is skipped

## Judge Output

`JudgeOutput` is the contract between judge and API response assembly.

```json
{
  "recommended": [
    {
      "rank": 1,
      "expert_id": "11008395",
      "name": "Hong Gildong",
      "organization": "Test Institute",
      "fit": "높음",
      "reasons": ["Fire-suppression publication evidence is available."],
      "evidence": [
        {
          "type": "paper",
          "title": "Fire Response Drone System",
          "date": "2024-09-01",
          "detail": "IEEE Access"
        }
      ],
      "risks": [],
      "rank_score": 95.5
    }
  ],
  "not_selected_reasons": [],
  "data_gaps": []
}
```

Judge rules:

- Judge uses the same low-variance invoke policy as planner and also receives a deterministic per-call seed.
- Single and reduce calls receive the full serialized shortlist in the user payload.
- Map calls receive a lightweight shortlist plus `selection_limit`.
- Reduce opens only when both gates pass:
  - candidate count is at or below `plan.top_k + 1`
  - token estimate is at or below `MAP_PHASE_MAX_TOKENS * 2`
- Map contracts the shortlist until the reduce gate opens.
- If an LLM batch returns too many survivors, no survivors, or invalid output, the system compresses deterministically by existing rank order.
- If a round still fails to contract enough after LLM output, the system force-compresses the full round to the target survivor count.
- Reduce never falls back to an oversized prompt after map contraction failure.
- Fallback judge is deterministic and uses existing card evidence only.
- Judge evidence is provisional. Final UI evidence is resolved after judge from the original expert payload.

## Evidence Resolution Contract

`RecommendationService` resolves final UI evidence after judge selection.

Current behavior:

- resolver input is `query`, verified `core_keywords`, judge reasons, recommendation metadata, and the original `ExpertPayload`
- the resolver selects only payload-backed evidence options and returns canonical `EvidenceItem` values
- invalid or unknown evidence option ids are rejected
- if aligned evidence cannot be resolved, the recommendation is dropped before the final API response
- shortlist preview fields such as `top_papers`, `top_patents`, and `top_projects` are not the final UI evidence source

## API Trace Contract

`POST /recommend` and `POST /search/candidates` expose trace data for run-to-run comparison.

Trace keys currently emitted:

- `planner`
- `planner_trace`
- `judge_trace` (`/recommend` only)
- `raw_query`
- `planner_raw_keywords`
- `verifier_keywords`
- `retrieval_keywords`
- `planner_retry_count`
- `verifier_applied`
- `retrieval_skipped_reason`
- `branch_queries`
- `include_orgs`
- `exclude_orgs`
- `candidate_ids`
- `evidence_resolution_trace` (`/recommend` only)
- `recommendation_evidence_summary` (`/recommend` only)
- `query_payload`
- `timers`

`planner_trace` includes normalized query, retry metadata, raw planner keywords, verifier-approved keywords, and retrieval keywords when available.

`judge_trace` includes round-level metadata such as:

- `context`
- `round`
- `batch`
- `candidate_count`
- `output_count`
- `selection_limit`
- `token_estimate`
- `seed`
- `status`
- `final_reduce_candidate_count`
- `final_reduce_token_estimate`
- `final_reduce_gate_reason`

Round summary entries also include:

- `input_count`
- `target_survivor_count`
- `post_llm_count`
- `post_compression_count`
- `reduce_candidate_limit`
- `reduce_token_limit`
- `forced_compression_applied`

## Response Semantics

- `POST /recommend` returns `200` even when no final recommendation survives.
- Empty recommendation responses are represented as `recommendations=[]` plus structured reasons in `not_selected_reasons` and `data_gaps`.
- Recommendation items without evidence are dropped before the final API response.
- Final `RecommendationDecision.evidence` shown to the UI is rebuilt from the original payload after judge, not copied directly from shortlist preview evidence.
