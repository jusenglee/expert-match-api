# NTIS Expert Recommendation Service

Current request flow:

1. Planner parses the user query into structured retrieval intent.
2. Verifier rewrites planner output into a retrieval-safe plan.
3. Query builder assembles retrieval text from verified `core_keywords` only.
4. Retriever runs Qdrant dense + sparse + RRF search.
5. Candidate cards are built and shortlisted.
6. Judge decides final recommendations with single-shot or map-reduce.
7. Evidence resolver rebuilds UI evidence from the original expert payload.

## Endpoints

- `POST /recommend`
- `POST /search/candidates`
- `POST /feedback`
- `GET /health`
- `GET /health/ready`
- `GET /playground`

## Current Retrieval Stability Notes

- Planner, verifier, and judge use deterministic seeds.
- Retrieval text is built from verifier-approved `core_keywords` only.
- Raw user query is kept in trace, not in retrieval text.
- `intent_summary` is UI and trace data only.
- `task_terms` preserves request-role or action meaning but is not used for retrieval.
- `branch_query_hints` is deprecated and not used by retrieval.
- Dense and sparse search use the same cleaned query text across all four branches.
- If planner and verifier still cannot produce safe `core_keywords` after retry, retrieval is skipped.
- Judge map contraction uses deterministic compression instead of oversized reduce fallback.
- Final UI evidence is resolved after judge from the original expert payload, not from card preview items alone.

## API Trace Notes

API trace includes:

- `planner_trace`
- `judge_trace`
- `raw_query`
- `planner_raw_keywords`
- `verifier_keywords`
- `retrieval_keywords`
- `planner_retry_count`
- `verifier_applied`
- `retrieval_skipped_reason`
- `branch_queries`
- `candidate_ids`
- `evidence_resolution_trace`
- `query_payload`
- `timers`

## Docs

- `apps/docs/CONTRACT.md`: planner, verifier, retrieval, judge, and trace contracts
- `apps/docs/SERVICE_FLOW.md`: pipeline behavior and runtime flow
- `apps/docs/RUNBOOK.md`: operating guidance
- `apps/docs/ENVIRONMENT.md`: environment settings
