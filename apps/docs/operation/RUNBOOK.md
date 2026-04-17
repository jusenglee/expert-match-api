# Runbook

## Runtime Checks

Recommended order:

1. `python -m apps.tools.validate_live`
2. `GET /health`
3. `GET /health/ready`

`validate_live` and `/health/ready` should be interpreted using the same sparse runtime resolution rules.

## Sparse Backend Expectations

- local `models/PIXIE-Splade-v1.0`
- online `telepix/PIXIE-Splade-v1.0`
- fallback `Qdrant/bm25`

Expected sparse modifier:

- custom PIXIE / SPLADE: `None`
- `Qdrant/bm25`: `IDF`

## Retrieval Expectations

- retrieval ranking is equal-weight RRF
- branch/path weight tables are not part of the active runtime
- path model is still `stable` / `expanded`
- dense and sparse query texts may differ by design

## Planner Log Checklist

Look for:

- `removed_meta_terms`
- `retained_contextual_terms`
- `retrieval_core`
- `must_aspects`
- `evidence_aspects`
- `intent_flags`
- `semantic_query`

Interpretation:

- `평가위원`, `전문가`, `추천` should not survive into retrieval fields
- `과제 평가` / `기술 평가` / `논문 평가` should appear in `intent_flags.review_targets`, not `must_aspects`
- `evidence_aspects` should contain bilingual (Korean + English) domain terms; if empty, the selector falls back to `must_aspects`
- `evidence_aspects` and `must_aspects` may differ: `evidence_aspects` includes English equivalents; `must_aspects` is Korean-only

## Retrieval Log Checklist

Look for:

- dense base source
- sparse base source
- query schema version
- stable/expanded execution count

If cache results look suspicious after deployment, invalidate retrieval cache artifacts because the query payload schema changed with dense/sparse split.

## Selector / Gate Log Checklist

Look for:

- `direct_match_count`
- `aspect_coverage`
- `aspect_source` — which planner field was used (`evidence_aspects` / `must_aspects` / `retrieval_core` / `core_keywords`)
- `matched_aspects`
- `selected_evidence_ids`
- `future_selected_evidence_ids`
- shortlist `kept`, `low_coverage`, `generic_only`, `dropped`

Interpretation:

- `aspect_source` should normally be `evidence_aspects`; if `must_aspects`, the LLM did not generate `evidence_aspects` (check planner output)
- `aspect_coverage` is phrase-based
- future-dated projects are still valid evidence

## Reasoner / Validator Log Checklist

Look for:

- batch candidate ids
- `returned_ratio`
- `retry_triggered`
- `retry_reason`
- fallback candidate ids

Validator warnings should reference:

- `other_candidate_name`
- `internal_evidence_id`
- `aspect_scope_miss`
- `strong_claim_without_direct_evidence`

`aspect_scope_miss` now means the aspect was absent across `title + detail + snippet`, not title-only.
