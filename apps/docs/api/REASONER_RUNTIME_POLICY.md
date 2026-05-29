# Reasoner Runtime Policy (chunk 재설계)

Last updated: 2026-05-28 (v2.0)

## Scope

This document records the active runtime behavior of recommendation reason generation on the chunk data model. Data model: [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md). Contract: [`DATA_CONTRACT.md`](DATA_CONTRACT.md).

## Active Policy

- Reason generation runs in sequential batches of up to `5` candidates.
- Candidate order is the retrieval order and is never changed by the reasoner.
- Evidence selection builds a per-candidate relevant pool of chunks, grouped by doc_type family, capped per family (default `10`).
- The reasoner uses a staged execution model:
  1. primary attempt with tool calling
  2. one retry with a smaller JSON-only payload
  3. deterministic server-side fallback if both attempts fail

## Evidence Selection (chunk-native)

- Chunks are first-class evidence. The selector reranks a candidate's matched chunks against the query (cross-encoder when a model is available, lexical fallback otherwise) and keeps the top chunks per family.
- The selector never reorders, drops, or invents candidates — it only narrows each candidate's grounding set.
- Default family caps (tunable): `achievement` 10, `assessment` 6, `expertise` 6, `identity` 1.

## Prompt Budgeting

- Selected chunks are the direct grounding set for `recommendation_reason`.
- Compact supporting context may also be included:
  - candidate head: `profile` + `researcher_meta` summary
  - assessment-history summary (`*_assessor`)
  - technical/specialty classifications
  - compact retrieval grounding (which doc_types matched, at what rank)
- Supporting context is trimmed aggressively to keep structured output stable.
- Each reason-generation LLM attempt uses an `8192` completion-token hint, forwarded as both `max_tokens` and `max_completion_tokens` by the OpenAI compatibility wrapper.

## Output Contract

The public recommendation response schema is unchanged in shape (see [`API_SPECIFICATION.md`](API_SPECIFICATION.md)). Internally, the LLM is expected to produce:

```json
{
  "items": [
    {
      "expert_id": "M1006328",
      "fit": "높음",
      "recommendation_reason": "짧고 근거 기반인 추천 사유",
      "selected_evidence_ids": ["PUB_M1006328_0001_c0", "PJT_M1006328_0001_c0"],
      "risks": []
    }
  ],
  "data_gaps": []
}
```

Evidence ID policy (v2.0):

- `selected_evidence_ids` MUST copy provided `chunk_id` values exactly.
- Valid id is any `chunk_id` present in the candidate's provided evidence pool. There is no positional format (`paper:N` is removed).
- If no direct evidence can be selected, the model returns an empty `selected_evidence_ids` array instead of inventing ids.
- The server resolves final `recommendation.evidence` from the selected `chunk_id`s; invalid or unresolved ids fall back deterministically to the highest-ranked chunks of the candidate.

## Trace Signals

Batch-level: `mode`, `retry_count`, `returned_ratio`, `prompt_budget_mode`, `trim_applied`, `payload_token_estimate`, `attempts`.

Top-level: `reason_generation_trace.reason_generation_failed`, `reason_generation_trace.server_fallback_reasons`.

Per-candidate evidence-resolution:
- `selected_evidence_ids` (chunk_ids returned by the model)
- `resolver_available_evidence_ids` (chunk_ids actually offered to the model)
- `invalid_selected_evidence_ids` (ids not present in the pool)
- `resolved_evidence_ids`
- `relevant_bundle_empty`
- `fallback`

## Notes

- Evidence id migration from positional (`paper:N`) to `chunk_id` is the only contract change in v2.0 for this layer; batch size, staged execution, and trim behavior are unchanged.
- Reason generation does not require new runbook steps beyond the chunk-collection readiness checks in [`../operation/RUNBOOK.md`](../operation/RUNBOOK.md).
