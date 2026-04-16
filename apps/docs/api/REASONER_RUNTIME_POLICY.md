# Reasoner Runtime Policy

Last updated: 2026-04-15

## Scope

This document records the active runtime behavior of recommendation reason generation after the tool-calling stabilization update.

## Active Policy

- Reason generation runs in sequential batches of up to `5` candidates.
- Evidence selection builds a per-candidate relevant pool capped at:
  - papers: `10`
  - projects: `10`
  - patents: `10`
- The reasoner uses a staged execution model:
  1. primary attempt with tool calling
  2. one retry with a smaller JSON-only payload
  3. deterministic server-side fallback if both attempts fail

## Prompt Budgeting

- `relevant_*` evidence is the direct grounding set for `recommendation_reason`.
- Compact supporting context may also be included:
  - compact `all_*` previews
  - compact retrieval grounding
  - technical classifications
  - evaluation activity summaries
- Supporting context is trimmed aggressively to keep structured output stable.

## Output Contract

The public recommendation response schema is unchanged. Internally, the LLM is expected to produce this structured payload:

```json
{
  "items": [
    {
      "expert_id": "12345678",
      "fit": "높음",
      "recommendation_reason": "짧고 근거 기반인 추천 사유",
      "selected_evidence_ids": ["paper:0", "project:1"],
      "risks": []
    }
  ],
  "data_gaps": []
}
```

Evidence ID policy:

- `selected_evidence_ids` must copy provided evidence ids exactly.
- Valid id format is limited to:
  - `paper:<number>`
  - `project:<number>`
  - `patent:<number>`
- If no direct evidence can be selected for a candidate, the model should return an empty `selected_evidence_ids` array instead of inventing ids.

## Trace Signals

Batch-level trace fields:

- `mode`
- `retry_count`
- `returned_ratio`
- `prompt_budget_mode`
- `trim_applied`
- `payload_token_estimate`
- `attempts`

Top-level trace fields:

- `reason_generation_trace.reason_generation_failed`
- `reason_generation_trace.server_fallback_reasons`

Per-candidate evidence-resolution trace fields:

- `selected_evidence_ids`
- `resolver_available_evidence_ids`
- `invalid_selected_evidence_ids`
- `unresolved_selected_evidence_ids`
- `resolved_evidence_ids`
- `relevant_bundle_empty`
- `fallback`

## Notes

- This update does not add any new environment variables.
- This update does not require new runbook steps.
