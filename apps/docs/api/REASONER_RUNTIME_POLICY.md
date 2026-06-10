# Reasoner Runtime Policy (flat chunk 모델)

Last updated: 2026-06-02 (v2.1)

## Scope

This document records the active runtime behavior of recommendation reason generation on the flat chunk data model. Data model: [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md). Contract: [`DATA_CONTRACT.md`](DATA_CONTRACT.md).

> Data unit: 1 chunk = 1 Qdrant Point. Payload root `chunk_id` is the authoritative evidence id; Qdrant Point ID may be the same `chunk_id` or an operational UUID. Payload is **flat** — researcher common meta (`researcher_id`, `researcher_name`, `affiliated_organization`, `highest_degree`, and the five counts `publication_count` / `scie_publication_count` / `intellectual_property_count` / `research_project_count` / `researcher_assessor_activity_count`) lives at the **root**; doc_type-specific detail lives in `doc_attrs{}`. There is a single date field `doc_date` (string, may be `"NONE"`). doc_type is exactly one of **paper / patent / project / assessor_activity / specialty**.

## Active Policy

- Reason generation runs in sequential batches of up to `5` candidates.
- Candidate order is the retrieval order and is never changed by the reasoner.
- Evidence selection builds a per-candidate relevant pool of chunks, grouped by doc_type and capped per family (default `achievement` 10).
- The reasoner uses a staged execution model:
  1. primary attempt with tool calling
  2. one retry with a smaller JSON-only payload
  3. deterministic server-side fallback if both attempts fail

## Evidence Selection (chunk-native)

- Chunks are first-class evidence. The selector reranks a candidate's matched chunks against the query (cross-encoder when a model is available, lexical fallback otherwise) and keeps the top chunks per family.
- The selector never reorders, drops, or invents candidates — it only narrows each candidate's grounding set.
- Default family caps (tunable via `NTIS_EVIDENCE_FAMILY_CAP`): `achievement` 10, `assessment` 6, `expertise` 6, `identity` 1.
- family membership: `achievement` = {paper, patent, project}, `assessment` = {assessor_activity}, `expertise` = {specialty}, `identity` = synthetic profile evidence (no doc_type; built from the flat root fields shared by every chunk).

## Prompt Budgeting

- Selected chunks are the direct grounding set for `recommendation_reason`.
- Compact supporting context may also be included:
  - candidate head: `profile` + flat-root researcher meta summary (organization / degree / counts)
  - assessment-history summary (`researcher_assessor_activity_count` + matched `assessor_activity` chunks)
  - technical/specialty classifications (`specialty` chunks)
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
      "selected_evidence_ids": ["paper_100000045256_c000", "project_100000099812_c000"],
      "risks": []
    }
  ],
  "data_gaps": []
}
```

Evidence ID policy (v2.x):

- `selected_evidence_ids` MUST copy provided `chunk_id` values exactly.
- A `chunk_id` follows the codec `<doc_type>_<doc_id_body>_c<NNN>`; the `<NNN>` chunk index is zero-padded to 3 digits and the codec is anchored ONLY by the trailing `_c<NNN>`. `<doc_id_body>` may be numeric (`paper_100000045256_c000`) or a researcher-id form (`specialty_M1013800_c000`). The owning doc is `doc_id = <doc_type>_<doc_id_body>`.
- There is no positional format (`paper:N` is removed). The server enforces **only the codec** on returned ids with the regex `^(?:paper|patent|project|assessor_activity|specialty)_.+_c\d+$` (the middle doc_id body is `.+`, anchored only by the trailing `_c<NNN>`) and records ids that fail it in trace. Pool membership is **not** enforced, because `selected_evidence_ids` does not drive evidence assembly (see below).
- If no direct evidence can be cited, the model returns an empty `selected_evidence_ids` array instead of inventing ids.
- The final `recommendation.evidence` is assembled deterministically from each candidate's selector-built relevant chunk pool (the full family-capped bundle), **independent of `selected_evidence_ids`**. `selected_evidence_ids` is only a citation hint for grounding the reason text and is recorded in trace; it never assembles or filters `recommendation.evidence`. Only candidates whose relevant pool is empty fall back to profile (or empty) evidence.
- Up to `4` evidence ids may be cited per candidate (`MAX_SELECTED_EVIDENCE_IDS`).
- `recommendation_reason` is truncated server-side to `320` characters (`REASON_MAX_CHARS`, suffix `...`) when the model exceeds it; affected candidates are recorded in trace.
- The server also strips any leaked evidence_id/chunk_id token from `recommendation_reason` prose (deterministic backstop; the model is instructed to keep ids only in `selected_evidence_ids`). Scrubbed candidates are recorded in trace.

## Trace Signals

Batch-level: `mode`, `retry_count`, `returned_ratio`, `prompt_budget_mode`, `trim_applied`, `payload_token_estimate`, `attempts`.

Top-level: `reason_generation_trace.reason_generation_failed`, `reason_generation_trace.server_fallback_reasons`.

Per-candidate evidence resolution (`reason_generation_trace.batches[*]` for model output, `reason_generation_trace.selected_evidence[*]` for assembly):
- `provided_evidence_ids` (chunk_ids of the candidate's selector-built relevant pool — these become `recommendation.evidence`)
- `selected_evidence_ids` (chunk_ids the model cited; tracking only — not used to assemble or filter evidence)
- `resolved_evidence_ids` (chunk_ids actually emitted as `recommendation.evidence`; equals `provided_evidence_ids`, or `["profile"]`/`[]` on fallback)
- `fallback` (`none` | `profile` | `empty`)
- `invalid_selected_evidence_ids_by_candidate` (model-returned ids failing the chunk_id codec; trace only)
- `empty_selected_evidence_candidate_ids`, `empty_reason_candidate_ids`, `missing_candidate_ids`
- `truncated_reason_candidate_ids` (reasons truncated to the 320-char server cap)
- `leaked_reason_candidate_ids` (candidates whose prose had a leaked evidence_id stripped)

## Notes

- Evidence id is the `chunk_id` codec, not a positional reference (`paper:N`); batch size, staged execution, and trim behavior are unchanged from the prior iteration.
- Reason generation does not require new runbook steps beyond the chunk-collection readiness checks in [`../operation/RUNBOOK.md`](../operation/RUNBOOK.md).
