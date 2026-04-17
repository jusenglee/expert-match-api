# Reasoner Runtime Policy

Last updated: 2026-04-17

## Scope

This document describes the active runtime policy for recommendation reason generation.

## Active Policy

- evidence selection is server-owned
- the LLM is summary-only
- generation runs in batches of `5`
- the primary attempt uses tool calling
- one compact retry is allowed for partial batch output
- unresolved candidates fall back to server-generated reasons

## Input Contract

Top-level inputs include:

- `query`
- `must_aspects`
- `generic_terms`

Per-candidate inputs include:

- `expert_id`
- `candidate_name`
- `selected_evidence`
- `selected_evidence_summary`
- `retrieval_grounding`
- `do_not_mention`

`do_not_mention` contains:

- other candidate names
- internal evidence ids

## Output Contract

The LLM returns only:

- `expert_id`
- `fit`
- `recommendation_reason`
- `risks`

The LLM does not return or choose `selected_evidence_ids`.

## Retry Policy

Compact retry is triggered when any of these occur:

- `missing_candidate_ids` is non-empty
- `returned_ratio < 1.0`
- empty recommendation reasons are returned

Retry keeps the same server-selected evidence set and only reduces prompt budget.

## Validator Policy

After reason generation, the deterministic validator checks:

- other candidate name leakage
- internal evidence id leakage
- aspect/evidence consistency
- strong claims without direct evidence

Aspect checks use the selector-equivalent evidence scope:

- `title`
- `detail`
- `snippet`

## Trace Fields

Batch trace fields:

- `mode`
- `retry_count`
- `returned_ratio`
- `prompt_budget_mode`
- `selected_evidence_count`
- `retry_triggered`
- `retry_trigger`
- `retry_reason`

Top-level trace fields:

- `reason_generation_trace.batches`
- `reason_generation_trace.selected_evidence`
- `reason_generation_trace.server_fallback_reasons`
- `reason_generation_trace.reason_sync_validator`
