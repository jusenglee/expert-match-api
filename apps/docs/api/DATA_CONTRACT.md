# Data Contract

## Planner Output

`PlannerOutput` is the shared contract between planning, retrieval, evidence selection, and recommendation assembly.

```json
{
  "intent_summary": "AI 기반 의료영상 분석 기술 개발 과제 추천",
  "hard_filters": {},
  "include_orgs": [],
  "exclude_orgs": [],
  "task_terms": [],
  "core_keywords": ["의료영상 분석", "AI 기반", "기술"],
  "retrieval_core": ["의료영상 분석", "AI 기반", "기술"],
  "must_aspects": ["의료영상 분석"],
  "generic_terms": ["경험"],
  "role_terms": ["전문가"],
  "action_terms": ["추천"],
  "bundle_ids": [],
  "intent_flags": {
    "review_context": true,
    "review_targets": ["과제 평가"]
  },
  "semantic_query": "AI 기반 의료영상 분석 기술 개발 과제",
  "top_k": 5
}
```

## Active Meanings

- `retrieval_core` is the sparse / lexical retrieval basis (Korean-only).
- `core_keywords` remains a compatibility alias of `retrieval_core`.
- `semantic_query` is a dense-only natural-language query.
- `must_aspects` is the Korean-only semantic quality description and evidence gate fallback. No longer the primary gate basis when `evidence_aspects` is present.
- `evidence_aspects` is the bilingual (Korean + English) evidence matching basis. Takes priority over `must_aspects` in the evidence selector and shortlist gate. See the Evidence Aspects Contract section below.
- `intent_flags.review_context` and `intent_flags.review_targets` capture review/evaluation context such as `과제 평가`, `기술 평가`, `논문 평가`.
- Meta request terms such as `평가위원`, `전문가`, `추천` do not remain in `retrieval_core`, `must_aspects`, or `evidence_aspects`.

## Evidence Aspects Contract

`evidence_aspects` is a new bilingual field in `PlannerOutput` (planner v0.7.0+).

```json
{
  "retrieval_core":    ["의료영상 분석", "AI 기반 의료영상"],
  "must_aspects":      ["의료영상 분석"],
  "evidence_aspects":  ["의료영상 분석", "medical image analysis",
                        "딥러닝", "deep learning",
                        "영상 진단", "image segmentation"]
}
```

Field responsibilities:

- `retrieval_core`: Korean-only, for sparse BM25 retrieval query.
- `must_aspects`: Korean-only, for semantic quality description and fallback.
- `evidence_aspects`: **Bilingual** (Korean + English), for direct evidence text matching.

The evidence selector prefers `evidence_aspects` over `must_aspects`. If empty, it falls back to `must_aspects → retrieval_core → core_keywords`.

The shortlist gate's `_required_aspect_coverage` computes the threshold from the same priority chain the selector uses: `evidence_aspects → must_aspects → retrieval_core → core_keywords`. Threshold = `min(2, len(target_aspects))`.

### Evidence selector search scope by item type

| Item type | Title field searched | Body fields searched |
|---|---|---|
| paper | `publication_title` | `abstract`, `journal_name`, `korean_keywords`, `english_keywords` |
| project | `display_title` (Korean-first) | `project_title_english` (supplement), `research_objective_summary`, `research_content_summary`, `managing_agency` |
| patent | `intellectual_property_title` | `application_registration_type`, `application_country` |

`project_title_english` is added to body_parts so that English evidence_aspects can match English project titles even when the Korean title takes precedence in display_title.

## Query Builder Contract

`QueryTextBuilder` compiles two modalities per branch:

- `stable_dense`
- `stable_sparse`
- `expanded_dense`
- `expanded_sparse`

The public `stable` / `expanded` strings remain sparse-oriented summaries for trace compatibility.

Priority rules:

- Sparse base: `retrieval_core -> core_keywords -> raw query`
- Dense base: `semantic_query -> retrieval_core -> core_keywords -> raw query`

Path model:

- Stable path: `stable_dense + stable_sparse`
- Expanded path: only the modality that actually changed is expanded; the other modality reuses the stable value

Branch context model:

- Branch hints are **not** appended to query text inside QueryTextBuilder.
- Branch context is carried exclusively by the branch-specific instruct prefix in `QdrantHybridRetriever`.
- This keeps query text clean for embedding (no awkward `keyword_list\nbranch_hints` suffix).

Expansion policy for dense when `semantic_query` is active:

- Expanded keywords are **not** appended to `semantic_query`-based dense text.
- Appending technical keyword suffixes to a natural-language sentence degrades embedding quality.
- Expansion applies only to the sparse modality when `dense_base_source == "semantic_query"`.

## Evidence Selector Contract

`KeywordEvidenceSelector` selects only direct-match evidence.

Selection rules:

- direct lexical match only
- dedup key: `normalized title + year`
- max `2` items per aspect
- max `4` total items per candidate
- future-dated projects are allowed

Output bundle fields:

- `matched_aspects`
- `matched_generic_terms`
- `direct_match_count`
- `aspect_coverage`
- `generic_only`
- `dedup_dropped_count`
- `future_selected_evidence_ids`

`aspect_coverage` is phrase-based, not token-count based.

## Recommendation Contract

`RecommendationDecision` is assembled by the server.

- `evidence` comes from server-selected evidence only
- `recommendation_reason` may come from the LLM or a server fallback
- `fit` remains the LLM summary label, but quality control is enforced by retrieval and deterministic gates

LLM input per candidate includes both `selected_evidence` and `retrieval_grounding`:

```json
{
  "retrieval_grounding": {
    "primary_branch": "art",
    "final_score": 0.0312,
    "branch_matches": [
      {"branch": "art", "rank": 2, "score": 0.021},
      {"branch": "pjt", "rank": 5, "score": 0.018}
    ]
  }
}
```

The LLM uses `retrieval_grounding` to:
- Assign `fit` level (높음 if final_score is high and 3+ branches match)
- Note multi-branch retrieval breadth in natural language when direct evidence is limited
- Must NOT quote raw score numbers or rank numbers in the reason text

## Validator Contract

The reason sync validator checks:

- other candidate names leaking into the reason
- internal evidence ids leaking into the reason
- `must_aspects` touching the same evidence scope the selector used (Korean-only; the validator checks Korean reason text against Korean aspects)
- strong claims without direct evidence

Note: the validator intentionally uses `must_aspects` (Korean-only) for the aspect scope check, not `evidence_aspects`. The reason text is generated in Korean, so bilingual English terms from `evidence_aspects` would produce false positives in the scope check.

Evidence scope for aspect checks is:

- `title`
- `detail`
- `snippet`

This avoids selector/validator disagreement caused by title-only checks.

## Trace Contract

Important trace fields:

- `planner_trace.removed_meta_terms`
- `planner_trace.retained_contextual_terms`
- `planner_trace.raw_must_aspects`
- `planner_trace.normalized_must_aspects`
- `planner_trace.pruned_must_aspects`
- `planner_trace.must_aspect_prune_reasons`
- `planner_trace.intent_flags`
- `reason_generation_trace.evidence_selection`
- `reason_generation_trace.shortlist_gate`
- `reason_generation_trace.reason_sync_validator`
- `reason_generation_trace.server_fallback_reasons`

## 2026-04-17 Contract Notes

- Retrieval score fusion is equal-weight RRF (schema version: `v3_branch_instruct_prefix`).
- `semantic_query` is active for the dense retrieval base (priority over `retrieval_core`).
- When `dense_base_source == "semantic_query"`, expansion keywords are NOT appended to the dense query.
- Branch query hints removed from query text; branch context moved to retriever-side instruct prefix.
- Each branch (`basic`, `art`, `pat`, `pjt`) now has a dedicated e5-instruct prefix in the retriever.
- `must_aspects` is no longer a direct copy of `retrieval_core`.
- Contextual evaluation phrases are tracked in `intent_flags`, not `must_aspects`.
- Phrase normalization is shared between selector, gate, and validator.
- LLM reasoner can reference `retrieval_grounding` for `fit` assignment and breadth notes.
