# Service Flow

## 1. Planner

`RecommendationService.search_candidates()` starts with the planner.

Planner responsibilities:

- normalize the query into `PlannerOutput`
- strip meta request terms from retrieval fields
- produce `retrieval_core` (Korean-only, for sparse BM25)
- produce dense-only `semantic_query`
- derive pruned `must_aspects` (Korean-only, semantic quality description)
- produce bilingual `evidence_aspects` (Korean + English, for actual evidence text matching)
- record review/evaluation context in `intent_flags.review_context` and `intent_flags.review_targets`

Field responsibilities are separated by design:

| Field | Language | Purpose |
|---|---|---|
| `retrieval_core` | Korean only | Sparse BM25 retrieval query |
| `must_aspects` | Korean only | Semantic quality fallback and gate |
| `evidence_aspects` | Korean + English | Direct text matching in evidence selector |

When the LLM does not populate `evidence_aspects`, the selector falls back to `must_aspects → retrieval_core → core_keywords`.

Planner does not:

- rank candidates
- choose evidence
- apply branch/path weights

## 2. Query Builder

`QueryTextBuilder` creates per-branch query text for two modalities.

- sparse uses `retrieval_core`
- dense prefers `semantic_query`

The runtime still uses only two paths:

- `stable`
- `expanded`

Expanded execution is runtime-driven. If the expanded modality text is identical to the stable one, that modality is reused.

## 3. Retrieval

`QdrantHybridRetriever` executes hybrid retrieval on `basic`, `art`, `pat`, `pjt`.

Per path:

- dense prefetch (query encoded with branch-specific instruct prefix)
- sparse prefetch
- branch-local RRF fusion

Across paths / branches:

- equal-weight RRF

There is no active branch/path weighting table in the runtime.

### 3-1. Branch-specific Instruct Prefix

`multilingual-e5-large-instruct` requires asymmetric encoding: the query side receives a
task-specific `Instruct: ...\nQuery: ` prefix; the document side is indexed as raw text.

Each branch uses a different prefix to align the query embedding direction with the branch's
document space:

- `basic`: academic background, affiliation, technical expertise
- `art`: academic publications and research papers
- `pat`: patents and inventions
- `pjt`: government-funded research projects

Branch hints are no longer appended to the query text itself. The instruct prefix in the
retriever is the sole carrier of branch context for the dense modality.

## 4. `/search/candidates`

`/search/candidates` stops after retrieval and card assembly.

It returns:

- planner output
- compiled branch queries
- retrieval score traces
- candidate cards

It does not run:

- evidence selector
- shortlist gate
- reason generation
- validator

## 5. `/recommend`

`/recommend` builds on top of retrieval.

### 5-1. Evidence Selection

`KeywordEvidenceSelector` runs over all retrieved candidates first.

Rules:

- direct match evidence only
- phrase-based aspect matching
- `title + year` dedup
- max `2` items per aspect
- max `4` items total

#### Bilingual aspect matching

The selector uses `evidence_aspects` as the primary match source. Because the research database
contains mixed-language data (English paper titles/abstracts, Korean project summaries), `evidence_aspects`
is bilingual so evidence in either language can be matched.

| Item type | Title field | Body fields searched |
|---|---|---|
| paper | `publication_title` | `abstract`, `journal_name`, `korean_keywords`, `english_keywords` |
| project | `display_title` (Korean-first) | `project_title_english` (supplement), `research_objective_summary`, `research_content_summary`, `managing_agency` |
| patent | `intellectual_property_title` | `application_registration_type`, `application_country` |

`project_title_english` is searched separately from `display_title` so that English aspects can
match English project titles even when the Korean title takes precedence in display.

Fallback priority when `evidence_aspects` is empty: `must_aspects → retrieval_core → core_keywords`.

### 5-2. Deterministic Shortlist Gate

Gate order:

1. `direct_match_count == 0` → drop
2. `aspect_coverage < min(2, len(target_aspects))` → demote
3. `generic_only == true` → bottom group

`aspect_coverage` uses phrase count, not token count.

`target_aspects` for the gate threshold follows the same priority chain as the selector:
`evidence_aspects → must_aspects → retrieval_core → core_keywords`.

### 5-3. Reason Generation

Reason generation runs in batches of `5`.

The LLM receives two data sources per candidate:

- `selected_evidence`: server-chosen direct-match evidence items (papers, patents, projects)
- `retrieval_grounding`: compact retrieval signals — `primary_branch`, `final_score`, `branch_matches`

The LLM:

- does not choose evidence
- summarizes `selected_evidence` as the primary basis for `recommendation_reason`
- uses `retrieval_grounding` to inform the `fit` level assignment (높음/중간/보통)
- may mention retrieval breadth (e.g., multi-branch match) in natural language when direct evidence is limited
- must not quote raw numeric scores or rank numbers in the reason text

If a batch returns partial output, the runtime runs one compact retry before falling back to server-generated reasons.

### 5-4. Reason Sync Validator

Validator checks:

- other candidate names
- internal evidence ids
- aspect/evidence consistency
- strong claims without direct evidence

Aspect validation uses the same evidence scope as the selector:

- `title`
- `detail`
- `snippet`

This avoids selector/validator disagreement from title-only checks.

## 6. Fallback Paths

Fallbacks exist at two points:

- selected-evidence fallback when the LLM omits or empties a reason
- validator fallback when the LLM returns a reason that fails deterministic checks

If every candidate is removed by the shortlist gate, the response returns no recommendations with a gate-specific reason.

## 7. Trace and Logging

Key runtime trace fields:

- `planner_trace.raw_must_aspects`
- `planner_trace.normalized_must_aspects`
- `planner_trace.pruned_must_aspects`
- `planner_trace.intent_flags`
- `reason_generation_trace.evidence_selection.aspect_source` — which field the selector used (`evidence_aspects` / `must_aspects` / `retrieval_core` / `core_keywords`)
- `reason_generation_trace.shortlist_gate.aspect_source` — which field the gate threshold was derived from
- `reason_generation_trace.evidence_selection`
- `reason_generation_trace.shortlist_gate`
- `reason_generation_trace.reason_sync_validator`

Key logs:

- planner contract summary
- branch query source summary
- selector candidate summary
- shortlist gate outcome
- reason batch retry/fallback
- validator fallback
