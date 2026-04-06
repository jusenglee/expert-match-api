# Contract

## Planner Output

`PlannerOutput`

```json
{
  "intent_summary": "Recommend AI semiconductor reviewers",
  "hard_filters": {
    "degree_slct_nm": "박사",
    "art_sci_slct_nm": "SCIE",
    "art_recent_years": 5,
    "project_cnt_min": 1
  },
  "exclude_orgs": ["A Organization"],
  "soft_preferences": ["recent achievements"],
  "branch_query_hints": {
    "basic": "profile-focused query",
    "art": "publication-focused query",
    "pat": "patent-focused query",
    "pjt": "project-focused query"
  },
  "top_k": 5
}
```

Planner behavior:

- The LLM planner must return a single JSON object matching `PlannerOutput`.
- `branch_query_hints` must be an object with `basic`, `art`, `pat`, and `pjt` keys, not a list.
- If the LLM output is not valid `PlannerOutput`, the runtime falls back to the heuristic planner.

## Judge Output

`JudgeOutput`

```json
{
  "recommended": [
    {
      "rank": 1,
      "expert_id": "11008395",
      "name": "Hong Gildong",
      "fit": "높음",
      "reasons": ["Publication evidence is strong."],
      "evidence": [
        {"type": "paper", "title": "Example paper", "date": "2024-09", "detail": "SCIE"},
        {"type": "project", "title": "Example project", "date": "2020-04-06", "detail": "Lead researcher"}
      ],
      "risks": ["Patent evidence is missing."]
    }
  ],
  "not_selected_reasons": ["Other candidates had broader evidence coverage."],
  "data_gaps": ["Patent evidence is missing."]
}
```

Judge behavior:

- If the LLM judge output is not valid `JudgeOutput`, the runtime falls back to the heuristic judge.

## API

### `POST /recommend`

Request:

```json
{
  "query": "Recommend reviewers with strong AI semiconductor paper and project histories",
  "top_k": 5,
  "filters_override": {
    "degree_slct_nm": "박사",
    "art_sci_slct_nm": "SCIE"
  },
  "exclude_orgs": ["A Organization"]
}
```

Response fields:

- `intent_summary`
- `applied_filters`
- `searched_branches`
- `retrieved_count`
- `recommendations`
- `data_gaps`
- `not_selected_reasons`
- `trace`

Behavior:

- `POST /recommend` returns `200` even when no final recommendation can be produced.
- In that case `recommendations` is an empty list and the reason is described in `not_selected_reasons` and/or `data_gaps`.
- Recommendations without evidence are excluded from the final response instead of causing a server error.

### `POST /search/candidates`

Response fields:

- `intent_summary`
- `applied_filters`
- `searched_branches`
- `retrieved_count`
- `candidates`
- `trace`

### `POST /feedback`

Stores operator selections and notes in SQLite.

### `GET /health/ready`

Response fields:

- `ready`
- `checks`
- `issues`
- `collection_name`
- `sample_point_id`

Behavior:

- `200` means the runtime is ready.
- `503` means readiness failed, but the body still uses the same top-level fields as the success response.
- Failure payloads are returned directly as `ReadinessResponse`; they are not wrapped in `detail`.
- If runtime initialization fails before the live validator exists, `/health/ready` returns `503` with `checks.startup_runtime_initialized=false` and the startup error string in `issues`.

## Required Payload Structure

The representative sample point selected by readiness validation must contain at least:

- root fields:
  - `basic_info`
  - `researcher_profile`
- nested evidence arrays:
  - `publications[]`
  - `research_projects[]`
- project date fields on every `research_projects[]` item:
  - `project_start_date`
  - `project_end_date`
  - `reference_year`

Readiness validation scans a bounded set of collection points and selects the most complete sample payload it can find before evaluating these requirements.

Legacy payload note:

- When Qdrant stores optional list or numeric fields as empty strings (`""`), the runtime normalizes them at read time.
- Required identifier/title fields are not auto-repaired and may still fail validation if malformed.

## Readiness Failure Conditions

Readiness is considered failed when any of these checks fail:

- LLM backend connectivity
- embedding backend connectivity
- Qdrant collection lookup
- required dense named vectors
- required sparse named vectors
- sparse vector IDF modifier
- required payload indexes
- representative sample point presence
- sample payload object shape
- `publications[]` or `research_projects[]` missing or empty
- missing project date fields on `research_projects[]`
