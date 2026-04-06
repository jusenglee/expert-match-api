# RUNBOOK

## 1. Install

```powershell
python -m pip install -e .[dev]
```

## 2. Prepare Qdrant

- Start Qdrant and make sure the configured URL is reachable.
- Use the target collection `researcher_recommend_test`, or override it through `NTIS_QDRANT_COLLECTION_NAME`.
- App startup will attempt to repair sparse vector modifiers to `IDF` on the configured collection before readiness validation runs.

## 3. Readiness Checks

Run the checks in this order:

1. `ntis-validate-live`
2. `GET /health`
3. `GET /health/ready`

If `ntis-validate-live` or `/health/ready` fails, inspect these items before calling recommendation APIs:

- collection existence
- required dense and sparse named vectors
- sparse vector IDF modifier
- required payload indexes
- representative sample point presence within the scanned window
- sample payload branches on the selected representative point: `publications[]`, `research_projects[]`
- project date fields: `project_start_date`, `project_end_date`, `reference_year`

The validator scans a bounded set of points and chooses the most complete sample payload it can find. If the collection contains a valid representative point later in the scan window, readiness will use that point instead of failing on the first incomplete record. `intellectual_properties[]` is still surfaced in `checks` when present, but it is not a readiness failure condition.

`/health/ready` now returns a structured `503` body instead of wrapping the report in `detail`. If startup runtime initialization fails before the validator is created, the service can still boot in degraded mode and `/health/ready` will expose `checks.startup_runtime_initialized=false`.

## 4. Run the Server

```powershell
uvicorn apps.api.main:app --reload
```

Notes:

- `NTIS_EMBEDDING_BACKEND=local` requires the bundled `multilingual-e5-large-instruct` directory to include `modules.json`, `1_Pooling/config.json`, and `2_Normalize/`.
- Legacy Qdrant payloads that store missing optional list or numeric values as `""` are normalized during retrieval. Malformed required fields are still treated as invalid payloads and skipped.
- If startup fails after the process begins listening, `POST /recommend`, `POST /search/candidates`, and `POST /feedback` remain unavailable until the startup issue is resolved.

## 5. Browser Playground

Open:

```text
http://127.0.0.1:8000/playground
```

Playground flow:

1. Check the readiness badge at the top.
2. Expand `Readiness details` when the status is red or you need the raw validator payload.
3. Choose `/recommend` or `/search/candidates`.
4. Enter a query and optionally fill `top_k`, `filters_override`, and `exclude_orgs`.
5. Inspect the response cards in the conversation panel.

`filters_override` must be a JSON object. `exclude_orgs` accepts one organization per line or comma-separated values.

## 6. curl Smoke Tests

```powershell
curl -X POST http://127.0.0.1:8000/search/candidates -H "Content-Type: application/json" -d "{\"query\":\"Recommend reviewers with recent SCIE publications in AI semiconductors\"}"
```

```powershell
curl -X POST http://127.0.0.1:8000/search/candidates -H "Content-Type: application/json" -d "{\"query\":\"Recommend experts with registered patents and commercialization experience\"}"
```

```powershell
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{\"query\":\"Recommend reviewers with AI semiconductor project delivery experience\"}"
```

```powershell
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{\"query\":\"Recommend recent SCIE and project experts in AI semiconductors\",\"exclude_orgs\":[\"A Organization\"]}"
```

## 7. Success Criteria

- `/health/ready` returns `200`
- `/playground` loads and can refresh readiness
- `/search/candidates` returns shortlist candidates
- `/recommend` returns ranked recommendations or a structured empty result with reasons
- evidence arrays are not empty for the selected path
- excluded organizations do not appear in the results

## 8. Logging

Useful runtime logs include:

- planner JSON
- planner fallback activation details
- judge fallback activation details
- branch query text
- applied hard filters
- exclude_orgs
- candidate IDs
- invalid payload skip warnings with point IDs
- recommendation evidence summary
- feedback persistence result
- readiness validation failures and startup initialization errors
