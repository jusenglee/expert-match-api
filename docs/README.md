# NTIS Evaluator Recommendation Service

FastAPI service for NTIS reviewer discovery and recommendation. The service combines Qdrant hybrid retrieval with planner/judge LLM stages and exposes both machine-facing APIs and a local browser playground.

## Current Endpoints

- `POST /recommend`: return final recommendations, reasons, evidence, and data gaps. If no evidence-backed recommendation is available, the endpoint still returns `200` with `recommendations=[]` and explanatory reasons.
- `POST /search/candidates`: return shortlist candidates before the final judge step.
- `POST /feedback`: store operator feedback in SQLite.
- `GET /health`: lightweight process-level health response.
- `GET /health/ready`: runtime readiness response for live dependencies and sample data.
- `GET /playground`: local browser UI for readiness inspection and API smoke testing.
- `ntis-validate-live`: CLI wrapper around the same live readiness validator.

## Readiness Behavior

- `/health/ready` returns `200` when the runtime is fully ready.
- `/health/ready` returns `503` when validation fails, but the JSON body still uses the same top-level fields as the success response: `ready`, `checks`, `issues`, `collection_name`, and `sample_point_id`.
- The live validator scans a bounded set of collection points and selects the most complete sample payload it can find before reporting payload-shape issues. Readiness requires root fields plus publication/project evidence and project date fields; patent evidence remains informational.
- If startup runtime initialization fails before the live validator is available, the app can still start in degraded mode. In that case `/health/ready` returns `503` with `checks.startup_runtime_initialized=false`, and recommendation endpoints remain unavailable until the startup issue is fixed.

## Docs

- [SERVICE_FLOW.md](/D:/Project/python_project/Ntis_person_API/docs/SERVICE_FLOW.md): pipeline walkthrough.
- [CONTRACT.md](/D:/Project/python_project/Ntis_person_API/docs/CONTRACT.md): request/response and payload contract.
- [RUNBOOK.md](/D:/Project/python_project/Ntis_person_API/docs/RUNBOOK.md): local run and troubleshooting checklist.
- [ENVIRONMENT.md](/D:/Project/python_project/Ntis_person_API/docs/ENVIRONMENT.md): environment variables and runtime defaults.

## Runtime Notes

- Qdrant collection: `researcher_recommend_test`
- Search branches: `basic`, `art`, `pat`, `pjt`
- Retrieval mode: dense + sparse + RRF
- Startup bootstrap attempts to repair sparse vector modifiers to `IDF` on existing collections before readiness validation runs.
- Retrieval normalizes legacy Qdrant payloads that encode missing optional list or numeric fields as empty strings (`""`).
- Strict runtime validation blocks heuristic/hash fallbacks from serving live traffic.

## Planner Notes

- The OpenAI-compatible planner is expected to emit a single JSON object matching `PlannerOutput`.
- Invalid planner output falls back to the heuristic planner instead of failing the request immediately.

## Quick Start

1. `python -m pip install -e .[dev]`
2. Prepare Qdrant and point the app at the target collection.
3. Configure the required `NTIS_` environment variables.
4. Load or verify the collection data.
5. Run `ntis-validate-live`
6. Run `uvicorn apps.api.main:app --reload`
7. Check `GET /health/ready`
8. Open `http://127.0.0.1:8000/playground`
9. Run `POST /search/candidates` or `POST /recommend` smoke tests if needed

If `uvicorn` starts but `/health/ready` returns `503`, inspect the structured readiness response before calling recommendation APIs. The browser playground shows the same readiness payload in a details panel.
