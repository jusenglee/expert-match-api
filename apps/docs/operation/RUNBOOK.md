# 운영 Runbook

이 문서는 현재 추천 런타임 기준의 설치, 실행, readiness 점검, 로그 확인 절차를 정리한다.

## 1. 설치

```powershell
python -m pip install -e .[dev]
```

## 2. Qdrant 준비

점검 항목:

- Qdrant URL 접근 가능 여부
- 대상 collection 존재 여부
- required named vector 존재 여부
- sparse modifier 상태
- 샘플 point 존재 여부

현재 sparse runtime은 실제 선택된 backend에 따라 modifier 기대값이 달라진다.

- PIXIE / custom SPLADE: sparse modifier `None`
- `Qdrant/bm25` FastEmbed builtin: sparse modifier `IDF`

## 3. Readiness 점검

추천 API 호출 전 다음 순서로 점검한다.

1. `python -m apps.tools.validate_live`
2. `GET /health`
3. `GET /health/ready`

`validate_live`와 `/health/ready`는 같은 sparse runtime resolver를 사용한다.

로컬 PIXIE와 online PIXIE가 모두 실패해 `Qdrant/bm25`로 fallback된 경우에도 서비스는 정상 기동될 수 있다. 이때 modifier 기대값은 `IDF`다.

## 4. 서버 실행

```powershell
uvicorn apps.api.main:app --host 0.0.0.0 --port 8011 --reload
```

운영 중 확인할 핵심 흐름:

- planner가 메타어를 제거하고 `retrieval_core`를 생성
- retrieval이 branch query를 구성하고 후보를 수집
- evidence selector가 전체 후보에서 직접 매치 evidence를 선별
- shortlist gate가 drop / low coverage / generic-only를 분리
- reason generator가 최대 5명씩 batch 실행
- validator가 사유를 검증하고 필요 시 fallback

## 5. Playground / API 확인

브라우저:

```text
http://127.0.0.1:8011/
```

`/search/candidates` 예시:

```powershell
curl -X POST http://127.0.0.1:8011/search/candidates `
     -H "Content-Type: application/json" `
     -d "{\"query\":\"AI 반도체 설계 경험이 있는 후보를 찾아줘\"}"
```

`/recommend` 예시:

```powershell
curl -X POST http://127.0.0.1:8011/recommend `
     -H "Content-Type: application/json" `
     -d "{\"query\":\"AI 반도체 설계 과제 경험이 있는 전문가를 추천해줘\"}"
```

## 6. 로그 확인 포인트

### Planner

INFO 로그:

- `retrieval_core`
- `must_aspects`
- `removed_meta_terms`
- fallback 진입 여부

이 로그로 메타어가 실제 검색어에서 제거됐는지 확인한다.

### Evidence Selector

INFO 로그:

- 후보별 `direct_match_count`
- 후보별 `aspect_coverage`
- 후보별 `generic_only`
- 후보별 `selected_evidence_ids`
- 후보별 `dedup_dropped_count`

이 로그로 반복 evidence, 0매치 후보, aspect 부족 후보를 바로 확인할 수 있다.

### Shortlist Gate

INFO 로그:

- `kept`
- `low_coverage`
- `generic_only`
- `dropped`
- 최종 shortlist candidate ids

이 로그는 왜 특정 후보가 LLM 단계에 들어갔는지 설명하는 기준 로그다.

### Reason Generation

INFO 로그:

- batch index
- batch size
- batch candidate ids
- returned candidate 수
- missing candidate 수
- empty reason candidate 수

WARNING 로그:

- empty reason fallback 발생

### Validator / Fallback

WARNING 로그:

- `reason_sync_validator` fallback 발생
- violation 종류
- fallback에 사용된 evidence ids

주요 violation 종류:

- `other_candidate_name`
- `internal_evidence_id`
- `aspect_title_miss`
- `strong_claim_without_direct_evidence`

## 7. Trace 확인 포인트

추천 응답 trace에서 우선 확인할 필드:

- `planner_trace.removed_meta_terms`
- `planner_trace.must_aspects`
- `planner_trace.generic_terms`
- `reason_generation_trace.evidence_selection`
- `reason_generation_trace.shortlist_gate`
- `reason_generation_trace.selected_evidence`
- `reason_generation_trace.server_fallback_reasons`
- `reason_generation_trace.reason_sync_validator`

이 조합으로 후보 탈락, evidence 선택, 사유 fallback 원인을 대부분 재구성할 수 있다.
## 2026-04-17 observability notes

- Retrieval logs should be read as equal-weight RRF. Do not expect branch/path weight tables in the active runtime.
- Planner logs now separate `removed_meta_terms` from `retained_contextual_terms`.
- Evidence selector logs may include `future_selected_evidence_ids` when a future-dated project is selected.
- Reason generation batch logs now include `retry_triggered` and `retry_reason` for partial-output compact retries.
