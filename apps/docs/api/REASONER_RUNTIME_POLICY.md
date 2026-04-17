# Reasoner Runtime Policy

Last updated: 2026-04-16

## Scope

이 문서는 추천 사유 생성의 현재 런타임 정책을 기록한다.

핵심 방향은 다음 한 줄이다.

- evidence 선택은 서버가 한다.
- LLM은 요약만 한다.

## Active Policy

- reason generation은 최대 `5`명씩 순차 batch로 실행된다.
- selector가 후보별 `selected_evidence`를 먼저 확정한다.
- LLM은 확정된 evidence만 보고 `fit`, `recommendation_reason`, `risks`를 생성한다.
- 1차는 tool calling, 2차는 compact JSON retry, 둘 다 실패하면 서버 fallback을 사용한다.

## Prompt Budgeting

### primary payload

- `selected_evidence` 최대 `4`건
- `matched_keywords` 최대 `5`개
- snippet 최대 `1000`자
- detail 최대 `200`자

### retry payload

- `selected_evidence` 최대 `2`건
- `matched_keywords` 최대 `3`개
- snippet 최대 `280`자
- detail 최대 `120`자

retry에서는 evidence 선택 기준을 바꾸지 않고 payload만 줄인다.

## LLM Input Contract

배치 payload는 다음 정보를 포함한다.

- top-level
  - `query`
  - `must_aspects`
  - `generic_terms`
- per candidate
  - `expert_id`
  - `candidate_name`
  - `selected_evidence`
  - `selected_evidence_summary`
  - `retrieval_grounding`
  - `do_not_mention`

`do_not_mention`에는 다른 후보 이름과 내부 evidence id가 들어간다.

## LLM Output Contract

```json
{
  "items": [
    {
      "expert_id": "12345678",
      "fit": "높음",
      "recommendation_reason": "선택된 evidence를 요약한 사유",
      "risks": []
    }
  ],
  "data_gaps": []
}
```

현재 정책상 다음은 금지한다.

- `selected_evidence_ids`를 새로 생성하는 것
- `do_not_mention`에 포함된 이름이나 id를 사유에 쓰는 것
- `selected_evidence` 밖의 내용을 근거처럼 단정하는 것

## Reason Sync Validator

LLM 출력 뒤에는 deterministic validator가 항상 한 번 더 돈다.

검사 항목은 다음과 같다.

- 다른 후보 이름 포함 여부
- 내부 evidence id 노출 여부
- 핵심 aspect가 evidence title에 전혀 닿지 않는지
- direct evidence 없이 과한 추천 문장이 생성됐는지

검증 실패 시 recommendation_reason은 서버 fallback 문장으로 치환된다.

## Trace Signals

배치 trace 필드

- `mode`
- `retry_count`
- `returned_ratio`
- `prompt_budget_mode`
- `trim_applied`
- `payload_token_estimate`
- `selected_evidence_count`
- `attempts`

최상위 trace 필드

- `reason_generation_trace.batches`
- `reason_generation_trace.selected_evidence`
- `reason_generation_trace.server_fallback_reasons`
- `reason_generation_trace.reason_sync_validator`

validator trace 필드

- `checked_candidate_ids`
- `fallback_count`
- `fallback_ratio`
- `violations`

## Logging Policy

INFO 로그

- batch 시작/종료
- returned/missing candidate 수
- retry 여부

WARNING 로그

- empty reason fallback
- reason sync validator fallback

이 fallback 비율은 upstream 품질 문제를 추적하는 운영 지표로 본다.
## 2026-04-17 retry policy update

- Compact retry is no longer exception-only.
- The retry condition is any partial batch outcome: `missing_candidate_ids`, empty recommendation reasons, or `returned_ratio < 1.0`.
- Batch trace now exposes `retry_triggered`, `retry_trigger`, and `retry_reason`.
