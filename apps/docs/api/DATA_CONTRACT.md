# 데이터 계약 (Data Contract)

## Planner Output

`PlannerOutput`은 검색과 추천 파이프라인의 공용 입력 계약이다.

```json
{
  "intent_summary": "AI 반도체 설계 경험이 있는 후보 추천",
  "hard_filters": {},
  "include_orgs": [],
  "exclude_orgs": [],
  "task_terms": [],
  "core_keywords": ["AI 반도체", "설계 경험"],
  "retrieval_core": ["AI 반도체", "설계 경험"],
  "must_aspects": ["AI 반도체", "설계 경험"],
  "generic_terms": [],
  "role_terms": ["전문가"],
  "action_terms": ["추천"],
  "bundle_ids": [],
  "semantic_query": "",
  "top_k": 5
}
```

현재 계약 규칙은 다음과 같다.

- `retrieval_core`가 실제 검색 기준이다.
- `must_aspects`는 shortlist gate와 evidence selector가 쓰는 질의 충족 기준이다.
- `generic_terms`는 범용 요청어 추적용이며 직접 검색어로 쓰지 않는다.
- `core_keywords`는 하위 호환 alias다.
- `평가위원`, `전문가`, `추천` 같은 메타어는 `retrieval_core`, `must_aspects`, `core_keywords`에 남기지 않는다.

## Query Builder Contract

`QueryTextBuilder`는 planner 출력에서 branch query를 만든다.

- 우선순위: `retrieval_core -> core_keywords -> raw query`
- planner가 정제한 검색어가 있으면 raw query를 다시 검색어로 쓰지 않는다.
- 모든 branch는 같은 stable base를 공유하고, branch hint만 뒤에 붙는다.

## Evidence Selector Contract

`KeywordEvidenceSelector`는 `CandidateCard` 미리보기에서 후보별 relevant evidence를 고른다.

### 입력

- `candidates: list[CandidateCard]`
- `plan: PlannerOutput`

### 출력

```json
{
  "expert_id": "12345678",
  "papers": [],
  "projects": [],
  "patents": [],
  "matched_aspects": ["AI 반도체", "설계 경험"],
  "matched_generic_terms": [],
  "direct_match_count": 2,
  "aspect_coverage": 2,
  "generic_only": false,
  "dedup_dropped_count": 1
}
```

선택 규칙은 다음과 같다.

- 직접 매치 evidence만 relevant pool에 포함
- dedup 기준은 `normalized title + year`
- aspect별 최대 2건
- 후보당 전체 최대 4건

## Reason Generator Contract

공개 응답 스키마는 유지되지만, 내부 LLM 계약은 바뀌었다.

### LLM 입력

- `must_aspects`
- `generic_terms`
- 후보별 `candidate_name`
- 후보별 `selected_evidence`
- 후보별 `selected_evidence_summary`
- 후보별 `retrieval_grounding`
- 후보별 `do_not_mention`

### LLM 출력

```json
{
  "items": [
    {
      "expert_id": "12345678",
      "fit": "높음",
      "recommendation_reason": "직접 evidence를 요약한 추천 사유",
      "risks": []
    }
  ],
  "data_gaps": []
}
```

중요한 변경점은 다음과 같다.

- `selected_evidence_ids`는 더 이상 LLM 출력 필수 필드가 아니다.
- evidence 선택 책임은 서버가 가진다.
- LLM은 요약만 한다.

## Recommendation Assembly Contract

`RecommendationService`는 최종 `RecommendationDecision`을 서버에서 조립한다.

- `evidence`는 항상 selector가 확정한 evidence에서 나온다.
- LLM reason이 비면 `selected_evidence` 기반 fallback 사유를 만든다.
- LLM reason이 validator를 통과하지 못하면 `reason_sync_validator` fallback 사유로 치환한다.

공개 응답 예시는 기존과 동일하다.

```json
{
  "rank": 1,
  "expert_id": "12345678",
  "name": "홍길동",
  "fit": "높음",
  "recommendation_reason": "직접 evidence 기반 추천 사유",
  "evidence": [
    {
      "type": "project",
      "title": "AI 반도체 설계 플랫폼",
      "date": "2025-01-01",
      "detail": "주관기관: NTIS"
    }
  ],
  "risks": [],
  "rank_score": 95.8
}
```

## Trace Contract

추천 응답의 trace에서 이번 변경으로 중요해진 필드는 다음과 같다.

- `planner_trace.removed_meta_terms`
- `planner_trace.must_aspects`
- `planner_trace.generic_terms`
- `planner_trace.fallback_terms`
- `reason_generation_trace.evidence_selection`
- `reason_generation_trace.shortlist_gate`
- `reason_generation_trace.selected_evidence`
- `reason_generation_trace.server_fallback_reasons`
- `reason_generation_trace.reason_sync_validator`

`selected_evidence_ids`를 기준으로 evidence를 해석하는 구형 계약은 활성 계약이 아니다.
## 2026-04-17 contract deltas

- Retrieval score fusion is equal-weight RRF. Branch/path weight tables are not part of the active contract.
- `평가` is contextual: `평가위원`, `전문가 추천`, `추천` stay out of retrieval keywords, while phrases such as `과제 평가`, `기술 평가`, `논문 평가` may remain in `retrieval_core` and `must_aspects`.
- `RelevantEvidenceBundle` may expose `future_selected_evidence_ids` for observability. Future-dated project evidence is still allowed.
- Reasoner batch trace now records `retry_triggered`, `retry_trigger`, and `retry_reason` when compact retry is scheduled for partial output.
