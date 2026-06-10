# ADR 0004: evidence 참조를 `chunk_id`로 통일

## 상태

승인 (2026-05-28). 갱신 (2026-06-02, flat payload 계약 v2.1).

## 맥락

v1.x는 LLM이 고른 근거를 `paper:0`/`project:1`/`patent:2` 같은 **배열 인덱스 기반 id**로 참조했다. 이 형식은 직렬화 순서·캡이 바뀌면 의미가 달라지고, 무효/미해결 id를 구분하는 별도 로직이 필요했다. chunk 모델에서는 모든 근거가 이미 불변 식별자 `chunk_id`를 갖는다.

chunk_id 코덱은 `<doc_type>_<숫자doc_id>_c<NNN>`이다(예: `paper_100000045256_c000`). 같은 문서의 doc_id는 `<doc_type>_<숫자doc_id>`(예: `paper_100000045256`), `c` 뒤 chunk_index는 3자리 zero-pad다. doc_type은 5종(paper/patent/project/assessor_activity/specialty)으로 앵커링되므로 `assessor_activity`처럼 `_`가 포함된 prefix도 모호성 없이 파싱된다.

## 결정

- LLM 입력 evidence 풀의 각 항목과 출력 `selected_evidence_ids`는 **`chunk_id`** 를 그대로 사용한다(evidence 참조 id == chunk_id).
- 위치 기반 형식(`paper:N` 등)은 폐기한다.
- ~~서버는 선택된 `chunk_id`로 최종 `recommendation.evidence`를 resolve하고, 풀에 없는/형식 무효 id는 trace에 기록한 뒤 **결정론적 chunk fallback**(후보 최상위 chunk)으로 대체한다.~~ *(아래 2026-06-09 갱신으로 대체됨 — evidence는 `selected_evidence_ids`와 무관하게 선별 풀 전체로 결정론적 조립)*
- 외부 응답 `recommendation.evidence[*]`에 `chunk_id`와 doc_type 문자열 `type`을 노출한다. `type`은 5 doc_type 또는 합성 `profile`(identity family) 중 하나다.

> 갱신(2026-06-09, 구현 정합): 현재 구현은 위 "선택 `chunk_id`로 evidence resolve" 대신, 최종 `recommendation.evidence`를 EvidenceSelector가 family 캡으로 선별한 후보별 relevant chunk 풀 **전체**로 결정론적으로 조립한다. `selected_evidence_ids`는 사유 문장의 인용 힌트이자 trace 기록 용도이며 evidence 조립·필터에는 사용되지 않는다(코덱 형식만 검증, 풀 멤버십 비검증). 사유가 인용한 증거는 항상 evidence에 포함되고(상위집합), 선별 풀이 빈 후보만 profile/빈 fallback으로 대체한다. `chunk_id` 코덱 통일·외부 노출 결정 자체는 유효하다. 상세: [`../../api/REASONER_RUNTIME_POLICY.md`](../../api/REASONER_RUNTIME_POLICY.md).

## 결과

- 근거 식별·dedupe·운영 추적이 단순·안정적이 된다(Qdrant에서 Point ID==`chunk_id`로 직접 조회 가능).
- 외부 계약 변경(BREAKING): [`../../api/EXTERNAL_API_CHANGELOG.md`](../../api/EXTERNAL_API_CHANGELOG.md) B항. 런타임 정책: [`../../api/REASONER_RUNTIME_POLICY.md`](../../api/REASONER_RUNTIME_POLICY.md).
