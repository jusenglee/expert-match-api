# ADR 0003: 모든 doc_type을 항상 검색 가능하게 둔다

## 상태

승인 (2026-05-28). 갱신 (2026-06-02, flat payload 계약 v2.1).

## 맥락

v1.x는 4브랜치(basic/art/pat/pjt)를 모든 질의에서 항상 검색했다. chunk 모델에서는 브랜치 대신 **5개 doc_type(4 family)**이 검색 분기다. 단일 dense+sparse 벡터로 회수하고 **`doc_type`은 payload 필터**로 분기한다(named vector 분기가 아니다). "어떤 doc_type을 검색할지"를 planner가 켜고 끄게 하면 recall이 흔들리고, 특정 근거(특히 신규 신호인 평가이력 `assessor_activity`)가 누락된 후보가 검색 단계에서 사라질 수 있다.

실제 적재 데이터의 doc_type 분포는 paper 64.6% / patent 18.7% / assessor_activity 9.7% / project 6.5% / specialty 0.5%로 매우 불균형하다(소수 family 보호가 더 중요해지는 근거).

## 결정

- 모든 doc_type은 **항상 검색 가능**하다(운영 축소는 `NTIS_RETRIEVAL_DOC_TYPES` 화이트리스트로만). 미설정 시 5종(paper/patent/project/assessor_activity/specialty) 전부.
- planner는 doc_type on/off를 결정하지 않고, intent flag로 **prior 힌트**만 줄 수 있다.
- doc_type 중요도는 weighted RRF가 아니라 **앱단 집계 prior(기본 equal, 옵트인 `NTIS_DOC_TYPE_PRIORS`)** + LLM 비교로 반영한다([`../DESIGN_GUIDELINES.md §6.2`](../DESIGN_GUIDELINES.md)).

## 결과

- recall 안정성 유지, 평가이력 family(`assessor_activity`)가 누락 없이 후보·evidence에 반영.
- doc_type on/off 로직을 planner에서 분리해, 필터/집계 버그가 doc_type 선택 문제와 혼동되지 않는다.
