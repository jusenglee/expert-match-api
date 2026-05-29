# ADR 0003: 모든 doc_type을 항상 검색 가능하게 둔다

## 상태

승인 (2026-05-28).

## 맥락

v1.x는 4브랜치(basic/art/pat/pjt)를 모든 질의에서 항상 검색했다. chunk 모델에서는 브랜치 대신 11개 doc_type(4 family)이 검색 분기다. "어떤 doc_type을 검색할지"를 planner가 켜고 끄게 하면 recall이 흔들리고, 특정 근거(특히 신규 신호인 평가이력)가 누락된 후보가 검색 단계에서 사라질 수 있다.

## 결정

- 모든 doc_type은 **항상 검색 가능**하다(운영 축소는 `NTIS_RETRIEVAL_DOC_TYPES` 화이트리스트로만).
- planner는 doc_type on/off를 결정하지 않고, intent flag로 **prior 힌트**만 줄 수 있다.
- doc_type 중요도는 weighted RRF가 아니라 **앱단 집계 prior(기본 equal, 옵트인)** + LLM 비교로 반영한다([`../DESIGN_GUIDELINES.md §6.2`](../DESIGN_GUIDELINES.md)).

## 결과

- recall 안정성 유지, 평가이력 family가 누락 없이 후보·evidence에 반영.
- doc_type on/off 로직을 planner에서 분리해, 필터/집계 버그가 doc_type 선택 문제와 혼동되지 않는다.
