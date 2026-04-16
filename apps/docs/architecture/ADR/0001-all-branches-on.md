# ADR 0001: Always Search All Branches

## 상태

승인

## 결정

`basic`, `art`, `pat`, `pjt` branch를 모든 질의에서 항상 검색한다.

## 배경

평가위원 추천은 특정 근거 한 종류만으로 품질을 보장하기 어렵다. branch를 planner가 on/off 하면 recall이 흔들리고, 특정 근거가 누락된 후보가 검색 단계에서 사라질 수 있다.

## 결과

- planner는 branch 선택이 아니라 query hint 생성만 담당한다.
- retrieval은 4개 branch를 모두 조립한다.
- branch 중요도는 weighted RRF가 아니라 query rewriting과 judge 판단으로 반영한다.
