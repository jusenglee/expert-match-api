# ADR 0005: 융합·가중·리랭커 재검토 결정

## 상태

승인 (2026-05-28). 갱신 (2026-06-02, flat payload 계약 v2.1 — 결정 내용 불변, 단일 벡터 명칭 정합만 반영). chunk 재설계 시 "일부 고정 제약 재검토 개방"(사용자 결정)에 따른 명문화.

## 맥락

chunk 단위 score가 생기면서 weighted fusion과 후보(연구자) cross-encoder 리랭커가 기술적으로 더 자연스러워졌다. v1.x의 고정 제약("weighted RRF 금지", "리랭커를 최종 판단 핵심에 두지 않는다")을 그대로 둘지, 새 데이터에 맞춰 풀지 재검토했다. 동시에 v1.x 반복에서 얻은 "판단을 한 레이어에 몰면 경계가 무너진다"는 교훈이 여전히 유효하다.

## 결정

1. **Qdrant 융합 = equal RRF 유지.** 단일 컬렉션 내 dense(`vector_e5i`) + sparse(`vector_splade`) 융합은 `FusionQuery(RRF)` 고정. Qdrant 단 weighted RRF/score 가중합은 쓰지 않는다(스케일 차이에 강건, 재현성). doc_type은 named vector가 아니라 payload 필터이므로 융합 자체는 doc_type과 무관하게 equal하다.
2. **집계 가중(doc_type prior) = 옵트인, 기본 equal.** 연구자 집계 단계의 family/doc_type 가중은 앱단 랭크 누적 가중으로 조정 가능하되 기본 equal(`NTIS_DOC_TYPE_PRIORS` 미설정 → None). v1.x도 앱단 `BRANCH_WEIGHTS`를 이미 썼으므로 일관적이다.
3. **후보 cross-encoder 리랭커 = 기본 OFF, 옵트인 실험만.** 후보 순위의 척추는 RRF로 유지. `NTIS_CANDIDATE_RERANKER` 기본값은 `off`이며, `band`로 켜더라도 **score 동률 밴드 내 재배열만** 허용하고 탈락·생성은 금지.
4. **evidence cross-encoder 리랭커 = 1급 채택.** 후보 **내부** chunk을 query 관련도로 재랭크해 family별 top-N만 LLM에 전달(모델 부재 시 lexical 강등). 목적은 토큰 절감·grounding 품질·정규화이며 후보 순위에 영향 없음.

## 결과

- "재검토 개방"의 결론은 **레버를 설계에 명시하고 끌 수 있게 두되, 추천 순위의 토대는 단순·결정론적(RRF)으로 유지**다.
- 위반 판단 기준: 후보 순위를 RRF가 아닌 가중합/리랭커로 결정하거나, LLM이 후보를 재정렬·탈락시키면 기준선 위반.
- 관련: [`../DESIGN_GUIDELINES.md §6`](../DESIGN_GUIDELINES.md).
