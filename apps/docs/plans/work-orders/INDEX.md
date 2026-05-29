# 마이그레이션 업무 지시서 (Work Orders) — INDEX

**대상:** NTIS 평가위원 추천 API · v1.x → v2.0 (chunk 재설계) 마이그레이션
**상위 문서:** [`../MIGRATION_PLAN.md`](../MIGRATION_PLAN.md) (Phase A~D 로드맵) · [`../../architecture/DATA_MODEL.md`](../../architecture/DATA_MODEL.md) (chunk 스키마) · [`../../architecture/DESIGN_GUIDELINES.md`](../../architecture/DESIGN_GUIDELINES.md) (고정 제약) · ADR [0002](../../architecture/ADR/0002-chunk-level-point-model.md)/[0003](../../architecture/ADR/0003-all-doc-types-searchable.md)/[0004](../../architecture/ADR/0004-chunk-id-evidence-contract.md)/[0005](../../architecture/ADR/0005-fusion-and-reranker-reconsidered.md)

> **출발 상황(2026-05-29 코드 직접 확인):** 설계 문서는 v2.0 chunk 모델로 전면 재작성 완료. **코드/Qdrant는 아직 v1.x 그대로.** `apps/` 코드 전역에서 `chunk_id`/`dense_e5i`/`sparse_splade`/`doc_type`/`ntis_researcher_chunks` 출현 = **0회**. 마이그레이션은 코드 차원에서 **미착수**다. 추가로 `tests/test_cross_encoder_evidence_selector.py`(untracked)가 존재하지 않는 `CrossEncoderEvidenceSelector`를 import해 **현재 pytest suite가 red**다 → WO-0에서 최우선 복구.

---

## 업무 지시서 목록

| WO | 제목 | 파일 | VPN | 선행 | 한 줄 |
|---|---|---|---|---|---|
| **WO-0** | 안정화 & 계약 고정 | [`WO-0_stabilize.md`](WO-0_stabilize.md) | 불필요 | — | CI green 복구 + `chunk_id` 코덱 + 11 doc_type/4 family 상수 + 네이밍 정리. 비파괴·로컬. |
| **WO-A** | 데이터 적재 (소스→chunk 변환) | [`WO-A_ingestion.md`](WO-A_ingestion.md) | 불필요* | WO-0 | 11 doc_type 변환기 + `researcher_meta`(8 count) 비정규화 + `event_year` 정규화 + §6 불변식 검증. |
| **WO-B** | Qdrant 컬렉션 생성 & 인덱스 | [`WO-B_collection.md`](WO-B_collection.md) | **필요** | WO-0 | `ntis_researcher_chunks` 생성(단일 dense_e5i+sparse_splade) + payload 인덱스 + sparse modifier + smoke. capability flag로 blue/green. |
| **WO-C** | 코드 변경 (런타임 chunk화) | [`WO-C_code.md`](WO-C_code.md) | 불필요 | WO-0 (통합검증 시 WO-B) | config/schema/retriever/filters/planner/evidence/reasoner/cards/schemas/validator 전면 재작성. **최대 작업.** |
| **WO-D** | 컷오버 & 검증 | [`WO-D_cutover.md`](WO-D_cutover.md) | **필요** | WO-A·B·C 전부 | golden 7~16 + v1.x↔v2.0 품질 비교 + `/health/ready` 그린 + env flip + 롤백 런북. |

\* WO-A는 변환·검증은 로컬, dense/sparse 임베딩 실제 생성 단계만 임베딩 서버 접근 필요.

---

## 의존성 & 권장 순서

```
        ┌──────────────────────────────┐
        │  WO-0  안정화 & 계약 고정      │  (로컬, 비파괴 — 반드시 먼저)
        └───────────────┬──────────────┘
            ┌───────────┼───────────┐
            ▼           ▼           ▼
        ┌───────┐   ┌───────┐   ┌───────────────┐
        │ WO-A  │   │ WO-B  │   │ WO-C 코드 골격 │   (병렬 가능)
        │ 적재  │   │ 컬렉션 │   │ (C1~C8)        │
        └───┬───┘   └───┬───┘   └───────┬───────┘
            └───────────┴──── WO-C 통합 검증 ◄┘ (WO-B 컬렉션 필요)
                        │
                        ▼
              ┌───────────────────┐
              │  WO-D  컷오버·검증  │  (A·B·C 완료 후)
              └───────────────────┘
```

1. **WO-0** — 단일 의존점(계약 상수) 확정 + CI 복구. **이 WO 완료 전 후속 착수 금지.**
2. **WO-A ∥ WO-B ∥ WO-C(코드 골격)** — 병렬 진행. A는 로컬, B는 VPN, C는 단위테스트까지 로컬.
3. **WO-C 통합 검증** — WO-B 컬렉션이 떠야 live 경로 확인 가능.
4. **WO-D** — 모든 게이트 통과 후 게이트형 컷오버. 롤백은 컬렉션명 revert로 즉시.

---

## 모든 WO 공통 — HARD 제약 (위반 = 기준선 위반)

각 WO의 §8 "리스크 & 가드레일"에서 해당 항목을 구체적으로 재인용한다.

1. **equal RRF 고정** — Qdrant 융합은 `FusionQuery(RRF)` equal weight만. 가중 RRF/score 가중합 금지. doc_type 중요도는 앱단 `NTIS_DOC_TYPE_PRIORS`(기본 equal)로만. (ADR 0005)
2. **후보 리랭커 기본 OFF** — `NTIS_CANDIDATE_RERANKER=off`. 옵트인(`band`) 시 score 동률 밴드 내 재배열만, 탈락·생성 금지. 후보 순위 척추는 RRF 고정.
3. **LLM 권한 제한** — 후보 재정렬/탈락/새 ID 생성 금지. 이유 생성 + evidence(`chunk_id`) 선택만.
4. **다중 doc_type recency = OR** — `min_should(min_count=1)`. AND 금지(과거 0건 장애 교훈). `filters.py:169-175`에 이미 존재 → **보존 필수**.
5. **evidence 리랭커는 grounding 선별만** — 후보(연구자) 순위에 영향 0. (CrossEncoder evidence selector ≠ candidate reranker, 혼동 금지)
6. **`researcher_meta` 비정규화 불변식** — 한 연구자의 모든 chunk에 동일. 증분 적재 시 lockstep 갱신, ingest 단계 검증.

---

## 진행 체크리스트

- [ ] **WO-0** — pytest green, `chunk_id` 코덱·doc_type/family 상수 모듈 머지, 네이밍 가이드 확정
- [ ] **WO-A** — 11 doc_type 변환기 + §6 불변식 검증기 통과(GOLDEN 시나리오 1)
- [ ] **WO-B** — `ntis_researcher_chunks` 생성 + 인덱스 + modifier + smoke 통과 (`researcher_recommend_proto` 불변)
- [ ] **WO-C** — C1~C8 머지, 단위테스트 green, `searched_doc_types`/`doc_type_coverage`/`evidence[*].chunk_id` 계약 반영 + `EXTERNAL_API_CHANGELOG.md` 갱신
- [ ] **WO-D** — staging `/health/ready` 그린 + golden 7~16 + 품질 비교 게이트 통과 → 운영 env flip (롤백 런북 준비)

---

*생성: 2026-05-29 · 근거: 13개 병렬 리더 기반 갭 분석 + 5개 WO 작업자의 실제 코드 정독. 각 WO 본문의 file:line 인용은 작성 시점 SPLADE 브랜치 기준.*
