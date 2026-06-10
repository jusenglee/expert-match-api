# WO-D · 컷오버 & 검증

> 공통 마이그레이션 컨텍스트(정체성/v1.x↔v2.0/HARD 제약/WO 의존성 그래프/코드 앵커)는 별도 공통 블록을 참조한다. 본 문서는 반복 서술하지 않고 WO-D 고유 지시만 기술한다.

---

## 1. 개요

- **목적:** WO-A(데이터 적재) · WO-B(Qdrant 컬렉션) · WO-C(런타임 코드 chunk화)가 모두 완료된 산출물을 **게이트형 절차**로 검증하고, staging에서 prod으로 `ntis_researcher_chunks` 컬렉션을 무중단·무손실로 컷오버한다.
- **한 줄 요약:** golden 7~16 CI 연결 → v1.x↔v2.0 품질 비교 게이트 → `/health/ready` 그린 게이트 → 운영 env flip → (실패 시) 컬렉션명 revert 롤백.
- **완료 시 달성 상태:**
  - 운영 `NTIS_QDRANT_COLLECTION_NAME=ntis_researcher_chunks`에서 `ntis-validate-live` → `GET /health` → `GET /health/ready` 3단이 모두 그린.
  - GOLDEN_TESTS 시나리오 7~16이 CI에 결선되어 green(`apps/docs/operation/GOLDEN_TESTS.md:33-71`).
  - v1.x↔v2.0 회귀 하니스가 합격 게이트를 통과(허용 회귀 임계 이내).
  - 외부 소비자에 v2.0 BREAKING(`searched_doc_types`/`doc_type_coverage`/`evidence[*].chunk_id`/`*_count_min`)이 사전 공지됨(`apps/docs/api/EXTERNAL_API_CHANGELOG.md:24`).
  - 5초 이내 롤백 가능한 런북 확보(컬렉션명 1개 환경변수 revert).

---

## 2. 선행조건 & 의존성

- **선행 WO (전부 완료 필수):**
  - **WO-0** — chunk_id 코덱(`PREFIX_researcher_id_seq_c#`), 11 doc_type / 4 family 상수, CI green 베이스라인. WO-D의 회귀 하니스/golden 어서션은 WO-0에서 고정한 상수·코덱을 그대로 import한다.
  - **WO-A** — 소스→chunk 변환 + `researcher_meta`(8 count) 비정규화 + 불변식 검증기. WO-D는 이 산출물을 staging 컬렉션에 적재한 상태를 전제로 한다.
  - **WO-B** — `ntis_researcher_chunks` 생성, payload 인덱스, sparse modifier 정합, 표본 upsert 스모크. **★ qdrant_bootstrap이 v2.0 스키마를 만들 수 있어야 한다(WO-B 완료).** 컷오버 전 빈/잘못된 스키마로 env flip 시 strict validation이 `/recommend`를 오프라인시키는 리스크가 본 WO의 핵심 가드 대상이다(§8).
  - **WO-C** — 런타임 파이프라인의 chunk화(`retriever`/`filters`/`evidence_selector`/`reasoner`/`cards`/`schemas`/`live_validator`/`validate_live`). **WO-C 미완 상태로는 본 WO의 어떤 게이트도 의미가 없다** — 예: `apps/search/live_validator.py`는 현재 v1.x(아래 STEP1 근거)이며 v2.0 컬렉션을 검증할 수 없다.
- **VPN/환경:** **VPN 필요.** Qdrant(`NTIS_QDRANT_URL`) 및 임베딩 서버 접근이 staging/prod 검증 전 구간에서 요구됨. 운영 권장값: `NTIS_STRICT_RUNTIME_VALIDATION=true`, `NTIS_CANDIDATE_RERANKER=off`, `NTIS_EVIDENCE_RERANKER_BACKEND=cross_encoder`, `NTIS_SEED_ON_STARTUP=false`(`apps/docs/operation/ENVIRONMENT.md:110`).
- **차단 요소:**
  - WO-C가 `apps/search/live_validator.py`를 v2.0으로 바꾸지 않으면 `/health/ready` 게이트가 v1.x 스키마를 기대해 항상 red → **컷오버 불가**.
  - `apps/api/main.py:346`의 `/health`가 여전히 `searched_branches: list(BRANCHES)`를 반환하면 §6 health 그린 조건(`searched_doc_types`)을 충족할 수 없음 → WO-C 종속.
  - staging Qdrant에 WO-A 적재 데이터가 없으면 `_select_sample_point()`가 빈 결과를 내 readiness가 503(`apps/search/live_validator.py:283-285`).

---

## 3. 범위

### In Scope
- GOLDEN_TESTS 시나리오 7~16의 CI 결선(테스트 파일/픽스처 연결, CI job에 추가).
- v1.x↔v2.0 품질 비교 하니스(동일 질의 세트 → 두 컬렉션 → 후보·evidence 회귀 탐지) 신규 작성.
- `/health/ready` 그린 게이트 절차 정의 및 실행(strict validation 하에서).
- 운영 env flip 절차(`NTIS_QDRANT_COLLECTION_NAME`) + flip 전 사전조건 체크리스트.
- 롤백 런북(컬렉션명 revert) 작성 및 RUNBOOK 반영.
- 외부 BREAKING 사전 공지 실행(EXTERNAL_API_CHANGELOG의 v2.0 항목 통지).

### Out of Scope
- chunk 변환 로직 자체(→ WO-A), 컬렉션 생성/인덱스(→ WO-B), 런타임 파이프라인 코드 변경(→ WO-C).
- `live_validator.py`/`validate_live.py`/`main.py`의 v2.0 화 **구현**은 WO-C 책임. 본 WO는 그 결과를 **검증·게이트·컷오버**한다(단, WO-C 누락분이 발견되면 §8에 따라 컷오버를 보류하고 WO-C로 반려).
- LLM 프롬프트/리즈너 정책 변경(→ WO-C, REASONER_RUNTIME_POLICY).

---

## 4. 상세 작업 항목

### D-1. GOLDEN_TESTS 시나리오 7~16 CI 결선

1. **[대상: `apps/docs/operation/GOLDEN_TESTS.md:33-71` + `tests/` 신규/수정 테스트]** 아래 시나리오를 자동화 테스트로 결선한다. 각 시나리오는 WO-0의 chunk_id 코덱·doc_type/family 상수를 import해 어서션한다. (시나리오 번호/이름은 GOLDEN_TESTS.md 원문 인용.)
   - **시나리오 7. "chunk = 1 Point 적재"(`GOLDEN_TESTS.md:33-35`):** publication 3 + research_project 2 → Qdrant 5 Point, 각 ID=`chunk_id`, 동일 `researcher_id`/`researcher_meta`. → WO-A 불변식 검증기 출력을 어서트하거나 표본 upsert 후 scroll 검증.
   - **시나리오 8. "연구자 집계와 dedupe"(`:37-39`):** 한 연구자 여러 doc_type chunk 동시 hit → 결과 연구자당 1건, 점수=RRF 누적, `matched_doc_types` 채워짐. → WO-C `retriever.py` 신설 집계(RRF 누적+dedupe) 단위 검증.
   - **시나리오 9. "doc_type별 chunk 캡"(`:41-43`):** 다작 vs 소수 고관련 → doc_type별 상위 `chunk_cap`(기본 3)만 집계 기여, chunk 수 독식 방지.
   - **시나리오 10. "통합 recency는 OR 결합"(`:45-47`):** `event_year >= 올해-3`이 doc_type들에 **OR(min_should, min_count=1)** 결합, AND 회귀 0건 가드. → **과거 0건 장애 회귀 방지**(HARD 제약 4, `filters.py:169-175` OR 가드 보존 확인).
   - **시나리오 11. "researcher_meta 기반 hard filter"(`:49-51`):** `publication_count_min`/`highest_degree`로 chunk 단계 deterministic 필터, 위반 후보 0건.
   - **시나리오 12. "제외 기관(cross-chunk)"(`:53-55`):** `exclude_orgs`가 `researcher_meta.affiliated_organization` + 매칭 chunk의 `performing_organization`/`managing_agency`/`appointing_organization`/`evaluation_agency_name` 어디에도 없는 후보만.
   - **시나리오 13. "평가이력 신호"(`:57-59`):** `researcher_assessor`/`expert_assessor` chunk이 검색·evidence에 1급 포함, (옵트인 시) assessment family prior 반영.
   - **시나리오 14. "chunk_id 기반 evidence"(`:61-63`):** LLM에 family별 캡 적용 chunk 풀 전달 → LLM은 풀의 `chunk_id`만 `selected_evidence_ids`로 인용 → 최종 `recommendation.evidence`는 `selected_evidence_ids`와 무관하게 선별 relevant 풀 전체로 결정론적으로 조립, `selected_evidence_ids`는 trace에만 기록.
   - **시나리오 15. "evidence id 계약과 fallback"(`:65-67`):** 코덱 위반 id는 trace(`invalid_selected_evidence_ids_by_candidate`)에만 기록되고 evidence 조립에는 영향 없음, 선별 풀이 빈 후보만 profile(또는 빈) fallback.
   - **시나리오 16. "evidence 선별 캡"(`:69-71`):** family별 캡(`achievement:10`/`assessment:6`/`expertise:6`/`identity:1`) 적용하되 **후보 순위 불변**(grounding 한정). → HARD 제약 5.
2. **[대상: CI 워크플로 + `pyproject.toml`]** 위 테스트를 CI에 추가한다. WO-0에서 `tests/test_cross_encoder_evidence_selector.py:16-19`가 참조하던 `CrossEncoderEvidenceSelector`/`rerank_source`가 WO-C에서 구현되어 있어야 시나리오 16이 통과한다(미구현 시 CI red → 컷오버 보류). **근거:** GOLDEN_TESTS Acceptance Criteria(`:85-96`).

### D-2. v1.x↔v2.0 품질 비교 하니스 (신규 생성)

3. **[대상: `tests/` 또는 `apps/tools/` 하 신규 스크립트 — 예: `apps/tools/compare_collections.py`(신규)]** 동일 질의 세트를 두 컬렉션에 돌려 후보·evidence 회귀를 탐지하는 게이트 하니스를 신규 작성한다.
   - **질의 세트:** RUNBOOK 예시 질의(`apps/docs/operation/RUNBOOK.md:72-79`: "AI 반도체 분야 SCIE 논문…", "AI 반도체 설계 과제 경험과 평가위원 활동…")를 포함한 대표 N개(추천: 도메인별 ≥10) 고정 세트.
   - **실행:** 한 번은 `NTIS_QDRANT_COLLECTION_NAME=researcher_recommend_proto`(v1.x), 한 번은 `=ntis_researcher_chunks`(v2.0)로 `service.recommend(...)`/`search_weighted_candidates(...)`(`apps/api/main.py:408,476`) 호출 후 결과를 캡처.
   - **회귀 메트릭:** ① 후보 집합 overlap@k(연구자 id 기준, evidence id 형식 차이 무시), ② 상위 k 순위 상관(Kendall/Spearman), ③ evidence 커버리지(추천당 evidence ≥1 비율), ④ empty-result 비율 변동.
   - **합격 게이트(컷오버 전 통과 필수):** 후보 overlap@5 ≥ 임계(권장 0.7, 운영 합의), empty-result 비율 v2.0 ≤ v1.x, evidence 커버리지 v2.0 ≥ v1.x. 미달 시 **컷오버 보류**.
   - **주의(계약 비대칭):** v1.x는 evidence id가 `paper:N`/`project:N`/`patent:N`(`reasoner.py:44` 패턴), v2.0은 `chunk_id`. **id 문자열 동치 비교는 금지**, 반드시 연구자 id 및 doc_type 레벨로 정규화해 비교한다.
   - **근거:** MIGRATION_PLAN Phase D-3 "동일 질의 셋으로 v1.x 대비 후보·근거 비교"(`apps/docs/plans/MIGRATION_PLAN.md:61`).

### D-3. `/health/ready` 그린 게이트 (staging 우선)

4. **[대상: `apps/search/live_validator.py` (WO-C가 v2.0화) + `apps/tools/validate_live.py:23-60`]** staging에서 `NTIS_STRICT_RUNTIME_VALIDATION=true` 하에 readiness 3단을 통과시킨다.
   - **현재(v1.x) 근거 — WO-C 종속 확인 항목:** `live_validator.py:21`이 `BRANCHES`/`PAYLOAD_INDEX_FIELDS`를 import하고, `:239` `dense_vector_by_branch[branch]`, `:246` `sparse_vector_by_branch[branch]`로 **4 named-vector**를 검사하며, `:137-148`이 `basic_info`/`researcher_profile` 루트 + `publications`/`research_projects` 리스트를 표본 검사한다(`sample_art_present`/`sample_pjt_present`/`sample_project_dates_valid`, `:37-44`). **이 v1.x 검사는 v2.0 chunk Point(단일 `dense_e5i`/`sparse_splade`, `doc_type`/`researcher_meta`/`domain_attrs`)를 통과시킬 수 없다.** WO-C가 이 모듈을 RUNBOOK §3(`apps/docs/operation/RUNBOOK.md:43-48`) 기준으로 교체했는지 게이트 진입 전 확인.
   - **그린 조건(RUNBOOK §3 인용):** ① 컬렉션 존재, ② named vector `dense_e5i`(1024,Cosine)/`sparse_splade` 존재, ③ sparse modifier 정합(SPLADE=none / bm25 fallback=IDF — `_modifier_matches_expected`, `live_validator.py:130-133`), ④ 필수 payload 인덱스(`researcher_id`,`doc_type`,`event_year`,`researcher_meta.*_count` 등), ⑤ 유효 샘플 Point 구조(`chunk_id`,`doc_type`,`researcher_meta`,`domain_attrs`).
   - **실행 순서(RUNBOOK §3, `:36-39`):** `ntis-validate-live`(CLI, `pyproject.toml:23` → `apps/tools/validate_live.py:main`) → `GET /health` → `GET /health/ready`.
   - **`/health` 그린 조건:** `main.py:341-347` `/health`가 `searched_doc_types`를 반환(WO-C가 `searched_branches`→`searched_doc_types`로 교체, `:346` `list(BRANCHES)` 제거)하고 `collection_name`이 `ntis_researcher_chunks`. 
   - **`/health/ready` 그린 조건:** `main.py:354-391` readiness가 503이 아닌 200, `report.ready=true`. 503이면 `validator.validate()`(`live_validator.py:206-328`)의 `checks`/`issues`로 원인 분류(RUNBOOK `:43-48`).
   - **근거:** MIGRATION_PLAN Phase D-1(`:59`), RUNBOOK §3.

### D-4. 운영 env flip (컷오버)

5. **[대상: 운영 환경변수 `NTIS_QDRANT_COLLECTION_NAME` → `apps/core/config.py:63` `qdrant_collection_name`]** staging 게이트(D-1·D-2·D-3) 전부 통과 후에만 운영 환경변수를 `ntis_researcher_chunks`로 전환한다(현재 default는 `researcher_recommend_proto`, `config.py:63`).
   - **★ flip 전 필수 사전조건(HARD 가드):** 운영 Qdrant에 v2.0 스키마가 **이미 정상 존재**해야 한다(WO-B 완료 + WO-A 적재 완료). `apps/api/main.py:191` `bootstrapper.ensure_collection(recreate=settings.seed_allow_recreate_collection)`가 startup에 호출되므로, **빈/잘못된 스키마 상태에서 flip 시** `NTIS_STRICT_RUNTIME_VALIDATION=true`의 `validator.validate()`가 실패→`/health/ready` 503→`get_service()`(`main.py:327-339`)가 503→**`/recommend`/`/search/candidates` 오프라인**. 따라서 flip은 반드시 D-3 staging 그린 + 운영 컬렉션 사전 적재 확인 후에만.
   - **flip 직후:** D-3 순서(`ntis-validate-live`→`/health`→`/health/ready`)를 운영에서 재실행해 그린 재확인. 하나라도 red면 즉시 D-5 롤백 트리거.
   - **근거:** MIGRATION_PLAN Phase D-4(`:62`), 전환 원칙 blue/green(`:14`).

### D-5. 롤백 런북

6. **[대상: `apps/docs/operation/RUNBOOK.md` (신규 §"컷오버 & 롤백" 추가)]** 무손실 즉시 롤백 절차를 RUNBOOK에 추가한다.
   - **롤백 동작:** 운영 `NTIS_QDRANT_COLLECTION_NAME`을 `researcher_recommend_proto`로 revert → 앱 재기동 → `ensure_collection`이 v1.x 컬렉션을 다시 바인딩 → v1.x 동작 즉시 복귀. **신규/구 컬렉션은 독립이므로 데이터 손실 없음**(MIGRATION_PLAN `:14`,`:66`).
   - **롤백 트리거(명시):** (a) D-4 flip 직후 `/health/ready` 503 또는 `report.ready=false`, (b) D-2 회귀 게이트 사후 미달(예: empty-result 급증), (c) `/recommend` 5xx 비율 급등 또는 evidence 커버리지 급락. 트리거 발동 시 **추가 분석 없이 즉시 revert**(빠른 복구 우선).
   - **롤백 검증:** revert 후 동일 3단(`ntis-validate-live`→`/health`→`/health/ready`)으로 v1.x 그린 확인.
   - **근거:** MIGRATION_PLAN §5 롤백(`:64-67`).

### D-6. 외부 BREAKING 사전 공지

7. **[대상: `apps/docs/api/EXTERNAL_API_CHANGELOG.md:24-60` v2.0 항목]** 컷오버 **전** 외부 소비자에 BREAKING을 통지한다. 통지에 포함할 확정 변경(원문 인용):
   - 필드 이름 변경(`:30-37`, Section A): `searched_branches`→`searched_doc_types`(`/recommend`,`/search/candidates`,`/health`), `candidates[*].branch_presence_flags`→`doc_type_coverage`(family 단위), `trace.query_payload.*_branch_counts`→`*_doc_type_counts`.
   - evidence 구조 변경(`:39-43`, Section B): `evidence[*].type` 4종→**11 doc_type 문자열**, `evidence[*]`에 **`chunk_id` 추가**, 내부 `selected_evidence_ids`가 `paper:N`→`chunk_id`.
   - 신규 필드(`:45-48`, Section C): `candidates[*].matched_doc_types`, `evidence[*].chunk_id`.
   - 컬렉션/스키마(`:50-54`, Section D): default 컬렉션 `researcher_recommend_proto`→`ntis_researcher_chunks`, 4쌍 named vector→`dense_e5i`+`sparse_splade`. **요청 스키마는 불변**(`:54`).
   - 추가로 본 WO 스코프 명시 항목: `*_count_min` 계열 필터 키(`publication_count_min` 등, GOLDEN `:49-51` / MIGRATION `:50`)와 `doc_type_coverage` 노출.
   - **공지 액션:** EXTERNAL_API_CHANGELOG의 v2.0 행 커밋 식별자를 `pending`/`v2.0`에서 실제 컷오버 커밋으로 확정하고, 소비자 가이드(`:56-60`)를 발송. **소비자 ack 또는 마감 통지 후** D-4 flip 진행.
   - **근거:** MIGRATION_PLAN §5(`:67`), EXTERNAL_API_CHANGELOG v2.0 재설계 섹션.

---

## 5. 변경/생성 대상 파일

| 파일 | 변경유형 | 요지 |
|---|---|---|
| `tests/test_golden_chunk_scenarios.py` (또는 기존 golden 테스트 확장) | 신규/수정 | GOLDEN 시나리오 7~16 자동화 결선(D-1). WO-0 코덱·상수 import. |
| `tests/test_cross_encoder_evidence_selector.py` | 검증(수정 가능) | 시나리오 16용. WO-C가 `CrossEncoderEvidenceSelector` 구현 후 green 확인(`:16-19`). |
| `apps/tools/compare_collections.py` | 신규 | v1.x↔v2.0 품질 비교 게이트 하니스(D-2). |
| CI 워크플로 (`.github/workflows/*` 등) | 수정 | golden 7~16 + 비교 하니스 게이트를 CI에 추가(D-1·D-2). |
| `apps/docs/operation/RUNBOOK.md` | 수정 | "컷오버 & 롤백" 섹션 추가(D-4·D-5): env flip 절차, 롤백 트리거/검증. |
| `apps/docs/api/EXTERNAL_API_CHANGELOG.md` | 수정 | v2.0 행 커밋 식별자 확정 + 공지 완료 표기(D-6, `:24`). |
| `apps/docs/plans/MIGRATION_PLAN.md` | 수정(선택) | Phase D 항목 체크 완료 표기(`:57-62`). |
| (검증 대상, **본 WO에서 미수정**) `apps/search/live_validator.py`, `apps/tools/validate_live.py`, `apps/api/main.py` | — | WO-C가 v2.0화한 결과를 게이트로 검증. 미완 발견 시 WO-C 반려(§8). |

---

## 6. 수용 기준 (Acceptance Criteria)

- [ ] GOLDEN 시나리오 7~16(`GOLDEN_TESTS.md:33-71`)이 CI에 결선되어 green. 특히 #10 OR recency, #14 chunk_id evidence, #16 evidence 캡(순위 불변) 포함.
- [ ] `tests/test_cross_encoder_evidence_selector.py`가 green(WO-C 구현 확인, `:16-19`).
- [ ] v1.x↔v2.0 비교 하니스 게이트 통과: 후보 overlap@5 ≥ 임계, empty-result 비율 v2.0 ≤ v1.x, evidence 커버리지 v2.0 ≥ v1.x.
- [ ] staging `NTIS_STRICT_RUNTIME_VALIDATION=true`에서 `ntis-validate-live` 종료코드 0(`validate_live.py:60`), `GET /health`가 `searched_doc_types`+`collection_name=ntis_researcher_chunks` 반환, `GET /health/ready`가 200·`ready=true`.
- [ ] `/health/ready` `checks`에 `dense_e5i`/`sparse_splade` 존재, payload 인덱스(`researcher_id`/`doc_type`/`event_year`/`researcher_meta.*_count`), 샘플 Point가 `chunk_id`/`doc_type`/`researcher_meta`/`domain_attrs` 구조로 검증됨(v1.x `publications`/`research_projects` 표본 검사 아님).
- [ ] 운영 컬렉션에 v2.0 스키마+데이터가 사전 존재함을 flip 전 확인(빈 스키마 flip 금지).
- [ ] 운영 env flip 후 동일 3단 재실행 그린.
- [ ] 롤백 런북이 RUNBOOK에 존재하고, 컬렉션명 revert만으로 v1.x 그린이 재현됨(드라이런 1회).
- [ ] EXTERNAL_API_CHANGELOG v2.0 BREAKING이 외부 소비자에 공지 완료(커밋 식별자 확정).

---

## 7. 검증 방법

### 단위/Golden (로컬·CI, VPN 불필요분)
```powershell
python -m pip install -e .[dev]
python -m pytest tests/ -q
python -m pytest tests/test_golden_chunk_scenarios.py tests/test_cross_encoder_evidence_selector.py -q
```

### 품질 비교 하니스 (VPN 필요 — 두 컬렉션 모두 staging에 존재)
```powershell
$env:NTIS_QDRANT_COLLECTION_NAME = "researcher_recommend_proto"; python -m apps.tools.compare_collections --baseline
$env:NTIS_QDRANT_COLLECTION_NAME = "ntis_researcher_chunks";     python -m apps.tools.compare_collections --candidate --gate
```
- `--gate`는 합격 임계 미달 시 비-0 종료(컷오버 보류 신호).

### Readiness 3단 (VPN 필요 — staging→prod 동일)
```powershell
$env:NTIS_STRICT_RUNTIME_VALIDATION = "true"
$env:NTIS_QDRANT_COLLECTION_NAME = "ntis_researcher_chunks"
ntis-validate-live   # apps/tools/validate_live.py:main, 종료코드 0 기대
uvicorn apps.api.main:app --host 0.0.0.0 --port 8011
# 별 셸:
curl http://127.0.0.1:8011/health         # searched_doc_types + collection_name 확인
curl http://127.0.0.1:8011/health/ready    # 200 / ready=true 기대 (503이면 checks/issues로 분류)
```

### 수동 스모크 (VPN 필요, RUNBOOK §6 인용)
```powershell
curl -X POST http://127.0.0.1:8011/recommend `
     -H "Content-Type: application/json" `
     -d "{\"query\":\"AI 반도체 설계 과제 경험과 평가위원 활동 이력이 있는 전문가를 추천하고 특정 기관은 제외해줘\", \"exclude_orgs\":[\"A기관\"]}"
```
- 응답에서 `recommendations[*].evidence[*].chunk_id` 존재, `searched_doc_types` 노출, `candidates[*].doc_type_coverage`/`matched_doc_types` 확인.

### 롤백 드라이런 (VPN 필요)
```powershell
$env:NTIS_QDRANT_COLLECTION_NAME = "researcher_recommend_proto"  # revert
# 앱 재기동 후 3단 재실행 → v1.x 그린 확인
```

---

## 8. 리스크 & 가드레일

### 본 WO가 준수해야 할 HARD 제약 (위반 시 기준선 위반)
1. **equal RRF 고정 (제약 1·2):** 비교 하니스/golden은 v2.0 후보 순위가 **equal RRF(FusionQuery) 누적**으로 산출됨을 전제로 검증한다. 회귀가 보여도 가중 RRF/score 가중합으로 "교정" 금지. doc_type 중요도는 앱단 `NTIS_DOC_TYPE_PRIORS`(기본 equal)로만.
2. **후보 리랭커 기본 OFF (제약 2):** 컷오버 환경은 `NTIS_CANDIDATE_RERANKER=off`(`ENVIRONMENT.md:110`). 비교 하니스도 off로 실행. band 옵트인 검증이 필요하면 별도 실행으로 격리(동률 밴드 내 재배열만, 탈락/생성 금지).
3. **OR recency 보존 (제약 4):** golden #10이 `filters.py:169-175`의 OR(min_should, min_count=1) 가드를 검증한다. **과거 0건 장애 회귀 방지** — AND로 좁아지면 게이트 fail 처리.
4. **LLM no-rerank (제약 3):** golden #5/#14/#15가 LLM이 후보 재정렬·탈락·신규 id 생성을 하지 않고 chunk_id 인용만 함을 검증. 비교 시 `recommendation.evidence`는 `selected_evidence_ids`와 무관하게 선별 relevant 풀 전체로 결정론적으로 조립된다(LLM 선택이 evidence를 좌우하지 않음).
5. **evidence 리랭커는 grounding 한정 (제약 5):** golden #16이 family 캡 적용 후 **후보 순위 불변**을 검증. evidence 선별이 순위에 영향 0임을 어서트.
6. **researcher_meta 비정규화 (제약 6):** golden #7/#11/#12가 한 연구자의 모든 chunk에서 `researcher_meta` 동일, hard filter deterministic을 검증. 적재 lockstep은 WO-A 책임이나 본 WO 표본 점검으로 재확인.

### 회귀 위험 · 회피책
- **빈 스키마 flip → 서비스 오프라인:** `ensure_collection`(`main.py:191`)+strict validation(`/health/ready` 503→`get_service()` 503)이 `/recommend`를 죽인다. → **회피:** flip 전 운영 컬렉션 v2.0 스키마+데이터 사전 존재 확인(D-4 사전조건), staging 그린 선행.
- **WO-C 미완 검증 모듈:** `live_validator.py`가 v1.x(`BRANCHES`/`dense_vector_by_branch`/`publications` 표본)인 채면 v2.0 컬렉션을 영구 503 처리. → **회피:** 게이트 진입 전 `live_validator.py`/`/health`(`main.py:346`)가 v2.0화됐는지 확인, 미완 시 컷오버 보류·WO-C 반려.
- **id 형식 비대칭 오탐:** v1.x `paper:N`(`reasoner.py:44`) vs v2.0 `chunk_id` 직접 비교는 가짜 회귀를 만든다. → **회피:** 비교 하니스는 연구자 id/doc_type 레벨 정규화 후 비교(D-2).
- **OR→AND 회귀:** recency 다중 doc_type가 AND로 굳으면 과거 장애 재현. → golden #10이 게이트.
- **외부 소비자 미통지 컷오버:** `searched_branches`/`branch_presence_flags` 직참 프런트가 깨짐. → D-6 사전 공지 완료를 flip 선행조건으로.

---

## 9. 작업 분할 & 예상 규모

**순서/병렬성:**
- **선행:** WO-A·WO-B·WO-C 완료 확인(체크포인트). 미완 시 본 WO 착수 불가.
- **병렬 가능:** D-1(golden 결선, 로컬/CI) ∥ D-2(비교 하니스 작성) ∥ D-6(외부 공지 문안 준비). VPN 불필요분은 먼저 진행.
- **순차(게이트 체인):** D-1·D-2 green → D-3 staging readiness 그린 → (D-6 공지 완료) → D-4 운영 flip → flip 후 3단 재확인 → (이상 시) D-5 롤백.

**rough 난이도:**
- D-1 golden 결선: 중 (시나리오 10개, WO-0/WO-C 산출물에 의존하는 어서션).
- D-2 비교 하니스: 중~상 (두 컬렉션 호출 + id 정규화 + 메트릭/게이트 로직, 신규).
- D-3 readiness 게이트: 하~중 (실행·진단 절차, 코드 변경 없음 — WO-C 산출물 검증).
- D-4 flip: 하 (환경변수 1개) — 단 사전조건 검증이 핵심.
- D-5 롤백 런북 + 드라이런: 하.
- D-6 공지: 하 (문서·통지, 합의 대기 변동).

---

## 10. 산출물 (Deliverables)

- GOLDEN 7~16 자동화 테스트(`tests/test_golden_chunk_scenarios.py` 등) + CI green 결선.
- v1.x↔v2.0 품질 비교 하니스(`apps/tools/compare_collections.py`) + 게이트 통과 리포트(후보 overlap/순위 상관/evidence 커버리지/empty 비율).
- staging·prod 각각의 readiness 3단(`ntis-validate-live`→`/health`→`/health/ready`) 그린 증빙(JSON 스냅샷 — `validate_live.py:57` 출력 포함).
- RUNBOOK 갱신("컷오버 & 롤백" 섹션: env flip 절차 + 롤백 트리거/검증).
- EXTERNAL_API_CHANGELOG v2.0 BREAKING 공지 완료(커밋 식별자 확정) + 소비자 가이드 발송 기록.
- 컷오버 실행 로그(flip 시각, 검증 결과, 롤백 여부) 및 롤백 드라이런 1회 증빙.
