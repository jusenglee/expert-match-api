# WO-A · 데이터 적재 (소스→chunk 변환)

> 본 WO는 공통 마이그레이션 컨텍스트(정체성·v1.x→v2.0 핵심·HARD 제약·WO 의존성 그래프)를 전제로 한다. 해당 내용은 반복 서술하지 않고 참조만 한다. 권위 문서: `apps/docs/architecture/DATA_MODEL.md`, `apps/docs/plans/MIGRATION_PLAN.md`(Phase A), `apps/docs/operation/GOLDEN_TESTS.md`, `apps/docs/전체 샘플 payload (도메인별).txt`.

---

## 1. 개요

- **목적:** 외부에서 도착하는 11종 doc_type 소스 문서를 v2.0 "chunk 1개 = 1 Point" payload(`apps/docs/architecture/DATA_MODEL.md` §3 3층 구조)로 변환하는 **오프라인 변환·검증 파이프라인**을 신규 작성한다. 임베딩 생성(`dense_e5i`/`sparse_splade`)을 제외한 모든 단계는 로컬에서 수행한다.
- **한 줄 요약:** `소스 문서 → (doc_type별 변환기) → chunk_text 직렬화 + 3층 payload + researcher_meta 비정규화 + event_year 정규화 + tags 정규화 → chunk_id 부여 → (임베딩) → 불변식 검증` 의 변환기/집계기/검증기를 구현한다.
- **완료 시 달성 상태:**
  - 11 doc_type 각각에 대해 소스 레코드를 받아 `DATA_MODEL.md` §3.3에 정의된 `domain_attrs` 필드 집합을 정확히 채우는 변환 함수가 존재한다.
  - 한 연구자의 모든 chunk에 동일한 `researcher_meta`(8 count) + `researcher_name`이 주입된다(비정규화 불변식, `DATA_MODEL.md` §6-2).
  - 각 chunk가 WO-0에서 정의한 chunk_id 코덱으로 결정적 `chunk_id`를 갖는다(`DATA_MODEL.md` §1.1, §6-1).
  - `DATA_MODEL.md` §6 7개 불변식을 표본 점검하는 검증 스크립트가 통과한다(`GOLDEN_TESTS.md` 시나리오 1·7과 연결).
  - 산출물은 Qdrant upsert 직전 형태의 chunk payload 컬렉션(+벡터)이며, 실제 적재는 WO-B/WO-D 소관이다.

---

## 2. 선행조건 & 의존성

- **선행 WO:** **WO-0 (안정화 & 계약 고정)** 완료 필수.
  - WO-0이 확정하는 **chunk_id 코덱**(형식 `PREFIX_researcher_id_seq_c#`, 예 `PUB_M1006328_0001_c0` — 샘플 payload 7.1 line 10)을 본 WO가 그대로 호출한다. WO-A는 코덱을 재정의하지 않는다.
  - WO-0이 확정하는 **doc_type enum(11종)·family 매핑(4군)·doc_type→PREFIX 매핑**(`DATA_MODEL.md` §4) 상수를 import해서 사용한다.
  - WO-0의 **CI green** 전제(현재 `tests/test_cross_encoder_evidence_selector.py:16-19`가 미존재 심볼 import로 CI red — WO-0/WO-C 소관, WO-A 직접 책임 아님이나 CI 통과 기준선은 WO-0이 확보).
- **VPN/환경:**
  - **VPN 불필요(원칙).** 소스→chunk 변환, researcher_meta 집계, event_year/tags 정규화, 불변식 검증은 전부 로컬.
  - **예외:** STEP (6) 임베딩 호출(`dense_e5i` 임베딩 서버, `MIGRATION_PLAN.md` Phase A A-3 "VPN 필요")만 외부 의존. 로컬 SPLADE(`SpladeSparseEncoder`)는 `models/PIXIE-Splade-v1.0/`(repo 루트에 이미 존재) 사용 시 VPN 불필요. 따라서 본 WO는 **변환·검증(로컬, VPN 불필요)** 과 **임베딩(VPN 필요 가능)** 을 단계 분리한다.
- **차단 요소:**
  - WO-0 미완 시 chunk_id 코덱/상수가 없어 시작 불가 → 본 WO 착수 차단.
  - 실제 소스 데이터 스키마 샘플 부재 시: `apps/docs/전체 샘플 payload (도메인별).txt`(7.1~7.11, 연구자 `M1006328` 기준)와 `DATA_MODEL.md` §3.3을 계약 기준으로 사용. 소스 원천 필드명과 본 샘플 필드명의 매핑표가 필요하면 데이터 제공처에 확인(open question).

---

## 3. 범위

### In Scope
1. 11 doc_type별 `소스 레코드 → chunk payload` 변환기 신규 작성(`DATA_MODEL.md` §3.3 + 샘플 payload 7.1~7.11 기준 필드 매핑).
2. doc_type별 `chunk_text` 직렬화 규칙 정의(role/action 불용어 배제 불변식 준수, `DATA_MODEL.md` §2 임베딩 정제 원칙·§6-7).
3. `chunk_id` 부여(WO-0 코덱 호출) + 단일 doc → 다중 chunk 분할 트리거 정책 제안(open question).
4. `researcher_meta`(8 count) + `affiliated_organization`/`highest_degree` 집계기 + 모든 chunk 동일 주입(비정규화).
5. `event_date → event_year` 정규화 + dateless 6종(`researcher_tech`/`expert_tech`/`researcher_core`/`researcher_major`/`expert_specific`/`profile`) null 처리.
6. `tags` 소문자+trim 정규화, `specialty_names`/`keywords` 기반 tags 생성 규칙.
7. `dense_e5i`+`sparse_splade` 임베딩 생성 — 입력은 동일 `chunk_text`. 기존 `apps/search/encoders.py`/`apps/search/sparse_runtime.py` 재사용(신규 모델 불필요).
8. `DATA_MODEL.md` §6 불변식 검증 스크립트(7개 항목).

### Out of Scope
- Qdrant 컬렉션 생성·인덱스 생성·실제 upsert·sparse modifier 정합 부트스트랩 → **WO-B**.
- 런타임 검색 파이프라인의 chunk화(retriever 집계/RRF 누적/chunk_cap/dedupe, filters의 `researcher_meta`/`event_year` 기준 전환, evidence_selector/reasoner의 chunk_id 전환) → **WO-C**.
- 스테이징 컷오버·`/health/ready`·골든 통과·v1.x 대비 회귀 비교 → **WO-D**.
- 기존 `apps/search/seed_data.py`의 v1.x 시드 로직 변경/삭제(아래 §8 참조: v2.0 변환기는 **별도 신규 모듈**, seed_data는 v1.x 시드 전용으로 잔존).

---

## 4. 상세 작업 항목

> 모든 신규 심볼은 **신규 생성**임을 명시한다(현행 `apps/` 코드에 `chunk_id`/`doc_type`/`dense_e5i`/`sparse_splade`/`researcher_meta`/`event_year`/`domain_attrs` 출현 0회 — `apps/` 전 디렉터리 grep으로 확인, 문서에만 존재). 권장 신규 위치는 `apps/ingest/`(신규 패키지). 본 WO는 `apps/search/seed_data.py`를 수정하지 않는다.

### A-0. chunk payload 데이터 모델 정의 (신규)
- [ ] **[`apps/domain/models.py` — 신규 클래스 추가]** v2.0 chunk payload용 Pydantic 모델 `ChunkPayload`(신규), 하위 `ResearcherMeta`(신규, 8 count + `affiliated_organization` + `highest_degree`), `DomainAttrs`(doc_type별 union 또는 `dict[str, Any]`)를 추가한다.
  - 근거: 현행 `apps/domain/models.py`는 v1.x 전용 — `BasicInfo`(line 61-68), `ResearcherProfile`(line 71-88, 4 count만: publication/scie/intellectual_property/research_project, **assessor 2종 count 없음**), `ExpertPayload`(line 146-180, nested `publications[]`/`intellectual_properties[]`/`research_projects[]`), `SeedEvidencePoint`(line 182-187: `point_id`/`branch`/`content_text`)는 모두 v1.x 구조. `BranchName = Literal["basic","art","pat","pjt"]`(line 7)도 v1.x.
  - 구체: `ChunkPayload`는 `DATA_MODEL.md` §3 1·2·3층을 그대로 반영 — L1(`researcher_id`/`researcher_name`/`doc_type`/`doc_id`/`chunk_id`/`chunk_text`/`chunk_text_len`/`researcher_meta`), L2(`event_date: datetime|None`/`event_year: int|None`/`tags: list[str]`), L3(`domain_attrs`). `ResearcherMeta`는 `ResearcherProfile`의 4 count를 계승하고 `researcher_assessor_count`/`expert_assessor_count`(★v1.x에 없던 신규)를 추가.
  - 제약: 기존 v1.x 모델(BasicInfo/ExpertPayload/SeedEvidencePoint 등)은 **변경/삭제 금지**(seed_data가 의존, WO-D 롤백 경로 보존).

### A-1. doc_type별 source→chunk 변환기 (신규)
- [ ] **[`apps/ingest/converters.py`(신규) — `convert_<doc_type>(source) -> ChunkPayload` 11개]** 각 doc_type별 변환 함수를 신규 작성. `domain_attrs` 필드 집합은 `DATA_MODEL.md` §3.3 + 샘플 payload를 정확히 따른다:
  - `publication`(샘플 7.1, line 30-39): `title_primary`/`title_secondary`/`journal_name`/`journal_class`/`indexing_class`/`publication_year_month`/`abstract`/`keywords`.
  - `intellectual_property`(7.2, line 57-68): `ip_type`/`ip_title`/`application_registration_type`/`application_country`/`application_number`/`application_date`/`registration_number`/`registration_date`/`ip_summary`/`ip_foreign_class`.
  - `research_project`(7.3, line 86-95): `project_title_korean`/`project_title_english`/`performing_organization`/`managing_agency`/`research_summary_korean`/`research_summary_english`/`research_period_start`/`research_period_end`.
  - `researcher_assessor`(7.4, line 113-121, **신규**): `appointing_organization_type`/`appointment_period_type`/`appointing_organization`/`evaluation_committee_name`/`appointment_period_start`/`_end`/`appointment_date`.
  - `expert_assessor`(7.5, line 139-145, **신규**): `evaluation_agency_name`/`assessment_type_class`/`assessment_class`/`assessment_content`/`valid_period_start`/`_end`.
  - `researcher_tech`(7.6, line 164-168, **신규**): `tech_classification_system`/`tech_classification`/`keywords`.
  - `expert_tech`(7.7, line 186-190, **신규**): `tech_classification_system`/`tech_classification`/`tech_rank`.
  - `researcher_core`(7.8, line 208-217, **신규**): `specialty_names`/`specialty_count`.
  - `researcher_major`(7.9, line 235-244, **신규**): `specialty_names`/`specialty_count`.
  - `expert_specific`(7.10, line 262-265, **신규**): `specific_specialty_name`/`career_description`.
  - `profile`(7.11, line 283-294): `researcher_number`/`position_title`/`highest_degree`/`major_field` + 8 count(`researcher_meta`와 동일).
  - 근거/제약: assessment family(researcher_assessor/expert_assessor) 등 **7개 신규 doc_type은 v1.x 파싱이 전무** → 변환기 전부 신규(공수 큼). 현행 `seed_data.points_from_payload`(line 134-209)는 basic/art/pat/pjt 4종만 처리(v1.x 시드 전용)이므로 재사용 불가.

- [ ] **[`apps/ingest/converters.py`(신규) — doc_type별 `chunk_text` 직렬화 규칙]** `chunk_text`는 샘플 payload의 `chunk_text` 필드 형식을 따른다. 도메인별 `라벨: 값\n라벨: 값` 직렬화:
  - `publication`: `논문명 + 키워드 + 학술지 + 초록`(샘플 7.1 line 11).
  - `intellectual_property`: `지식재산권명 + 구분 + 출원국가 + 요약`(7.2 line 49).
  - `research_project`: `국문과제명 + 수행기관 + 전문기관 + 국문요약`(7.3 line 78).
  - `researcher_assessor`/`expert_assessor`: `임명/평가기관 + 위원회/평가구분 + 내용`(7.4 line 105 / 7.5 line 131).
  - `*_tech`: `기술분류 + (키워드) + 분류체계 (+ 기술순위)`(7.6 line 156 / 7.7 line 178).
  - `researcher_core`/`researcher_major`: `핵심/전공 전문분야: <specialty_names 나열>`(7.8 line 200 / 7.9 line 227).
  - `expert_specific`: `특정전문분야 + 경력`(7.10 line 254).
  - `profile`: `이름 + 소속 + 직위 + 학위 + 전공 + 누적실적`(7.11 line 275).
  - **불변식(HARD):** `chunk_text`에 "평가위원/추천" 등 요청 페르소나/role/action 불용어를 **절대 넣지 않는다**(`DATA_MODEL.md` §2 line 47 임베딩 정제 원칙, §6-7). 현행 v1.x `seed_data.build_source_texts`의 basic_text는 `심사평가위원 활동`/`심사참여건수`(seed_data.py line 107-108, 153-154)를 포함 — 이는 v1.x 잔재이며 **v2.0 변환기로 그대로 복사 금지**. 평가 활동 신호는 `researcher_assessor`/`expert_assessor` doc_type의 도메인 텍스트로만 표현한다.
  - [ ] `chunk_text_len = len(chunk_text)` 자동 산출(샘플의 `chunk_text_len` 필드, 7.1 line 12).

### A-2. chunk_id 생성 + 분할 트리거 정책 (open question 포함)
- [ ] **[`apps/ingest/converters.py`(신규) — WO-0 코덱 호출]** chunk_id는 WO-0이 정의한 코덱 함수(예 `encode_chunk_id(doc_type, researcher_id, doc_seq, chunk_seq)`)로 생성. 형식 `PREFIX_researcher_id_doc_seq_c#`(예 `PUB_M1006328_0001_c0`). PREFIX는 WO-0 doc_type→PREFIX 매핑(PUB/IP/PJT/RAS/EAS/RTC/ETC/RCO/RMJ/ESP/PRF — 샘플 7.1~7.11 `doc_id`/`chunk_id` 접두 관찰값) 사용. **본 WO에서 코덱 재정의 금지.**
- [ ] **[정책 제안 — 단일 doc → 다중 chunk 분할 트리거]** `DATA_MODEL.md`가 명시적 분할 임계를 정의하지 않은 **open question**. 모든 샘플이 `_c0` 단일 chunk인 점을 기준으로 다음 정책을 제안(WO-0 계약 확정 시 반영):
  - 기본: doc 1건 = chunk 1건(`_c0`). 분할은 예외.
  - 분할 트리거 후보: `chunk_text_len`이 임베딩 모델 토큰 한계(`SpladeSparseEncoder.embed`는 `truncation=True`로 잘림 — encoders.py line 88; e5 dense도 토큰 한계 존재)를 초과하는 긴 본문(예 `abstract`/`ip_summary`/`research_summary_korean`이 매우 긴 경우)에 한해 의미 경계(문단/문장) 단위로 분할, `_c1`/`_c2`… 부여.
  - 분할 시 불변식: 같은 doc의 모든 chunk가 동일 `doc_id`·동일 `event_date`/`event_year`·동일 `researcher_meta`를 공유, `chunk_id`만 `c#`로 구분(전역 유일 — §6-1).
  - **권장(보수적):** v2.0 초기 컷오버에서는 분할을 끄고(=항상 `_c0`) truncation 손실을 모니터링 → 데이터로 임계 결정. 본 WO는 분할 함수의 hook만 만들고 기본 OFF.

### A-3. researcher_meta 집계기 (신규)
- [ ] **[`apps/ingest/meta_aggregator.py`(신규) — `build_researcher_meta(researcher_id, source_records) -> ResearcherMeta`]** 연구자별 1회 집계 후 그 연구자의 **모든 chunk에 동일 주입**. 산출 필드(`DATA_MODEL.md` §3.1 line 70-79, 샘플 7.1 line 13-22):
  - `publication_count` / `scie_publication_count` / `intellectual_property_count` / `research_project_count` / `researcher_assessor_count` / `expert_assessor_count`(6 count) + `affiliated_organization` + `highest_degree`.
  - 산출 출처 우선순위: `profile` doc의 8 count(샘플 7.11 line 288-293)를 1차 신뢰원으로 사용. profile 부재/불일치 시 각 doc_type 소스 레코드 수를 카운트(예 publication 레코드 N → `publication_count`)하되, profile과 어긋나면 **검증 단계에서 경고**(open question: 어느 값을 truth로 둘지 데이터 제공처 확인).
  - `affiliated_organization`/`highest_degree`는 profile doc에서(7.11: `domain_attrs.highest_degree`/`researcher_meta.affiliated_organization`).
- [ ] **[비정규화 주입]** 변환 파이프라인이 한 연구자의 모든 chunk를 만들 때 동일 `ResearcherMeta` 인스턴스(또는 동일 값)와 동일 `researcher_name`을 주입(`DATA_MODEL.md` §6-2). 연구자 단위 처리 루프 구조 권장: `for researcher in researchers: meta = build_researcher_meta(...); for record in researcher.records: chunk = convert(record, meta=meta, name=...)`.
  - 근거/제약: §6-2 불변식(한 researcher_id의 모든 chunk에서 researcher_meta/researcher_name 동일). 위반 시 chunk 단계 hard filter(WO-C `filters.py`)가 비결정적. v1.x `ResearcherProfile`(models.py line 71-88)에는 assessor 2종 count가 없으므로 신규 집계 로직이 반드시 필요.

### A-4. event_date → event_year 정규화 + null 처리 (신규)
- [ ] **[`apps/ingest/normalizers.py`(신규) — `normalize_event(doc_type, source) -> (event_date, event_year)`]** 도메인 통합 대표 일자를 `event_date`로, 그 연도를 `event_year`로(`DATA_MODEL.md` §3.2 line 87-88). doc_type별 대표 일자 선정:
  - `publication`: `publication_year_month`(7.1 `2024-05` → `event_date 2024-05-01`/`event_year 2024`).
  - `intellectual_property`: `registration_date` 우선, 없으면 `application_date`(7.2 → `2025-10-31`/2025).
  - `research_project`: `research_period_start`(7.3 → `2019-10-07`/2019).
  - `researcher_assessor`: `appointment_date`/`appointment_period_start`(7.4 → `2025-03-11`/2025).
  - `expert_assessor`: `valid_period_start`(7.5 → `2025-03-11`/2025).
  - **dateless 6종 null 허용 명시(★):** `researcher_tech`/`expert_tech`/`researcher_core`/`researcher_major`/`expert_specific`/`profile`는 `event_date=null`, `event_year=null`(샘플 7.6~7.11 모두 `null` — 7.6 line 160-161 등).
  - 불변식: `event_year == year(event_date)`(둘 다 null 가능, §6-4). 정규화 함수가 둘을 동시 산출해 불일치 원천 차단.

### A-5. tags 정규화 + specialty 기반 생성 (신규)
- [ ] **[`apps/ingest/normalizers.py`(신규) — `build_tags(doc_type, domain_attrs) -> list[str]`]** `tags`는 소문자+trim 정규화(`DATA_MODEL.md` §3.2 line 89, §6-5):
  - `publication`: `keywords` 소문자화(7.1 `["RAG","LLM","Vector DB"]` → `["rag","llm","vector db"]` line 27·38).
  - `researcher_tech`: `keywords` 소문자화(7.6 → `["식품","안전","검출"]` line 162).
  - `researcher_core`/`researcher_major`: `specialty_names` 소문자화(7.8 line 206, 7.9 line 233).
  - `expert_specific`: `specific_specialty_name`을 단일 원소 tag로(7.10 → `["기초(의학) 연구자"]` line 260).
  - tags 없는 doc_type(`intellectual_property`/`research_project`/`*_assessor`/`expert_tech`/`profile`): 빈 배열 `[]`(7.2 line 55, 7.3 line 84, 7.4 line 111, 7.5 line 137, 7.7 line 184, 7.11 line 281).
  - 규칙: `tag.strip().lower()`, 빈 문자열·중복 제거, 원순서 유지. (참고: `_normalize_string_list` models.py line 14-28은 trim만 하고 lowercase는 안 함 → tags 전용 lowercase 함수 신규 필요.)

### A-6. 임베딩 생성 — 기존 인코더 재사용 (신규 호출부)
- [ ] **[`apps/ingest/embedder.py`(신규) — 기존 인코더 재사용]** chunk별 `dense_e5i`와 `sparse_splade`를 **동일 `chunk_text` 단일 입력**에서 생성(`DATA_MODEL.md` §2 line 38·42-43, §6-6).
  - **dense_e5i:** 기존 `apps/search/encoders.py`의 `LocalSentenceTransformerEncoder`(line 142-165) 또는 `OpenAIEmbeddingEncoder`(line 118-138) 재사용. `vector_size=1024`(e5-large-instruct, §1 line 26). e5 instruct 프리픽스는 쿼리 측만 — 적재 본문(passage)은 §2 line 47 정제 원칙대로 순수 도메인 텍스트.
  - **sparse_splade:** 기존 `apps/search/sparse_runtime.py`의 `resolve_sparse_runtime`(line 63) 3단 fallback 체인(로컬 PIXIE → online `telepix/PIXIE-Splade-v1.0`(line 27) → `Qdrant/bm25` IDF(line 28·168))과 `apps/search/encoders.py`의 `SpladeSparseEncoder`(line 45-104, `log(1+ReLU)` max-pool line 95-98)를 그대로 재사용. **신규 모델·신규 인코더 불필요.**
  - modifier 정합: SPLADE backend면 `requires_idf_modifier=False`(sparse_runtime line 99·199-200), bm25 fallback이면 `IDF`(line 173-177) — 단, **modifier 적용/컬렉션 부트스트랩은 WO-B 소관**. 본 WO는 동일 `chunk_text`로 sparse weight map만 생성.
  - 불변식(HARD §6-6): dense·sparse 입력 `chunk_text` 바이트 동일 보장 — 두 인코더에 같은 문자열 객체를 전달하고, 직렬화 후 변형 금지.
  - VPN: 로컬 PIXIE(`models/PIXIE-Splade-v1.0/`) + 로컬 dense 모델이면 VPN 불필요. 임베딩 서버 호출(`OpenAIEmbeddingEncoder`) 시에만 VPN 필요(이 단계만 분리 실행).

### A-7. 불변식 검증 스크립트 (신규)
- [ ] **[`apps/ingest/validate_chunks.py`(신규) — `validate_chunks(chunks) -> ValidationReport`]** `DATA_MODEL.md` §6 7개 불변식을 표본/전수 점검(`MIGRATION_PLAN.md` A-4, `GOLDEN_TESTS.md` 시나리오 1·7):
  1. **chunk_id 전역 유일** == Point ID(§6-1). 중복 chunk_id 0건 단언.
  2. **researcher_meta/researcher_name 연구자별 동일**(§6-2). 같은 researcher_id의 chunk 그룹에서 meta 8필드 + name 불일치 0건.
  3. **doc_type enum(11종)**(§6-3). WO-0 doc_type 상수 집합 외 값 0건.
  4. **event_year == year(event_date)**(§6-4). 둘 다 null 허용, 한쪽만 값 있으면 위반.
  5. **tags 소문자·trim**(§6-5). `tag != tag.strip().lower()` 0건.
  6. **dense·sparse 동일 chunk_text**(§6-6). 임베딩 단계에서 입력 해시를 chunk에 부착 → dense_input_hash == sparse_input_hash == hash(chunk_text).
  7. **chunk_text role/action 불용어 미포함**(§6-7). WO-0/planner의 role/action 용어 사전(참고: planner.py `role_terms`/`action_terms`, models.py `PlannerOutput` line 198-199) 대조, 위반 chunk 리포트.
- [ ] 검증 스크립트는 CLI 진입점 제공(예 `python -m apps.ingest.validate_chunks <chunks.jsonl>`), exit code 비0 + 위반 리포트 출력. `GOLDEN_TESTS.md` 시나리오 7("publication 3 + research_project 2 → 5 Point, 모두 동일 researcher_id/researcher_meta") 케이스를 단위 테스트로 포함.

---

## 5. 변경/생성 대상 파일

| 파일 | 변경유형 | 요지 |
|---|---|---|
| `apps/domain/models.py` | 수정(추가만) | `ChunkPayload`/`ResearcherMeta`(8 count, assessor 2종 신규)/`DomainAttrs` 신규 클래스 추가. 기존 v1.x 모델(BasicInfo/ExpertPayload/SeedEvidencePoint 등) 불변 |
| `apps/ingest/__init__.py` | 신규 | 신규 적재 패키지 |
| `apps/ingest/converters.py` | 신규 | 11 doc_type별 `convert_<doc_type>` + doc_type별 `chunk_text` 직렬화 + chunk_id(WO-0 코덱 호출) + 분할 hook(기본 OFF) |
| `apps/ingest/meta_aggregator.py` | 신규 | `build_researcher_meta` 8필드 집계 + 모든 chunk 동일 주입(비정규화) |
| `apps/ingest/normalizers.py` | 신규 | `normalize_event`(event_date→event_year, dateless 6종 null) + `build_tags`(lowercase+trim, specialty 기반) |
| `apps/ingest/embedder.py` | 신규 | 기존 `encoders.py`/`sparse_runtime.py` 재사용한 dense_e5i+sparse_splade 생성(동일 chunk_text) |
| `apps/ingest/validate_chunks.py` | 신규 | §6 7개 불변식 검증 CLI + 리포트 |
| `tests/test_ingest_converters.py` | 신규 | 11 doc_type 변환 + chunk_text 직렬화(불용어 배제) 단위 테스트 |
| `tests/test_ingest_invariants.py` | 신규 | §6 불변식 + GOLDEN 시나리오 7 케이스 테스트 |
| `apps/search/seed_data.py` | **불변(수정 금지)** | v1.x 시드 전용으로 잔존(WO-D 롤백 경로). v2.0 변환기는 별도 `apps/ingest/` |

---

## 6. 수용 기준 (Acceptance Criteria)

- [ ] 11 doc_type 각각에 변환 함수가 존재하고, 샘플 payload 7.1~7.11의 `domain_attrs` 키 집합을 정확히 재현한다(누락/오타 0건).
- [ ] assessment family 신규 4종(`researcher_assessor`/`expert_assessor`) + 전문성 5종(`*_tech`/`*_core`/`*_major`/`expert_specific`) 변환기가 동작(v1.x 미존재 → 신규 구현 완료).
- [ ] `chunk_text`에 role/action 불용어("평가위원","추천","심사" 등)가 포함되지 않는다(§6-7, 검증 스크립트 PASS). v1.x `심사평가위원 활동`/`심사참여건수` 텍스트가 v2.0 chunk_text에 0건.
- [ ] 한 연구자의 모든 chunk에서 `researcher_meta`(8필드) + `researcher_name`이 동일하다(§6-2, 검증 PASS).
- [ ] `researcher_meta`에 `researcher_assessor_count`/`expert_assessor_count`가 산출된다(v1.x `ResearcherProfile`에 없던 신규 count).
- [ ] 모든 chunk의 Point ID == `chunk_id`이고 전역 유일하다(§6-1).
- [ ] `event_year == year(event_date)`이며 dateless 6종은 둘 다 null이다(§6-4).
- [ ] `tags`가 전부 소문자+trim이며 specialty/keywords 기반 생성 규칙을 따른다(§6-5).
- [ ] `dense_e5i`(1024d)·`sparse_splade`가 동일 `chunk_text`에서 생성된다(§6-6, 입력 해시 일치 단언).
- [ ] 임베딩 단계에서 신규 모델/신규 인코더를 만들지 않고 기존 `encoders.py`/`sparse_runtime.py`만 재사용한다.
- [ ] 검증 스크립트(`validate_chunks`)가 §6 7개 불변식을 모두 점검하고 위반 시 비0 exit + 리포트를 낸다.
- [ ] `GOLDEN_TESTS.md` 시나리오 1(role/action 배제)·7(chunk=1 Point, 동일 meta) 대응 테스트 통과.
- [ ] 기존 `apps/search/seed_data.py` 및 v1.x 모델이 변경되지 않아 CI/v1.x 시드가 깨지지 않는다.

---

## 7. 검증 방법

> 본 WO는 로컬 검증이 원칙. 임베딩 호출(A-6)만 로컬 모델 부재 시 VPN 필요.

- **단위 테스트(로컬):**
  - `pytest tests/test_ingest_converters.py -v` — 11 doc_type 변환 + chunk_text 직렬화 + 불용어 배제.
  - `pytest tests/test_ingest_invariants.py -v` — §6 7개 불변식 + GOLDEN 시나리오 7.
  - `pytest -q` — 전체 회귀(기존 v1.x 테스트 그대로 green 유지, seed_data 미변경 확인).
- **불변식 검증 CLI(로컬):**
  - `python -m apps.ingest.validate_chunks <generated_chunks.jsonl>` → exit 0 + "all invariants passed".
  - 의도적 위반 픽스처(meta 불일치/대문자 tag/event_year 불일치)로 비0 exit + 리포트 확인.
- **임베딩 스모크(로컬 우선, 모델 부재 시 VPN):**
  - 로컬 `models/PIXIE-Splade-v1.0/`로 `SpladeSparseEncoder` 1 chunk 임베딩 → non-empty weight map(encoders.py line 82-104).
  - dense 1024d 길이 단언(`LocalSentenceTransformerEncoder.embed`의 dim mismatch raise — encoders.py line 161-164).
  - 동일 chunk_text로 dense·sparse 호출 후 입력 해시 일치 단언.
- **하류 연결(참고, 본 WO 비차단):** 생성 chunk는 WO-B가 `ntis_researcher_chunks`에 upsert → WO-D에서 `uvicorn apps.api.main:app --port 8011 --reload` + `ntis-validate-live`로 `GOLDEN_TESTS.md` 시나리오 7~16 통과 확인. 본 WO는 그 입력 산출물의 정합성까지만 책임.

---

## 8. 리스크 & 가드레일

### 준수해야 할 HARD 제약(이 WO 직접 관련)
- **chunk_id 결정성·전역 유일(§6-1):** WO-0 코덱만 사용, 본 WO에서 재정의 금지. Point ID == chunk_id. 위반 시 멱등 적재·evidence 참조(WO-C/WO-D) 붕괴.
- **researcher_meta 비정규화 lockstep(§6-2, 공통 제약 6):** 한 연구자 모든 chunk 동일. 증분 적재 시 한 연구자의 meta가 바뀌면 그 연구자 전 chunk 재적재(부분 갱신 금지). ingest 단계에서 불변식 검증 필수.
- **chunk_text 임베딩 정제(§2 line 47, §6-7):** role/action 불용어 배제. 본 WO가 임베딩 입력 텍스트를 직렬화하므로 이 제약의 1차 책임 지점.
- **dense·sparse 동일 입력(§6-6):** 두 벡터는 같은 chunk_text에서. 직렬화 후 변형 금지.

### 본 WO가 직접 건드리지 않지만 하류에서 전제되는 제약(데이터 형태로 보장)
- **equal RRF / 후보 리랭커 기본 OFF / LLM no-rerank**(공통 제약 1·2·3): 본 WO는 검색·집계·LLM 단계가 아니므로 직접 위반 여지 없음. 단, `researcher_meta` count를 가중치/score로 쓰지 않고 **payload 필터 신호로만** 산출해야 한다(중요도 표현은 WO-C 앱단 `NTIS_DOC_TYPE_PRIORS`(기본 equal) 소관). chunk_text에 인위적 가중(키워드 반복 stuffing 등) 금지.
- **OR recency(공통 제약 4, filters.py:169-175):** 본 WO는 단일 `event_year`만 산출. 다중 doc_type recency OR 결합은 WO-C 보존 책임. 단, dateless 6종 null을 정확히 표기해야 WO-C가 recency 대상에서 올바르게 제외.

### 회귀 위험 & 회피책
- **assessment family 신규 7종 공수 과소평가:** v1.x 파싱 전무 → 변환기 전부 신규. 회피: 샘플 7.4~7.10을 픽스처로 고정하고 doc_type별 테이블 매핑을 테스트로 잠금. 일정은 §9에서 분리 추정.
- **v1.x 모델/seed_data 오염:** v2.0 모델을 기존 클래스에 in-place 변형하면 v1.x 시드/롤백이 깨짐. 회피: `apps/ingest/` 신규 패키지 + `models.py` 추가-only, `BranchName`(models.py line 7)·`SeedEvidencePoint`(line 182-187)·`points_from_payload`(seed_data.py line 134-209) 불변.
- **meta truth 충돌(open question):** profile의 count vs 실제 레코드 수 불일치 시 비결정. 회피: profile을 1차 truth로 고정하되 검증 스크립트가 불일치를 경고로 노출, 정책은 데이터 제공처 확인 후 WO-0 계약에 반영.
- **분할 트리거 미정(open question):** 임의 분할은 chunk_id/evidence 안정성 위협. 회피: 초기 컷오버 분할 OFF(항상 `_c0`), truncation 손실 모니터링 후 데이터 기반 결정(§A-2).
- **lowercase 누락:** 기존 `_normalize_string_list`(models.py line 14-28)는 trim만 — 그대로 쓰면 §6-5 위반. 회피: tags 전용 lowercase 함수 신규.

---

## 9. 작업 분할 & 예상 규모

권장 순서(WO-0 완료 후 착수). 본 WO 내부 병렬성 존재:

1. **A-0 모델 정의**(선행, 직렬) — `ChunkPayload`/`ResearcherMeta`/`DomainAttrs`. 난이도 中. 이후 단계의 타입 기반.
2. 다음 3트랙 **병렬 가능**:
   - **트랙 1 — A-1 변환기(가장 큰 작업).** achievement 3종 + identity/expertise 6종 + assessment 2종. 난이도 高(assessment family 신규로 비중 큼). chunk_text 직렬화 포함.
   - **트랙 2 — A-3 meta 집계기 + A-4 event 정규화 + A-5 tags.** 난이도 中. 독립 유틸.
   - **트랙 3 — A-6 임베딩(기존 인코더 wrapper).** 난이도 低~中(기존 재사용). 단 임베딩 실호출은 로컬 모델/VPN 확보 후.
3. **A-7 검증 스크립트 + 테스트**(통합, 직렬) — 1·2·3 산출물 합류 지점. `GOLDEN_TESTS.md` 시나리오 1·7 연결. 난이도 中.
4. **A-2 분할 트리거**는 hook만(기본 OFF) → 난이도 低, 정책 결정은 open question으로 WO-0/데이터 제공처와 협의.

전체 rough 규모: **中~大**(11 doc_type 신규 변환기, 특히 v1.x 미존재 7종이 비용 견인). 코드 위험은 낮으나(검색·LLM 미접촉) 매핑 정확도·불변식 보장이 품질 척추.

---

## 10. 산출물 (Deliverables)

1. **chunk payload 생성기:** `apps/ingest/converters.py`(11 doc_type 변환 + chunk_text 직렬화 + chunk_id) + `apps/ingest/meta_aggregator.py` + `apps/ingest/normalizers.py` + `apps/ingest/embedder.py`.
2. **v2.0 chunk 데이터 모델:** `apps/domain/models.py`에 추가된 `ChunkPayload`/`ResearcherMeta`/`DomainAttrs`(v1.x 모델 불변).
3. **불변식 검증기:** `apps/ingest/validate_chunks.py`(§6 7개 점검 CLI + 리포트).
4. **테스트:** `tests/test_ingest_converters.py`, `tests/test_ingest_invariants.py`(GOLDEN 시나리오 1·7 포함).
5. **산출 데이터:** Qdrant upsert 직전 형태의 chunk payload 컬렉션(+ `dense_e5i`/`sparse_splade` 벡터) — WO-B(컬렉션·인덱스)·WO-D(컷오버)의 입력.
6. **정책 메모:** chunk 분할 트리거 정책 제안(open question) + researcher_meta truth 출처 정책 제안 — WO-0 계약/`MIGRATION_PLAN.md` Phase A 반영용.
