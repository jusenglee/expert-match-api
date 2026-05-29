# WO-0 · 안정화 & 계약 고정

> 공통 마이그레이션 컨텍스트(정체성/파이프라인/v1.x↔v2.0/HARD 제약/WO 그래프)는 별도 공통 문서를 참조한다. 본 WO 본문에서는 반복하지 않고 필요한 지점에서만 인용한다.

---

## 1. 개요

- **목적:** 데이터(WO-A)·Qdrant(WO-B)를 손대기 전에, v2.0 전환의 **단일 의존점이 되는 코드 계약 상수**를 확정하고 **CI를 green으로 복구**한다. 본 WO는 전부 **로컬·비파괴**다 — 기존 v1.x 런타임 동작과 `researcher_recommend_proto` 컬렉션을 일절 건드리지 않는다.
- **한 줄 요약:** `chunk_id` 코덱 + 11 doc_type/4 family 상수 모듈 + CrossEncoder 계약 표면(스텁 또는 스킵) + 네이밍 정리 가이드를 만들어, WO-A(코덱 사용)·WO-C(enum/family/네이밍 사용)가 의존할 안정적 표면을 고정한다.
- **이 WO 완료 시 달성 상태:**
  - `pytest` 전체 suite가 **import 실패 없이** 수집·통과(green). 현재 `tests/test_cross_encoder_evidence_selector.py`(git `??` untracked)가 존재하지 않는 `CrossEncoderEvidenceSelector`/`rerank_source`를 import해 suite 전체가 red인 상태가 해소된다.
  - `chunk_id` 생성/파싱 함수와 11 doc_type→PREFIX 매핑이 코드 상수로 존재하고, 샘플 payload(`전체 샘플 payload (도메인별).txt`)의 실제 chunk_id와 라운드트립한다.
  - 11 doc_type enum + 4 family 매핑이 **단일 출처 코드 상수**로 존재(신규 모듈 `apps/search/doc_types.py`).
  - `researcher_id`/`expert_id` 혼재 정리 가이드와 `service._chunk_candidates()` 개명 지시가 WO-C 진입 전 문서로 확정된다.

---

## 2. 선행조건 & 의존성

- **선행 WO:** 없음. **본 WO는 WO-A / WO-B / WO-C 전체의 선행이다. 본 WO 완료(=수용 기준 §6 전부 충족) 전에는 WO-A/B/C 착수 금지.**
- **VPN/환경:** **불필요(완전 로컬).** Qdrant 접속·임베딩·LLM 호출 없음. 파이썬 가상환경 + `pytest`만 있으면 된다.
- **차단 요소:** 현재 `pytest`가 import 단계에서 깨져 있어 어떤 회귀 검증도 신뢰할 수 없다(아래 §4-A에서 최우선 복구). 이 복구 전에는 본 WO 내 다른 작업의 "테스트 green" 확인이 불가능하므로 **§4-A를 가장 먼저 처리**한다.

---

## 3. 범위

### In Scope (이 WO에서 한다)

1. **CI 복구** — `tests/test_cross_encoder_evidence_selector.py`의 import 실패 해소(아래 두 갈래 중 택1, §4-A).
2. **`chunk_id` 코덱** — `PREFIX_researcher_id_seq_c#` 생성/파싱 결정적 유틸 + 11 doc_type→PREFIX 매핑(신규).
3. **doc_type/family 상수 모듈** — 11 doc_type enum + 4 family 매핑 단일 출처(신규 `apps/search/doc_types.py`).
4. **네이밍 정리 방침(문서)** — `researcher_id`↔`expert_id` 혼재 정리 가이드 + `service._chunk_candidates()` 개명 지시.

### Out of Scope (이 WO에서 하지 않는다 — 후속 WO로)

- 실제 Qdrant 컬렉션 생성·인덱스(→ **WO-B**).
- 소스→chunk 변환·적재(→ **WO-A**, 단 본 WO의 `chunk_id` 코덱을 사용).
- 런타임 파이프라인 chunk화: `retriever.py` RRF 집계의 chunk→researcher 전환, `filters.py`의 `doc_type` 필터/`event_year` recency 전환, `seed_data.py` 1연구자=N point 폐기, `BRANCHES`/`BRANCH_WEIGHTS`/`PATH_WEIGHTS`/`*_vector_e5i`/`*_vector_splade`/named-vector 분기 **제거**, `reasoner.py`의 `VALID_EVIDENCE_ID_PATTERN`(`reasoner.py:44`)을 `chunk_id` 패턴으로 교체, `evidence_selector`의 `paper:`/`project:`/`patent:` item_id(→ `chunk_id`) 교체, family cap 실제 적용 등 **동작 변경 일체**(→ **WO-C**).
- 컷오버·live 검증(→ **WO-D**).
- 본 WO는 **신규 상수/유틸 추가만** 한다. 기존 v1.x 상수(`schema_registry.BRANCHES` 등) **삭제·수정 금지** — 삭제는 WO-C.

---

## 4. 상세 작업 항목

> 각 항목은 STEP 1에서 직접 확인한 실제 file:line 심볼에 근거한다. "신규 생성"으로 명시된 심볼은 현재 repo에 존재하지 않는다(확인 완료).

### A. CI 복구 — `test_cross_encoder_evidence_selector.py` import 실패 해소 (최우선)

**현황(확인):**
- `tests/test_cross_encoder_evidence_selector.py:16-19`가 `from apps.recommendation.evidence_selector import (CrossEncoderEvidenceSelector, KeywordEvidenceSelector,)`를 import한다. 그러나 `apps/recommendation/evidence_selector.py`에는 `KeywordEvidenceSelector`(:126)와 `EvidenceSelector` Protocol(:47)만 있고 **`CrossEncoderEvidenceSelector`는 없다**. → 모듈 import 단계에서 `ImportError` → **suite 전체 수집 실패(red)**.
- 또한 테스트는 `papers[0].rerank_source`(:106) 등 결과 아이템의 `rerank_source` 속성을 단언하나, `RelevantEvidenceItem`(`evidence_selector.py:23-31`)에 `rerank_source` 필드가 **없다**(repo 전역 `rerank_source` 출현 0회 확인). `match_score`(:31)는 존재.
- 테스트가 기대하는 `CrossEncoderEvidenceSelector` 생성자 계약(`tests/...:79-87`): `scorer=`, `fallback=KeywordEvidenceSelector(...)`, `top_n_per_type=5`, `relevance_floor=0.30`, `pregate_per_type=20`, `max_pairs_per_request=256`. 동작 단언: 점수 정렬(`:102-107`), floor drop + `last_trace["candidate_evidence_counts"][0]["dropped_below_floor"]`(`:122-124`), top-N cap(`:138-140`), pregate(`scorer.total_pairs == 2`, `:151`), dedup(`dedup_dropped`, `:166-168`), scorer 없음/예외 시 lexical fallback(`last_trace["mode"]=="lexical_fallback"`, `fallback_reason`=`"no_scorer"`/`"scorer_error"`, `:177-191`), semantic_query 우선(`last_trace["query"]`, `scorer.calls[0][0][0]`, `:204-206`).
- git 상 이 파일은 **untracked(`??`)** (확인 완료).

**택1 — 두 갈래와 선택 기준:**

- **(a) 빠른 green: 테스트를 skip/제거.** `tests/test_cross_encoder_evidence_selector.py` 최상단에 모듈 레벨 `pytestmark = pytest.mark.skip(reason="CrossEncoderEvidenceSelector는 WO-C에서 구현. WO-0은 계약 표면만 고정. WO-C STEP에서 skip 해제할 것.")`를 추가하되, **`import pytest`만 두고 깨지는 import(:16-19의 `CrossEncoderEvidenceSelector`)를 모듈 collect 시점에 평가되지 않도록** 한다(skip 마크만으로는 모듈 본문 import가 먼저 실행되어 `ImportError`가 그대로 발생하므로, `pytest.importorskip` 또는 `try/except ImportError → pytest.skip`로 import 자체를 가드해야 실효). untracked 파일이므로 제거(파일 삭제)도 동등하게 허용한다.
  - **선택 기준:** WO-C에서 CrossEncoder 구현 일정이 임박하지 않거나, WO-0의 목표가 "CI 신호 복구"에만 한정될 때. **단점:** 계약 표면(클래스/필드)이 실제로 존재하지 않으므로 WO-A/WO-C가 의존할 "고정점"이 약해진다.
- **(b) 권장: 최소 스텁으로 실제 통과.** `evidence_selector.py`에 `CrossEncoderEvidenceSelector`를 **신규 생성**하고, `RelevantEvidenceItem`(`evidence_selector.py:23-31`)에 `rerank_source: str = "lexical"` 필드를 **신규 추가**해, 위 테스트가 **실제로 통과**하도록 구현한다.
  - **선택 기준:** WO-C가 CrossEncoder를 1급 selector로 승격하는 것이 확정되어 있으므로(공통 컨텍스트 v2.0 목표), 계약 표면을 지금 고정하면 WO-C는 "내부 채움"만 하면 된다. **이 WO는 (b)를 채택한다.**

**A-1. `RelevantEvidenceItem`에 `rerank_source` 필드 신규 추가** — `[apps/recommendation/evidence_selector.py:23-31]`
- `match_score: float = 0.0`(:31) 다음에 `rerank_source: str = "lexical"` 추가. `KeywordEvidenceSelector`가 만드는 아이템(`:209-223`, `:250-265`, `:290-305`)은 기본값 `"lexical"`을 그대로 갖는다(명시 변경 불필요). CrossEncoder 경로에서만 `"cross_encoder"`로 설정.
- **근거:** 테스트 `:106` 단언. 외부 계약 노출 대상 아님(내부 trace/품질 메타). ADR 0005/REASONER_RUNTIME_POLICY와 무관한 grounding-only 메타이며 **후보 순위에 영향 0**(HARD 제약 5).

**A-2. `CrossEncoderEvidenceSelector` 최소 스텁 신규 생성** — `[apps/recommendation/evidence_selector.py:47 이후, EvidenceSelector Protocol 준수]`
- 생성자 시그니처는 테스트(`:79-87`)와 일치: `(*, scorer, fallback, top_n_per_type=5, relevance_floor=0.30, pregate_per_type=20, max_pairs_per_request=256)`. `last_trace: dict` 보유.
- `select(*, candidates, plan)`은 `EvidenceSelector` Protocol(`:47-53`) 시그니처를 만족하고 `dict[str, RelevantEvidenceBundle]` 반환.
- 동작(테스트가 단언하는 최소 집합):
  1. **scorer 부재/예외 → lexical fallback.** `scorer is None`이면 `last_trace["mode"]="lexical_fallback"`, `fallback_reason="no_scorer"`; `scorer.score(...)`가 예외면 `fallback_reason="scorer_error"`. 어느 쪽이든 `fallback`(KeywordEvidenceSelector) 결과를 반환(`:171-191`).
  2. **query 선택:** `plan.semantic_query`가 비어있지 않으면 그것을, 아니면 `plan.core_keywords` 조합을 query로 사용. `last_trace["query"]`와 scorer에 전달되는 pair의 query 측이 일치(`:204-206`).
  3. **dedup→pregate→score→floor→cap 순서:** 동일 `(title, year)` dedup(`dedup_dropped` 집계, `:154-168`) → 타입별 `pregate_per_type`개로 자른 뒤에만 scorer 호출(`total_pairs` 제한, `:143-151`) → `relevance_floor` 미만 drop(`dropped_below_floor` 집계, `:110-124`) → score 내림차순 정렬 후 `top_n_per_type` cap(`:127-140`). 결과 아이템 `rerank_source="cross_encoder"`, `match_score`=정규화 점수.
  4. `last_trace["candidate_evidence_counts"]`는 `[{... dedup_dropped, dropped_below_floor ...}, ...]` 형태로 후보별 카운트를 담는다(테스트가 `[0]` 인덱스로 접근).
- **scorer 프로토콜:** `score(pairs: list[tuple[str,str]]) -> list[float]`(테스트 `FakeScorer.score`, `:33-45`). 본 WO 스텁은 **실제 cross-encoder 모델을 로드하지 않는다** — 주입된 scorer만 사용. 실제 모델 어댑터·DI 와이어링·family cap(achievement10/assessment6/expertise6/identity1) 실제 적용은 **WO-C**.
- **HARD 제약(가드):** evidence 리랭커는 **grounding 선별만** 한다 — 후보(연구자) 순위·탈락·생성에 영향 0(제약 5). 후보 cross-encoder 리랭커(`NTIS_CANDIDATE_RERANKER`)와 **별개**이며 혼동 금지(제약 2).

**A-3. skip 해제 메모(택a를 쓸 경우만):** WO-C 진입 STEP에 "이 테스트 skip 해제 + CrossEncoder 실구현" 항목이 들어가도록 본 WO 산출물에 명시. (b)를 채택하면 불필요.

### B. `chunk_id` 코덱 — 신규 생성

**현황(확인):**
- 샘플 payload의 실제 chunk_id 형식(라운드트립 기준 진실):
  - publication → `PUB_M1006328_0001_c0` (`전체 샘플…txt:10`), doc_id `PUB_M1006328_0001` (:9)
  - intellectual_property → `IP_M1006328_0001_c0` (:48), doc_id `IP_M1006328_0001` (:47) — **주의: PREFIX는 `IP`** (목록 표기 "IP"와 일치)
  - research_project → `PJT_M1006328_0001_c0` (:77)
  - researcher_assessor → `RAS_..._c0` (:104), expert_assessor → `EAS_..._c0` (:130)
  - researcher_tech → `RTC_..._c0` (:155), expert_tech → `ETC_..._c0` (:177)
  - researcher_core → `RCO_..._c0` (:199), researcher_major → `RMJ_..._c0` (:226)
  - expert_specific → `ESP_..._c0` (:253), profile → `PRF_..._c0` (:274)
- 형식 분해: `{PREFIX}_{researcher_id}_{seq}_c{chunk_index}`. 단, `researcher_id`는 `M1006328`처럼 `_`를 포함하지 않으나 **방어적으로** 파서는 "끝에서부터 `_c{n}`과 직전 `_{seq}`를 떼고, 첫 토큰을 PREFIX, 그 사이를 researcher_id로" 해석하도록 한다(researcher_id에 `_`가 들어와도 깨지지 않게). `seq`는 doc 단위 시퀀스(예 `0001`), `chunk_index`는 chunk 분할 인덱스(예 `0`).

**B-1. PREFIX↔doc_type 매핑 상수 신규 생성** — `[apps/search/doc_types.py (신규)]`
- 11 doc_type → PREFIX 양방향 매핑(빠짐없이): `publication→PUB`, `intellectual_property→IP`, `research_project→PJT`, `researcher_assessor→RAS`, `expert_assessor→EAS`, `researcher_tech→RTC`, `expert_tech→ETC`, `researcher_core→RCO`, `researcher_major→RMJ`, `expert_specific→ESP`, `profile→PRF`.
- `DOC_TYPE_TO_PREFIX: dict[str,str]`과 역매핑 `PREFIX_TO_DOC_TYPE`을 두되 **단일 출처**에서 파생(한쪽을 dict로 두고 다른쪽은 `{v:k for ...}`로 생성, 양쪽 손수 작성 금지). PREFIX 전역 유일성을 모듈 로드 시 assert로 검증.

**B-2. 생성/파싱 결정적 유틸 신규 생성** — `[apps/search/doc_types.py 또는 apps/search/chunk_id.py (신규)]`
- `build_chunk_id(doc_type: str, researcher_id: str, seq: int|str, chunk_index: int) -> str`:
  - `f"{DOC_TYPE_TO_PREFIX[doc_type]}_{researcher_id}_{seq:04d 정규화}_c{chunk_index}"`. `seq`는 4자리 zero-pad(`0001`)로 정규화(샘플과 일치). doc_type 미정의 시 명시적 `ValueError`(unknown doc_type).
  - **순수 함수·부수효과 없음.** 동일 입력 → 동일 출력(결정성).
- `parse_chunk_id(chunk_id: str) -> ChunkIdParts(prefix, doc_type, researcher_id, seq, chunk_index)`:
  - 끝의 `_c{int}` → `chunk_index`, 직전 `_{seq}` → `seq`, 첫 토큰 → `prefix`(→ `PREFIX_TO_DOC_TYPE`로 `doc_type` 복원), 나머지 가운데 → `researcher_id`. 알 수 없는 PREFIX/형식 위반 시 `ValueError`.
- `build_doc_id(doc_type, researcher_id, seq)`(선택): chunk 접미사 `_c{n}` 없이 `PUB_M1006328_0001` 형태. 샘플 `doc_id`(`:9` 등)와 일치.
- **계약 명문화(docstring/주석):** ① 결정성(같은 chunk → 같은 id) ② 전역 유일성(researcher_id+doc_type+seq+chunk_index 조합이 유일) ③ idempotent upsert(Point ID=chunk_id이므로 재적재 시 덮어쓰기). 이는 DATA_MODEL §6-1(Point ID==chunk_id, 전역 유일), ADR 0002(멱등 적재), ADR 0004(evidence 참조 안정성)의 코드 반영.
- **라운드트립 불변식:** 모든 11 샘플 chunk_id에 대해 `parse(build(parse(x))) == parse(x)` 및 `build(*parse(x)) == x`가 성립(테스트로 고정, §7).

**B-3. WO 간 연결 명시:** 본 코덱은 **WO-A가 소스→chunk 변환 시 chunk_id/doc_id 생성에 사용**하고, **WO-C가 evidence id(`chunk_id`)·`reasoner.VALID_EVIDENCE_ID_PATTERN` 교체·trace 파싱에 사용**한다. 두 WO의 **단일 의존점**이다.

### C. doc_type / family 상수 모듈 — 신규 생성

**현황(확인):** v1.x는 `apps/search/schema_registry.py:19-24`의 `BRANCHES = ("basic","art","pat","pjt")`와 그에 묶인 `DENSE_VECTOR_BY_BRANCH`(:31-36)/`SPARSE_VECTOR_BY_BRANCH`(:43-48), `models.py:7`의 `BranchName = Literal["basic","art","pat","pjt"]`로 구성된다. v2.0 doc_type/family 상수는 **현재 코드에 없다**(repo 전역 `doc_type`/`ntis_researcher_chunks` 앱 코드 출현 0회).

**C-1. 11 doc_type enum 신규 생성** — `[apps/search/doc_types.py (신규, B와 동일 모듈)]`
- 11 doc_type 문자열을 단일 출처로(예 `StrEnum DocType` 또는 `Literal` + tuple `DOC_TYPES`). DATA_MODEL §4 표 및 §6-3 불변식("doc_type은 11종 enum 중 하나")의 근거값과 1:1.

**C-2. 4 family 매핑 신규 생성** — `[같은 모듈]`
- `DOC_TYPE_TO_FAMILY: dict[doc_type→family]`, family = `identity / achievement / assessment / expertise`:
  - **identity**: `profile`
  - **achievement**: `publication, intellectual_property, research_project`
  - **assessment**: `researcher_assessor, expert_assessor` (★v1.x 4브랜치에 없던 신규 신호)
  - **expertise**: `researcher_tech, expert_tech, researcher_core, researcher_major, expert_specific`
  - 근거: DATA_MODEL.md §4 family 표(:180-185), `0002` ADR.
- family별 chunk_cap/cap 기본값 상수(**값만 정의, 적용은 WO-C**): `FAMILY_EVIDENCE_CAP = {achievement:10, assessment:6, expertise:6, identity:1}`(공통 컨텍스트 v2.0 목표) — 본 WO는 상수 선언만, 실제 cap 적용 로직은 WO-C.
- `NTIS_DOC_TYPE_PRIORS` 기본값(equal)을 표현하기 위한 doc_type 키 목록도 본 모듈에서 파생(실제 prior 주입/소비는 WO-C). **HARD 제약 1:** doc_type 중요도는 가중 RRF가 아니라 앱단 prior로만 표현, 기본 equal.
- 모듈 로드 시 invariant assert: family 매핑이 11 doc_type을 빠짐없이/중복없이 덮는지(`set(DOC_TYPE_TO_FAMILY) == set(DOC_TYPES)`).

**C-3. 기존 BRANCHES와의 관계(이 WO에서는 공존):** `schema_registry.BRANCHES`(:19-24) 등 v1.x 상수는 **수정·삭제 금지**. 신규 doc_type/family는 별도 모듈에 **추가만** 한다. v1.x→v2.0 매핑(DATA_MODEL §7: basic→profile+전문성, art→publication, pat→intellectual_property, pjt→research_project)은 주석/문서로만 남긴다. 실제 `BRANCHES`/named-vector/`BranchName` 제거·치환은 **WO-C**.

### D. 네이밍 정리 방침 (산출=가이드 문서; 코드 개명은 충돌 항목만)

**현황(확인):**
- `researcher_id`: `BasicInfo.researcher_id`(`models.py:62`), `SeedEvidencePoint.researcher_id`(`models.py:184`), `schema_registry.py:59/80`(`basic_info.researcher_id` 인덱스), 샘플 payload L1 `researcher_id`.
- `expert_id`: `SearchHit.expert_id`(`models.py:222`), `CandidateCard.expert_id`(`models.py:249`), `RecommendationDecision.expert_id`(`models.py:272`), `RelevantEvidenceBundle.expert_id`(`evidence_selector.py:35`) 및 그 사용처(`evidence_selector.py:148/153/157/161` 등). 즉 **저장/적재 측은 `researcher_id`, 검색결과/API/카드/근거 측은 `expert_id`**로 혼재.
- `_chunk_candidates`: `service.py:510-518` 정의(배치 분할 헬퍼) + 호출 `service.py:392`, `service.py:531`. 실제로는 "후보 카드를 batch_size로 쪼개는" 함수이며 chunk(=Point) 의미와 **이름 충돌**.

**D-1. `researcher_id`↔`expert_id` 정리 가이드(문서 산출, 코드 변경은 WO-C):**
- 방침 결정: v2.0 데이터 1층 키가 `researcher_id`(DATA_MODEL §3.1, 샘플 payload)이고 집계·dedupe 기준 키이므로 **내부 식별자는 `researcher_id`로 통일**한다. 외부 API 응답 필드(`SearchHit`/`CandidateCard`/`RecommendationDecision`의 `expert_id`)는 **외부 계약**이므로 변경 여부·alias는 **EXTERNAL_API_CHANGELOG / API_SPECIFICATION을 권위로** WO-C에서 결정(BREAKING 여부 명시). 본 WO에서는 "어디가 내부/외부 경계인지" 매핑표만 확정하고 **코드는 손대지 않는다**(비파괴).
- 가이드 산출물에 위 file:line 목록을 그대로 첨부해 WO-C가 일괄 치환 대상으로 사용.

**D-2. `service._chunk_candidates()` 개명 지시:**
- `service.py:510`의 `_chunk_candidates` → **`_batch_candidates`**(또는 `_split_into_batches`)로 개명. 호출부 `service.py:392`, `service.py:531` 동반 수정. 파라미터 `batch_size`(:511), 사용 상수 `REASON_GENERATION_BATCH_SIZE`(:44)는 의미 유지.
- **근거:** v2.0에서 "chunk"는 1 Point=1 chunk라는 1급 도메인 용어가 되므로, "후보 배치 분할"을 `chunk`로 부르면 코드 전반에서 의미 충돌. 이 개명은 **순수 리네이밍(동작 불변)** 이라 본 WO에서 수행해도 비파괴.
- (이름 외 시그니처/로직 변경 금지. 배치 동작 자체 변경은 WO-C 범위.)

---

## 5. 변경/생성 대상 파일

| 파일 | 변경유형 | 요지 |
|---|---|---|
| `tests/test_cross_encoder_evidence_selector.py` | 수정(택a) / 무변경·통과(택b) | (a) import 가드+skip; (b) 채택 시 코드만 추가하고 테스트는 실제 통과 (untracked `??` → 본 WO에서 git add 대상) |
| `apps/recommendation/evidence_selector.py` | 수정 | `RelevantEvidenceItem`에 `rerank_source: str="lexical"`(:31 이후); `CrossEncoderEvidenceSelector` 최소 스텁 신규(:47 Protocol 준수) |
| `apps/search/doc_types.py` | **신규** | 11 doc_type enum + 4 family 매핑 + `DOC_TYPE_TO_PREFIX`/역매핑 + `build_chunk_id`/`parse_chunk_id`/`build_doc_id` + `FAMILY_EVIDENCE_CAP` 상수 + 로드시 invariant assert (단일 출처) |
| `apps/recommendation/service.py` | 수정 | `_chunk_candidates`(:510) → `_batch_candidates`로 개명, 호출부(:392, :531) 동반 수정 |
| (가이드 문서, 예 `apps/docs/plans/` 하위 또는 본 WO 산출 노트) | 신규/수정 | `researcher_id`↔`expert_id` 경계 매핑표 + WO-C 치환 대상 file:line 목록 |
| `tests/test_chunk_id_codec.py` (또는 기존 test에 추가) | **신규** | chunk_id 라운드트립/매핑/family invariant 단위테스트(§7) |

> v1.x 상수(`schema_registry.py:19-48`/`models.py:7` 등)는 **본 WO에서 변경하지 않는다**(표에 없음 = 의도적).

---

## 6. 수용 기준 (Acceptance Criteria)

- [ ] `pytest -q` 전체 suite가 **import 실패 없이 수집**되고 **green**이다(현재 red 원인인 `test_cross_encoder_evidence_selector.py` import 해소 확인).
- [ ] (택b 채택 시) `pytest tests/test_cross_encoder_evidence_selector.py -q`의 9개 테스트(정렬/floor/cap/pregate/dedup/no_scorer/scorer_error/semantic_query) 전부 통과. `papers[0].rerank_source == "cross_encoder"`(:106) 단언 통과.
- [ ] `RelevantEvidenceItem`에 `rerank_source` 필드가 존재하고 기본값 `"lexical"`이다.
- [ ] `apps/search/doc_types.py`가 존재하고 `from apps.search.doc_types import DocType(또는 DOC_TYPES), DOC_TYPE_TO_PREFIX, DOC_TYPE_TO_FAMILY, build_chunk_id, parse_chunk_id` 가 성공한다.
- [ ] 11 doc_type 전부에 대해 `build_chunk_id`가 샘플 payload(`전체 샘플…txt`)의 실제 chunk_id 11종과 정확히 일치한다(예: `build_chunk_id("intellectual_property","M1006328","0001",0) == "IP_M1006328_0001_c0"`).
- [ ] 11 샘플 chunk_id 전부에 대해 `parse_chunk_id`가 올바른 `(doc_type, researcher_id, seq, chunk_index)`를 복원하고 라운드트립(`build(*parse(x))==x`)이 성립한다. 알 수 없는 PREFIX/형식 위반은 `ValueError`.
- [ ] `DOC_TYPE_TO_FAMILY`가 11 doc_type을 빠짐없이/중복없이 4 family로 덮는다(`set(keys)==set(DOC_TYPES)`, family 값 ∈ {identity,achievement,assessment,expertise}). `FAMILY_EVIDENCE_CAP == {achievement:10, assessment:6, expertise:6, identity:1}`.
- [ ] `service._chunk_candidates`가 더 이상 존재하지 않고 `_batch_candidates`(개명)만 존재하며 호출부 2곳(:392, :531)이 갱신됐다(`grep _chunk_candidates` → 0건).
- [ ] `researcher_id`↔`expert_id` 경계 매핑표와 WO-C 치환 대상 목록이 산출물로 존재한다.
- [ ] **비파괴 확인:** `schema_registry.BRANCHES`/`DENSE_VECTOR_BY_BRANCH`/`SPARSE_VECTOR_BY_BRANCH`/`models.BranchName`/`reasoner.VALID_EVIDENCE_ID_PATTERN`/`seed_data` 1연구자=N point 로직이 **변경 없이 그대로**다(v1.x 런타임 동작·기존 Qdrant 컬렉션 불변).

---

## 7. 검증 방법

**단위 테스트(주 검증 — 전부 로컬):**
```bash
# 1) CI 복구 확인 — 전체 수집/통과
pytest -q

# 2) CrossEncoder 계약(택b)
pytest tests/test_cross_encoder_evidence_selector.py -q

# 3) chunk_id 코덱 + doc_type/family invariant (신규 테스트)
pytest tests/test_chunk_id_codec.py -q
```
신규 `tests/test_chunk_id_codec.py`가 검증할 것: ① 11 doc_type build 결과 == 샘플 11 chunk_id ② parse 라운드트립 ③ unknown PREFIX/형식 → `ValueError` ④ `DOC_TYPE_TO_FAMILY` 커버리지·중복 없음 ⑤ `DOC_TYPE_TO_PREFIX` 값 전역 유일.

**개명 회귀 확인:**
```bash
# 0건이어야 한다
rg -n "_chunk_candidates" apps/
# 신규 모듈 import 가능 여부
python -c "from apps.search.doc_types import DOC_TYPE_TO_PREFIX, DOC_TYPE_TO_FAMILY, build_chunk_id, parse_chunk_id; print(build_chunk_id('publication','M1006328','0001',0))"
```

**health/수동(스모크, 비파괴 확인용 — 동작 변경 없으므로 통과해야 정상):**
```bash
uvicorn apps.api.main:app --port 8011 --reload   # 기동/헬스만 확인. v1.x 동작 그대로
```

**golden/live(본 WO에서는 회귀 가드로만):** 본 WO는 동작을 바꾸지 않으므로 기존 golden 테스트가 **변동 없이** 통과해야 한다. `ntis-validate-live`는 Qdrant/네트워크가 필요하므로 본 WO 수용 기준에는 포함하지 않는다(=WO-D). 단 로컬에서 기존 golden suite가 깨지지 않음을 `pytest`로 확인한다(GOLDEN_TESTS.md 참조).

---

## 8. 리스크 & 가드레일

**이 WO에서 위반 금지 HARD 제약(전부 "보존"이 목표 — 본 WO는 비파괴):**
1. **equal RRF 보존:** 후보 순위 척추는 equal RRF(FusionQuery). 본 WO는 RRF 코드(`retriever.py:687` rrf_contribution 등)를 **건드리지 않는다**. doc_type 중요도는 `FAMILY_EVIDENCE_CAP`/`NTIS_DOC_TYPE_PRIORS`(상수 선언만, 기본 equal)로만 표현 — 가중 RRF/score 가중합 금지.
2. **후보 리랭커 OFF 유지:** `CrossEncoderEvidenceSelector`는 **evidence(근거) 선별용**이며 후보 cross-encoder 리랭커(`NTIS_CANDIDATE_RERANKER=off`)와 **완전 별개**다. 본 WO 스텁이 후보 순위/탈락/생성에 손대지 않음을 코드 리뷰로 확인(제약 2·5).
3. **evidence 리랭커는 grounding 선별만, 후보 순위 영향 0:** `rerank_source`/`CrossEncoderEvidenceSelector`는 어느 후보를 추천할지에 영향 0. select()는 `dict[expert_id→bundle]`만 만든다.
4. **OR recency 가드 보존:** `filters.py:169-175`의 OR(min_should, min_count=1) 가드는 본 WO에서 **읽지도 건드리지도 않는다**(WO-C/B 대상). 과거 다중 doc_type AND로 0건 난 장애 교훈 — 후속 WO가 절대 AND로 회귀하지 않도록 본 WO 산출 가이드에 경고 인용.
5. **LLM no-rerank:** 본 WO는 reasoner를 변경하지 않음. `VALID_EVIDENCE_ID_PATTERN`(`reasoner.py:44`, 현재 `^(paper|project|patent):\d+$`)의 `chunk_id` 패턴 교체는 **WO-C**다 — 본 WO에서 chunk_id 코덱을 만들되 reasoner에 연결하지 않는다(연결 시 v1.x 런타임이 깨짐).
6. **chunk_id 계약:** 결정성·전역 유일성·idempotent upsert를 코덱 docstring/테스트로 고정(DATA_MODEL §6, ADR 0002/0004). WO-A/WO-C가 이 단일 출처를 사용.
7. **researcher_meta 비정규화(불변식 인지):** 본 WO는 적재를 안 하지만, `researcher_meta`가 한 연구자의 모든 chunk에 동일해야 한다는 불변식(DATA_MODEL §6-2)을 WO-A 검증 항목으로 가이드에 명시(lockstep 갱신).

**회귀 위험 & 회피책:**
- **위험: `evidence_selector.py` 수정으로 `KeywordEvidenceSelector` 동작 변동.** → 회피: `RelevantEvidenceItem`에 **기본값 필드 추가만**(기존 생성자 호출 무변), `KeywordEvidenceSelector`(:126-371) 로직 **무수정**. 기존 evidence 관련 테스트 재실행으로 확인.
- **위험: import 가드(택a) 부작용으로 다른 테스트가 조용히 skip.** → 회피: import 가드는 해당 파일 한정. `pytest -q` 수집 카운트를 변경 전후 비교.
- **위험: `_chunk_candidates` 개명 시 호출부 누락.** → 회피: `rg _chunk_candidates apps/` 0건 확인을 수용 기준화.
- **위험: 코덱 PREFIX 오타(특히 IP vs PJT).** → 회피: 11 샘플 chunk_id 전수 라운드트립 테스트(§7)로 고정. `intellectual_property→IP`, `research_project→PJT` 분리 확인.

---

## 9. 작업 분할 & 예상 규모

권장 순서(A를 가장 먼저 — CI가 살아야 이후 검증 신뢰 가능):

1. **A. CI 복구** (택b: `rerank_source` 필드 + `CrossEncoderEvidenceSelector` 스텁) — `pytest` green 확보. *난이도 中*(테스트가 정의한 동작 계약이 구체적이라 스텁이라도 dedup/pregate/floor/cap 로직 필요). A-1↔A-2 순차.
2. **B + C. doc_types.py 신규 모듈** (코덱 + 11 enum + family + 상수 + invariant) — A와 **병렬 가능**(독립 파일). *난이도 中*(매핑/파서 결정성·테스트가 핵심).
3. **D-2. `_chunk_candidates` 개명** — A/B/C와 **병렬 가능**(독립). *난이도 低*(순수 리네이밍).
4. **D-1. 네이밍 가이드 문서** — B/C 완료 후가 자연스러움(매핑표에 신규 상수 참조). *난이도 低*.
5. **테스트 작성** `test_chunk_id_codec.py` — B/C 직후. *난이도 低~中*.

전체 rough 규모: **小~中.** 신규 코드는 한 모듈(`doc_types.py`) + selector 스텁 한 클래스 + 테스트 한 파일 수준. 동작 변경이 없어 회귀 표면이 좁다. 가장 큰 변경(런타임 chunk화)은 전부 WO-C로 이연된다.

---

## 10. 산출물 (Deliverables)

1. **green CI:** `pytest -q` 전체 통과(스크린샷/로그). 택b 시 `test_cross_encoder_evidence_selector.py` 9 테스트 통과.
2. **`apps/search/doc_types.py`** (신규): 11 doc_type enum + `DOC_TYPE_TO_FAMILY`(4 family) + `DOC_TYPE_TO_PREFIX`/역매핑 + `build_chunk_id`/`parse_chunk_id`/`build_doc_id` + `FAMILY_EVIDENCE_CAP` + 로드시 invariant assert. **WO-A·WO-C의 단일 의존점.**
3. **`apps/recommendation/evidence_selector.py`** (수정): `RelevantEvidenceItem.rerank_source` 필드 + `CrossEncoderEvidenceSelector` 최소 스텁(WO-C에서 실모델 어댑터·family cap·DI 채움).
4. **`apps/recommendation/service.py`** (수정): `_chunk_candidates` → `_batch_candidates` 개명 + 호출부 갱신.
5. **`tests/test_chunk_id_codec.py`** (신규): chunk_id 라운드트립·11 샘플 일치·family/PREFIX invariant 단위테스트.
6. **네이밍 정리 가이드** (문서): `researcher_id`↔`expert_id` 내부/외부 경계 매핑표 + WO-C 일괄 치환 대상 file:line 목록 + (택a 채택 시) "WO-C에서 CrossEncoder 테스트 skip 해제" 메모.
7. **WO-A/WO-C로의 연결 노트:** WO-A는 chunk_id 코덱을 적재 변환에 사용, WO-C는 enum/family/코덱을 런타임 파이프라인·`reasoner.VALID_EVIDENCE_ID_PATTERN` 교체에 사용. WO-B(컬렉션·인덱스)는 DATA_MODEL §5 인덱스 권장안과 본 모듈 doc_type 값을 정렬 기준으로 사용.

> **다음 단계:** 본 WO 수용 기준 §6 전부 충족 후에만 **WO-A(데이터 적재) ∥ WO-B(Qdrant 컬렉션) ∥ WO-C(코드 변경) 코드 골격** 병렬 착수 가능. 그 전 착수 금지.
