# 평가위원 추천 RAG 챗봇
## 설계 · 구현 · 제한 · 방향성 지침서

**기준선:** Qdrant v1.16.0 / Named Vector + Hybrid Retrieval / LLM Recommendation  
**문서 버전:** v1.3  
**기준일:** 2026-04-09

---

## 문서 개요

| 항목 | 내용 |
|---|---|
| 문서 목적 | 평가위원 추천 RAG 챗봇의 확정 설계, 구현 원칙, 제약 사항, 향후 방향을 개발팀·기획팀 기준으로 일관되게 문서화 |
| 고정 기준 | Qdrant v1.16.0 고정 / named vector 기반 / prefetch 동시검색 / 가중치 미사용 / 최종 추천은 LLM 판단 |
| 검색 철학 | Qdrant는 후보 생성과 정렬 보조, LLM은 비교·추천과 설명 생성 |
| 설계 단위 | 전문가 1명 = 1 point, 기본정보(root) + 논문/특허/과제 nested payload |
| 문서 버전 | v1.3 (LLM 구조적 토큰 다이어트, 기관명 정규화 튜닝, 동적 최신성 가산점 반영, 2026-04-14) |

> **이번 문서에서 확정한 핵심 원칙**
>
> - 리랭커(예: cross-encoder)를 최종 의사결정의 핵심으로 두지 않는다.
> - 하이브리드 검색 결과를 넓게 확보한 뒤, 최종 추천과 설명은 LLM이 수행한다.
> - Qdrant v1.16.0 범위 안에서 `prefetch + RRF`를 사용하며, weighted RRF는 설계 범위에서 제외한다.
> - 숫자/날짜/기관 제외 같은 hard filter는 LLM이 아니라 시스템이 보장한다.
> - 추천 결과에는 반드시 근거, 제외 사유, 데이터 공백을 함께 제시한다.
> - 구현 기본값은 `basic/art/pat/pjt` 전 branch 항상 검색이며, planner는 branch on/off 대신 branch별 query hint를 생성한다.

> **용어 정리**
>
> - **Retrieval:** Qdrant에서 후보군을 찾는 단계
> - **Recommendation:** LLM이 후보군을 비교하여 최종 추천을 만드는 단계
> - **Hard filter:** 숫자·날짜·기관 제외처럼 시스템이 강제 보장해야 하는 조건
> - **Evidence card:** LLM 입력용으로 압축한 후보 요약 카드

---

## 1. 문서 목적과 범위

이 문서는 평가위원 추천 RAG 챗봇의 기준 설계를 고정하기 위한 실행 문서이다. 목적은 기술 스택, 데이터 모델, 검색 전략, LLM 추천 규칙, 운영상 제한과 향후 방향을 한 문서에 통합하여 설계 흔들림을 줄이는 데 있다.

- **포함 범위:** 컬렉션 설계, ingestion, retrieval, LLM 추천, 필터링, 평가 지표, 운영 가이드, 확장 방향
- **비포함 범위:** 프런트엔드 UI 상세 설계, 인프라 IaC 스크립트, 보안 인증 체계의 세부 구현
- **문서의 우선순위:** 정확한 기준선 수립 > 운영 가능성 > 확장성 > 시각적 완성도

---

## 2. 확정된 설계 기준

이 기준은 이후 구현 세부사항보다 상위에 있는 설계 원칙이다. 즉, 구현 편의를 위해 가중치나 외부 리랭커를 다시 들여오는 것은 기본선 위반으로 본다.

| 항목 | 최종 결정 | 설계 의미 |
|---|---|---|
| Qdrant 버전 | v1.16.0 고정 | 기능 범위와 제약을 명확히 고정 |
| 검색 방식 | named vector + sparse vector 하이브리드 | 기본정보/논문/특허/과제별 표현 유지 |
| 동시검색 | `prefetch` 사용 | 여러 representation을 한 요청에서 조합 |
| 가중치 | 미사용 | weighted RRF 및 직접 score 가중합은 설계 범위 제외 |
| 최종 판단 | LLM 추천 | 검색 결과를 근거로 비교·판단 |
| 후보 단위 | 전문가 1명 = 1 point | 검색과 추천 단위를 일치시킴 |
| 필터 처리 | 시스템에서 deterministic 보장 | LLM 환각과 조건 누락 방지 |
| 출력 원칙 | 추천 + 근거 + 제외 사유 + 데이터 공백 | 운영자 검토 가능성 확보 |
| 임베딩 정제 | 원문 질의 배제 및 순수 도메인 키워드 조합 | "평가위원" 등 불용어로 인한 임베딩 벡터 오염을 원천 차단하여 검색 무결성 보장 |

---

## 3. 서비스 목표와 성공 기준

### 3.1 서비스 목표

이 시스템의 목표는 자연어 요청을 입력받아 적합한 평가위원 후보군을 탐색하고, 검색 결과를 근거로 LLM이 최종 추천과 설명을 생성하는 것이다. 일반 QA형 RAG와 달리, 핵심은 **정답 문장 회수**가 아니라 **후보 비교와 추천 근거 제시**에 있다.

- 사용자 질의의 주제, 맥락, 제외 조건, 최근성 기준을 이해한다.
- 논문·특허·과제·기본 프로필을 함께 반영하여 후보군을 넓게 확보한다.
- LLM이 후보를 비교하고 최종 Top-N 추천을 생성한다.
- 각 추천 결과에 대해 근거, 주의점, 데이터 공백을 명시한다.

### 3.2 성공 기준

- Hard filter 위반 없이 후보를 반환할 것
- 추천 결과 상위권에 실제로 검토 가능한 전문가가 포함될 것
- 추천 이유가 payload 근거와 일치할 것
- 운영자가 수작업으로 수정하는 비율이 점차 감소할 것

---

## 4. 전체 아키텍처

```text
[사용자]
   ↓
[LLM Query Planner]
   - 질의 요약
   - hard filter 추출
   - soft preference 추출
   - branch query hint 생성
   ↓
[Retrieval Orchestrator]
   - dense/sparse query 생성
   - prefetch 조립
   - filter 조립
   ↓
[Qdrant researcher_recommend_proto]
   - basic/art/pat/pjt dense vectors
   - basic/art/pat/pjt sparse vectors
   - root + nested payload
   ↓
[Candidate Card Builder]
   - top-N 후보를 LLM 입력용 카드로 압축
   ↓
[LLM Judge / Recommender — Map-Reduce 2단계]
   - Map: 경량 직렬화 + 경량 프롬프트로 배치 병렬 예비 심사 (max_tokens=3000)
   - Reduce: 생존자만 전체 직렬화 + 상세 프롬프트로 최종 비교
   - 추천/제외 사유 생성
   - 데이터 공백 명시
   ↓
 [응답 생성기]
    - 최종 추천 결과 반환
    ↓
 [가시성 계층 (Observability Layer)]
    - Trace ID 생성 및 비동기 컨텍스트 전파
    - 실시간 로그 캡처 및 UI 콘솔 출력
```

| 구성요소 | 역할 | 비고 |
|---|---|---|
| LLM Query Planner | 질의 요약, core_keywords 추출, hard filter 추출, branch_weights/branch_query_hints 생성 | JSON-only 출력 강제, Few-shot 예시 2건, `<thinking>` 태그 미사용 |
| Retrieval Orchestrator | dense/sparse 질의 생성, Qdrant prefetch 조립 | 원발성 질의(Raw Query)의 어투를 벗겨내고 순수 추출된 도메인 키워드로만 쿼리 뼈대를 조립하여 검색 노이즈 사전 차단 |
| Qdrant | 후보 생성, fusion, 필터 적용 | 최종 추천 판단은 담당하지 않음 |
| Candidate Card Builder | LLM 입력용 증거 요약 | 토큰 비용과 잡음을 줄이는 단계 |
| LLM Judge / Recommender | Map-Reduce 2단계 심사: Map(경량 병렬 예비 심사) → Reduce(상세 최종 추천) | 최종 의사결정의 핵심 |
| Visibility Layer | Trace ID 부여 및 실시간 한글 로깅 | **(v1.1 추가)** 전 과정 추적성 보장 |
| Audit Logger | 질의, 필터, 후보, 추천 결과 저장 | 운영 개선과 평가셋 구축에 필수 |

---

## 5. 데이터 및 컬렉션 설계

### 5.1 컬렉션 단위

권장 컬렉션은 `researcher_recommend_proto` 하나로 시작한다. 한 point는 한 명의 전문가를 의미한다. 이 구조는 추천 단위와 저장 단위를 일치시켜 retrieval 이후 후처리를 단순화한다.

### 5.2 named vector 구성

| 구분 | 벡터명 | 권장 source text | 사용 목적 |
|---|---|---|---|
| Dense | `basic_vector_e5i` | 이름/소속/직위/학위/전공/기술분류 요약 | 전문가 프로필 의미 검색 |
| Dense | `art_vector_e5i` | 대표 논문명/학술지명/초록/키워드 요약 | 학술 성과 의미 검색 |
| Dense | `pat_vector_e5i` | 발명명/특허 구분/출원·등록 정보 요약 | 지식재산 기반 의미 검색 |
| Dense | `pjt_vector_e5i` | 과제명/목표/내용/기관 요약 | 과제 경험 의미 검색 |
| Sparse | `basic_vector_bm25` | 기본정보 기반 토큰화 텍스트 | exact keyword 보강 |
| Sparse | `art_vector_bm25` | 논문명/학술지/키워드 | 논문 keyword 매칭 |
| Sparse | `pat_vector_bm25` | 발명명/특허 분류/기관명 | 특허 keyword 매칭 |
| Sparse | `pjt_vector_bm25` | 과제명/과제내용/기관명 | 과제 keyword 매칭 |

### 5.3 payload 구조

루트 레벨(`basic_info`, `researcher_profile`)과 Nested 레벨(`publications[]`, `intellectual_properties[]`, `research_projects[]`)로 구성된다. 전체 필드 목록과 필터 적용 규약은 [`CONTRACT.md`](CONTRACT.md)를 참조한다.

### 5.4 구현 전 정합성 확인 포인트

> **업로드 설계안 기준 구현 전 확인 필요**
>
> - 정정: 과제 영역 sparse vector 컬럼명은 `pjt_vector_bm25`를 사용한다.
> - 정정: 과제 날짜 매핑은 `project_start_date = TOT_RSCH_START_DT`, `project_end_date = TOT_RSCH_END_DT`, `reference_year = STAN_YR`로 구현한다.
> - 권장: 기술분류는 최소한 `basic_vector` source text에는 반드시 합성한다.
> - 권장: IRIS 계열 데이터는 1차 검색 스키마와 분리하고, candidate card enrichment 용도로 결합한다.

---

## 6. 검색 설계 (Qdrant v1.16.0 기준)

### 6.1 검색 철학

검색 단계의 목적은 **정답 1명을 바로 고르는 것**이 아니라, **적절한 후보군을 넓게 확보하는 것**이다. 따라서 retrieval은 recall 우선, recommendation은 reasoning 우선으로 설계한다.

- 기본정보/논문/특허/과제는 서로 다른 representation으로 유지한다.
- 한 질의에서 `basic/art/pat/pjt` branch는 모두 항상 검색한다.
- 가중치 대신 branch별 query hint와 candidate card 비교로 검색 전략을 보정한다.
- 최종 Top-N는 LLM이 판단하므로, retrieval 단계에서는 증거가 풍부한 후보를 충분히 수집하는 것이 중요하다.

### 6.2 Query Planner의 역할

- **질의 요약** (`intent_summary`): 사용자의 평가 주제와 추천 목적을 한 문장으로 정규화
- **핵심 키워드** (`core_keywords`): 기술 도메인 키워드 배열 (전공, 연구 분야 단어는 hard_filters가 아닌 여기에 포함)
- **Hard filter 추출** (`hard_filters`): 허용 키만 사용 — `degree_slct_nm`, `art_recent_years`, `pat_recent_years`, `pjt_recent_years`, `article_cnt_min`, `patent_cnt_min`, `project_cnt_min`, `art_sci_slct_nm`
- **제외 기관** (`exclude_orgs`): 추천 목록에서 배제할 기관명 배열
- **Soft preference 추출** (`soft_preferences`): 학술 중심, 산업체 경험, 특허 선호 등
- **브랜치 가중치** (`branch_weights`): `basic/art/pat/pjt`별 상대 중요도 (0.0~1.0). 현재 가중 RRF로는 미사용, query hint 참고용
- **Branch query hint 생성** (`branch_query_hints`): `basic / art / pat / pjt`별 벡터 유사도를 높이기 위한 한국어 검색 보조 문구
- **Top-K 정책 결정** (`top_k`): 질의 난이도와 제약 수에 따라 후보 수를 조절

#### Planner 프롬프트 설계 (v1.2)

- **JSON-only 출력 강제**: "첫 글자는 반드시 `{`, 마지막 글자는 반드시 `}`" 규칙 명시. 설명, 마크다운, 코드 펜스, `<thinking>` 태그 출력 금지.
- **스키마 + Few-shot 예시 2건**: 실제 질의와 정답 JSON을 2건 포함하여 출력 형식을 정확히 유도.
- **hard_filters 허용 키 제한**: 허용되지 않은 키 사용 시 Pydantic 검증에서 거부됨.
- **`<thinking>` 태그 미사용**: CoT 사고 과정을 별도 태그로 유도하지 않음 — LLM이 태그 안에 `{}`를 생성하여 JSON 추출을 방해하는 문제를 원천 차단.

### 6.3 권장 retrieval 파이프라인

| 단계 | 설명 | 권장 목적 |
|---|---|---|
| 1차 | branch별 dense + sparse prefetch 후 RRF | 각 영역 내부에서 의미 검색과 키워드 검색을 균형 있게 결합 |
| 2차 | branch 결과를 다시 RRF로 결합 | `basic/art/pat/pjt` 후보를 하나의 전문가 리스트로 통합 |
| 3차 | payload filter 적용 | 기관 제외, 날짜, 최소 실적 등 보장 |
| 4차 | top-N 후보 카드 생성 | LLM 입력 품질 확보 |
| 5차 | LLM 추천 | 최종 추천과 설명 생성 |

### 6.4 필터와 payload index 원칙

- 필터에 자주 쓰는 필드는 반드시 payload index 대상에 포함한다.
- 논문/특허/과제의 배열 객체 조건은 nested filter로 묶어야 한다.
- exact match가 필요한 기관명/학위/구분 값은 `keyword` 취급이 안전하다.
- 날짜와 집계값은 임베딩에 녹이지 말고 payload filter로 처리한다.
- 여러 strict filter가 겹칠 때는 ACORN 사용 여부를 성능 테스트로 결정한다.

> **권장 인덱스 후보**
>
> - root: `basic_info.researcher_id`, `researcher_profile.highest_degree`, `researcher_profile.publication_count`, `researcher_profile.scie_publication_count`, `researcher_profile.intellectual_property_count`, `researcher_profile.research_project_count`
> - `publications[]`: `journal_index_type`, `publication_year_month`
> - `intellectual_properties[]`: `application_registration_type`, `application_country`, `application_date`, `registration_date`
> - `research_projects[]`: `project_start_date`, `project_end_date`, `reference_year`, `performing_organization`, `managing_agency`

---

## 7. LLM 추천 전략

### 7.1 핵심 원칙

- LLM은 벡터 점수 자체를 신뢰의 근거로 삼지 않는다.
- LLM은 retrieval 결과 안에 존재하는 근거만 사용한다.
- 추천 사유는 반드시 payload evidence와 연결한다.
- 충분한 근거가 없으면 `추천 보류` 또는 `추가 조건 필요`를 출력한다.

### 7.2 Map-Reduce 2단계 심사 (v1.2 신규)

`NTIS_USE_MAP_REDUCE_JUDGING=true`(기본값)이면 Judge는 Map(경량 직렬화 + 경량 프롬프트, `max_tokens=3000`, `survivors` 배열 반환) → Reduce(전체 직렬화 + 상세 프롬프트, `max_tokens` 제한 없음) 2단계로 동작한다. 이 구조로 입력 토큰을 절약하고 심사 대상을 축소한 뒤 집중 비교가 이루어진다.

Map/Reduce 각 라운드의 직렬화 내용, 출력 스키마, 병렬화 설정, JSON 추출 방식은 [`SERVICE_FLOW.md §1-6~1-7`](SERVICE_FLOW.md)에 상세히 기술되어 있다.

### 7.3 Candidate Card 설계

| 카드 종류 | 용도 |
|---|---|
| Map용 경량 카드 | 프로필 핵심 + 논문/특허/과제 제목만. Map 라운드 예비 심사 입력 |
| Reduce용 전체 카드 | 초록(150자), 키워드(5개), 연구 요약(150자) 포함. Reduce 라운드 최종 심사 입력 |

I/O 필드 규격은 [`CONTRACT.md`](CONTRACT.md)를 참조한다.

### 7.4 프롬프트 가드레일 (LLM 출력 품질 보장)

- 질문에 없는 조건을 임의로 추가하지 말 것
- Hard filter를 충족하지 않는 후보는 추천하지 말 것
- 동일한 근거를 여러 번 반복하지 말 것
- 근거가 약한 경우 `추측`이 아니라 `근거 부족`으로 표현할 것
- 추천 결과뿐 아니라 왜 1순위/2순위인지 비교 문장을 포함할 것
- `rank_score`는 상대적 RRF 점수이며 절대적 적합도 백분율이 아님을 명시

---

## 8. 구현 가이드

### 8.1 Ingestion 파이프라인

| 단계 | 작업 | 비고 |
|---|---|---|
| 원천 적재 | NTIS / IRIS 원천 테이블 수집 | 버전, 수집일, 원천 식별자 관리 |
| 정규화 | 기관명/학위/국가/구분값 표준화 | 필터 오작동 방지 |
| 집계 생성 | 논문수·SCIE수·특허수·과제수 계산 | root payload에 저장 |
| source text 생성 | `basic/art/pat/pjt`별 벡터 입력 텍스트 생성 | 필드 그대로 이어붙이지 않고 목적별 요약 |
| 임베딩 생성 | dense + sparse 벡터 생성 | 모델 버전 고정 필요 |
| 적재 | point upsert | 벡터/페이로드 정합성 검증 |
| 검증 | 샘플 질의 기반 스모크 테스트 | 필터, evidence, 추천 결과 동시 확인 |

### 8.2 서비스 레이어 권장 모듈

- `schema_registry`: 벡터명, payload 필드, 필터 허용 목록 관리
- `query_planner`: LLM structured output 생성
- `retrieval_service`: prefetch query 조립과 Qdrant 호출
- `candidate_card_service`: top-N 후보 요약과 evidence 추출
- `recommendation_service`: LLM Judge 호출
- `explanation_service`: 사용자 응답 문장화
- `audit_logger`: 질의, 필터, 후보, 추천 결과, 운영자 수정 결과 저장

### 8.3 API 권장안

| API | 설명 | 응답 핵심 |
|---|---|---|
| `POST /recommend` | 질의 기반 추천 | 추천 결과, 근거, 제외 사유, 데이터 공백 |
| `POST /search/candidates` | retrieval 디버깅 | raw 후보와 각 branch 근거 |
| `POST /explain` | 기존 추천 결과 설명 확장 | 추천 사유 상세화 |
| `POST /feedback` | 운영자 수정/채택 결과 저장 | 후속 평가 데이터 축적 |

### 8.4 권장 운영 로그 (Trace ID 기반)

모든 로그는 **Trace ID(Request ID)**를 포함하며, 운영자가 즉시 이해할 수 있는 **한글 기반 메시지**로 기록한다. Planner/Retriever/Judge(Map·Reduce 라운드 별) 단계의 주요 로깅 포인트, 로그 형식, 실제 예시는 [`RUNBOOK.md §7`](RUNBOOK.md)를 참조한다.

---

## 9. 제한 사항 및 기술 제약

### 9.1 요약 표

| 구분 | 제한 사항 | 영향 | 대응 방향 |
|---|---|---|---|
| 버전 제약 | v1.16.0에서는 weighted RRF를 사용하지 않음 | branch 간 상대 강도를 수치로 조정하지 못함 | 모든 branch 검색 + query hint + judge 비교로 대응 |
| 버전 제약 | 각 named vector raw similarity의 직접 가중합은 범위 밖 | 정밀한 score engineering이 제한됨 | LLM 추천으로 최종 판단 이동 |
| 모델 제약 | LLM은 검색 밖의 사실을 알지 못함 | 환각 위험 | evidence card와 금지 규칙 강화 |
| 구조 제약 | 전문가 1 point 구조는 item-level ranking이 약함 | 특정 논문 1건이 아주 좋아도 전문가 전체 점수에 묻힐 수 있음 | 향후 `expert_id` grouping 구조를 검토 |
| 데이터 제약 | 기관명/날짜/동일인 식별 품질에 민감 | 필터/추천 품질 하락 가능 | 정규화와 검증 파이프라인 강화 |
| 운영 제약 | 질의 유형이 다양함 | branch selection 실패 가능 | planner 평가셋과 운영자 피드백 누적 |

### 9.2 추천 시스템 관점의 주의점

- 좋은 검색 점수 = 좋은 평가위원은 아니다.
- 추천에는 적합성 외에도 최근성, 실적 분포, 기관 중복, 데이터 공백이 함께 고려되어야 한다.
- 따라서 retrieval 점수 상위 순서를 그대로 사용자에게 노출하면 품질이 낮아질 수 있다.

### 9.3 데이터 품질 관점의 주의점

- 기관명 표기가 흔들리면 제외기관 필터가 실패할 수 있다.
- 날짜 필드 매핑이 잘못되면 최근 3년/5년 조건이 왜곡된다.
- 논문/특허/과제 텍스트가 지나치게 길면 임베딩 입력이 불안정해질 수 있다.
- IRIS와 NTIS의 동일 인물 식별 키 정합성이 약하면 enrichment 결과가 불안정해질 수 있다.

---

## 10. 방향성 지침

### 10.1 설계 원칙

- 단순한 retrieval 구조를 먼저 안정화하고, 추천 판단은 LLM에 위임한다.
- 필터는 항상 deterministic 하게 구현하고, 벡터는 의미 검색에만 집중시킨다.
- 한 번에 많은 기능을 넣기보다, 운영 로그를 모아 오류 유형을 먼저 분류한다.
- 추천 품질 개선은 score 미세조정보다 evidence quality와 prompt 품질 개선에서 시작한다.
- 모델 변경과 스키마 변경은 같은 시점에 동시에 하지 않는다.

### 10.2 단계별 로드맵

| 단계 | 목표 | 주요 작업 |
|---|---|---|
| MVP | 검색과 추천의 기본 루프 완성 | named vector 컬렉션, prefetch, LLM planner, LLM judge, 운영 로그 |
| Stabilization | 필터 정확도와 설명 품질 안정화 | 기관명 정규화, payload index, candidate card 개선, evaluation 셋 구축 |
| Expansion | 추천 근거 다양화 | IRIS enrichment, 평가위원 활동 데이터 반영, explain API 고도화 |
| Future | 구조적 확장 검토 | `expert_id` grouping, item-level collection, 버전 업그레이드 검토 |

### 10.3 나중에 검토할 수 있으나 지금은 고정하지 않는 항목

- Weighted RRF 도입 여부
- `FormulaQuery` 기반 payload score boost 도입 여부
- Cross-encoder 또는 별도 re-ranker 도입 여부
- expert와 item을 분리한 2컬렉션 구조
- feedback-driven retrieval 개선

---

## 11. 품질 평가와 수용 기준

### 11.1 Retrieval 평가

- Recall@50 / Recall@100
- NDCG@10
- 필터 정확도(기관 제외, 최근성, 최소 실적)
- branch별 기여도 관찰

### 11.2 Recommendation 평가

- Top-5 추천 적중률
- 운영자 수정률
- 추천 사유의 faithfulness
- 데이터 공백 탐지율
- 동일 질의 반복 시 일관성

### 11.3 수용 기준 예시(초안)

| 항목 | 초기 목표 | 설명 |
|---|---|---|
| Hard filter 위반률 | 0% | 기관 제외, 학위 조건 등 위반 금지 |
| 운영자 전면 재선정 비율 | 20% 미만 | 추천 결과가 완전히 부적합한 경우 비율 |
| 응답 시간 | 실운영 기준에서 합의 | 토큰 비용과 latency를 함께 고려 |
| 근거 누락 | 상위 추천 100% 근거 포함 | 추천마다 논문/특허/과제 등 최소 1개 이상 근거 |

---

## 12. 최종 권고안 요약

> **최종 권고**
>
> - 전문가 1명 = 1 point 구조를 유지한다.
> - `basic / art / pat / pjt` named vector와 sparse vector를 함께 쓴다.
> - Qdrant v1.16.0에서는 `prefetch + equal RRF`를 기본으로 한다.
> - 가중치와 직접 score 융합은 설계에서 제외한다.
> - 최종 추천은 LLM이 하며, retrieval은 후보 확보에 집중한다.
> - Hard filter는 시스템이 강제 보장한다.
> - Candidate card 단계로 LLM 입력을 압축한다.
> - 운영 로그와 평가셋을 먼저 쌓고, 개선은 점진적으로 한다.

---

## 부록 A. 업로드 설계안 정합성 메모

아래 항목은 업로드된 스프레드시트 기준으로 구현 전에 한 번 더 확인해야 하는 부분이다. 이 항목들은 설계 방향의 오류라기보다, 실제 구현 시 버그로 이어질 수 있는 매핑 이슈 후보이다.

| 항목 | 관찰 내용 | 판단 |
|---|---|---|
| 과제 sparse vector명 | 구현 기준은 `pjt_vector_bm25`로 확정 | 정정 반영 |
| 과제 날짜 매핑 | 구현 기준은 `project_start_date = TOT_RSCH_START_DT`, `project_end_date = TOT_RSCH_END_DT`, `reference_year = STAN_YR` | 정정 반영 |
| 기술분류 반영 | 기술분류 시트는 있으나 메인 스키마 반영이 약함 | 권장 보완 |
| IRIS 결합 방식 | IRIS payload는 많지만 retrieval 스키마로 바로 넣기엔 복잡도 큼 | enrichment 용도 권장 |

---

## 부록 B. 참고 자료 (공식 문서)

- https://api.qdrant.tech/api-reference/search/query-points
- https://qdrant.tech/documentation/concepts/hybrid-queries/
- https://qdrant.tech/documentation/manage-data/collections/
- https://qdrant.tech/documentation/search/filtering/
- https://qdrant.tech/documentation/manage-data/indexing/
- https://qdrant.tech/documentation/search/search/
- https://qdrant.tech/documentation/search-precision/automate-filtering-with-llms/
- https://qdrant.tech/blog/qdrant-1.16.x/
- https://qdrant.tech/blog/qdrant-1.17.x/
