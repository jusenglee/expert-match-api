# 운영 매뉴얼 (RUNBOOK) — chunk 재설계

**문서 버전:** v2.0 (2026-05-28)

전문가 추천 시스템의 설치·실행·점검·모니터링 지침. 데이터 모델은 [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md), 환경 변수는 [`ENVIRONMENT.md`](ENVIRONMENT.md).

## 1. 패키지 설치

Python 3.12+ 환경에서:
```powershell
python -m pip install -e .[dev]
```

## 2. Qdrant 및 데이터 준비 (chunk 컬렉션)

- Qdrant 서버 구동 및 `NTIS_QDRANT_URL` 접근 확인.
- 기본 컬렉션 이름은 `ntis_researcher_chunks`(`NTIS_QDRANT_COLLECTION_NAME`으로 override).
- 컬렉션 스키마(필수):
  - Point ID = `chunk_id` 문자열
  - named vector: `dense_e5i`(1024, Cosine) + `sparse_splade`
  - payload 인덱스: `researcher_id`, `doc_type`, `tags`, `event_year`, `researcher_meta.*_count`(8종), 주요 기관/구분 keyword 필드 ([`DATA_MODEL.md §5`](../architecture/DATA_MODEL.md))
- 시작 시, 실제 선택된 sparse backend에 맞춰 sparse vector modifier(`IDF` 또는 없음)를 자동 확인·복구한다.

### 2.1 적재(ingestion) 불변식 점검

적재 데이터는 [`DATA_MODEL.md §6`](../architecture/DATA_MODEL.md)의 불변식을 만족해야 한다. 운영 표본 점검 항목:
- Point ID == `chunk_id`, 전역 유일.
- 한 `researcher_id`의 모든 chunk에서 `researcher_meta`/`researcher_name` 동일.
- `doc_type` ∈ 정의된 11종.
- `event_year` == `event_date`의 연도(둘 다 null 허용).
- dense/sparse가 동일 `chunk_text`에서 생성.
- `chunk_text`에 요청 어투(role/action 불용어) 미포함.

## 3. 준비 상태 점검 (Readiness)

추천 호출 전 순서:
1. `ntis-validate-live` (CLI)
2. `GET /health`
3. `GET /health/ready`

`ntis-validate-live`는 앱 startup과 동일한 sparse backend 선택 로직을 쓴다. local/online PIXIE가 모두 실패해 `Qdrant/bm25` fallback이면 sparse modifier 기대값은 `IDF`다.

`/health/ready`가 `503`/`ready:false`면 점검:
- 컬렉션 존재 여부
- named vector `dense_e5i`/`sparse_splade` 존재 여부
- sparse vector modifier 값(IDF/none)
- 필수 payload 인덱스(`researcher_id`, `doc_type`, `event_year`, `researcher_meta.*_count` 등) 생성 여부
- 유효 샘플 Point 존재 및 구조(`chunk_id`, `doc_type`, `researcher_meta`, `domain_attrs`)

## 4. 서버 실행

```powershell
uvicorn apps.api.main:app --host 0.0.0.0 --port 8011 --reload
```

- **LLM 일관성 모드:** 플래너/리즈너는 고정 저변동 샘플링(`temperature=0.0`, `top_p=0.2`, `reasoning_effort=low`, `include_reasoning=false`, `disable_thinking=true`)을 사용. 튜닝은 코드 변경 필요.
- **임베딩:** `NTIS_EMBEDDING_BACKEND=local`이면 `multilingual-e5-large-instruct` 폴더가 온전해야 함. chunk dense 입력은 `chunk_text` 단일 필드.
- **Sparse:** `로컬 PIXIE → online PIXIE(telepix/PIXIE-Splade-v1.0) → Qdrant/bm25` 순. SPLADE면 modifier 없음, bm25 fallback이면 `IDF`.
- **추천 사유:** `/recommend` 시 LLM이 숏리스트(Top-k) 대상으로 chunk 근거 기반 사유를 생성.

## 5. 브라우저 플레이그라운드

```text
http://127.0.0.1:8011/
```
상태 배지(녹색=정상), 분석 결과 창, 실시간 Trace ID 기반 한글 로그 콘솔.

## 6. API 테스트 (curl)

```powershell
curl -X POST http://127.0.0.1:8011/search/candidates `
     -H "Content-Type: application/json" `
     -d "{\"query\":\"AI 반도체 분야 SCIE 논문 실적이 우수한 평가위원 후보를 찾아줘\"}"
```
```powershell
curl -X POST http://127.0.0.1:8011/recommend `
     -H "Content-Type: application/json" `
     -d "{\"query\":\"AI 반도체 설계 과제 경험과 평가위원 활동 이력이 있는 전문가를 추천하고 특정 기관은 제외해줘\", \"exclude_orgs\":[\"A기관\"]}"
```

## 7. 가시성 및 로깅

모든 요청에 Trace ID를 부여한다. 로그 형식:
`[HH:MM:SS.mmm] [레벨] [trace=TraceID] [METHOD /path] [모듈명] 메시지`

주요 로그 포인트:
- **요청 컨텍스트:** Trace ID, method/path, client, content type/length, user agent, 처리 시간
- **사용자 질의:** endpoint, 정규화 전/후 길이, `top_k`, include/exclude 기관 수, filter override key
- **Planner:** `retrieval_core`, `core_keywords`, `role_terms`, `action_terms`, `semantic_query`, hard filter 값
- **Retriever:** 실제 `retrieval_keywords`, 1차 sparse 키워드 쿼리, 2차 하이브리드 쿼리, `researcher_id` 풀 미리보기, doc_type 경로별 hit count, hard filter 통과/탈락 수, 연구자 집계 후보 수
- **Recommendation:** Top-k 확정, evidence 선별 chunk 수, 사유 생성 배치 수, 최종 추천/데이터 공백 수
- **Fallback:** LLM 오류/JSON 파싱 실패 시 휴리스틱·결정론적 전환 안내
- **Data Gap:** 특정 연구자의 데이터 누락 경고

### 로그 예시 (질의 → 플래너 → 1차 → 2차 → 집계)
```
[09:45:51.125] [INFO] [trace=abc123] [POST /recommend] [apps.api.main] 사용자 질의 수신: endpoint=/recommend top_k=5 exclude_orgs=1 query='드론 화재 진압 평가위원 추천'
[09:45:52.010] [INFO] [trace=abc123] [POST /recommend] [apps.recommendation.planner] 플래너 완료: retrieval_core=['드론','화재 진압'] core_keywords=['드론','화재 진압'] role_terms=['평가위원'] action_terms=['추천'] semantic_query='드론 기반 화재 진압 기술 전문가' exclude_orgs=['A기관'] hard_filters={} top_k=5
[09:45:52.080] [INFO] [trace=abc123] [POST /recommend] [apps.search.retriever] 검색 컴파일: mode=keyword_pool_then_hybrid retrieval_keywords=['드론','화재','진압'] doc_types=11 limits={prefetch:100, output:50, chunk_cap:3, retrieval:80}
[09:45:52.095] [INFO] [trace=abc123] [POST /recommend] [apps.search.retriever] 1차 키워드 검색 완료: elapsed_ms=82.1 researcher_pool=37 doc_type_counts={'publication':15,'research_project':9,'researcher_assessor':6,'intellectual_property':2}
[09:45:52.220] [INFO] [trace=abc123] [POST /recommend] [apps.search.retriever] 2차 하이브리드+집계 완료: raw_doc_type_counts={...} aggregated_candidates=24 support_pass=15 support_filtered=9 final=15
[09:45:54.330] [INFO] [trace=abc123] [POST /recommend] [apps.api.main] 추천 응답 준비: retrieved_count=15 recommendations=5 data_gaps=0 top_k_used=5 timers={'plan_ms':880,'search_ms':210,'total_ms':3205}
```

`trace.query_payload`에서 `retrieval_mode`, `retrieval_keywords`, `semantic_query`, `keyword_stage_queries`, `hybrid_stage_queries`, `keyword_stage_candidate_count`, `keyword_stage_doc_type_counts`, `hybrid_stage_raw_doc_type_counts`, `aggregated_candidate_count`, `support_pass_count`, `support_filtered_count`를 확인할 수 있다(벡터 값·전체 payload는 미노출).
