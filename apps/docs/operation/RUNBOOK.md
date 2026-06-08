# 운영 매뉴얼 (RUNBOOK) — flat chunk 모델

**문서 버전:** v2.1 (flat payload 정렬, 2026-06-02)

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
  - payload root `chunk_id` 문자열 (`<doc_type>_<숫자doc_id>_c<NNN>`, 예: `paper_100000045256_c000`)이 authoritative evidence id.
  - Point ID는 신규/멱등 컬렉션에서는 `chunk_id`를 권장하나, 운영 컬렉션이 UUID Point ID를 쓰더라도 런타임은 `payload.chunk_id`를 기준으로 evidence를 resolve한다.
  - named vector: `vector_e5i`(1024, Cosine) + `vector_splade`. doc_type은 named vector가 아니라 payload 필터.
  - payload 인덱스: `researcher_id`(keyword), `doc_type`(keyword), `affiliated_organization`/`highest_degree`(keyword), 연구자 공통 count 5종(`publication_count`/`scie_publication_count`/`intellectual_property_count`/`research_project_count`/`researcher_assessor_activity_count`, integer), `doc_date`(datetime, recency). `doc_attrs.*`는 유동 필드라 필터/인덱스 대상이 아니다. ([`DATA_MODEL.md §5`](../architecture/DATA_MODEL.md))
- 시작 시, 실제 선택된 sparse backend에 맞춰 sparse vector modifier(`IDF` 또는 없음)를 자동 확인·복구한다.

### 2.1 적재(ingestion) 불변식 점검

적재는 **외부 제공자 소관**이며(구 `apps/ingest`는 `legacy_v1x/ingest/`로 격리), 적재 데이터는 [`DATA_MODEL.md §6`](../architecture/DATA_MODEL.md)의 불변식을 만족해야 한다. 운영 표본 점검 항목:
- `payload.chunk_id`가 전역 유일하고 evidence id 코덱을 따른다. Point ID와 다르면 `payload.chunk_id`를 기준으로 진단한다.
- 한 `researcher_id`의 모든 chunk에서 flat root 공통 메타(`researcher_name`/`affiliated_organization`/`highest_degree` + count 5종) 동일.
- `doc_type` ∈ 정의된 5종(`paper`/`patent`/`project`/`assessor_activity`/`specialty`).
- `doc_date`는 단일 문자열(`"NONE"` 또는 결측 허용); recency는 datetime 인덱스에 매칭되는 값만 대상.
- dense/sparse가 동일 `chunk_text`에서 생성.
- `chunk_text`에 요청 어투(role/action 불용어) 미포함.

### 2.2 컬렉션 부트스트랩 CLI (WO-B, VPN 필요)

`apps/tools/bootstrap_chunks.py` — `ntis_researcher_chunks` 컬렉션 생성·점검·스모크.

```bash
# 컬렉션 생성(단일 vector_e5i + vector_splade + flat payload 인덱스)
NTIS_QDRANT_COLLECTION_NAME=ntis_researcher_chunks python -m apps.tools.bootstrap_chunks ensure
# 스키마/인덱스 확인 + 레거시 컬렉션 Point 수(보존) 확인
... python -m apps.tools.bootstrap_chunks inspect
# 표본 chunk upsert + query_points(dense/sparse/doc_type 필터/OR recency) + 멱등 확인
... python -m apps.tools.bootstrap_chunks smoke
```

- **blue/green 가드:** 부트스트래퍼는 레거시 `researcher_recommend_proto`를 **절대 재생성/삭제하지 않는다**(`recreate=True`라도 no-op). v2.0 컬렉션만 단일 벡터 스키마로 생성·재생성.
- **스모크 벡터:** 임베딩 서버 없이 스키마·필터 동작만 검증하도록 pseudo 벡터를 쓴다(실제 임베딩 적재는 WO-A/WO-D). 신규 멱등 적재 스모크에서는 Point ID를 `chunk_id`로 쓰면 동일 chunk_id 재upsert 시 Point 수 불변을 확인할 수 있다. 운영 진단과 evidence resolve는 Point ID가 아니라 payload `chunk_id` 기준으로 수행한다.

#### sparse modifier 정합 — PATCH vs drop&recreate (B-5)

- **권고:** 신규 v2.0 컬렉션은 **처음부터 올바른 modifier로 생성**(SPLADE=none / `Qdrant/bm25` fallback=IDF)해 PATCH 의존을 회피한다. 운영 중 backend가 바뀌어(예: 로컬 PIXIE 실패 → bm25 fallback) modifier 기대값이 달라지면 `update_collection`(PATCH) 시도 → readiness가 `sparse_vectors_idf` 불일치로 잡으면 **drop&recreate**(blue/green이라 구 컬렉션 무영향).
- **실측 (TODO, VPN 필요):** 현 환경 Qdrant 버전에서 sparse modifier PATCH가 실시간 교정되는지는 **아직 미측정**(VPN 미연결). `bootstrap_chunks smoke` 1회 실행 후 결과를 이 줄에 기록할 것.

## 3. 준비 상태 점검 (Readiness)

추천 호출 전 순서:
1. `ntis-validate-live` (CLI)
2. `GET /health`
3. `GET /health/ready`

`ntis-validate-live`는 앱 startup과 동일한 sparse backend 선택 로직을 쓴다. local/online PIXIE가 모두 실패해 `Qdrant/bm25` fallback이면 sparse modifier 기대값은 `IDF`다.

`/health/ready`가 `503`/`ready:false`면 점검:
- 컬렉션 존재 여부
- named vector `vector_e5i`/`vector_splade` 존재 여부
- sparse vector modifier 값(IDF/none)
- 필수 payload 인덱스(`researcher_id`, `doc_type`, `doc_date`, `affiliated_organization`, `highest_degree`, 연구자 count 5종 등) 생성 여부
- 유효 샘플 Point 존재 및 구조(flat root: `chunk_id`, `doc_type`, `researcher_id`, 공통 메타 + count 5종, doc_type별 `doc_attrs`)

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
- **Retriever:** 실제 `retrieval_keywords`, `search_query_plan`(dense/raw, sparse joint, sparse concept queries), `researcher_id` 그룹 수, hard filter 통과/탈락 수, post-group relevance gate 활성 개념과 필터링 수, 연구자 집계 후보 수
- **Recommendation:** Top-k 확정, evidence 선별 chunk 수, 사유 생성 배치 수, 최종 추천/데이터 공백 수
- **Fallback:** LLM 오류/JSON 파싱 실패 시 휴리스틱·결정론적 전환 안내
- **Data Gap:** 특정 연구자의 데이터 누락 경고

### 로그 예시 (질의 → 플래너 → 1차 → 2차 → 집계)
```
[09:45:51.125] [INFO] [trace=abc123] [POST /recommend] [apps.api.main] 사용자 질의 수신: endpoint=/recommend top_k=5 exclude_orgs=1 query='드론 화재 진압 평가위원 추천'
[09:45:52.010] [INFO] [trace=abc123] [POST /recommend] [apps.recommendation.planner] 플래너 완료: retrieval_core=['드론','화재 진압'] core_keywords=['드론','화재 진압'] role_terms=['평가위원'] action_terms=['추천'] semantic_query='드론 기반 화재 진압 기술 전문가' exclude_orgs=['A기관'] hard_filters={} top_k=5
[09:45:52.080] [INFO] [trace=abc123] [POST /recommend] [apps.search.retriever] 검색 쿼리 컴파일: mode=grouped_hybrid_rrf retrieval_keywords=['드론','화재','진압'] dense_query='드론 화재 진압 평가위원 추천' sparse_queries={'sparse_joint':'드론 화재 진압'} limits={prefetch:256,group_size:10,groups:80}
[09:45:52.095] [INFO] [trace=abc123] [POST /recommend] [apps.search.retriever] relevance gate: version=v1 active_concepts=[]
[09:45:52.220] [INFO] [trace=abc123] [POST /recommend] [apps.search.retriever] 검색 집계 완료: elapsed_ms=210.0 groups=37 candidates=24 org_filtered=1 final_hits=23
[09:45:54.330] [INFO] [trace=abc123] [POST /recommend] [apps.api.main] 추천 응답 준비: retrieved_count=15 recommendations=5 data_gaps=0 top_k_used=5 timers={'plan_ms':880,'search_ms':210,'total_ms':3205}
```

`trace.query_payload`에서 `retrieval_mode`, `retrieval_keywords`, `search_query_plan`, `semantic_query`, `group_count`, `aggregated_candidate_count`, `relevance_gate_active_concepts`, `relevance_kept_chunk_count`, `relevance_dropped_chunk_count`, `relevance_filtered_candidate_count`, `org_filtered_count`를 확인할 수 있다(벡터 값·전체 payload는 미노출).
