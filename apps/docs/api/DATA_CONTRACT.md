# 데이터 규약 (Data Contract) — chunk 파이프라인

**문서 버전:** v2.0 (chunk 재설계, 2026-05-28)
**데이터 전제:** [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md)

내부 단계 간 구조화된 계약을 정의한다. 외부 노출 응답 규격은 [`API_SPECIFICATION.md`](API_SPECIFICATION.md), 변경 이력은 [`EXTERNAL_API_CHANGELOG.md`](EXTERNAL_API_CHANGELOG.md).

---

## 1. 플래너 출력 (Planner Output)

`PlannerOutput`은 플래너와 검색 엔진 사이의 계약이다.

```json
{
  "intent_summary": "드론 화재 진압 전문가 찾기",
  "retrieval_core": ["화재 진압", "드론"],
  "core_keywords": ["화재 진압", "드론"],
  "semantic_query": "드론 기반 화재 진압 기술 전문가",
  "task_terms": ["전문가 추천"],
  "role_terms": ["전문가"],
  "action_terms": ["추천"],
  "hard_filters": {},
  "include_orgs": [],
  "exclude_orgs": [],
  "intent_flags": {},
  "top_k": 5
}
```

**규칙:**
- 하나의 JSON 객체만 반환한다(첫 글자 `{`, 끝 글자 `}`). 마크다운/`<thinking>` 금지.
- `core_keywords`/`retrieval_core`는 검색에 안전한 도메인 명사·명사구만.
- `role_terms`/`action_terms`("평가위원", "추천" 등)는 검색 텍스트에 넣지 않는다(벡터 오염 방지).
- `hard_filters`는 허용 키만 사용한다(아래 §1.1). 미허용 키는 Pydantic 검증에서 거부.
- 명시 요청 파라미터(`top_k`, `filters_override`, `exclude_orgs`)는 자연어 추출보다 우선.
- 출력이 무효이거나 `core_keywords`가 비면 1회 재시도, 그래도 비면 검색 생략.

### 1.1 `hard_filters` 허용 키 (chunk 모델)

| 키 | 의미 | 적용 대상 |
|---|---|---|
| `highest_degree` | 학위 | `researcher_meta.highest_degree` |
| `recent_years` | 최근 N년 | `event_year >= 올해-N` (해당 doc_type, 여러 개면 OR) |
| `recent_doc_types` | 최근성 적용 family/doc_type | recency 결합 대상 지정 |
| `publication_count_min` | 최소 논문 수 | `researcher_meta.publication_count` |
| `scie_publication_count_min` | 최소 SCIE 수 | `researcher_meta.scie_publication_count` |
| `intellectual_property_count_min` | 최소 특허 수 | `researcher_meta.intellectual_property_count` |
| `research_project_count_min` | 최소 과제 수 | `researcher_meta.research_project_count` |
| `researcher_assessor_count_min` | 최소 평가위원 활동 | `researcher_meta.researcher_assessor_count` |
| `expert_assessor_count_min` | 최소 전문평가 활동 | `researcher_meta.expert_assessor_count` |
| `journal_class` | 등재구분 | `publication.domain_attrs.journal_class` |

> v1.x의 `art_recent_years`/`pat_recent_years`/`pjt_recent_years`는 통합 `recent_years` + `recent_doc_types`로 대체된다(`event_year` 단일 기준). `*_cnt_min` → `*_count_min`으로 명칭 통일.

---

## 2. 쿼리 빌더 규약 (Query Builder Contract)

`QueryTextBuilder`는 플래너 출력으로 검색 텍스트를 만든다.
- 원본 사용자 질의, `intent_summary`, `role_terms`, `action_terms`는 검색 텍스트에서 제외.
- 1단계 sparse 키워드 텍스트는 `retrieval_core`/`core_keywords` 결합으로 생성.
- 2단계 하이브리드 텍스트는 `semantic_query`가 있으면 사용, 없으면 동일 키워드 텍스트.
- doc_type(또는 family) 경로는 동일 기본 텍스트를 공유한다(경로별 별도 hint 생성 안 함).

---

## 3. 검색 규약 (Retrieval Contract)

`QdrantHybridRetriever`는 모드 `keyword_pool_then_hybrid` 고정.

- **1단계:** `sparse_splade` 키워드 검색 → `researcher_id` 후보 풀 수집(중복 제거).
- **2단계:** 풀을 `researcher_id MatchAny`로 제한 → doc_type 경로별 dense+sparse `prefetch` → `FusionQuery(RRF)` (equal). chunk hit 산출.
- **집계:** chunk hit을 `researcher_id`로 묶어 RRF 누적(연구자 점수). doc_type별 상위 N chunk만 기여(캡), prior 적용(기본 equal), 연구자당 1건 dedupe.
- 1단계 풀이 비면 2단계 생략, `keyword_stage_candidate_count=0`.
- `trace.query_payload`는 1차 풀 크기, 1차 doc_type별 count, 2차 doc_type별 raw count, 집계 후보 수, hard filter 통과/탈락 수를 포함한다.
- `trace.query_payload`는 검색 키워드/텍스트만 노출하며 dense/sparse 벡터 값과 전체 payload는 노출하지 않는다.

---

## 4. 후보 카드 규약 (Candidate Card Contract)

`CandidateCard`는 `/search/candidates`와 `/recommend` 공통 내부 계약.
- 카드 순서는 검색 정렬 순서를 엄격히 따른다.
- `rank_score`는 RRF 집계 점수를 0~100으로 정규화한 값.
- 각 후보는 family별 보유 플래그(`doc_type_coverage`)와 `researcher_meta` 기반 카운트, 매칭 chunk 미리보기(`chunk_id` 포함)를 보유.
- `/recommend` 직전, 후보 내부 chunk을 `core_keywords`/query 관련도로 재랭크한다.

---

## 5. evidence 선별 규약 (Evidence Selection Contract)

`EvidenceSelector` Protocol. 구현: `CrossEncoderEvidenceSelector`(기본, 모델 부재 시 lexical 강등) / `KeywordEvidenceSelector`(순수 lexical).

- 입력: 한 후보의 매칭 chunk 집합.
- 출력: family별 관련도 상위 chunk 묶음(`RelevantEvidenceBundle`). 각 항목은 `chunk_id`, `doc_type`, `title`, `event_date`, `snippet`, `match_score`를 보유.
- 캡: family별 상위 N (기본 10, 운영 튜닝 가능). 후보 순위에는 영향을 주지 않는다.

---

## 6. 추천 사유 생성 규약 (Reason Generation Contract)

`OpenAICompatReasonGenerator`는 정렬된 상위 K명만 수신한다.

**출력 스키마:**
```json
{
  "items": [
    {
      "expert_id": "M1006328",
      "fit": "높음",
      "recommendation_reason": "화재 대응 관련 논문·과제와 관련 평가위원 활동 이력이 있음.",
      "selected_evidence_ids": ["PUB_M1006328_0001_c0", "PJT_M1006328_0001_c0"],
      "risks": []
    }
  ],
  "data_gaps": []
}
```

**규칙:**
- LLM은 후보 순위를 바꾸지 않고, 후보를 누락시키지 않으며, 새 ID를 만들지 않는다.
- `selected_evidence_ids`는 제공된 풀의 **`chunk_id`를 정확히 복사**한다. 풀에 없거나 형식이 깨진 id는 무효 처리되고 결정론적 fallback으로 evidence를 조립한다.
- 후보별 evidence는 LLM 전달 전 관련도 재랭크 + family별 캡이 적용된다.
- 심사는 최대 5명 단위 순차 배치로 진행한다.

> v1.x의 evidence id 형식 `paper:N`/`project:N`/`patent:N`은 폐기되고 `chunk_id`로 통일된다. 상세 정책은 [`REASONER_RUNTIME_POLICY.md`](REASONER_RUNTIME_POLICY.md).
