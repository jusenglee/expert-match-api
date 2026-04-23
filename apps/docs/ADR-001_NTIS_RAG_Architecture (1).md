# ADR-001: NTIS 도메인 맞춤형 고성능 RAG 검색 아키텍처 도입

- 상태: Proposed
- 작성일: 2026-04-22
- 대상 시스템: NTIS 기반 전문가/평가위원 추천 검색 서비스
- 의사결정 유형: Architecture / Retrieval Pipeline / Ranking
- 관련 컴포넌트: Planner LLM, Qdrant, Dense Retriever, Sparse Retriever, Cross-Encoder, Redis Cache, Evaluation Pipeline

## 1. Context

사용자가 “화재 관련 컨소시엄 평가위원을 찾아주세요”와 같이 자연어로 질의할 때, 기존 단일 Dense Retrieval 방식은 실제 전문 분야보다 “평가위원”, “전문가”, “연구자”와 같은 직함성 단어에 끌려 결과를 반환하는 문제가 있다.

NTIS 도메인의 코퍼스는 이미 전문가, 연구자, 과제 수행 이력 중심으로 구성되어 있어 직함성 단어가 구분력을 갖지 못한다. 또한 단일 임베딩 검색은 필수 조건의 포함 여부를 보장하지 못하므로, 사용자의 핵심 의도와 다른 후보가 상위에 노출될 수 있다.

본 ADR은 NTIS 전문가 추천 검색의 정확도, 안정성, 설명 가능성, 운영 확장성을 개선하기 위해 Planner 기반 질의 구조화, Qdrant 필터/하이브리드 후보 생성, HyDE 보강, Cross-Encoder 재정렬, 사람 단위 집계를 결합한 검색 아키텍처를 채택할지 결정한다.

## 2. Decision

NTIS 도메인 전문가/평가위원 추천 검색에는 다음 구조를 채택한다.

```text
User Query
  -> Planner LLM
  -> Qdrant Candidate Generation
       - Explicit hard filters
       - Sparse keyword/synonym search
       - Dense search using semantic_query and bounded HyDE
       - Fusion by RRF or weighted score
  -> Person-level Aggregation
  -> Cross-Encoder Reranking
  -> Business Rules and Explanation
  -> Top-K Expert Recommendations
```

핵심 결정은 다음과 같다.

1. 원본 질의를 그대로 검색하지 않고 Planner LLM이 intent, hard_filters, domain_terms, soft_preference_terms, semantic_query, bounded_hyde_document, relaxation_policy를 구조화한다.
2. Qdrant 필터는 명시적이고 구조화된 제약에만 강하게 적용한다. 도메인 키워드는 단일 must AND가 아니라 동의어 그룹, sparse 검색, score boost, fallback 정책으로 처리한다.
3. Dense retrieval에는 원본 질의가 아니라 semantic_query 또는 제한된 HyDE 문서를 사용한다.
4. Sparse retrieval을 dense retrieval과 함께 수행하고, RRF 또는 weighted fusion으로 Top 100~200 후보를 만든다.
5. 검색 결과는 chunk/project 단위에서 끝내지 않고 person_id 기준으로 집계한다.
6. Cross-Encoder는 Top 30~100 수준의 후보에만 적용해 최종 관련성 점수를 산출한다.
7. 최종 추천에는 근거 snippet, 최근성, 역할 가중치, 공동연구/컨소시엄 경험, 이해상충 가능성 등을 반영한다.
8. 후보 수가 부족할 경우 조건 완화 정책을 단계적으로 적용하고, 사용자에게 완화 사실을 설명한다.

## 3. Decision Drivers

- Precision: 상위 추천 결과가 실제 사용자의 도메인 의도와 맞아야 한다.
- Recall: 좋은 전문가를 hard keyword filter 때문에 놓치지 않아야 한다.
- Stability: 같은 질의에 대해 결과 변동성이 낮아야 한다.
- Latency: 운영 환경에서 p95 응답 지연을 관리할 수 있어야 한다.
- Explainability: 추천 근거를 사용자에게 제시할 수 있어야 한다.
- Scalability: 부서, 직급, 연구 연차, 이해상충 등 조건 추가에 쉽게 대응해야 한다.

## 4. Considered Options

| Option | Description | Pros | Cons | Decision |
|---|---|---|---|---|
| A. Dense only | 원본 질의를 임베딩해 Qdrant 유사도 검색만 수행 | 구현 단순, 빠름 | 직함 노이즈와 semantic drift에 취약, 필수 조건 보장 불가 | Reject |
| B. Keyword hard filter + Dense | must_keywords를 AND 필터로 적용 후 dense 검색 | 안정성 향상, 노이즈 감소 | 동의어/표현 차이로 recall 손실 큼 | Partial |
| C. HyDE + Dense | 가상 문서를 생성해 dense 검색 | 짧은 질의 보강, 문맥 검색 개선 | HyDE 과구체화 시 hallucinated evidence 위험 | Partial |
| D. Hybrid + Cross-Encoder | sparse+dense 후보 생성 후 Cross-Encoder 재정렬 | precision/recall 균형, 최종 품질 우수 | 구현 복잡도, 모델 추론 비용 증가 | Accept |
| E. Fine-tuned Reranker only | 도메인 학습된 reranker 중심으로 처리 | 최종 정밀도 잠재력 높음 | 후보 생성 실패 시 보완 불가, 학습 데이터 필요 | Future |

## 5. Architecture

### 5.1 Planner LLM

Planner는 사용자의 원문 질의를 검색 실행 계획으로 변환한다. Planner 출력은 반드시 스키마 검증을 통과해야 하며, 실패 시 fallback parser 또는 보수적 기본 검색 계획을 사용한다.

예상 출력 스키마:

```json
{
  "intent": "expert_recommendation",
  "semantic_query": "화재·소방·재난안전 분야의 컨소시엄/공동연구 경험이 있는 평가위원 후보",
  "bounded_hyde_document": "화재, 소방, 재난안전, 방재, 연소 분야의 연구 수행 이력이 있으며 다기관 공동연구나 산학연 컨소시엄 형태의 국가 R&D 과제에 참여한 전문가 프로필.",
  "hard_filters": {
    "candidate_pool": "expert",
    "eligible_for_panel": true
  },
  "domain_term_groups": [
    {
      "name": "fire_domain",
      "mode": "at_least_one",
      "terms": ["화재", "소방", "재난안전", "방재", "연소", "산불", "화재안전"]
    }
  ],
  "soft_preference_groups": [
    {
      "name": "collaboration_experience",
      "terms": ["컨소시엄", "공동연구", "다기관", "협동연구", "산학연", "참여기관"]
    }
  ],
  "noise_terms": ["평가위원", "전문가", "연구자", "교수"],
  "relaxation_policy": {
    "min_candidates": 100,
    "fallback_order": [
      "domain_group_at_least_one",
      "remove_collaboration_hard_filter",
      "expand_domain_synonyms",
      "dense_only_with_domain_boost"
    ]
  }
}
```

Planner 운영 원칙:

- temperature는 0 또는 0.1로 고정한다.
- 원문에 없는 인명, 기관명, 과제명, 연도, 수상명, 직책을 HyDE가 생성하지 못하게 한다.
- “평가위원”, “전문가”, “연구자”는 검색 핵심어가 아니라 역할/목적 또는 noise term으로 처리한다.
- 출력은 JSON Schema로 검증하고, schema_version과 prompt_version을 함께 관리한다.

### 5.2 Qdrant Payload and Indexing

문서 원문 전체를 필터 대상으로 삼지 않고, 필터/검색/설명 목적별 payload를 분리한다.

| Field | Type | Purpose | Example |
|---|---|---|---|
| person_id | keyword | 사람 단위 집계 키 | P12345 |
| domain_tags | keyword array | 도메인 필터/boost | 화재, 소방, 재난안전 |
| collaboration_tags | keyword array | 공동연구/컨소시엄 선호 조건 | 공동연구, 다기관, 산학연 |
| project_roles | keyword array | 역할 가중치 | 주관책임자, 공동연구책임자 |
| profile_summary | text/vector source | dense embedding 대상 | 최근 5년 화재안전 연구 수행 |
| evidence_text | text/sparse source | sparse 검색 및 근거 추출 | 과제명, 요약, 키워드 |
| recent_project_years | integer array | 최근성 점수 | 2021, 2022, 2024 |
| eligible_for_panel | boolean | 명시적 hard filter | true |

권장 사항:

- 후보 수를 크게 줄이는 payload에는 Qdrant payload index를 생성한다.
- domain_tags, collaboration_tags는 원문에서 추출한 정규화 태그로 관리한다.
- evidence_text는 sparse retrieval 및 사용자 설명용 snippet 추출에 활용한다.

### 5.3 Candidate Generation

후보 생성은 hard filter, sparse retrieval, dense retrieval을 병렬 또는 순차적으로 수행한다.

```text
Candidate Generation
  1. Apply explicit hard filters
  2. Sparse retrieval using domain/synonym terms
  3. Dense retrieval using semantic_query embedding
  4. Dense retrieval using bounded HyDE embedding
  5. Fuse candidates by RRF or weighted score
  6. Keep Top 100~200 candidates
```

하드 필터는 다음과 같은 명시 조건에만 사용한다.

- 후보군 유형: expert/researcher
- 평가위원 후보 자격 여부
- 사용자가 명시한 지역, 기관 유형, 부서, 직급, 연차 등 구조화 가능한 조건
- 법적/운영 정책상 반드시 제외해야 하는 조건

도메인 단어는 다음 방식으로 처리한다.

- 동의어 그룹 중 at least one 조건
- sparse retrieval keyword boost
- dense retrieval semantic_query 반영
- reranking feature 반영
- 후보 부족 시 relaxation policy 적용

### 5.4 Person-level Aggregation

검색은 chunk 또는 project 단위로 수행될 수 있지만, 최종 추천은 person_id 기준으로 집계한다.

```text
person_score_pre_rerank =
  max(candidate_retrieval_score)
  + evidence_count_boost
  + recent_activity_boost
  + role_weight_boost
  + collaboration_experience_boost
```

집계 시 동일 인물이 여러 chunk로 중복 노출되지 않도록 하고, 상위 근거 snippet을 2~5개 유지한다.

### 5.5 Cross-Encoder Reranking

Cross-Encoder 입력은 원문 질의가 아니라 structured semantic query와 후보자의 evidence/profile을 사용한다.

권장 query format:

```text
검색 목적: 화재 관련 평가위원 후보 추천
핵심 전문성: 화재, 소방, 재난안전, 방재, 연소
선호 경험: 컨소시엄, 공동연구, 다기관 국가 R&D 과제 참여
제외할 노이즈: 단순히 평가위원이라는 직함만 있는 문서
판단 기준: 실제 연구/과제 수행 이력과 도메인 적합성
```

권장 document format:

```text
후보자 프로필 요약: ...
관련 과제 근거: ...
역할: 주관책임자/공동연구책임자/참여연구원
최근성: 최근 5년 내 수행 여부
공동연구 근거: ...
```

Cross-Encoder는 Top 30~100 후보에 대해서만 batch scoring한다. 최종 점수는 Cross-Encoder 점수와 비즈니스 점수를 조합한다.

```text
final_person_score =
  0.50 * cross_encoder_score
+ 0.15 * domain_evidence_score
+ 0.10 * collaboration_score
+ 0.10 * recent_activity_score
+ 0.10 * role_weight_score
+ 0.05 * diversity_or_policy_score
- conflict_of_interest_penalty
```

### 5.6 Fallback and Relaxation

검색 후보가 부족하거나 0건이면 다음 순서로 완화한다.

| Step | Relaxation | User-facing Explanation |
|---|---|---|
| 1 | collaboration 조건을 hard에서 soft로 이동 | 컨소시엄 조건을 선호 조건으로 완화했습니다. |
| 2 | domain exact term을 synonym group으로 확장 | 화재와 유사한 소방/방재/재난안전 표현까지 확장했습니다. |
| 3 | sparse+dense hybrid에서 dense 비중 증가 | 표현 차이를 보완하기 위해 의미 기반 후보를 추가했습니다. |
| 4 | 최소 필수 조건만 유지 | 엄격 조건 결과가 부족해 핵심 전문성 중심으로 추천했습니다. |

완화가 적용된 경우 최종 응답에 반드시 완화 사실을 표시한다.

## 6. Operational Controls

### 6.1 Consistency

- Planner LLM temperature를 0 또는 0.1로 고정한다.
- Planner output은 JSON Schema로 검증한다.
- Redis 또는 동등한 인메모리 캐시를 사용한다.
- 캐시 키에는 normalized_query, model_version, prompt_version, schema_version, synonym_dictionary_version을 포함한다.

```text
cache_key = hash(
  normalized_query,
  planner_model_version,
  prompt_version,
  schema_version,
  synonym_dictionary_version
)
```

### 6.2 Latency Budget

| Component | Target |
|---|---:|
| Planner LLM with cache hit | < 50 ms |
| Planner LLM with cache miss | 모델별 SLA 설정 |
| Qdrant candidate generation | p95 < 500 ms |
| Cross-Encoder reranking Top 50 | p95 < 1,500 ms |
| End-to-end | p95 < 3,000 ms |

### 6.3 Failure Handling

| Failure | Handling |
|---|---|
| Planner schema validation fail | 기본 semantic_query + conservative filters 사용 |
| Qdrant candidate count = 0 | relaxation policy 적용 |
| Cross-Encoder timeout | fusion score 기반 degraded response 반환 |
| Redis unavailable | LLM 직접 호출, circuit breaker 적용 |
| 모델 서버 unavailable | sparse+dense fallback 결과 반환 |

## 7. Consequences

### Positive

- 직함성 단어로 인한 semantic drift를 줄일 수 있다.
- hard filter, sparse, dense, reranking의 역할이 분리되어 운영 조정이 쉽다.
- Cross-Encoder를 후보군에만 적용하므로 GPU 비용을 통제할 수 있다.
- person-level aggregation으로 전문가 추천 서비스의 목적에 맞는 결과를 제공할 수 있다.
- 근거 snippet 기반 설명이 가능해 사용자 신뢰도가 높아진다.

### Negative / Trade-offs

- 단일 dense 검색보다 구현과 운영 복잡도가 증가한다.
- payload 정규화, synonym dictionary, 평가셋 구축이 필요하다.
- Cross-Encoder 추론 비용과 배치 운영 전략이 필요하다.
- Planner 품질에 따라 검색 계획이 흔들릴 수 있으므로 schema validation과 캐시 버전관리가 필수다.

### Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| must keyword 과강제 | 좋은 후보 누락 | hard filter와 soft preference 분리 |
| HyDE hallucination | 잘못된 검색 방향 | bounded HyDE, 구체 사실 생성 금지 |
| Cross-Encoder domain mismatch | 재정렬 품질 저하 | 도메인 평가셋 구축 및 fine-tuning 검토 |
| 후보 중복 | 같은 사람이 반복 노출 | person_id aggregation |
| 응답 지연 | 사용자 경험 저하 | Top-K 제한, batch scoring, cache, timeout fallback |

## 8. Implementation Plan

| Phase | Scope | Output |
|---|---|---|
| Phase 0 | 로그 분석 및 gold set 구축 | 대표 질의 100~300개, 적합 후보 라벨 |
| Phase 1 | payload 정규화 및 Qdrant index 구성 | domain_tags, collaboration_tags, evidence_text |
| Phase 2 | Planner schema/prompt 구현 | structured query plan |
| Phase 3 | sparse+dense hybrid candidate generation | Top 100~200 후보군 |
| Phase 4 | person-level aggregation | 중복 제거 및 근거 snippet |
| Phase 5 | Cross-Encoder reranking | Top 5~10 추천 결과 |
| Phase 6 | evaluation/monitoring dashboard | nDCG, Recall, latency, failure rate |
| Phase 7 | domain fine-tuning 검토 | 개선된 reranker 또는 embedding model |

## 9. Evaluation Plan

### 9.1 Offline Evaluation

| Experiment | Purpose |
|---|---|
| Baseline dense only | 현재 문제 재현 |
| keyword filter + dense | 필터 효과 확인 |
| HyDE dense | HyDE 단독 효과 확인 |
| sparse + dense hybrid | lexical 보강 효과 확인 |
| hybrid + Cross-Encoder | 최종 성능 확인 |
| hybrid + fine-tuned Cross-Encoder | 도메인 튜닝 효과 확인 |

### 9.2 Metrics

| Metric | Meaning | Target Direction |
|---|---|---|
| Recall@100 | 적합 후보가 후보군에 포함되는지 | Higher |
| Precision@5 | 최종 Top 5 품질 | Higher |
| nDCG@10 | 상위 랭킹 품질 | Higher |
| MRR | 첫 적합 후보 위치 | Higher |
| No-result rate | 필터 과강도 여부 | Lower |
| Candidate count distribution | 질의별 후보 수 안정성 | Stable |
| p50/p95 latency | 운영 응답 속도 | Lower |
| Planner parse failure rate | 구조화 출력 안정성 | Lower |
| Relaxation rate | 조건 완화 빈도 | Monitor |

## 10. Monitoring

운영 대시보드는 다음 항목을 추적한다.

- query volume, cache hit rate
- planner validation success/failure
- candidate count before/after fusion
- relaxation applied rate
- cross-encoder latency and timeout rate
- final Top-K click/selection rate
- user feedback positive/negative rate
- no-result and low-confidence response rate

## 11. Rollout Strategy

1. Shadow mode: 기존 검색 결과와 신규 파이프라인 결과를 동시에 생성하되 사용자에게는 기존 결과만 노출한다.
2. Internal review: 도메인 담당자가 대표 질의별 Top 10 결과를 비교 평가한다.
3. A/B test: 일부 사용자 또는 내부 사용자에게 신규 결과를 노출한다.
4. Gradual rollout: 질의 유형별로 적용 범위를 확대한다.
5. Full rollout: 모니터링 지표가 기준을 충족하면 기본 검색 파이프라인으로 전환한다.

## 12. Open Questions

- 평가위원 후보 자격을 판단할 수 있는 신뢰 가능한 payload가 존재하는가?
- 이해상충, 소속기관 제한, 최근 참여 과제 제한 등 정책 조건은 어디에서 관리할 것인가?
- Cross-Encoder는 범용 한국어 모델로 시작할 것인가, 도메인 fine-tuning 모델을 별도로 학습할 것인가?
- 최종 사용자에게 후보별 근거를 몇 개까지 노출할 것인가?
- 동의어 사전은 수동 관리, LLM 생성, 로그 기반 자동 확장 중 어떤 방식을 채택할 것인가?

## 13. Final Recommendation

본 ADR은 Option D, 즉 Planner 기반 질의 구조화와 Qdrant hybrid candidate generation, person-level aggregation, Cross-Encoder reranking을 결합한 구조를 채택한다.

단, `must_keywords`를 그대로 AND hard filter로 적용하는 방식은 채택하지 않는다. 하드 필터는 명시적 제약에만 사용하고, 도메인 키워드는 동의어 그룹, sparse retrieval, score boost, fallback policy로 처리한다.

최종 검색 단위는 문서가 아니라 전문가(person)이며, 추천 결과는 반드시 근거 snippet과 함께 제공한다.
