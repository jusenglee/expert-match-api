# Golden Tests

## Scenarios

### 1. Publication-focused query

- Input: `최근 5년 SCIE 논문 실적이 우수한 AI 반도체 평가위원을 추천해주세요`
- Expected:
  - publication-related filters are applied when explicitly requested
  - final evidence includes at least one paper item

### 2. Patent-focused query

- Input: `등록 특허가 많고 사업화 경험이 있는 전문가를 추천해주세요`
- Expected:
  - patent-related evidence is present in the final recommendation

### 3. Project-focused query

- Input: `AI 반도체 과제 수행 경험이 있는 전문가를 추천해주세요`
- Expected:
  - project-related evidence is present in the final recommendation

### 4. Excluded organization query

- Input: `전문가를 추천해주되 A기관 소속은 제외해주세요`
- Expected:
  - exclude-org filtering is applied
  - excluded organizations do not appear in final results

### 5. Partial-evidence query

- Input: `특허와 논문 모두 강점이 있는 전문가를 추천해주세요`
- Expected:
  - data gaps or risks are surfaced when one evidence type is missing
  - recommendations without evidence are removed from final output

### 6. Empty recommendation result

- Input: a query with very strict filters or insufficient evidence
- Expected:
  - `POST /recommend` returns HTTP 200
  - `recommendations=[]`
  - structured reasons appear in `not_selected_reasons` or `data_gaps`

### 7. Zero-hit retrieval

- Input: a query that produces no matching candidates
- Expected:
  - `POST /recommend` returns HTTP 200
  - judge is skipped
  - `recommendations=[]`
  - `not_selected_reasons` explains that no matching candidates were found

### 8. Planner pollution cleaned by verifier

- Input: `NTIS 국가과학기술지식정보서비스 운영 및 개발 사업의 제안 평가위원을 추천해주세요`
- Expected:
  - planner raw output may contain request-role terms
  - verifier removes non-retrieval terms from `core_keywords`
  - cleaned retrieval query excludes request-role terms
  - removed role/action terms are preserved in `task_terms` or trace

### 9. Retrieval skipped on empty verified keywords

- Input: a query whose planner/verifier pipeline cannot produce safe domain keywords
- Expected:
  - Qdrant retrieval is skipped
  - `retrieval_skipped_reason` is present in trace
  - `data_gaps` explains why retrieval did not run

### 10. UI evidence aligned after judge

- Input: a query whose winning expert has multiple papers or projects but only one directly supports the final recommendation reason
- Expected:
  - final UI evidence is rebuilt after judge from the original payload
  - final evidence may differ from judge placeholder evidence or card preview evidence
  - `evidence_resolution_trace` shows aligned evidence selection for the returned expert

## Acceptance Criteria

- Retrieval text is built only from verifier-approved `core_keywords`.
- `intent_summary`, `task_terms`, and deprecated branch hints do not enter retrieval text.
- `searched_branches` always contains `basic`, `art`, `pat`, `pjt`.
- `POST /search/candidates` and `POST /recommend` both return valid response structures.
- Trace exposes `planner_raw_keywords`, `verifier_keywords`, `retrieval_keywords`, `planner_retry_count`, and `retrieval_skipped_reason`.
- Final `RecommendationDecision.evidence` is resolved from original expert payload data, not copied directly from shortlist preview slices.
